"""Content-addressed private checkpoints; preserve raw bytes without daily copies."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from pathlib import Path
import hashlib,json,os,re,shutil,tarfile,gzip,threading,time

ARCHIVE='ayushverma1988/v10-multibagger-archive'

def is_raw(path):return any(part in {'raw','history_raw','filing_raw'} for part in Path(path).parts)

def object_path(digest):
    if not isinstance(digest,str) or not re.fullmatch('[0-9a-f]{64}',digest):
        raise ValueError('Invalid checkpoint object digest')
    # Separate namespace: even adding a subdirectory to the full legacy
    # objects directory could exceed the Hub's 10,000 entries limit.
    return f'v13/object_shards/{digest[:2]}/{digest[2:4]}/{digest}.bin'

def validate_object(item):
    digest=item['sha256'];sharded=object_path(digest)
    if item['object_path'] not in {sharded,'v13/objects/'+digest+'.bin'}:
        raise ValueError('Object identity mismatch')

def manifest(root):
    root=Path(root);old=root/'remote_manifest.json'
    prior=json.loads(old.read_text())['files'] if old.exists() else []
    for item in prior:validate_object(item)
    known={r['sha256']:r['object_path'] for r in prior}
    files={r['path']:r for r in prior}
    for p in sorted(root.rglob('*')):
        if not p.is_file() or p.name in {'state_manifest.json','remote_manifest.json'}:continue
        rel=str(p.relative_to(root));digest=hashlib.sha256(p.read_bytes()).hexdigest()
        files[rel]={'path':rel,'sha256':digest,'bytes':p.stat().st_size,'object_path':known.get(digest,object_path(digest)),'raw_source':is_raw(rel)}
    return {'format':'V13-content-addressed-checkpoint-1','saved_at_utc':datetime.now(timezone.utc).isoformat(),'files':sorted(files.values(),key=lambda r:r['path'])}

def materialize_one(item,root,fetch):
    rel=Path(item['path'])
    if rel.is_absolute() or '..' in rel.parts:raise ValueError('Unsafe checkpoint path')
    validate_object(item)
    raw=Path(fetch(item['object_path'])).read_bytes()
    if len(raw)!=item['bytes'] or hashlib.sha256(raw).hexdigest()!=item['sha256']:raise ValueError('Checkpoint object integrity failure')
    target=Path(root)/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)

def build_derived_bundle(root,data,path):
    """One deterministic private restart pack; original raw objects stay separate."""
    root=Path(root);path=Path(path);active=[r for r in data['files'] if not r['raw_source']]
    with path.open('wb') as dest,gzip.GzipFile(filename='',mode='wb',fileobj=dest,mtime=0,compresslevel=1) as gz,tarfile.open(fileobj=gz,mode='w|') as tar:
        for item in active:
            rel=Path(item['path'])
            if rel.is_absolute() or '..' in rel.parts:raise ValueError('Unsafe derived bundle path')
            source=root/rel
            if not source.is_file() or source.stat().st_size!=item['bytes'] or hashlib.sha256(source.read_bytes()).hexdigest()!=item['sha256']:
                raise ValueError('Derived bundle input differs from checkpoint manifest')
            info=tarfile.TarInfo(item['path']);info.size=item['bytes'];info.mode=0o644;info.mtime=0
            with source.open('rb') as src:tar.addfile(info,src)
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    return {'sha256':digest,'bytes':path.stat().st_size,'object_path':object_path(digest),
      'files':len(active),'uncompressed_bytes':sum(r['bytes'] for r in active)}

def materialize_bundle(bundle,data,root,fetch):
    validate_object(bundle);source=Path(fetch(bundle['object_path']))
    if source.stat().st_size!=bundle['bytes'] or hashlib.sha256(source.read_bytes()).hexdigest()!=bundle['sha256']:
        raise ValueError('Derived bundle digest or size mismatch')
    expected={r['path']:r for r in data['files'] if not r['raw_source']};seen=set();root=Path(root)
    if len(expected)!=bundle['files']:raise ValueError('Derived bundle member count mismatch')
    with tarfile.open(source,'r|gz') as tar:
        for member in tar:
            rel=Path(member.name)
            if not member.isfile() or rel.is_absolute() or '..' in rel.parts or member.name not in expected or member.name in seen:
                raise ValueError('Unexpected, duplicate or unsafe derived bundle member')
            item=expected[member.name];validate_object(item)
            raw=tar.extractfile(member).read()
            if len(raw)!=item['bytes'] or hashlib.sha256(raw).hexdigest()!=item['sha256']:
                raise ValueError('Derived bundle member integrity failure')
            target=root/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw);seen.add(member.name)
    if seen!=set(expected):raise ValueError('Derived bundle omitted checkpoint files')
    return len(seen)

class VerifiedObjectDownloader:
    """Legacy migration GETs with known hashes; no per-file metadata HEADs.

    The authenticated resolver is documented by the Hub. Requests are paced
    below its free resolver allowance; 429 responses pause the shared queue
    according to the server's reset header. Credentials are never logged.
    """
    def __init__(self,items,folder,token,requests_per_second=8):
        self.items={r['object_path']:r for r in items};self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=True)
        self.token=token;self.interval=1/requests_per_second;self.lock=threading.Lock();self.next=0.;self.blocked=0.;self.local=threading.local();self.done=0
        self.object_locks={path:threading.Lock() for path in self.items}
    def __call__(self,path):
        with self.object_locks[path]:return self.download(path)
    def download(self,path):
        from huggingface_hub import hf_hub_url
        import requests
        item=self.items[path];validate_object(item);out=self.folder/(item['sha256']+'.bin')
        if out.exists() and out.stat().st_size==item['bytes'] and hashlib.sha256(out.read_bytes()).hexdigest()==item['sha256']:return out
        if not hasattr(self.local,'session'):self.local.session=requests.Session()
        last_error='rate limit'
        for attempt in range(5):
            with self.lock:
                now=time.monotonic();slot=max(now,self.next,self.blocked);self.next=slot+self.interval
            while time.monotonic()<slot:time.sleep(max(0,min(1,slot-time.monotonic())))
            url=hf_hub_url(ARCHIVE,path,repo_type='dataset')
            try:
                with self.local.session.get(url,headers={'Authorization':'Bearer '+self.token},timeout=(8,60),stream=True) as response:
                    if response.status_code==429:
                        match=re.search(r';\s*t=(\d+)',response.headers.get('RateLimit',''))
                        delay=int(match.group(1)) if match else int(response.headers.get('Retry-After','300'))
                        with self.lock:self.blocked=max(self.blocked,time.monotonic()+max(delay,1))
                        last_error='rate limit';continue
                    if response.status_code in {408,425,500,502,503,504}:
                        last_error='HTTP '+str(response.status_code)
                        time.sleep(min(2**attempt,15));continue
                    if response.status_code!=200:raise RuntimeError('Private checkpoint resolver HTTP '+str(response.status_code))
                    temp=out.with_suffix('.part');digest=hashlib.sha256();size=0
                    with temp.open('wb') as target:
                        for chunk in response.iter_content(1024*1024):
                            size+=len(chunk)
                            if size>item['bytes']:raise ValueError('Checkpoint resolver exceeded recorded size')
                            digest.update(chunk);target.write(chunk)
                    if size!=item['bytes'] or digest.hexdigest()!=item['sha256']:raise ValueError('Checkpoint resolver integrity failure')
                    temp.replace(out)
            except requests.RequestException as exc:
                # Exception text may contain a signed redirect URL; retain
                # only its class. Fresh attempts overwrite partial bytes and
                # must pass the same recorded size/hash before acceptance.
                last_error=type(exc).__name__;time.sleep(min(2**attempt,15));continue
            with self.lock:
                self.done+=1
                if self.done%250==0:print('Verified checkpoint objects restored',self.done,flush=True)
            return out
        raise RuntimeError('Private checkpoint unavailable after bounded retries: '+last_error)

def restore(root):
    from huggingface_hub import HfApi,hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError
    root=Path(root);token=os.environ.get('HF_ARCHIVE_TOKEN')
    if not token:raise RuntimeError('Required existing private archive credential is unavailable')
    api=HfApi(token=token)
    if not api.repo_info(ARCHIVE,repo_type='dataset').private:raise RuntimeError('Checkpoint archive must be private')
    def fetch(path):return hf_hub_download(ARCHIVE,path,repo_type='dataset',token=token)
    try:path=fetch('v13/state/latest_manifest.json')
    except EntryNotFoundError:
        # Migrate the first deployed worker's archive without losing forecasts.
        try:legacy=fetch('v13/state/latest.tar.gz')
        except EntryNotFoundError:return False
        root.mkdir(parents=True,exist_ok=True)
        with tarfile.open(legacy,'r:gz') as tar:tar.extractall(root,filter='data')
        from v13_runner import verify_manifest
        verify_manifest(root);return True
    data=json.loads(Path(path).read_text())
    if data.get('format')!='V13-content-addressed-checkpoint-1':raise ValueError('Unknown private checkpoint format')
    root.mkdir(parents=True,exist_ok=True)
    active=[r for r in data['files'] if not r['raw_source']]
    # Original source objects remain immutable in the private archive. They
    # are fetched by hash only when auditing; inference restores derived data.
    started=time.monotonic()
    if data.get('derived_state_bundle'):
        count=materialize_bundle(data['derived_state_bundle'],data,root,fetch)
        method='HASH_VERIFIED_DERIVED_STATE_BUNDLE'
    else:
        paced=VerifiedObjectDownloader(active,root.parent/'v13_restore_objects',token)
        # Resolver/CDN round trips can exceed one second. More bounded I/O
        # workers fill that latency while the shared eight-request/second
        # limiter and server-directed 429 pauses remain unchanged.
        print('Checkpoint restore plan',json.dumps({'files':len(active),'workers':32,'requests_per_second':8}),flush=True)
        with ThreadPoolExecutor(max_workers=32) as pool:
            list(pool.map(lambda r:materialize_one(r,root,paced),active))
        count=len(active);method='PACED_LEGACY_OBJECT_MIGRATION'
    (root/'remote_manifest.json').write_text(json.dumps(data,indent=2))
    (root/'state_manifest.json').write_text(json.dumps({'files':active},indent=2))
    print('Checkpoint restore completed',json.dumps({'method':method,'files':count,'seconds':round(time.monotonic()-started,2)}),flush=True)
    return True

def save(root):
    from huggingface_hub import HfApi,CommitOperationAdd
    from v13_checkpoint_writes import MAX_BATCH,UploadProgress,spec,write,recovery_delta
    root=Path(root);token=os.environ['HF_ARCHIVE_TOKEN'];api=HfApi(token=token)
    if not api.repo_info(ARCHIVE,repo_type='dataset').private:raise RuntimeError('Checkpoint archive must be private')
    data=manifest(root);old=root/'remote_manifest.json'
    base_raw=old.read_bytes() if old.exists() else json.dumps({'format':'V13-content-addressed-checkpoint-1','files':[]}).encode()
    previous=json.loads(base_raw)
    known={r['sha256'] for r in previous['files']}
    objects={r['sha256']:r for r in data['files'] if r['sha256'] not in known}
    entries=list(objects.values())
    progress=UploadProgress(root.parent/'v13_checkpoint_upload_progress.json',ARCHIVE)
    pending=[r for r in entries if not progress.contains(r)]
    try:
        for start in range(0,len(pending),MAX_BATCH):
            specs=[spec(r['object_path'],root/r['path'],r['sha256'],r['bytes']) for r in pending[start:start+MAX_BATCH]]
            write(api,ARCHIVE,specs,'V13 immutable source/model objects',CommitOperationAdd,progress.confirm)
        bundle_path=root.parent/'v13_derived_state_bundle.tar.gz'
        bundle=build_derived_bundle(root,data,bundle_path);data['derived_state_bundle']=bundle
        previous_bundle=previous.get('derived_state_bundle',{})
        if previous_bundle.get('sha256')!=bundle['sha256'] and not progress.contains(bundle):
            write(api,ARCHIVE,[spec(bundle['object_path'],bundle_path,bundle['sha256'],bundle['bytes'])],
                'V13 verified derived restart bundle',CommitOperationAdd,progress.confirm)
        raw=json.dumps(data,sort_keys=True,indent=2).encode();sha=hashlib.sha256(raw).hexdigest()
        path=root.parent/'v13_checkpoint_manifest.json';path.write_bytes(raw)
        key=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+sha[:12]
        # Keep the immutable snapshot and latest pointer in one transaction.
        write(api,ARCHIVE,[spec('v13/state/snapshots/'+key+'.json',path),
            spec('v13/state/latest_manifest.json',path)],'V13 checkpoint '+key,
            CommitOperationAdd,split=False)
    except Exception as exc:
        try:
            recovery=recovery_delta(root,data,entries,base_raw)
            print('Checkpoint recovery delta preserved',json.dumps({
                'bytes':recovery['bytes'],'sha256':recovery['sha256'],
                'delta_objects':recovery['delta_objects']}),flush=True)
        except Exception as recovery_exc:
            (root.parent/'v13_checkpoint_recovery_error.json').write_text(json.dumps({
                'archive_error_type':type(exc).__name__,
                'recovery_error_type':type(recovery_exc).__name__}))
        raise
    (root/'remote_manifest.json').write_bytes(raw)
    (root/'state_manifest.json').write_text(json.dumps({'files':[r for r in data['files'] if (root/r['path']).exists()]},indent=2))
    # Small second checkpoint for GitHub artifacts; full immutable objects stay
    # private and are not re-uploaded once per day or exposed in public logs.
    archive=root.parent/'v13_state.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        tar.add(path,arcname='v13_checkpoint_manifest.json')
        if (root/'latest_run.json').exists():tar.add(root/'latest_run.json',arcname='latest_run.json')
    # All delta objects and the atomic pointer now have a successful or
    # digest-reconciled remote receipt. Earlier emergency bytes are redundant.
    for name in ('v13_checkpoint_recovery.tar.gz','v13_checkpoint_recovery.json',
                 'v13_checkpoint_recovery_error.json','v13_checkpoint_upload_progress.json'):
        (root.parent/name).unlink(missing_ok=True)
    return {'archive':ARCHIVE,'snapshot':key,'manifest_sha256':sha,'new_objects':len(entries),'total_objects':len({r['sha256'] for r in data['files']}),
      'derived_bundle_bytes':bundle['bytes'],'derived_bundle_files':bundle['files'],
      'maximum_object_commit_files':MAX_BATCH,'resumed_confirmed_objects':len(entries)-len(pending)}

def restore_review_originals(root,additional_hashes=None):
    """Fetch only original hashes required by reviewed evidence, using the
    worker's existing archive connection. No credential is exported or logged.
    """
    from huggingface_hub import hf_hub_download
    root=Path(root);mp=root/'remote_manifest.json'
    if not mp.exists():return
    required=set(additional_hashes or [])
    for rel in ('catalyst_evidence/reviewed_execution.json','action_audit/reviewed_actions.json','action_audit/reviewed_coverage.json','evidence/reviewed_earnings_scenarios.json'):
        p=root/rel
        if not p.exists():continue
        for r in json.loads(p.read_text()):
            if r.get('source_sha256'):required.add(r['source_sha256'])
            required.update(r.get('primary_source_hashes',[]));required.update(r.get('input_source_sha256',{}).values())
    for item in json.loads(mp.read_text())['files']:
        if item['sha256'] in required and not (root/item['path']).exists():
            materialize_one(item,root,lambda key:hf_hub_download(ARCHIVE,key,repo_type='dataset',token=os.environ.get('HF_ARCHIVE_TOKEN')))

