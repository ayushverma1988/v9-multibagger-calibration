"""Content-addressed private checkpoints; preserve raw bytes without daily copies."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from pathlib import Path
import hashlib,json,os,shutil,tarfile

ARCHIVE='ayushverma1988/v10-multibagger-archive'

def is_raw(path):return any(part in {'raw','history_raw','filing_raw'} for part in Path(path).parts)

def manifest(root):
    root=Path(root);old=root/'remote_manifest.json'
    prior=json.loads(old.read_text())['files'] if old.exists() else []
    files={r['path']:r for r in prior}
    for p in sorted(root.rglob('*')):
        if not p.is_file() or p.name in {'state_manifest.json','remote_manifest.json'}:continue
        rel=str(p.relative_to(root));digest=hashlib.sha256(p.read_bytes()).hexdigest()
        files[rel]={'path':rel,'sha256':digest,'bytes':p.stat().st_size,'object_path':'v13/objects/'+digest+'.bin','raw_source':is_raw(rel)}
    return {'format':'V13-content-addressed-checkpoint-1','saved_at_utc':datetime.now(timezone.utc).isoformat(),'files':sorted(files.values(),key=lambda r:r['path'])}

def materialize_one(item,root,fetch):
    rel=Path(item['path'])
    if rel.is_absolute() or '..' in rel.parts:raise ValueError('Unsafe checkpoint path')
    if item['object_path']!='v13/objects/'+item['sha256']+'.bin':raise ValueError('Object identity mismatch')
    raw=Path(fetch(item['object_path'])).read_bytes()
    if len(raw)!=item['bytes'] or hashlib.sha256(raw).hexdigest()!=item['sha256']:raise ValueError('Checkpoint object integrity failure')
    target=Path(root)/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)

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
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda r:materialize_one(r,root,fetch),active))
    (root/'remote_manifest.json').write_text(json.dumps(data,indent=2))
    (root/'state_manifest.json').write_text(json.dumps({'files':active},indent=2))
    return True

def save(root):
    from huggingface_hub import HfApi,CommitOperationAdd
    root=Path(root);token=os.environ['HF_ARCHIVE_TOKEN'];api=HfApi(token=token)
    if not api.repo_info(ARCHIVE,repo_type='dataset').private:raise RuntimeError('Checkpoint archive must be private')
    data=manifest(root);old=root/'remote_manifest.json'
    known={r['sha256'] for r in json.loads(old.read_text())['files']} if old.exists() else set()
    objects={r['sha256']:r for r in data['files'] if r['sha256'] not in known}
    entries=list(objects.values())
    for start in range(0,len(entries),200):
        operations=[CommitOperationAdd(path_in_repo=r['object_path'],path_or_fileobj=root/r['path']) for r in entries[start:start+200]]
        api.create_commit(repo_id=ARCHIVE,repo_type='dataset',commit_message='V13 immutable source/model objects',operations=operations)
    raw=json.dumps(data,sort_keys=True,indent=2).encode();sha=hashlib.sha256(raw).hexdigest()
    path=root.parent/'v13_checkpoint_manifest.json';path.write_bytes(raw)
    key=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+sha[:12]
    api.create_commit(repo_id=ARCHIVE,repo_type='dataset',commit_message='V13 checkpoint '+key,
      operations=[CommitOperationAdd(path_in_repo='v13/state/snapshots/'+key+'.json',path_or_fileobj=path),
                  CommitOperationAdd(path_in_repo='v13/state/latest_manifest.json',path_or_fileobj=path)])
    (root/'remote_manifest.json').write_bytes(raw)
    (root/'state_manifest.json').write_text(json.dumps({'files':[r for r in data['files'] if (root/r['path']).exists()]},indent=2))
    # Small second checkpoint for GitHub artifacts; full immutable objects stay
    # private and are not re-uploaded once per day or exposed in public logs.
    archive=root.parent/'v13_state.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        tar.add(path,arcname='v13_checkpoint_manifest.json')
        if (root/'latest_run.json').exists():tar.add(root/'latest_run.json',arcname='latest_run.json')
    return {'archive':ARCHIVE,'snapshot':key,'manifest_sha256':sha,'new_objects':len(entries),'total_objects':len({r['sha256'] for r in data['files']})}

def restore_review_originals(root):
    """Fetch only original hashes required by reviewed evidence, using the
    worker's existing archive connection. No credential is exported or logged.
    """
    from huggingface_hub import hf_hub_download
    root=Path(root);mp=root/'remote_manifest.json'
    if not mp.exists():return
    required=set()
    for rel in ('catalyst_evidence/reviewed_execution.json','action_audit/reviewed_actions.json','action_audit/reviewed_coverage.json','evidence/reviewed_earnings_scenarios.json'):
        p=root/rel
        if not p.exists():continue
        for r in json.loads(p.read_text()):
            if r.get('source_sha256'):required.add(r['source_sha256'])
            required.update(r.get('primary_source_hashes',[]));required.update(r.get('input_source_sha256',{}).values())
    for item in json.loads(mp.read_text())['files']:
        if item['sha256'] in required and not (root/item['path']).exists():
            materialize_one(item,root,lambda key:hf_hub_download(ARCHIVE,key,repo_type='dataset',token=os.environ.get('HF_ARCHIVE_TOKEN')))
