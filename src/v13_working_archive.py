"""Budgeted working-only checkpoints after explicit retirement of raw history."""
import hashlib,json,os,re,shutil,tempfile,threading,time,zipfile
from datetime import datetime,timezone
from pathlib import Path
from v13_archive_codec import sha_file

FORMAT='V13-content-addressed-checkpoint-4'
CODEC='V13_WORKING_CORE_ZIP'
BUDGET=1_000_000_000
POLICY={'format':'V13-working-core-retention-1','mode':'WORKING_CORE_ONLY',
 'target_stored_bytes':BUDGET,'full_checkpoint_versions_retained':1,
 'raw_source_originals_retained':False,'older_raw_evidence_and_snapshots_retired_by_user':True,
 'working_values_models_and_frozen_forecasts_preserved':True,
 'dated_fact_and_forecast_history_in_working_tables':True,'new_data_minimum_retention_days':730,
 'if_budget_exceeded':'KEEP_LAST_CONFIRMED_CHECKPOINT_AND_VERIFIED_LOCAL_RECOVERY'}
_guard=threading.RLock();_cache={};_indexes={}

def enabled(data=None):
    return os.environ.get('V13_WORKING_CORE_ARCHIVE')=='1' or (data or {}).get('format')==FORMAT

def validate_core(ref):
    sha=ref.get('sha256');size=ref.get('bytes')
    if (not isinstance(sha,str) or not re.fullmatch('[0-9a-f]{64}',sha)
        or ref.get('object_path')!=f'v13/working_core/{sha}.zip'
        or type(size) is not int or not 0<size<=BUDGET
        or not re.fullmatch('[0-9a-f]{64}',str(ref.get('source_manifest_sha256','')))):
        raise ValueError('Invalid working core reference')

def core_file(ref,fetch):
    validate_core(ref);path=Path(fetch(ref['object_path']))
    if path.stat().st_size!=ref['bytes'] or sha_file(path)!=ref['sha256']:
        raise ValueError('Working core physical identity mismatch')
    with zipfile.ZipFile(path) as archive:
        info=json.loads(archive.read('core_receipt.json'))
    if info['source_manifest_sha256']!=ref['source_manifest_sha256']:
        raise ValueError('Working core source manifest mismatch')
    return path

def identities(data):
    rows=[r for r in data['files'] if not r['raw_source']]
    result={r['path']:(r['sha256'],r['bytes']) for r in rows}
    if len(rows)!=len(result):raise ValueError('Duplicate working state member')
    return result

def migration_matches(source_raw,published):
    if published.get('format')!=FORMAT or published.get('migration_source_manifest_sha256')!=hashlib.sha256(source_raw).hexdigest():return False
    original=identities(json.loads(source_raw));current=identities(published)
    return (all(current.get(name)==value for name,value in original.items())
      and all(name in original or name.startswith('archive_provenance/') for name in current))

def materialize_core(ref,data,root,fetch):
    from v13_core_restore import restore_package
    source=core_file(ref,fetch);root=Path(root);root.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.core-read-',dir=root.parent) as folder:
        stage=Path(folder)/'state';restore_package(source,stage)
        inside=json.loads((stage/'remote_manifest.json').read_text())
        if identities(inside)!=identities(data):raise ValueError('Core does not match published working state')
        root.mkdir(parents=True,exist_ok=True)
        for name in identities(data):
            target=root/name;target.parent.mkdir(parents=True,exist_ok=True);(stage/name).replace(target)
    return len(identities(data))

def read_member(item,fetch):
    from v13_core_restore import restore_package
    ref=item['storage_core'];validate_core(ref)
    folder=Path(os.environ.get('V13_CORE_CACHE','v13_core_object_cache'))/ref['sha256']
    with _guard:
        key=str(folder)
        if key not in _cache or not folder.exists():
            folder.parent.mkdir(parents=True,exist_ok=True)
            if not folder.exists():restore_package(core_file(ref,fetch),folder)
            source=json.loads((folder/'remote_manifest.json').read_text())
            _cache[key]={r['sha256']:r for r in source['files'] if not r['raw_source']}
        old=_cache[key].get(item['sha256'])
    if old is None or old['bytes']!=item['bytes']:raise ValueError('Working member is not in its bound core')
    source=folder/old['path']
    if source.stat().st_size!=item['bytes'] or sha_file(source)!=item['sha256']:
        raise ValueError('Cached working member differs from original bytes')
    return source.read_bytes()

def fetch_by_sha(repo,sha,token=None):
    """Resolve an older current-file address without resurrecting retired data."""
    from huggingface_hub import hf_hub_download
    from v13_archive_codec import logical_download
    with _guard:
        if repo not in _indexes:
            path=hf_hub_download(repo,'v13/state/latest_manifest.json',repo_type='dataset',token=token)
            data=json.loads(Path(logical_download(path,'v13/state/latest_manifest.json')).read_text())
            if data.get('format')!=FORMAT:return None
            _indexes[repo]={r['sha256']:r for r in data['files']}
        item=_indexes[repo].get(sha)
    if item is None:return None
    raw=read_member(item,lambda key:hf_hub_download(repo,key,repo_type='dataset',token=token))
    folder=Path(os.environ.get('V13_DECODED_CACHE','v13_decoded_objects'));folder.mkdir(parents=True,exist_ok=True)
    target=folder/(sha+'.decoded');target.write_bytes(raw);return str(target)

def preserve_provenance(root,base_raw,previous):
    import zstandard as zstd
    root=Path(root);folder=root/'archive_provenance';folder.mkdir(exist_ok=True)
    if previous.get('format')!=FORMAT:
        digest=hashlib.sha256(base_raw).hexdigest();target=folder/(digest+'.source_manifest.json.zst')
        target.write_bytes(zstd.ZstdCompressor(level=12,write_checksum=True).compress(base_raw))
        (folder/'retention_policy.json').write_text(json.dumps({**POLICY,
          'original_source_manifest_sha256':digest,'original_source_references':sum(r['raw_source'] for r in previous['files']),
          'retirement_authorized_date_IST':'2026-11-05'},sort_keys=True,indent=2))
    # Retain small date-wise hash inventories for freshly collected originals,
    # while their bodies remain transient inputs rather than permanent copies.
    from v13_checkpoint import is_raw
    records=[]
    for p in sorted(root.rglob('*')):
        if p.is_file() and is_raw(p.relative_to(root)):
            records.append({'path':str(p.relative_to(root)),'sha256':sha_file(p),'bytes':p.stat().st_size,
              'original_body_archived':False})
    if records:
        raw='\n'.join(json.dumps(r,sort_keys=True) for r in records).encode()
        day=datetime.now(timezone.utc).strftime('%Y-%m-%d');name=hashlib.sha256(raw).hexdigest()
        target=folder/'capture_receipts'/(day+'_'+name+'.jsonl.zst');target.parent.mkdir(exist_ok=True)
        target.write_bytes(zstd.ZstdCompressor(level=12,write_checksum=True).compress(raw))

def save(root,api):
    from v13_checkpoint import ARCHIVE,manifest
    from v13_checkpoint_writes import spec,recovery_delta,CheckpointTemporaryError
    from v13_storage_migrate import commit
    from v13_compact_core import build_core
    from v13_archive_codec import archive_download
    root=Path(root);old=root/'remote_manifest.json';base_raw=old.read_bytes() if old.exists() else b'{"files":[]}'
    previous=json.loads(base_raw);preserve_provenance(root,base_raw,previous)
    data=manifest(root)
    data['files']=[{k:v for k,v in r.items() if k not in {'storage_pack','pack_offset','storage_core'}} for r in data['files'] if not r['raw_source']]
    data.update(format=FORMAT,storage_policy=POLICY,
      migration_source_manifest_sha256=previous.get('migration_source_manifest_sha256',hashlib.sha256(base_raw).hexdigest()),
      source_worker_run=json.loads((root/'latest_run.json').read_text()).get('workflow_run') if (root/'latest_run.json').exists() else None)
    known={r['sha256'] for r in previous.get('files',[])};entries=list({r['sha256']:r for r in data['files'] if r['sha256'] not in known}.values())
    recovery_delta(root,data,entries,base_raw)
    staging=root.parent/'v13_core_staging';shutil.rmtree(staging,ignore_errors=True);staging.mkdir()
    raw_inner=json.dumps(data,sort_keys=True,indent=2).encode()
    asof=json.loads((root/'latest_run.json').read_text()).get('asof') if (root/'latest_run.json').exists() else datetime.now(timezone.utc).date().isoformat()
    try:
        built=build_core(root,raw_inner,staging,asof,data['source_worker_run'],target_bytes=BUDGET-10_000_000)
        shutil.rmtree(staging/'fresh_verified_state')
        package=staging/built['package_file'];sha=built['package_sha256']
        core={'object_path':f'v13/working_core/{sha}.zip','sha256':sha,'bytes':built['package_bytes'],
          'source_manifest_sha256':built['source_manifest_sha256']}
        expected=api.repo_info(ARCHIVE,repo_type='dataset').sha
        expected=commit(api,expected,adds=[spec(core['object_path'],package)],message='Stage budgeted V13 working checkpoint before independent restart verification')
        def fresh(key):return archive_download(ARCHIVE,key,repo_type='dataset',token=os.environ['HF_ARCHIVE_TOKEN'],cache_dir=staging/'readback',force_download=True)
        materialize_core(core,data,staging/'remote_verified_state',fresh);shutil.rmtree(staging/'remote_verified_state')
        data['files']=[{**r,'storage_core':core} for r in data['files']]
        data['derived_state_bundle']={**core,'codec':CODEC,'files':len(data['files']),'uncompressed_bytes':built['state_uncompressed_bytes']}
        raw=json.dumps(data,sort_keys=True,indent=2).encode();digest=hashlib.sha256(raw).hexdigest()
        key=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+digest[:12]
        mf=staging/'published_manifest.json';mf.write_bytes(raw)
        # Encode metadata explicitly so parent-commit checks also cover the
        # physical bytes; the generic writer never publishes unbound pointers.
        from v13_archive_codec import physical_spec
        adds=[physical_spec(spec('v13/state/latest_manifest.json',mf)),physical_spec(spec('v13/state/snapshots/'+key+'.json',mf))]
        date=staging/'dated.json';date.write_text(json.dumps({'format':'V13-working-date-index-1','source_asof':asof,
          'saved_at_utc':data['saved_at_utc'],'manifest_sha256':digest,'snapshot':key,
          'full_snapshot_version_retained':False,'dated_tables_and_forecasts_carried_in_current_core':True},sort_keys=True))
        policy=staging/'policy.json';policy.write_text(json.dumps(POLICY,sort_keys=True,indent=2))
        adds += [spec('v13/core_dates/'+asof.replace('-','/')+'/'+key+'.json',date),spec('v13/storage_policy.json',policy)]
        expected=commit(api,expected,adds=adds,message='Publish verified V13 working core, dated index and one-GB retention policy atomically')
        confirmed=Path(fresh('v13/state/latest_manifest.json')).read_bytes()
        if confirmed!=raw:raise ValueError('Published working-core pointer was not independently confirmed')
    except Exception as exc:
        # The prior core remains readable. This preserves unpublished new
        # values without promoting the failed pointer or deleting old data.
        recovery_delta(root,data,entries,base_raw)
        if isinstance(exc,ValueError) and 'exceeds requested byte target' in str(exc):
            raise CheckpointTemporaryError('Working core exceeded one-GB target; last published checkpoint and local recovery retained') from None
        raise
    (root/'remote_manifest.json').write_bytes(raw)
    (root/'state_manifest.json').write_text(json.dumps({'files':data['files']},indent=2))
    with __import__('tarfile').open(root.parent/'v13_state.tar.gz','w:gz') as tar:
        tar.add(mf,arcname='v13_checkpoint_manifest.json')
        if (root/'latest_run.json').exists():tar.add(root/'latest_run.json',arcname='latest_run.json')
    for name in ('v13_checkpoint_recovery.tar.gz','v13_checkpoint_recovery.json','v13_checkpoint_recovery_error.json','v13_checkpoint_upload_progress.json'):
        (root.parent/name).unlink(missing_ok=True)
    return {'archive':ARCHIVE,'snapshot':key,'manifest_sha256':digest,'format':FORMAT,
      'new_objects':len(entries),'total_objects':len({r['sha256'] for r in data['files']}),
      'derived_bundle_bytes':core['bytes'],'derived_bundle_files':len(data['files']),
      'working_core_only':True,'raw_source_originals_archived':False,'core_budget_bytes':BUDGET,
      'remote_publication_confirmed':True,'independent_remote_core_restart_verified':True,'head':expected}

def retirement_plan(api,data):
    from v13_checkpoint import ARCHIVE
    from huggingface_hub.hf_api import RepoFolder
    validate_core(data['derived_state_bundle']);core=data['derived_state_bundle']['object_path']
    remove=[]
    for item in api.list_repo_tree(ARCHIVE,repo_type='dataset',path_in_repo='v13',recursive=False):
        if item.path in {'v13/state','v13/working_core','v13/core_dates','v13/storage_policy.json'}:continue
        remove.append((item.path,isinstance(item,RepoFolder)))
    for item in api.list_repo_tree(ARCHIVE,repo_type='dataset',path_in_repo='v13/working_core',recursive=False):
        if item.path!=core:remove.append((item.path,isinstance(item,RepoFolder)))
    for item in api.list_repo_tree(ARCHIVE,repo_type='dataset',path_in_repo='v13/state',recursive=True):
        if isinstance(item,RepoFolder):continue
        if item.path=='v13/state/latest_manifest.json':continue
        # The latest pointer is sufficient for restart. Retain its exact dated
        # duplicate; all other complete snapshot versions are retired.
        if item.path.endswith('_'+hashlib.sha256(json.dumps(data,sort_keys=True,indent=2).encode()).hexdigest()[:12]+'.json'):continue
        remove.append((item.path,False))
    if any(p in {'v13/state/latest_manifest.json',core,'v13'} for p,_ in remove):raise ValueError('Retirement would remove current working checkpoint')
    return remove

def retire(out='v13_storage_retention_receipt.json',purge=False):
    from huggingface_hub import HfApi,CommitOperationDelete
    from v13_checkpoint import ARCHIVE
    from v13_archive_cleanup import OWNER,value,record_file
    from v13_archive_codec import archive_download
    token=os.environ['HF_ARCHIVE_TOKEN'];api=HfApi(token=token)
    if api.whoami().get('name')!=OWNER or not api.repo_info(ARCHIVE,repo_type='dataset').private:raise ValueError('Private archive ownership mismatch')
    refs=api.list_repo_refs(ARCHIVE,repo_type='dataset')
    if [r.name for r in refs.branches]!=['main'] or refs.tags:raise ValueError('Unreviewed archive references')
    raw=Path(archive_download(ARCHIVE,'v13/state/latest_manifest.json',repo_type='dataset',token=token,force_download=True)).read_bytes()
    data=json.loads(raw)
    if data.get('format')!=FORMAT:raise ValueError('Only the approved working-only archive may be retired')
    if any(r['raw_source'] for r in data['files']) or data.get('storage_policy',{}).get('mode')!='WORKING_CORE_ONLY':
        raise ValueError('Published checkpoint does not follow the approved working-only policy')
    # Verify a new remote extraction immediately before retiring any original.
    with tempfile.TemporaryDirectory(prefix='v13-retirement-check-') as temp:
        materialize_core(data['derived_state_bundle'],data,Path(temp)/'state',lambda key:archive_download(ARCHIVE,key,repo_type='dataset',token=token,force_download=True))
    # Capture provider descriptors before history rewriting can hide retired
    # LFS entries. Purge only identities absent from the retained tree below.
    lfs_before=list(api.list_lfs_files(ARCHIVE,repo_type='dataset')) if purge else []
    removals=retirement_plan(api,data);head=api.repo_info(ARCHIVE,repo_type='dataset').sha
    for start in range(0,len(removals),200):
        now=Path(archive_download(ARCHIVE,'v13/state/latest_manifest.json',repo_type='dataset',token=token,force_download=True)).read_bytes()
        if now!=raw:raise ValueError('Working pointer advanced before retirement')
        ops=[CommitOperationDelete(path_in_repo=p,is_folder=folder) for p,folder in removals[start:start+200]]
        head=api.create_commit(ARCHIVE,repo_type='dataset',parent_commit=head,operations=ops,
          commit_message='Retire approved original evidence and old snapshots; retain verified current models and working data').oid
    api.super_squash_history(ARCHIVE,repo_type='dataset',branch='main',commit_message='Keep only approved compact working archive and lightweight date indexes')
    files=[record_file(r) for r in api.list_repo_tree(ARCHIVE,repo_type='dataset',recursive=True) if not isinstance(r,__import__('huggingface_hub.hf_api',fromlist=['RepoFolder']).RepoFolder)]
    size=sum(r['bytes'] or 0 for r in files)
    if size>BUDGET:raise ValueError('Retained archive exceeds approved one-GB target')
    live={r['lfs_sha256'] for r in files if r['lfs_sha256']};purged=0;purged_bytes=0
    if purge:
        batch=[]
        for item in lfs_before:
            oid=value(item,'file_oid',value(item,'oid'))
            if not isinstance(oid,str):raise ValueError('Unknown provider object identity; purge stopped')
            oid=oid.removeprefix('sha256:')
            if not re.fullmatch('[0-9a-f]{64}',oid):raise ValueError('Invalid provider object identity; purge stopped')
            if oid in live:continue
            batch.append(item);purged_bytes+=value(item,'size',0) or 0
            if len(batch)==1000:
                api.permanently_delete_lfs_files(ARCHIVE,batch,repo_type='dataset',rewrite_history=False);purged+=len(batch);batch=[]
                print(json.dumps({'retired_lfs_objects_purged':purged}),flush=True)
        if batch:api.permanently_delete_lfs_files(ARCHIVE,batch,repo_type='dataset',rewrite_history=False);purged+=len(batch)
    # Storage deletion is complete only after a second, independent restart.
    with tempfile.TemporaryDirectory(prefix='v13-after-retirement-') as temp:
        materialize_core(data['derived_state_bundle'],data,Path(temp)/'state',lambda key:archive_download(ARCHIVE,key,repo_type='dataset',token=token,force_download=True))
    if Path(archive_download(ARCHIVE,'v13/state/latest_manifest.json',repo_type='dataset',token=token,force_download=True)).read_bytes()!=raw:raise ValueError('Retirement changed the active checkpoint')
    info=api.repo_info(ARCHIVE,repo_type='dataset',expand=['usedStorage','sha'])
    result={'checked_at_utc':datetime.now(timezone.utc).isoformat(),'mode':'WORKING_CORE_ONLY',
      'restart_caches_target':1,'minimum_new_data_retention_days':730,'target_archive_bytes':BUDGET,
      'current_tree_stored_bytes':size,'current_tree_under_one_gb':True,'provider_used_storage_bytes':value(info,'used_storage'),
      'raw_originals_and_old_snapshots_retired':True,'working_values_and_frozen_forecasts_deleted':False,
      'deleted_paths':[p for p,_ in removals],'retired_lfs_objects_purged':purged,'retired_lfs_bytes_purged':purged_bytes,
      'fresh_restart_before_and_after_retirement_verified':True,'current_manifest_sha256':hashlib.sha256(raw).hexdigest(),
      'history_squashed':True,'storage_budget_warning':size>=900_000_000}
    Path(out).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True);return result
