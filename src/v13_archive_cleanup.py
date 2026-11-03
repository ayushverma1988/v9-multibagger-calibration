"""Inventory the user's archive before explicitly authorised, dependency-safe cleanup."""
import argparse, hashlib, json, os, re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

OWNER='ayushverma1988'
ARCHIVE=OWNER+'/v10-multibagger-archive'

def value(obj,key,default=None):
    return obj.get(key,default) if isinstance(obj,dict) else getattr(obj,key,default)

def record_file(item):
    lfs=value(item,'lfs')
    return {'path':value(item,'path',value(item,'rfilename')),'bytes':value(item,'size'),
      'blob_id':value(item,'blob_id'),'lfs_sha256':value(lfs,'sha256')}

def reference(item):
    pack=item.get('storage_pack')
    r=pack if pack else item
    return {'path':r['object_path'],'sha256':r['sha256'],'bytes':r['bytes']}

def root_fingerprint(entries):
    result={}
    for item in entries:
        path=value(item,'path',value(item,'rfilename'))
        identity=value(item,'tree_id') or value(item,'blob_id')
        if not re.fullmatch('[0-9a-f]{40}|[0-9a-f]{64}',str(identity)):
            raise ValueError('Missing trustworthy repository tree identity')
        result[path]=identity
    if 'v13' not in result:raise ValueError('Required V13 tree is missing')
    return result

def verify_retained_root(before,after,expected_v13=None):
    wanted={k:v for k,v in before.items() if k in {'v13','.gitattributes','README.md'}}
    if expected_v13 is not None:wanted['v13']=expected_v13
    if after!=wanted:raise ValueError('Retained archive root/V13 tree identity changed')

def git_tree_hash(files):
    tree={}
    for path,blob in files.items():
        if not re.fullmatch('[0-9a-f]{40}',str(blob)):raise ValueError('Unsupported Git blob identity')
        parts=path.split('/')
        if any(p in {'','..','.'} for p in parts):raise ValueError('Unsafe Git tree member')
        cursor=tree
        for part in parts[:-1]:cursor=cursor.setdefault(part,{})
        cursor[parts[-1]]=blob
    def encode(t):
        records=[]
        for name,item in sorted(t.items(),key=lambda x:(x[0]+('/' if isinstance(x[1],dict) else '')).encode()):
            folder=isinstance(item,dict);oid=encode(item) if folder else item
            records.append((b'40000 ' if folder else b'100644 ')+name.encode()+b'\0'+bytes.fromhex(oid))
        raw=b''.join(records)
        return hashlib.sha1(b'tree '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
    return encode(tree)

def inspect_archive(repo,files,download,out,encodings=None,catalog=None):
    by_path={r['path']:r for r in files}
    if 'v13/state/latest_manifest.json' not in by_path:raise ValueError('V13 checkpoint pointer is missing')
    names=[p for p in by_path if p=='v13/state/latest_manifest.json' or
      p.startswith('v13/state/snapshots/') and p.endswith('.json')]
    manifests=[];refs={};problems=[];required=set();optional_missing=[]
    from v13_checkpoint import validate_object
    for name in sorted(names):
        raw=Path(download(name)).read_bytes();digest=hashlib.sha256(raw).hexdigest()
        data=json.loads(raw)
        if data.get('format') not in {'V13-content-addressed-checkpoint-1','V13-content-addressed-checkpoint-2','V13-content-addressed-checkpoint-3'}:
            raise ValueError('Unsupported V13 manifest '+name)
        (out/'manifests').mkdir(exist_ok=True)
        (out/'manifests'/Path(name).name).write_bytes(raw)
        manifests.append({'path':name,'sha256':digest,'files':len(data['files'])})
        for item in data['files']:required.add(reference(item)['path'])
        if name=='v13/state/latest_manifest.json' and data.get('derived_state_bundle'):
            required.add(data['derived_state_bundle']['object_path'])
        for item in [*data['files'],*([data['derived_state_bundle']] if data.get('derived_state_bundle') else [])]:
            validate_object(item)
            r=reference(item)
            old=refs.get(r['path'])
            if old is not None and old!=r:raise ValueError('Conflicting immutable archive object')
            refs[r['path']]=r
    for path,r in refs.items():
        actual=by_path.get(path)
        encoded=(encodings or {}).get(path)
        compact=(catalog or {}).get(r['sha256'])
        if encoded and actual and encoded['stored_sha256']==actual.get('lfs_sha256'):
            if encoded['original_sha256']!=r['sha256'] or encoded['original_bytes']!=r['bytes']:
                problems.append({'path':path,'reason':'ENCODED_ORIGINAL_IDENTITY_MISMATCH'});continue
            actual={**actual,'bytes':encoded['original_bytes'],'lfs_sha256':encoded['original_sha256']}
        if actual is None and compact:
            packed=by_path.get(compact['pack_path'])
            if (not packed or packed['bytes']!=compact['pack_bytes'] or packed.get('lfs_sha256')!=compact['pack_sha256']
                or compact['original_bytes']!=r['bytes']):
                problems.append({'path':path,'reason':'COMPACT_ORIGINAL_IDENTITY_MISMATCH'});continue
            actual={'bytes':compact['original_bytes'],'lfs_sha256':compact['original_sha256']}
        if not actual:
            if path in required:problems.append({'path':path,'reason':'MISSING_REFERENCED_OBJECT'})
            else:optional_missing.append(path)
        elif actual['bytes']!=r['bytes']:problems.append({'path':path,'reason':'REFERENCED_OBJECT_SIZE_MISMATCH'})
        elif actual['lfs_sha256'] and actual['lfs_sha256']!=r['sha256']:
            problems.append({'path':path,'reason':'REFERENCED_OBJECT_HASH_MISMATCH'})
    # All V13 source vintages, frozen forecasts and checkpoint snapshots stay.
    # A historical namespace name alone is never a deletion criterion.
    protected={p for p in by_path if p.startswith('v13/')}|set(refs)|{'.gitattributes','README.md'}
    remove=[r for r in files if r['path'] not in protected]
    return {'manifests':manifests,'referenced_objects':list(refs.values()),
      'reference_problems':problems,'optional_historical_caches_already_missing':optional_missing,
      'protected_paths':sorted(protected),
      'delete_candidates':remove,'candidate_bytes':sum(r['bytes'] or 0 for r in remove)}

def inventory(out):
    from huggingface_hub import HfApi
    from v13_archive_codec import archive_download as hf_hub_download
    from huggingface_hub.hf_api import RepoFile
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    token=os.environ['HF_ARCHIVE_TOKEN'];api=HfApi(token=token)
    who=api.whoami()
    if who.get('name')!=OWNER:raise ValueError('Archive credential owner mismatch')
    repos=[]
    for kind,method in [('model',api.list_models),('dataset',api.list_datasets),('space',api.list_spaces)]:
        for item in method(author=OWNER):
            name=value(item,'id')
            if not name.startswith(OWNER+'/'):raise ValueError('Unexpected repository owner')
            info=api.repo_info(name,repo_type=kind)
            files=[record_file(x) for x in api.list_repo_tree(name,repo_type=kind,revision=info.sha,recursive=True) if isinstance(x,RepoFile)]
            refs=api.list_repo_refs(name,repo_type=kind)
            repos.append({'repo_id':name,'repo_type':kind,'private':info.private,'head':info.sha,
              'used_storage':value(info,'used_storage',value(info,'usedStorage')),
              'files':files,'current_file_bytes':sum(x['bytes'] or 0 for x in files),
              'branches':[{'name':x.name,'target_commit':x.target_commit} for x in refs.branches],
              'tags':[{'name':x.name,'target_commit':x.target_commit} for x in refs.tags]})
            print('Inventoried',kind,name,'files',len(files),'current_bytes',repos[-1]['current_file_bytes'],flush=True)
    archive=next((r for r in repos if r['repo_id']==ARCHIVE and r['repo_type']=='dataset'),None)
    if not archive or not archive['private']:raise ValueError('Expected private V13 archive unavailable')
    def download(path):return hf_hub_download(ARCHIVE,path,repo_type='dataset',revision=archive['head'],token=token)
    encodings={};catalog={}
    import pyarrow.parquet as pq
    for item in archive['files']:
        path=item['path']
        if not path.endswith('.parquet'):continue
        if path.startswith('v13/storage_index/'):
            for row in pq.read_table(download(path)).to_pylist():
                old=encodings.get(row['logical_path'])
                if old is None or row['recorded_at_utc']>old['recorded_at_utc']:encodings[row['logical_path']]=row
        elif path.startswith('v13/storage_catalog/'):
            from v13_archive_catalog import validate_row
            for row in pq.read_table(download(path)).to_pylist():
                validate_row(row);catalog[row['original_sha256']]=row
    dependencies=inspect_archive(archive,archive['files'],download,out,encodings,catalog)
    other=[{'repo_id':r['repo_id'],'repo_type':r['repo_type'],'head':r['head'],
      'bytes':r['current_file_bytes'],'private':r['private']} for r in repos if r['repo_id']!=ARCHIVE]
    result={'mode':'READ_ONLY_INVENTORY','owner':OWNER,'observed_at_utc':datetime.now(timezone.utc).isoformat(),
      'credential_role':value(value(who.get('auth',{}),'accessToken',{}),'role'),
      'protected_archive':ARCHIVE,'archive_head':archive['head'],'repositories':repos,
      'archive_dependencies':dependencies,'other_repository_candidates':other,
      'top_level_archive_bytes':dict(Counter())}
    totals=Counter()
    for r in archive['files']:totals[r['path'].split('/')[0]]+=r['bytes'] or 0
    result['top_level_archive_bytes']=dict(totals)
    raw=json.dumps(result,indent=2).encode();(out/'inventory.json').write_bytes(raw)
    summary={'mode':result['mode'],'owner':OWNER,'credential_role':result['credential_role'],
      'inventory_sha256':hashlib.sha256(raw).hexdigest(),'repositories':[{k:r[k] for k in
        ['repo_id','repo_type','head','used_storage','current_file_bytes']} for r in repos],
      'archive_head':archive['head'],'v13_manifests':len(dependencies['manifests']),
      'reference_problems':len(dependencies['reference_problems']),
      'top_level_archive_bytes':result['top_level_archive_bytes'],
      'legacy_file_candidates':len(dependencies['delete_candidates']),
      'legacy_candidate_bytes':dependencies['candidate_bytes'],'other_repositories':other,
      'remote_mutations_performed':False}
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)
    return result

def deletion_plan(data):
    if data.get('owner')!=OWNER or data.get('protected_archive')!=ARCHIVE:
        raise ValueError('Cleanup ownership/dependency contract mismatch')
    graph=data['archive_dependencies']
    if graph['reference_problems']:raise ValueError('Unresolved V13 archive references prohibit destructive cleanup')
    protected=set(graph['protected_paths'])
    optional={r['path']:r for r in data.get('optional_bundle_caches',[])}
    from v13_checkpoint import validate_object
    for r in optional.values():validate_object({'object_path':r['path'],'sha256':r['sha256'],'bytes':r['bytes']})
    required={r['path'] for r in graph['referenced_objects']}
    paths=[]
    for item in graph['delete_candidates']:
        path=item['path'];rel=Path(path)
        cache=optional.get(path)
        permitted_cache=bool(cache and cache.get('reconstructable_from_retained_members') is True and
          path not in required and path not in set(data.get('active_bundle_paths',[])))
        if rel.is_absolute() or '..' in rel.parts or path.startswith('v13/') and not permitted_cache or path in protected:
            raise ValueError('Cleanup candidate intersects V13 protection')
        paths.append(path)
    repos=[]
    for item in data['other_repository_candidates']:
        name=item['repo_id']
        if not name.startswith(OWNER+'/') or name==ARCHIVE or 'v13' in name.lower():
            raise ValueError('Repository is not an approved unrelated owned model/dataset')
        if item['repo_type'] not in {'model','dataset','space'}:raise ValueError('Unknown repository type')
        repos.append(item)
    return {'archive':ARCHIVE,'expected_head':data['archive_head'],
      'delete_paths':paths,'delete_repositories':repos,'protected_paths':sorted(protected)}

def apply_cleanup(inventory_path,expected_sha,out,squash=False):
    from huggingface_hub import HfApi,CommitOperationDelete
    from v13_archive_codec import archive_download as hf_hub_download
    from huggingface_hub.hf_api import RepoFile
    from v13_checkpoint import restore
    raw=Path(inventory_path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=expected_sha:raise ValueError('Reviewed inventory bytes changed')
    data=json.loads(raw);plan=deletion_plan(data);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    token=os.environ['HF_ARCHIVE_TOKEN'];api=HfApi(token=token)
    if api.whoami().get('name')!=OWNER:raise ValueError('Archive credential owner mismatch')
    def head():return api.repo_info(ARCHIVE,repo_type='dataset').sha
    if head()!=plan['expected_head']:raise ValueError('Archive changed since reviewed inventory; refresh before deletion')
    before_root=root_fingerprint(api.list_repo_tree(ARCHIVE,repo_type='dataset',recursive=False,revision=plan['expected_head']))
    original=next(r for r in data['repositories'] if r['repo_id']==ARCHIVE and r['repo_type']=='dataset')
    all_v13={r['path'].removeprefix('v13/'):r['blob_id'] for r in original['files'] if r['path'].startswith('v13/')}
    if git_tree_hash(all_v13)!=before_root['v13']:raise ValueError('Full audited V13 inventory differs from live Git tree')
    deleted_paths=set(plan['delete_paths'])
    kept_v13={p:b for p,b in all_v13.items() if 'v13/'+p not in deleted_paths}
    expected_v13=git_tree_hash(kept_v13)
    def storage_view():
        return [{'repo_id':r.id,'repo_type':r.type,'visibility':r.visibility,
          'storage_bytes':r.storage} for r in api.list_user_repos()]
    try:storage_before=storage_view()
    except Exception as exc:storage_before={'unavailable':type(exc).__name__}
    if isinstance(storage_before,list):
        known={(r['repo_id'],r['repo_type']) for r in data['repositories']}
        if {(r['repo_id'],r['repo_type']) for r in storage_before}-known:
            raise ValueError('Additional owned repositories require inventory before cleanup')
    # Verify the actual complete restart bundle before removing anything.
    state=out/'verified_state'
    if not restore(state):raise ValueError('V13 restart verification failed before cleanup')
    if hashlib.sha256((state/'active_heads.joblib').read_bytes()).hexdigest()!='c9208f89c46d043ee1d74e44dee1c85b9721b7a094e40a0fabf2c5506c246571':
        raise ValueError('Unexpected fixed champion; inventory must be reviewed again')
    import pandas as pd
    ledger=state/'prospective_observations.parquet'
    verification={'restart_verified':True,'forecast_records':len(pd.read_parquet(ledger)),
      'forecast_ledger_sha256':hashlib.sha256(ledger.read_bytes()).hexdigest(),
      'fixed_head_sha256':hashlib.sha256((state/'active_heads.joblib').read_bytes()).hexdigest()}
    (out/'deletion_plan.json').write_text(json.dumps(plan,indent=2))
    result={'mode':'AUTHORISED_V13_ONLY_CLEANUP','inventory_sha256':expected_sha,
      'started_at_utc':datetime.now(timezone.utc).isoformat(),'verification_before':verification,
      'deleted_repositories':[],'deleted_archive_paths':[],'history_squashed':False,
      'provider_storage_before':storage_before}
    def receipt():(out/'cleanup_receipt.json').write_text(json.dumps(result,indent=2))
    receipt()
    for r in plan['delete_repositories']:
        current=api.repo_info(r['repo_id'],repo_type=r['repo_type'])
        if current.sha!=r['head']:raise ValueError('Unrelated repository changed since inspection')
        api.delete_repo(r['repo_id'],repo_type=r['repo_type'])
        if api.repo_exists(r['repo_id'],repo_type=r['repo_type']):raise ValueError('Repository deletion was not confirmed')
        result['deleted_repositories'].append(r);receipt()
        print('Deleted unrelated',r['repo_type'],r['repo_id'],flush=True)
    expected=plan['expected_head']
    for start in range(0,len(plan['delete_paths']),250):
        paths=plan['delete_paths'][start:start+250]
        commit=api.create_commit(ARCHIVE,repo_type='dataset',parent_commit=expected,
          operations=[CommitOperationDelete(path_in_repo=p,is_folder=False) for p in paths],
          commit_message='Remove retired non-V13 archive data; retain V13 dependencies')
        expected=commit.oid;result['deleted_archive_paths']+=paths;result['head_after_deletions']=expected;receipt()
        print('Deleted retired archive paths',len(result['deleted_archive_paths']),flush=True)
    protected=set(plan['protected_paths']);wanted={r['path']:r for r in original['files'] if r['path'] in protected}
    if any(not (p.startswith('v13/') or p in {'.gitattributes','README.md'}) for p in protected):
        raise ValueError('Additional protected roots require explicit verification')
    def verify_tree():
        after=root_fingerprint(api.list_repo_tree(ARCHIVE,repo_type='dataset',recursive=False))
        verify_retained_root(before_root,after,expected_v13)
        return after
    verify_tree();result['all_retained_paths_byte_identities_unchanged']=True
    result['v13_merkle_tree_before']=before_root['v13'];result['v13_merkle_tree_after']=expected_v13
    result['retention_policy']=data.get('retention_policy');receipt()
    if squash:
        refs=api.list_repo_refs(ARCHIVE,repo_type='dataset')
        if [r.name for r in refs.branches]!=['main'] or refs.tags:
            raise ValueError('Additional branches/tags need dependency review before history pruning')
        if head()!=expected:raise ValueError('Archive changed before history pruning')
        api.super_squash_history(ARCHIVE,repo_type='dataset',branch='main',
          commit_message='Retain verified V13 archive tree after authorised removal of obsolete data/history')
        result['history_squashed']=True;result['head_after_history_pruning']=head();receipt()
        verify_tree()
    # Read and hash the pointer again: cleanup cannot alter current forecasts.
    p=hf_hub_download(ARCHIVE,'v13/state/latest_manifest.json',repo_type='dataset',token=token)
    previous=next(x for x in data['archive_dependencies']['manifests'] if x['path']=='v13/state/latest_manifest.json')
    if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=previous['sha256']:
        raise ValueError('V13 checkpoint pointer changed during cleanup')
    checkpoint=json.loads(Path(p).read_text())
    if checkpoint.get('derived_state_bundle'):
        from v13_checkpoint import materialize_bundle
        def fresh_download(path):
            return hf_hub_download(ARCHIVE,path,repo_type='dataset',token=token,
              cache_dir=out/'post_cleanup_hf_cache',force_download=True)
        verified=out/'post_cleanup_restore'
        materialize_bundle(checkpoint['derived_state_bundle'],checkpoint,verified,fresh_download)
        for name in ['active_heads.joblib','challenger/research_heads.joblib','prospective_observations.parquet']:
            if hashlib.sha256((verified/name).read_bytes()).hexdigest()!=hashlib.sha256((state/name).read_bytes()).hexdigest():
                raise ValueError('Fresh post-cleanup restart differs from original fixed state')
        result['fresh_post_cleanup_restart_verified']=True
    result.update(v13_checkpoint_pointer_unchanged=True,
      deleted_current_archive_bytes=sum(r['bytes'] or 0 for r in graph_candidates(data)),
      retained_archive_paths=len(wanted),completed_at_utc=datetime.now(timezone.utc).isoformat())
    try:
        info=api.repo_info(ARCHIVE,repo_type='dataset',expand=['usedStorage','sha','private'])
        result['provider_used_storage_after']=value(info,'used_storage',value(info,'usedStorage'))
    except Exception as exc:result['storage_counter_unavailable']=type(exc).__name__
    try:result['provider_storage_after']=storage_view()
    except Exception as exc:result['storage_counter_unavailable']=type(exc).__name__
    receipt();print(json.dumps({k:v for k,v in result.items() if k!='deleted_archive_paths'},indent=2),flush=True)
    return result

def graph_candidates(data):return data['archive_dependencies']['delete_candidates']

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--inventory')
    p.add_argument('--inventory-sha256');p.add_argument('--prune-obsolete-history',action='store_true');a=p.parse_args()
    if a.inventory:
        if not a.inventory_sha256:p.error('Reviewed inventory SHA256 is required')
        apply_cleanup(a.inventory,a.inventory_sha256,a.out,a.prune_obsolete_history)
    else:inventory(a.out)
