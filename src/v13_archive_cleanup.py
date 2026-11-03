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

def inspect_archive(repo,files,download,out):
    by_path={r['path']:r for r in files}
    if 'v13/state/latest_manifest.json' not in by_path:raise ValueError('V13 checkpoint pointer is missing')
    names=[p for p in by_path if p=='v13/state/latest_manifest.json' or
      p.startswith('v13/state/snapshots/') and p.endswith('.json')]
    manifests=[];refs={};problems=[]
    from v13_checkpoint import validate_object
    for name in sorted(names):
        raw=Path(download(name)).read_bytes();digest=hashlib.sha256(raw).hexdigest()
        data=json.loads(raw)
        if data.get('format') not in {'V13-content-addressed-checkpoint-1','V13-content-addressed-checkpoint-2'}:
            raise ValueError('Unsupported V13 manifest '+name)
        (out/'manifests').mkdir(exist_ok=True)
        (out/'manifests'/Path(name).name).write_bytes(raw)
        manifests.append({'path':name,'sha256':digest,'files':len(data['files'])})
        for item in [*data['files'],*([data['derived_state_bundle']] if data.get('derived_state_bundle') else [])]:
            validate_object(item)
            r=reference(item)
            old=refs.get(r['path'])
            if old is not None and old!=r:raise ValueError('Conflicting immutable archive object')
            refs[r['path']]=r
    for path,r in refs.items():
        actual=by_path.get(path)
        if not actual:problems.append({'path':path,'reason':'MISSING_REFERENCED_OBJECT'})
        elif actual['bytes']!=r['bytes']:problems.append({'path':path,'reason':'REFERENCED_OBJECT_SIZE_MISMATCH'})
        elif actual['lfs_sha256'] and actual['lfs_sha256']!=r['sha256']:
            problems.append({'path':path,'reason':'REFERENCED_OBJECT_HASH_MISMATCH'})
    # All V13 source vintages, frozen forecasts and checkpoint snapshots stay.
    # A historical namespace name alone is never a deletion criterion.
    protected={p for p in by_path if p.startswith('v13/')}|set(refs)|{'.gitattributes','README.md'}
    remove=[r for r in files if r['path'] not in protected]
    return {'manifests':manifests,'referenced_objects':list(refs.values()),
      'reference_problems':problems,'protected_paths':sorted(protected),
      'delete_candidates':remove,'candidate_bytes':sum(r['bytes'] or 0 for r in remove)}

def inventory(out):
    from huggingface_hub import HfApi,hf_hub_download
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
    dependencies=inspect_archive(archive,archive['files'],download,out)
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

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();inventory(a.out)
