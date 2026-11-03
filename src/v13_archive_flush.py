"""Publish the preserved worker state after cleanup without running inference."""
import argparse,hashlib,json,os
from pathlib import Path
from v13_archive_cleanup import ARCHIVE,OWNER
from v13_archive_retention import worker_base

CHAMPION='c9208f89c46d043ee1d74e44dee1c85b9721b7a094e40a0fabf2c5506c246571'

def member_identity(data):
    rows=data['files'];result={r['path']:(r['sha256'],r['bytes'],r['raw_source']) for r in rows}
    if len(result)!=len(rows):raise ValueError('Duplicate checkpoint member path')
    return result

def assert_same_members(expected,actual):
    if member_identity(expected)!=member_identity(actual):
        raise ValueError('Recovery publication would change original checkpoint members')

def recovery_raw_aliases(planned,scanned,base):
    expected=member_identity(planned);actual=member_identity(scanned)
    extra=set(actual)-set(expected)
    changed={p for p in set(actual)&set(expected) if actual[p]!=expected[p]}
    if extra or changed:
        raise ValueError(f'Restored checkpoint has {len(extra)} extra and {len(changed)} changed members')
    known={(r['sha256'],r['bytes']) for r in base['files'] if r['raw_source']}
    aliases=[]
    for item in planned['files']:
        if item['path'] in actual:continue
        if not item['raw_source'] or (item['sha256'],item['bytes']) not in known:
            raise ValueError('Missing recovery member is not an existing raw-source alias')
        aliases.append(item)
    return aliases

def run(artifact,out):
    from huggingface_hub import HfApi
    from v13_archive_codec import archive_download as hf_hub_download
    from v13_recovery_restore import restore_artifact
    from v13_checkpoint import manifest,save,materialize_bundle
    import pandas as pd
    out=Path(out);out.mkdir(parents=True,exist_ok=True);root=out/'state'
    token=os.environ['HF_ARCHIVE_TOKEN'];api=HfApi(token=token)
    info=api.repo_info(ARCHIVE,repo_type='dataset')
    if api.whoami().get('name')!=OWNER or not info.private:raise ValueError('Private archive ownership mismatch')
    def fetch(path):return hf_hub_download(ARCHIVE,path,repo_type='dataset',token=token)
    base=worker_base(artifact)
    current=Path(fetch('v13/state/latest_manifest.json')).read_bytes()
    if base is None or hashlib.sha256(base).hexdigest()!=hashlib.sha256(current).hexdigest():
        raise ValueError('Remote checkpoint advanced beyond the preserved recovery base')
    result={'source_worker_run':41547044971,'remote_head_before':info.sha,
      'base_manifest_sha256':hashlib.sha256(base).hexdigest(),'remote_publication_confirmed':False}
    def receipt():(out/'flush_receipt.json').write_text(json.dumps(result,indent=2))
    receipt();result['restore']=restore_artifact(artifact,root,fetch);receipt()
    planned=json.loads((root/'pending_checkpoint_manifest.json').read_text())
    aliases=recovery_raw_aliases(planned,manifest(root),json.loads(base))
    assert_same_members(planned,manifest(root,preserved_raw_aliases=aliases))
    result['preserved_existing_raw_aliases']=len(aliases);receipt()
    if hashlib.sha256((root/'active_heads.joblib').read_bytes()).hexdigest()!=CHAMPION:
        raise ValueError('Frozen champion differs from reviewed state')
    result['forecast_records']=len(pd.read_parquet(root/'prospective_observations.parquet'))
    result['member_count']=len(planned['files'])
    result['fixed_champion_sha256']=CHAMPION;receipt()
    # No collection, scoring, model fitting or historical outcome updates occur.
    (out/'v13_archive_capacity.json').unlink(missing_ok=True)
    try:published=save(root,preserved_raw_aliases=aliases)
    except Exception as exc:
        result['failure_type']=type(exc).__name__
        result['http_status']=getattr(getattr(exc,'response',None),'status_code',None)
        result['original_github_recovery_retained']=True;receipt();raise
    result['archive_receipt']=published;receipt()
    remote=Path(hf_hub_download(ARCHIVE,'v13/state/latest_manifest.json',repo_type='dataset',token=token,
      force_download=True)).read_bytes()
    if hashlib.sha256(remote).hexdigest()!=published['manifest_sha256']:
        raise ValueError('Published remote pointer was not independently confirmed')
    checkpoint=json.loads(remote);assert_same_members(planned,checkpoint)
    def fresh(path):return hf_hub_download(ARCHIVE,path,repo_type='dataset',token=token,
      cache_dir=out/'fresh_cache',force_download=True)
    fresh_root=out/'fresh_state'
    materialize_bundle(checkpoint['derived_state_bundle'],checkpoint,fresh_root,fresh)
    for item in checkpoint['files']:
        if item['raw_source']:continue
        if hashlib.sha256((fresh_root/item['path']).read_bytes()).hexdigest()!=item['sha256']:
            raise ValueError('Fresh restart differs from the preserved worker')
    result.update(remote_publication_confirmed=True,all_forecast_and_model_bytes_unchanged=True,
      fresh_restart_verified=True,remote_head_after=api.repo_info(ARCHIVE,repo_type='dataset').sha)
    receipt();print(json.dumps(result,indent=2),flush=True);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--artifact',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();run(a.artifact,a.out)
