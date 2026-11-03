"""Identify reconstructable restart caches; preserve every original snapshot member."""
import argparse,hashlib,json,os,tarfile,zipfile
from pathlib import Path
from v13_archive_cleanup import ARCHIVE,reference,value

def load_json(raw):
    try:
        import orjson
        return orjson.loads(raw)
    except ImportError:return json.loads(raw)

def download_artifact(artifact_id,digest,path):
    import requests
    repo=os.environ['GITHUB_REPOSITORY']
    if repo!='ayushverma1988/v9-multibagger-calibration':raise ValueError('Unexpected GitHub archive project')
    path=Path(path);h=hashlib.sha256();size=0
    with requests.get(f'https://api.github.com/repos/{repo}/actions/artifacts/{int(artifact_id)}/zip',
      headers={'Authorization':'Bearer '+os.environ['GITHUB_TOKEN'],'Accept':'application/vnd.github+json'},
      stream=True,timeout=(15,120)) as response:
        response.raise_for_status()
        with path.open('wb') as f:
            for chunk in response.iter_content(1024*1024):f.write(chunk);h.update(chunk);size+=len(chunk)
    if h.hexdigest()!=digest.removeprefix('sha256:'):raise ValueError('GitHub artifact digest mismatch')
    print('Verified artifact',artifact_id,'bytes',size,flush=True)

def worker_base(archive):
    with zipfile.ZipFile(archive) as z:
        if 'v13_checkpoint_recovery.json' not in z.namelist():return None
        descriptor=load_json(z.read('v13_checkpoint_recovery.json'))
        with z.open('v13_checkpoint_recovery.tar.gz') as compressed:
            with tarfile.open(fileobj=compressed,mode='r|gz') as tar:
                for member in tar:
                    if member.name=='base_manifest.json':
                        raw=tar.extractfile(member).read()
                        if hashlib.sha256(raw).hexdigest()!=descriptor['base_manifest_sha256']:
                            raise ValueError('Pending recovery base manifest hash mismatch')
                        return raw
    raise ValueError('Recovery artifact has no verified base manifest')

def cache_plan(data,manifests,pending_raw=None):
    archive=next(r for r in data['repositories'] if r['repo_id']==ARCHIVE)
    physical={r['path']:r for r in archive['files']}
    bundles={};members=set();latest=None;manifest_hashes=set()
    for name,raw in manifests:
        expected=next(m['sha256'] for m in data['archive_dependencies']['manifests'] if Path(m['path']).name==name)
        actual=hashlib.sha256(raw).hexdigest()
        if actual!=expected:raise ValueError('Audited snapshot manifest changed')
        manifest_hashes.add(actual);m=load_json(raw)
        for item in m['files']:members.add(reference(item)['path'])
        b=m.get('derived_state_bundle')
        if b:bundles[b['object_path']]={'path':b['object_path'],'sha256':b['sha256'],'bytes':b['bytes']}
        if name=='latest_manifest.json':latest=m
    if latest is None:raise ValueError('Missing current V13 checkpoint manifest')
    active={latest['derived_state_bundle']['object_path']} if latest.get('derived_state_bundle') else set()
    pending_sha=None
    if pending_raw:
        pending_sha=hashlib.sha256(pending_raw).hexdigest()
        if pending_sha not in manifest_hashes:raise ValueError('Pending recovery base is outside the audited snapshots')
        pending=load_json(pending_raw)
        if pending.get('derived_state_bundle'):active.add(pending['derived_state_bundle']['object_path'])
    optional=[];already_pruned=[]
    for path,b in bundles.items():
        if path in active or path in members:continue
        f=physical.get(path)
        if f is None:already_pruned.append(path);continue
        if f['bytes']!=b['bytes'] or f.get('lfs_sha256') and f['lfs_sha256']!=b['sha256']:
            raise ValueError('Historical restart cache identity mismatch')
        optional.append({**b,'reconstructable_from_retained_members':True})
    optional_paths={r['path'] for r in optional}
    if optional_paths&members or optional_paths&active:raise ValueError('Required original data intersects cache deletion')
    graph=data['archive_dependencies']
    graph['protected_paths']=[p for p in graph['protected_paths'] if p not in optional_paths]
    graph['referenced_objects']=[r for r in graph['referenced_objects'] if r['path'] not in optional_paths]
    graph['delete_candidates'] += [physical[p] for p in sorted(optional_paths)]
    graph['candidate_bytes']=sum(r['bytes'] or 0 for r in graph['delete_candidates'])
    data['optional_bundle_caches']=optional
    data['active_bundle_paths']=sorted(active)
    data['retention_policy']={'all_original_snapshot_members_retained':True,
      'frozen_forecasts_and_input_snapshots_retained':True,'all_snapshot_manifest_bytes_retained':True,
      'pending_recovery_base_sha256':pending_sha,'manifest_count':len(manifest_hashes),
      'unique_original_member_objects':len(members),'unique_restart_bundles':len(bundles),
      'redundant_restart_caches':len(optional),'redundant_cache_bytes':sum(r['bytes'] for r in optional),
      'previously_pruned_caches':already_pruned}
    return data

def reconstruct_restart_cache(manifest,folder,fetch):
    """Rebuild a retired cache from preserved hash-bound member objects."""
    from v13_checkpoint import materialize_one,build_derived_bundle
    folder=Path(folder);state=folder/'members';state.mkdir(parents=True,exist_ok=True)
    for item in manifest['files']:
        if not item['raw_source']:materialize_one(item,state,fetch)
    path=folder/'reconstructed.tar.gz';result=build_derived_bundle(state,manifest,path)
    old=manifest.get('derived_state_bundle')
    if old and (result['sha256']!=old['sha256'] or result['bytes']!=old['bytes']):
        raise ValueError('Reconstructed restart cache differs from recorded original')
    return path

def run(audit_zip,worker_zip,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(audit_zip) as z:
        key=next(n for n in z.namelist() if Path(n).name=='inventory.json')
        original=z.read(key)
        if hashlib.sha256(original).hexdigest()!='c4f47a6ce1cf224b51689d89f04b65e252ad5ab8712f1d72334a7c753a6ae9b3':
            raise ValueError('Unreviewed source inventory')
        data=load_json(original)
        names={Path(n).name:n for n in z.namelist() if '/manifests/' in '/'+n}
        def snapshots():
            for item in data['archive_dependencies']['manifests']:
                name=Path(item['path']).name;yield name,z.read(names[name])
        updated=cache_plan(data,snapshots(),worker_base(worker_zip) if worker_zip else None)
    raw=json.dumps(updated,indent=2).encode();(out/'retention_inventory.json').write_bytes(raw)
    summary={'inventory_sha256':hashlib.sha256(raw).hexdigest(),'source_inventory_sha256':hashlib.sha256(original).hexdigest(),
      'archive_head':updated['archive_head'],'retention_policy':updated['retention_policy'],
      'active_bundle_paths':updated['active_bundle_paths'],'delete_files':len(updated['archive_dependencies']['delete_candidates']),
      'delete_current_bytes':updated['archive_dependencies']['candidate_bytes'],'other_repositories':updated['other_repository_candidates']}
    (out/'retention_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--audit-zip',required=True);p.add_argument('--worker-zip');p.add_argument('--out',required=True)
    a=p.parse_args();run(a.audit_zip,a.worker_zip,a.out)
