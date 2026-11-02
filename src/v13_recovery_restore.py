"""Restore an unpublished delta without falsely publishing a remote pointer."""
import hashlib, json, tarfile
from pathlib import Path
from v13_checkpoint import materialize_bundle, materialize_one, validate_object

def digest(raw):return hashlib.sha256(raw).hexdigest()

def restore_delta(bundle, root, fetch, expected_sha256=None):
    root=Path(root);source=Path(bundle);actual=digest(source.read_bytes())
    if expected_sha256 and actual!=expected_sha256:raise ValueError('Recovery archive hash mismatch')
    with tarfile.open(source,'r:gz') as tar:
        names=tar.getnames()
        if len(names)!=len(set(names)) or any(not m.isfile() for m in tar.getmembers()):raise ValueError('Duplicate or non-file recovery member')
        def read(name):return tar.extractfile(name).read()
        meta=json.loads(read('recovery.json'));base_raw=read('base_manifest.json');planned_raw=read('planned_manifest.json')
        if meta['format']!='V13-checkpoint-recovery-delta-1':raise ValueError('Unknown recovery format')
        if digest(base_raw)!=meta['base_manifest_sha256'] or digest(planned_raw)!=meta['planned_manifest_sha256']:raise ValueError('Recovery manifest binding mismatch')
        base,planned=json.loads(base_raw),json.loads(planned_raw)
        prior={x['sha256'] for x in base['files']};needed={x['sha256']:x for x in planned['files'] if x['sha256'] not in prior}
        expected={'recovery.json','base_manifest.json','planned_manifest.json'}|{'objects/'+h+'.bin' for h in needed}
        if set(names)!=expected or len(needed)!=meta['delta_objects']:raise ValueError('Recovery object set mismatch')
        for item in planned['files']:
            rel=Path(item['path'])
            if rel.is_absolute() or '..' in rel.parts:raise ValueError('Unsafe recovery path')
            validate_object(item)
        objects={}
        for h,item in needed.items():
            raw=read('objects/'+h+'.bin')
            if len(raw)!=item['bytes'] or digest(raw)!=h:raise ValueError('Recovery object integrity failure')
            objects[h]=raw
        root.mkdir(parents=True,exist_ok=True)
        if base.get('derived_state_bundle'):materialize_bundle(base['derived_state_bundle'],base,root,fetch)
        else:
            for item in base['files']:
                if not item['raw_source']:materialize_one(item,root,fetch)
        for item in planned['files']:
            if item['sha256'] in objects:
                target=root/item['path'];target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(objects[item['sha256']])
        for item in planned['files']:
            target=root/item['path']
            if not item['raw_source'] or item['sha256'] in objects:
                valid=target.is_file() and target.stat().st_size==item['bytes'] and digest(target.read_bytes())==item['sha256']
                if not valid and item['sha256'] in prior:
                    # A known object can legitimately acquire a new local path.
                    # The delta stores new hashes, not every changed pathname.
                    materialize_one(item,root,fetch)
                    valid=target.is_file() and target.stat().st_size==item['bytes'] and digest(target.read_bytes())==item['sha256']
                if not valid:raise ValueError('Restored checkpoint differs from planned manifest: '+item['path'])
        (root/'remote_manifest.json').write_bytes(base_raw)
        (root/'pending_checkpoint_manifest.json').write_bytes(planned_raw)
        (root/'state_manifest.json').write_text(json.dumps({'files':[x for x in planned['files'] if (root/x['path']).is_file()]},indent=2))
    result={'archive_sha256':actual,'base_manifest_sha256':digest(base_raw),'planned_manifest_sha256':digest(planned_raw),
      'delta_objects':len(needed),'remote_publication_confirmed':False,'restored_local_state_verified':True}
    (root/'recovery_restore_receipt.json').write_text(json.dumps(result,indent=2));return result

def restore_artifact(archive,root,fetch):
    import zipfile
    root=Path(root)
    with zipfile.ZipFile(archive) as z:
        if 'v13_checkpoint_recovery.json' not in z.namelist():return {'recovery_present':False}
        desc=json.loads(z.read('v13_checkpoint_recovery.json'));raw=z.read('v13_checkpoint_recovery.tar.gz')
        if len(raw)!=desc['bytes'] or digest(raw)!=desc['sha256']:raise ValueError('Artifact recovery receipt mismatch')
        path=root.parent/'previous_verified_recovery.tar.gz';path.write_bytes(raw)
        result=restore_delta(path,root,fetch,desc['sha256'])
        if 'v13_archive_capacity.json' in z.namelist():
            quota=json.loads(z.read('v13_archive_capacity.json'))
            if quota.get('archive_capacity_blocked') is True:(root.parent/'v13_archive_capacity.json').write_text(json.dumps(quota))
    return {'recovery_present':True,**result}

if __name__=='__main__':
    import argparse,os
    from huggingface_hub import HfApi,hf_hub_download
    from v13_checkpoint import ARCHIVE
    p=argparse.ArgumentParser();p.add_argument('--artifact',required=True);p.add_argument('--state',required=True);a=p.parse_args()
    token=os.environ['HF_ARCHIVE_TOKEN'];api=HfApi(token=token)
    if not api.repo_info(ARCHIVE,repo_type='dataset').private:raise RuntimeError('Private archive required')
    print(json.dumps(restore_artifact(a.artifact,a.state,lambda x:hf_hub_download(ARCHIVE,x,repo_type='dataset',token=token))))
