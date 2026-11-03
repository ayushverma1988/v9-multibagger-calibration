"""Retain two disposable restart caches and every original dated data version."""
import json,os
from datetime import datetime,timezone
from pathlib import Path
from v13_archive_cleanup import ARCHIVE,reference,value
from v13_archive_codec import archive_download

def cache_inventory(data,key,previous):
    from huggingface_hub.errors import EntryNotFoundError
    try:
        p=archive_download(ARCHIVE,'v13/storage/restart_caches.json',repo_type='dataset',token=os.environ['HF_ARCHIVE_TOKEN'])
        result=json.loads(Path(p).read_text())
    except EntryNotFoundError:result={'format':'V13-restart-cache-inventory-1','bundles':[]}
    if result.get('format')!='V13-restart-cache-inventory-1':raise ValueError('Unexpected restart cache inventory')
    existing={r['object_path']:r for r in result['bundles']}
    for snapshot,m in [('previous_published',previous),(key,data)]:
        b=m.get('derived_state_bundle')
        if b:existing.setdefault(b['object_path'],{**b,'snapshot':snapshot,'kind':'DERIVED_RESTART_CACHE',
          'saved_at_utc':m.get('saved_at_utc',''),'cached':True,'ever_referenced_as_member':False})
    members={reference(r)['path'] for r in data['files']}
    for row in existing.values():
        row['ever_referenced_as_member'] |= row['object_path'] in members
    result['bundles']=list(existing.values());return result

def retention(out='v13_storage_retention_receipt.json'):
    from huggingface_hub import HfApi,CommitOperationDelete,CommitOperationAdd
    from v13_checkpoint_writes import spec,write
    token=os.environ['HF_ARCHIVE_TOKEN'];api=HfApi(token=token)
    if not api.repo_info(ARCHIVE,repo_type='dataset').private:raise ValueError('Private V13 archive required')
    p=archive_download(ARCHIVE,'v13/storage/restart_caches.json',repo_type='dataset',token=token,force_download=True)
    inventory=json.loads(Path(p).read_text())
    if inventory.get('format')!='V13-restart-cache-inventory-1':raise ValueError('Invalid restart inventory')
    latest=json.loads(Path(archive_download(ARCHIVE,'v13/state/latest_manifest.json',repo_type='dataset',token=token)).read_text())
    rows=[r for r in inventory['bundles'] if r.get('cached')]
    keep={r['object_path'] for r in sorted(rows,key=lambda r:r['saved_at_utc'])[-2:]}
    keep.add(latest['derived_state_bundle']['object_path'])
    keep|={reference(r)['path'] for r in latest['files']}
    retired=[r for r in rows if r['object_path'] not in keep and not r.get('ever_referenced_as_member')]
    for row in retired:
        if row.get('kind')!='DERIVED_RESTART_CACHE':raise ValueError('Only reconstructed caches may be retired')
        info=api.get_paths_info(ARCHIVE,repo_type='dataset',paths=[row['object_path']])
        if info:
            api.create_commit(ARCHIVE,repo_type='dataset',parent_commit=api.repo_info(ARCHIVE,repo_type='dataset').sha,
              operations=[CommitOperationDelete(path_in_repo=row['object_path'],is_folder=False)],
              commit_message='Retain two restart caches; preserve every dated original source and forecast')
        row['cached']=False
    target=Path(out).with_name('v13_cache_inventory_updated.json');target.write_text(json.dumps(inventory,sort_keys=True,indent=2))
    write(api,ARCHIVE,[spec('v13/storage/restart_caches.json',target)],'Record V13 restart-cache retention',CommitOperationAdd)
    # Data versioning lives in immutable dated manifests; prune only redundant
    # repository history that would otherwise retain retired cache binaries.
    refs=api.list_repo_refs(ARCHIVE,repo_type='dataset')
    squashed=False
    if [r.name for r in refs.branches]==['main'] and not refs.tags:
        api.super_squash_history(ARCHIVE,repo_type='dataset',branch='main',commit_message='Preserve dated V13 data without duplicate retired-cache history')
        squashed=True
    info=api.repo_info(ARCHIVE,repo_type='dataset',expand=['usedStorage','sha'])
    used=value(info,'used_storage')
    result={'checked_at_utc':datetime.now(timezone.utc).isoformat(),'restart_caches_target':2,
      'restart_caches_removed':len(retired),'dated_data_and_forecasts_deleted':False,
      'minimum_new_data_retention_days':730,'provider_used_storage_bytes':used,
      'storage_budget_warning':bool(used is not None and used>=80*10**9),'history_squashed':squashed}
    Path(out).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));return result

if __name__=='__main__':retention()
