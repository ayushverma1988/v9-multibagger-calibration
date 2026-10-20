"""One-time verified restart-bundle migration; no model or forecast changes."""
import argparse,hashlib,json
from pathlib import Path
from datetime import datetime,timezone
from v13_checkpoint import restore,save


def compact(root,receipt):
    root=Path(root);started=datetime.now(timezone.utc).isoformat()
    if not restore(root):raise RuntimeError('Existing checkpoint is required; migration cannot initialize a new state')
    old=json.loads((root/'remote_manifest.json').read_text())
    before={r['path']:(r['sha256'],r['bytes'],r['object_path']) for r in old['files']}
    published=save(root)
    new=json.loads((root/'remote_manifest.json').read_text())
    after={r['path']:(r['sha256'],r['bytes'],r['object_path']) for r in new['files']}
    if before!=after:raise ValueError('Bundle migration changed original checkpoint contents')
    verify=root.with_name(root.name+'_bundle_verify')
    if not restore(verify):raise RuntimeError('Bundle restart verification failed')
    for r in new['files']:
        if r['raw_source']:continue
        p=verify/r['path']
        if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest()!=r['sha256']:raise ValueError('Restored bundle differs from original derived state')
    result={'started_at_utc':started,'completed_at_utc':datetime.now(timezone.utc).isoformat(),
      'passed':True,'original_checkpoint_records_unchanged':True,'frozen_forecasts_and_heads_changed':False,
      'bundle_restore_verified':True,'checkpoint_files':len(new['files']),**published}
    Path(receipt).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--state',default='v13_compaction_state');p.add_argument('--receipt',default='checkpoint_compaction_receipt.json');a=p.parse_args();compact(a.state,a.receipt)
