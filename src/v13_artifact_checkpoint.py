"""Explicit recovery-only checkpoint mode for a confirmed archive quota stop."""
import hashlib,json,os
from pathlib import Path
from v13_checkpoint import manifest
from v13_checkpoint_writes import recovery_delta

def is_storage_limit(exc):
    return 'private storage limit reached' in str(exc).lower()

def recovery_checkpoint(root):
    root=Path(root);base=root/'remote_manifest.json'
    if not base.exists():raise RuntimeError('Artifact recovery requires an existing verified archive base')
    base_raw=base.read_bytes();previous=json.loads(base_raw)
    known={r['sha256'] for r in previous['files']};data=manifest(root)
    entries={r['sha256']:r for r in data['files'] if r['sha256'] not in known}
    recovery=recovery_delta(root,data,list(entries.values()),base_raw)
    return {'storage_mode':'VERIFIED_RECOVERY_PENDING_GITHUB_ARTIFACT_UPLOAD',
      'remote_publication_confirmed':False,'archive_capacity_blocked':True,
      'base_manifest_sha256':hashlib.sha256(base_raw).hexdigest(),
      'recovery':recovery,'github_artifact_upload_required':True}

def save_with_quota_fallback(root,save):
    enabled=os.environ.get('V13_ARTIFACT_CHECKPOINT_FALLBACK')=='1'
    marker=Path(root).parent/'v13_archive_capacity.json'
    if enabled and marker.exists():return recovery_checkpoint(root)
    try:return save(root)
    except Exception as exc:
        if not enabled or not is_storage_limit(exc):raise
        marker.write_text(json.dumps({'archive_capacity_blocked':True,'fallback':'GITHUB_ARTIFACT_RECOVERY','remote_checkpoint_updated':False}))
        return recovery_checkpoint(root)
