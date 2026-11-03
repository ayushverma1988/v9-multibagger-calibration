"""Read immutable legacy or unpublished-recovery worker artifacts safely."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import tarfile
import zipfile


def digest(data):
    return hashlib.sha256(data).hexdigest()


def _safe(name):
    p = PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts:
        raise ValueError('Unsafe worker member')
    return p


def _read_metadata(tar, wanted):
    found = {}
    for item in tar:
        if item.name in wanted:
            if not item.isfile() or item.size > 512*1024*1024 or item.name in found:
                raise ValueError('Invalid or duplicate worker metadata')
            found[item.name] = tar.extractfile(item).read()
            if set(found) == wanted:
                break
    if set(found) != wanted:
        raise ValueError('Worker manifest metadata missing')
    return found


def read_manifest(artifact):
    with zipfile.ZipFile(artifact) as z:
        if len(z.namelist()) != len(set(z.namelist())):
            raise ValueError('Duplicate worker ZIP member')
        receipt = json.loads(z.read('archive_receipt.json'))
        recovery = receipt.get('remote_publication_confirmed') is False
        if recovery:
            desc = json.loads(z.read('v13_checkpoint_recovery.json'))
            h = hashlib.sha256();size = 0
            with z.open('v13_checkpoint_recovery.tar.gz') as source:
                while chunk := source.read(4*1024*1024):
                    h.update(chunk);size += len(chunk)
            if h.hexdigest() != desc['sha256'] or size != desc['bytes']:
                raise ValueError('Recovery container integrity mismatch')
            if receipt.get('recovery', {}).get('sha256', desc['sha256']) != desc['sha256']:
                raise ValueError('Archive/recovery receipt mismatch')
            with z.open('v13_checkpoint_recovery.tar.gz') as source:
                with tarfile.open(fileobj=source, mode='r|gz') as tar:
                    raw = _read_metadata(tar, {'recovery.json', 'base_manifest.json', 'planned_manifest.json'})
            meta = json.loads(raw['recovery.json'])
            if (meta.get('format') != 'V13-checkpoint-recovery-delta-1'
                or digest(raw['base_manifest.json']) != meta['base_manifest_sha256']
                or digest(raw['planned_manifest.json']) != meta['planned_manifest_sha256']):
                raise ValueError('Recovery manifest binding mismatch')
            manifest_raw = raw['planned_manifest.json']
            receipt = {**receipt, 'manifest_sha256': digest(manifest_raw),
                       'manifest_kind': 'VERIFIED_RECOVERY_PLANNED_NOT_REMOTE_PUBLISHED',
                       'snapshot': None, 'remote_publication_confirmed': False,
                       'recovery_container_sha256': desc['sha256']}
        else:
            with z.open('v13_state.tar.gz') as source:
                with tarfile.open(fileobj=source, mode='r|gz') as tar:
                    manifest_raw = _read_metadata(tar, {'v13_checkpoint_manifest.json'})['v13_checkpoint_manifest.json']
            if digest(manifest_raw) != receipt['manifest_sha256']:
                raise ValueError('Snapshot manifest hash mismatch')
            receipt = {**receipt, 'manifest_kind': 'PUBLISHED_PRIVATE_CHECKPOINT'}
    manifest = json.loads(manifest_raw)
    paths = set()
    for item in manifest['files']:
        _safe(item['path'])
        if (item['path'] in paths or not re.fullmatch('[0-9a-f]{64}', item['sha256'])
            or not isinstance(item['bytes'], int) or item['bytes'] < 0):
            raise ValueError('Invalid or duplicate manifest object')
        paths.add(item['path'])
    return manifest, receipt, manifest_raw


def verified_direct(z, manifest, name):
    _safe(name)
    if not name.startswith('v13_state/'):
        raise ValueError('Not a direct state export')
    rel = name[len('v13_state/'):]
    ref = next((r for r in manifest['files'] if r['path'] == rel), None)
    if ref is None:
        raise ValueError('Direct export not bound to source manifest: '+rel)
    raw = z.read(name)
    if len(raw) != ref['bytes'] or digest(raw) != ref['sha256']:
        raise ValueError('Direct export integrity mismatch: '+rel)
    return raw


def extract_delta_histories(artifact, manifest, items, root):
    """Materialize only requested hashes from verified emergency bytes."""
    root = Path(root)
    wanted = {}
    for item in items:
        _safe(item['path'])
        wanted.setdefault(item['sha256'], []).append(item)
    with zipfile.ZipFile(artifact) as z:
        if 'v13_checkpoint_recovery.tar.gz' not in z.namelist():
            return 0
        count = 0
        with z.open('v13_checkpoint_recovery.tar.gz') as source:
            with tarfile.open(fileobj=source, mode='r|gz') as tar:
                for member in tar:
                    if not member.name.startswith('objects/') or not member.name.endswith('.bin'):
                        continue
                    sha = member.name[8:-4]
                    if sha not in wanted:
                        continue
                    if not member.isfile() or member.size != wanted[sha][0]['bytes']:
                        raise ValueError('Invalid recovery history member')
                    raw = tar.extractfile(member).read()
                    if digest(raw) != sha:
                        raise ValueError('Recovery history hash mismatch')
                    for item in wanted.pop(sha):
                        target = root/item['path'];target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(raw);count += 1
                    if not wanted:
                        break
    return count


def audit_exports(artifact, output):
    manifest, receipt, raw = read_manifest(artifact)
    out = Path(output);out.mkdir(parents=True, exist_ok=True)
    (out/'source_manifest.json').write_bytes(raw)
    (out/'source_checkpoint_receipt.json').write_text(json.dumps(receipt, indent=2))
    copied = []
    with zipfile.ZipFile(artifact) as z:
        for name in z.namelist():
            if name.startswith('v13_state/') and not name.endswith('/'):
                data = verified_direct(z, manifest, name)
                target = out/_safe(name);target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data);copied.append(name)
        for name in ('archive_receipt.json', 'v13_checkpoint_recovery.json',
                     'v13_archive_capacity.json', 'v13_worker_attempt.json', 'previous_worker_receipt.json'):
            if name in z.namelist():
                (out/name).write_bytes(z.read(name))
    result = {'source_manifest_sha256': receipt['manifest_sha256'],
              'source_manifest_kind': receipt['manifest_kind'],
              'direct_exports_verified': len(copied), 'manifest_references': len(manifest['files']),
              'all_manifest_references_individually_downloaded': False,
              'remote_publication_confirmed': receipt.get('remote_publication_confirmed', True),
              'original_source_objects_or_forecasts_modified': False, 'copied': copied}
    (out/'audit_export_receipt.json').write_text(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser();p.add_argument('--artifact', required=True);p.add_argument('--output', required=True)
    a = p.parse_args();print(json.dumps(audit_exports(a.artifact, a.output), indent=2))
