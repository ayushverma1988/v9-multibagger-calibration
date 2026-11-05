"""Restore a compact working checkpoint without executing serialized models."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
import zipfile


def restore_package(package, target):
    import zstandard as zstd
    target = Path(target)
    if target.exists():
        raise ValueError('Restore target already exists; choose a new directory')
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(package) as archive:
        if len(archive.namelist()) != len(set(archive.namelist())):
            raise ValueError('Duplicate package member')
        receipt = json.loads(archive.read('core_receipt.json'))
        if receipt.get('format') != 'V13-compact-working-core-1':
            raise ValueError('Unsupported compact core')
        packed = archive.getinfo('v13_state.tar.zst')
        if packed.file_size != receipt['payload_bytes']:
            raise ValueError('Compressed payload length mismatch')
        h = hashlib.sha256()
        with archive.open(packed) as source:
            while chunk := source.read(1024*1024):
                h.update(chunk)
        if h.hexdigest() != receipt['payload_sha256']:
            raise ValueError('Compressed payload hash mismatch')
        with tempfile.TemporaryDirectory(prefix='.v13-core-', dir=target.parent) as folder:
            root = Path(folder)/'state';root.mkdir()
            seen = set();total = 0
            with archive.open(packed) as source, zstd.ZstdDecompressor().stream_reader(source) as decoded:
                with tarfile.open(fileobj=decoded, mode='r|') as tar:
                    first = tar.next()
                    if first.name != 'source_manifest.json' or not first.isfile() or first.size != receipt['source_manifest_bytes'] or first.size > 512*1024**2:
                        raise ValueError('Invalid source manifest member')
                    raw = tar.extractfile(first).read()
                    if hashlib.sha256(raw).hexdigest() != receipt['source_manifest_sha256']:
                        raise ValueError('Source manifest hash mismatch')
                    data = json.loads(raw)
                    expected = {r['path']:r for r in data['files'] if not r['raw_source']}
                    del data
                    if len(expected) != receipt['state_files'] or sum(r['bytes'] for r in expected.values()) != receipt['state_uncompressed_bytes']:
                        raise ValueError('Working state coverage mismatch')
                    while member := tar.next():
                        rel = PurePosixPath(member.name)
                        if (not member.isfile() or rel.is_absolute() or '..' in rel.parts
                            or not member.name.startswith('state/') or member.name[6:] not in expected
                            or member.name[6:] in seen):
                            raise ValueError('Unexpected, unsafe or duplicate core member')
                        name = member.name[6:];wanted = expected[name]
                        if member.size != wanted['bytes']:
                            raise ValueError('Working member size mismatch')
                        destination = root/name;destination.parent.mkdir(parents=True, exist_ok=True)
                        digest = hashlib.sha256();count = 0
                        with tar.extractfile(member) as original, destination.open('wb') as output:
                            while chunk := original.read(1024*1024):
                                digest.update(chunk);output.write(chunk);count += len(chunk)
                        if count != wanted['bytes'] or digest.hexdigest() != wanted['sha256']:
                            raise ValueError('Working member hash mismatch')
                        seen.add(name);total += count
            if seen != set(expected):
                raise ValueError('Core omitted working checkpoint members')
            (root/'remote_manifest.json').write_bytes(raw)
            root.replace(target)
    return {'state_files_verified':len(seen), 'state_bytes_verified':total,
            'source_manifest_sha256':receipt['source_manifest_sha256'],
            'all_raw_source_evidence_included':False, 'models_executed':False}


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('package');p.add_argument('--out', required=True)
    args = p.parse_args();print(json.dumps(restore_package(args.package, args.out), indent=2))
