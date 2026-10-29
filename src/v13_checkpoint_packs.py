"""Bounded immutable byte packs; each member retains its own SHA-256 identity.

Packs reduce archive commit operations, never source checks or stored evidence.
The manifest binds each member's digest, length and offset to the pack digest.
"""
import hashlib
from pathlib import Path

MAX_PACK_BYTES = 32 * 1024 * 1024
MAX_PACK_MEMBERS = 500


def pack_path(digest):
    import re
    if not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest):
        raise ValueError('Invalid checkpoint pack digest')
    return f'v13/packed_objects/{digest[:2]}/{digest[2:4]}/{digest}.bin'


def validate_pack_member(item):
    pack = item.get('storage_pack')
    if pack is None:
        if 'pack_offset' in item:
            raise ValueError('Checkpoint member offset without pack')
        return
    if not isinstance(pack, dict) or set(pack) != {'sha256', 'bytes', 'object_path'}:
        raise ValueError('Invalid checkpoint pack descriptor')
    if pack['object_path'] != pack_path(pack['sha256']):
        raise ValueError('Checkpoint pack identity mismatch')
    offset, size, total = item.get('pack_offset'), item['bytes'], pack['bytes']
    if any(type(n) is not int for n in (offset, size, total)) or min(offset, size, total) < 0:
        raise ValueError('Invalid checkpoint pack bounds')
    if total > MAX_PACK_BYTES or offset + size > total:
        raise ValueError('Checkpoint member outside pack')


def build_packs(root, entries, folder):
    """Return upload descriptors and member storage mappings, deterministically."""
    root, folder = Path(root), Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    uploads, members, group, size = [], {}, [], 0

    def flush():
        if not group:
            return
        temp = folder / 'building.pack'
        digest = hashlib.sha256()
        offsets = []
        offset = 0
        try:
            with temp.open('wb') as dest:
                for item in group:
                    raw = (root / item['path']).read_bytes()
                    if len(raw) != item['bytes'] or hashlib.sha256(raw).hexdigest() != item['sha256']:
                        raise ValueError('Checkpoint pack input changed')
                    offsets.append((item['sha256'], offset))
                    dest.write(raw)
                    digest.update(raw)
                    offset += len(raw)
            sha = digest.hexdigest()
            target = folder / (sha + '.bin')
            temp.replace(target)
        finally:
            temp.unlink(missing_ok=True)
        pack = {'sha256': sha, 'bytes': offset, 'object_path': pack_path(sha)}
        uploads.append({**pack, 'file': str(target)})
        for sha, offset in offsets:
            members[sha] = {'storage_pack': dict(pack), 'pack_offset': offset}

    for item in sorted(entries, key=lambda r: r['sha256']):
        if item['bytes'] > MAX_PACK_BYTES:
            raise ValueError('Oversized object must use direct storage')
        if group and (size + item['bytes'] > MAX_PACK_BYTES or len(group) >= MAX_PACK_MEMBERS):
            flush()
            group, size = [], 0
        group.append(item)
        size += item['bytes']
    flush()
    return uploads, members


def read_member(item, fetch):
    """Verify the complete pack and the exact member, including zero-byte data."""
    validate_pack_member(item)
    pack = item['storage_pack']
    path = Path(fetch(pack['object_path']))
    if path.stat().st_size != pack['bytes']:
        raise ValueError('Checkpoint pack size mismatch')
    # Packs are bounded to 32 MiB; never retain an unbounded archive in memory.
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != pack['sha256']:
        raise ValueError('Checkpoint pack integrity failure')
    start = item['pack_offset']
    value = raw[start:start + item['bytes']]
    if len(value) != item['bytes'] or hashlib.sha256(value).hexdigest() != item['sha256']:
        raise ValueError('Checkpoint packed member integrity failure')
    return value
