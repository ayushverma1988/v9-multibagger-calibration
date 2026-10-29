"""Bounded checkpoint uploads with timeout reconciliation and recovery deltas.

HF documents that a timed-out commit can have completed server-side. Inspect
the exact file digests before repeating it, and recreate mutated operations.
"""
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import gzip
import hashlib
import json
from pathlib import Path
import re
import tarfile
import time

MAX_BATCH = 50
MAX_ATTEMPTS = 4
TRANSIENT = {408, 425, 429, 500, 502, 503, 504}


class CheckpointTemporaryError(RuntimeError):
    pass


def status(exc):
    return getattr(getattr(exc, 'response', None), 'status_code', None)


def temporary(exc):
    if status(exc) in TRANSIENT:
        return True
    module = type(exc).__module__.split('.')[0]
    name = type(exc).__name__
    return isinstance(exc, (TimeoutError, ConnectionError)) or (
        module in {'httpx', 'httpcore', 'requests'} and
        any(word in name for word in ('Timeout', 'Connection', 'Connect', 'ReadError', 'Transport', 'RemoteProtocol')))


def check_deadline(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise CheckpointTemporaryError('Checkpoint save time budget exhausted; local recovery delta retained')


def pause(exc, attempt, deadline=None):
    headers = getattr(getattr(exc, 'response', None), 'headers', {}) or {}
    seconds = None
    value = headers.get('Retry-After')
    if value:
        try:
            seconds = max(0., float(value))
        except (TypeError, ValueError):
            try:
                seconds = max(0., (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds())
            except (TypeError, ValueError):
                pass
    if seconds is None and status(exc) == 429:
        match = re.search(r';\s*t=(\d+)', headers.get('RateLimit', ''))
        seconds = float(match.group(1)) if match else 60.
    if seconds is None:
        seconds = min(2 ** attempt, 30)
    # Honor throttling without one long uninterruptible sleep.
    while seconds > 0:
        check_deadline(deadline)
        chunk = min(seconds, 30.)
        time.sleep(chunk)
        seconds -= chunk


def spec(path, file, sha256=None, size=None):
    file = Path(file)
    raw = file.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if (sha256 is not None and sha256 != digest) or (size is not None and size != len(raw)):
        raise ValueError('Checkpoint upload source changed')
    blob = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
    return {'remote':path, 'file':str(file), 'sha256':digest, 'bytes':len(raw), 'git_blob':blob}


def _get(obj, key, default=None):
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


def committed(api, archive, specs):
    infos = api.get_paths_info(repo_id=archive, repo_type='dataset',
                               paths=[s['remote'] for s in specs])
    by_path = {_get(r, 'path', _get(r, 'rfilename')):r for r in infos}
    for s in specs:
        item = by_path.get(s['remote'])
        if item is None or _get(item, 'size') != s['bytes']:
            return False
        lfs = _get(item, 'lfs')
        if lfs:
            if _get(lfs, 'sha256', _get(lfs, 'oid')) != s['sha256']:
                return False
        elif _get(item, 'blob_id') != s['git_blob']:
            return False
    return True


def write(api, archive, specs, message, operation_type, confirmed=None, split=True, deadline=None):
    """Fresh operations per request; never split an atomic manifest publish."""
    if not specs:
        return
    for attempt in range(MAX_ATTEMPTS):
        check_deadline(deadline)
        # create_commit mutates CommitOperationAdd; do not reuse after failure.
        operations = []
        for s in specs:
            raw = Path(s['file']).read_bytes()
            if len(raw) != s['bytes'] or hashlib.sha256(raw).hexdigest() != s['sha256']:
                raise ValueError('Checkpoint upload source changed during retry')
            operations.append(operation_type(path_in_repo=s['remote'], path_or_fileobj=Path(s['file'])))
        try:
            api.create_commit(repo_id=archive, repo_type='dataset',
                commit_message=message, operations=operations)
        except Exception as exc:
            if not temporary(exc):
                raise
            # A 504 may mean the server committed but its reply was lost.
            # If metadata is temporarily unavailable, retry the inspection
            # first; never treat an unreadable postcondition as success.
            seen = None
            for inspection in range(MAX_ATTEMPTS):
                check_deadline(deadline)
                try:
                    seen = committed(api, archive, specs)
                    break
                except Exception as check_exc:
                    if not temporary(check_exc):
                        raise
                    if inspection == MAX_ATTEMPTS - 1:
                        raise CheckpointTemporaryError('Checkpoint commit status remains unavailable') from None
                    pause(check_exc, inspection, deadline)
            if seen:
                if confirmed is not None:
                    confirmed(specs)
                return
            if split and len(specs) > 1 and status(exc) in {408, 500, 502, 503, 504}:
                middle = len(specs) // 2
                write(api, archive, specs[:middle], message, operation_type, confirmed, True, deadline)
                write(api, archive, specs[middle:], message, operation_type, confirmed, True, deadline)
                return
            if attempt == MAX_ATTEMPTS - 1:
                code = status(exc)
                raise CheckpointTemporaryError('Checkpoint upload failed after bounded attempts: ' +
                    ('HTTP ' + str(code) if code else type(exc).__name__)) from None
            pause(exc, attempt, deadline)
        else:
            if confirmed is not None:
                confirmed(specs)
            return


class UploadProgress:
    """Only confirmed immutable object hashes are reusable between save calls."""
    def __init__(self, path, archive):
        self.path, self.archive = Path(path), archive
        old = json.loads(self.path.read_text()) if self.path.exists() else {}
        if old and (old.get('format') != 'V13-confirmed-object-uploads-1' or old.get('archive') != archive):
            raise ValueError('Checkpoint upload journal identity mismatch')
        self.objects = old.get('objects', {})

    def contains(self, item):
        expected = {'bytes':item['bytes'], 'object_path':item['object_path']}
        return self.objects.get(item['sha256']) == expected

    def confirm(self, specs):
        for s in specs:
            self.objects[s['sha256']] = {'bytes':s['bytes'], 'object_path':s['remote']}
        raw = json.dumps({'format':'V13-confirmed-object-uploads-1',
            'archive':self.archive, 'objects':self.objects}, sort_keys=True).encode()
        temp = self.path.with_suffix('.tmp')
        temp.write_bytes(raw);temp.replace(self.path)


def recovery_delta(root, data, entries, base_raw):
    """Preserve unpublished local bytes even if the archive stays unavailable.

    A base manifest plus a complete delta is sufficient to reconstruct the
    attempted state using the base's hash-verified derived bundle. The old
    remote pointer is never promoted by creating this local recovery copy.
    """
    root = Path(root)
    target = root.parent / 'v13_checkpoint_recovery.tar.gz'
    temporary_path = target.with_suffix('.tmp')
    import io
    planned = json.dumps(data, sort_keys=True, indent=2).encode()
    metadata = {'format':'V13-checkpoint-recovery-delta-1',
        'base_manifest_sha256':hashlib.sha256(base_raw).hexdigest(),
        'planned_manifest_sha256':hashlib.sha256(planned).hexdigest(),
        'delta_objects':len(entries), 'remote_publication_confirmed':False}
    with temporary_path.open('wb') as dest, gzip.GzipFile(filename='', mode='wb', fileobj=dest, mtime=0, compresslevel=1) as gz, tarfile.open(fileobj=gz, mode='w|') as tar:
        payloads = [('recovery.json', json.dumps(metadata, sort_keys=True).encode()),
            ('base_manifest.json', base_raw), ('planned_manifest.json', planned)]
        for name, raw in payloads:
            info = tarfile.TarInfo(name);info.size=len(raw);info.mtime=0
            tar.addfile(info, io.BytesIO(raw))
        for r in entries:
            rel=Path(r['path'])
            if rel.is_absolute() or '..' in rel.parts:
                raise ValueError('Unsafe recovery delta path')
            source=root/rel
            raw=source.read_bytes()
            if len(raw)!=r['bytes'] or hashlib.sha256(raw).hexdigest()!=r['sha256']:
                raise ValueError('Recovery delta source integrity failure')
            info=tarfile.TarInfo('objects/'+r['sha256']+'.bin');info.size=len(raw);info.mtime=0
            tar.addfile(info,io.BytesIO(raw))
    temporary_path.replace(target)
    metadata.update(bytes=target.stat().st_size,
                    sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    (root.parent/'v13_checkpoint_recovery.json').write_text(json.dumps(metadata,indent=2))
    return metadata
