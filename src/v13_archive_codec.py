"""Lossless Zstandard storage with original byte identities and dated snapshots.

The metadata is a standard Zstandard skippable frame; ordinary zstd tools can
decode the complete stream. Original checkpoint names, hashes and source clocks
are unchanged. Compression affects physical storage, never numeric precision.
"""
import hashlib,os,re,struct,tempfile,threading
from pathlib import Path

HEADER=struct.Struct('<4sI8sQ32s')
MAGIC=b'\x50\x2a\x4d\x18'
TAG=b'V13COMP1'
MAX_DECODED_BYTES=16*1024**3
_locks={};_guard=threading.Lock()

def enabled():return os.environ.get('V13_COMPACT_ARCHIVE')=='1'

def sha_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk:=f.read(1024*1024):h.update(chunk)
    return h.hexdigest()

def envelope(path):
    with Path(path).open('rb') as f:raw=f.read(HEADER.size)
    if len(raw)<HEADER.size:return None
    magic,length,tag,size,digest=HEADER.unpack(raw)
    if magic!=MAGIC or tag!=TAG:return None
    if length!=HEADER.size-8 or size>MAX_DECODED_BYTES:raise ValueError('Invalid compressed archive envelope')
    return {'sha256':digest.hex(),'bytes':size,'codec':'zstd','version':1}

def encode_file(source,target,expected_sha256=None,level=9):
    import zstandard as zstd
    source,target=Path(source),Path(target);size=source.stat().st_size
    digest=sha_file(source)
    if expected_sha256 and digest!=expected_sha256:raise ValueError('Original bytes changed before compression')
    target.parent.mkdir(parents=True,exist_ok=True)
    h=hashlib.sha256()
    with target.open('wb') as dest:
        dest.write(HEADER.pack(MAGIC,HEADER.size-8,TAG,size,bytes.fromhex(digest)))
        with zstd.ZstdCompressor(level=level,write_checksum=True,write_content_size=True).stream_writer(dest,size=size,closefd=False) as writer:
            with source.open('rb') as src:
                while chunk:=src.read(1024*1024):writer.write(chunk);h.update(chunk)
    if h.hexdigest()!=digest:target.unlink(missing_ok=True);raise ValueError('Original bytes changed while compressing')
    return {'original_sha256':digest,'original_bytes':size,'sha256':sha_file(target),'bytes':target.stat().st_size}

def decode_file(source,target,expected_sha256=None,expected_bytes=None):
    import zstandard as zstd
    source,target=Path(source),Path(target);meta=envelope(source)
    wanted_sha=expected_sha256 or (meta or {}).get('sha256')
    wanted_size=expected_bytes if expected_bytes is not None else (meta or {}).get('bytes')
    if meta and (wanted_sha!=meta['sha256'] or wanted_size!=meta['bytes']):
        raise ValueError('Compressed archive metadata differs from original checkpoint')
    if wanted_size is not None and wanted_size>MAX_DECODED_BYTES:raise ValueError('Decoded archive exceeds allowed size')
    target.parent.mkdir(parents=True,exist_ok=True);h=hashlib.sha256();total=0
    handle=tempfile.NamedTemporaryFile(dir=target.parent,prefix='.decode-',delete=False);temporary=Path(handle.name)
    try:
        with source.open('rb') as src,handle as dest:
            reader=zstd.ZstdDecompressor().stream_reader(src,read_across_frames=True,closefd=False) if meta else src
            try:
                while chunk:=reader.read(1024*1024):
                    total+=len(chunk)
                    if total>MAX_DECODED_BYTES or wanted_size is not None and total>wanted_size:
                        raise ValueError('Decoded archive exceeds original checkpoint length')
                    h.update(chunk);dest.write(chunk)
            finally:
                if meta:reader.close()
        if wanted_size is not None and total!=wanted_size or wanted_sha and h.hexdigest()!=wanted_sha:
            raise ValueError('Decoded archive differs from original checkpoint bytes')
        temporary.replace(target)
    finally:temporary.unlink(missing_ok=True)
    return target

def logical_download(path,filename,cache=None):
    """Decode an encoded Hub object, or return an untouched legacy object."""
    path=Path(path);meta=envelope(path)
    if meta is None:return str(path)
    name=Path(filename).name
    canonical=re.fullmatch(r'([0-9a-f]{64})\.bin',name)
    snapshot=re.fullmatch(r'\d{8}T\d{6}Z_([0-9a-f]{12})\.json',name)
    if canonical and canonical.group(1)!=meta['sha256'] or snapshot and not meta['sha256'].startswith(snapshot.group(1)):
        raise ValueError('Compressed archive original identity differs from its canonical name')
    folder=Path(cache or os.environ.get('V13_DECODED_CACHE','v13_decoded_objects'))
    target=folder/(meta['sha256']+'.decoded')
    with _guard:lock=_locks.setdefault(str(target),threading.Lock())
    with lock:
        if not target.is_file() or target.stat().st_size!=meta['bytes']:
            decode_file(path,target,meta['sha256'],meta['bytes'])
    return str(target)

def archive_download(repo_id,filename,**kwargs):
    from huggingface_hub import hf_hub_download
    canonical=re.fullmatch(r'v13/(?:objects/|object_shards/[0-9a-f]{2}/[0-9a-f]{2}/|packed_objects/[0-9a-f]{2}/[0-9a-f]{2}/)([0-9a-f]{64})\.bin',filename)
    if canonical and enabled():
        from v13_archive_catalog import fetch_member
        found=fetch_member(repo_id,canonical.group(1),kwargs.get('token'))
        if found is not None:return found
    try:physical=hf_hub_download(repo_id,filename,**kwargs)
    except Exception as exc:
        from huggingface_hub.errors import EntryNotFoundError
        if not canonical or not isinstance(exc,EntryNotFoundError):raise
        from v13_archive_catalog import fetch_member
        found=fetch_member(repo_id,canonical.group(1),kwargs.get('token'),refresh=True)
        if found is None:
            from v13_working_archive import fetch_by_sha
            found=fetch_by_sha(repo_id,canonical.group(1),kwargs.get('token'))
        if found is None:raise
        return found
    base=kwargs.get('cache_dir')
    return logical_download(physical,filename,Path(base)/'decoded' if base else None)

def compressible(remote):
    return (remote.startswith(('v13/objects/','v13/object_shards/','v13/packed_objects/')) and remote.endswith('.bin')
      or remote=='v13/state/latest_manifest.json'
      or remote.startswith('v13/state/snapshots/') and remote.endswith('.json'))

def physical_spec(item):
    if not enabled() or not compressible(item['remote']) or item['bytes']<256:return item
    source=Path(item['file'])
    if sha_file(source)!=item['sha256'] or source.stat().st_size!=item['bytes']:
        raise ValueError('Checkpoint source changed before physical encoding')
    if envelope(source):return item
    folder=Path(os.environ.get('V13_STORAGE_STAGING_DIR','v13_encoded_uploads'))
    target=folder/(item['sha256']+'.zst')
    if not target.exists():encode_file(source,target,item['sha256'])
    meta=envelope(target)
    if not meta or meta['sha256']!=item['sha256'] or meta['bytes']!=item['bytes']:
        raise ValueError('Existing encoded upload has another original identity')
    if target.stat().st_size>=int(item['bytes']*.98):return item
    length=target.stat().st_size;blob=hashlib.sha1(b'blob '+str(length).encode()+b'\0')
    with target.open('rb') as f:
        while chunk:=f.read(1024*1024):blob.update(chunk)
    return {**item,'file':str(target),'bytes':length,'sha256':sha_file(target),'git_blob':blob.hexdigest()}
