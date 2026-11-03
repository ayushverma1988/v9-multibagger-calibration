"""Resolve original content hashes from bounded compressed migration packs."""
import hashlib,json,threading,tempfile
from pathlib import Path
from v13_archive_codec import archive_download,logical_download,sha_file,envelope

_prefixes={};_guard=threading.RLock();_verified=set();_batches={};_pack_files={};_pack_locks={}

def validate_row(row):
    from v13_checkpoint import object_path
    sha=row['original_sha256'];object_path(sha)
    pack=row['pack_sha256'];object_path(pack)
    if row['pack_path']!=f'v13/compact_packs/{pack[:2]}/{pack}.zst':raise ValueError('Invalid compact pack address')
    nums=[row[k] for k in ('original_bytes','offset','pack_original_bytes','pack_bytes')]
    if any(type(x) is not int or x<0 for x in nums) or row['offset']+row['original_bytes']>row['pack_original_bytes']:
        raise ValueError('Compact pack member is outside its original bounds')
    if row['pack_original_bytes']>128*1024**2:raise ValueError('Unbounded compact migration pack')

def prefix_rows(repo,prefix,token=None,refresh=False):
    from huggingface_hub.errors import EntryNotFoundError
    import pyarrow.parquet as pq
    key=(repo,prefix)
    with _guard:
        if key in _prefixes and not refresh:return _prefixes[key]
        try:p=archive_download(repo,f'v13/storage_catalog/{prefix}/index.json',repo_type='dataset',token=token,force_download=refresh)
        except EntryNotFoundError:_prefixes[key]={};return {}
        index=json.loads(Path(p).read_text())
        if index.get('format')!='V13-compressed-object-catalog-1' or index.get('prefix')!=prefix:
            raise ValueError('Unexpected compressed object catalog')
        result={}
        for descriptor in index['batches']:
            identity=(repo,descriptor['sha256'])
            if identity not in _batches:
                path=archive_download(repo,descriptor['path'],repo_type='dataset',token=token)
                if sha_file(path)!=descriptor['sha256']:raise ValueError('Compact catalog batch identity mismatch')
                _batches[identity]=pq.read_table(path).to_pylist()
            for row in _batches[identity]:
                validate_row(row);sha=row['original_sha256']
                if not sha.startswith(prefix) or sha in result and result[sha]!=row:
                    raise ValueError('Conflicting compact original identity')
                result[sha]=row
        _prefixes[key]=result;return result

def fetch_member(repo,sha,token=None,cache=None,refresh=False):
    from huggingface_hub import hf_hub_download
    from v13_archive_codec import decode_file
    row=prefix_rows(repo,sha[:2],token,refresh).get(sha)
    if row is None:return None
    validate_row(row)
    folder=Path(cache or 'v13_decoded_objects');folder.mkdir(parents=True,exist_ok=True)
    target=folder/(sha+'.decoded')
    if target.is_file() and target.stat().st_size==row['original_bytes']:return str(target)
    identity=(repo,row['pack_path'],str(folder))
    with _guard:lock=_pack_locks.setdefault(identity,threading.Lock())
    with lock:
        saved=_pack_files.get(identity)
        if saved is None or not Path(saved['path']).exists():
            physical=hf_hub_download(repo,row['pack_path'],repo_type='dataset',token=token,cache_dir=folder/'compressed_cache')
            if Path(physical).stat().st_size!=row['pack_bytes'] or sha_file(physical)!=row['pack_sha256']:
                raise ValueError('Compressed compact pack physical identity mismatch')
            meta=envelope(physical)
            if not meta or meta['sha256']!=row['pack_original_sha256'] or meta['bytes']!=row['pack_original_bytes']:
                raise ValueError('Compressed pack decoded identity mismatch')
            saved={'path':logical_download(physical,row['pack_path'],folder/'packs'),
              'sha256':meta['sha256'],'bytes':meta['bytes'],'stored_bytes':row['pack_bytes']};_pack_files[identity]=saved
        if saved['sha256']!=row['pack_original_sha256'] or saved['bytes']!=row['pack_original_bytes'] or saved['stored_bytes']!=row['pack_bytes']:
            raise ValueError('Conflicting compact pack decoded metadata')
        decoded=saved['path']
    with Path(decoded).open('rb') as f:f.seek(row['offset']);raw=f.read(row['original_bytes'])
    if len(raw)!=row['original_bytes'] or hashlib.sha256(raw).hexdigest()!=sha:
        raise ValueError('Compact source differs from its original frozen hash')
    with tempfile.NamedTemporaryFile(dir=folder,prefix='.member-',delete=False) as f:
        f.write(raw);temporary=Path(f.name)
    temporary.replace(target);return str(target)
