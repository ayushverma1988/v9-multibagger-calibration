"""Small date-wise indexes over one shared compressed immutable object pool."""
import hashlib,json,re
from datetime import datetime,timezone
from pathlib import Path
from zoneinfo import ZoneInfo

POLICY={'format':'V13-lossless-storage-policy-1','codec':'zstandard','compression_level':9,
  'tabular_codec':'parquet-zstd-9','new_data_minimum_retention_days':730,
  'training_baseline_and_frozen_forecasts_retained':True,'restart_cache_copies':2,
  'original_numeric_precision_and_missing_values_preserved':True,
  'archive_git_history_is_not_the_dated_data_index':True}

def snapshot_index(key,data,root=None):
    match=re.fullmatch(r'(\d{8}T\d{6}Z)_([0-9a-f]{12})',key)
    if not match:raise ValueError('Invalid dated snapshot key')
    saved=datetime.strptime(match.group(1),'%Y%m%dT%H%M%SZ').replace(tzinfo=timezone.utc)
    source=None
    if root and (Path(root)/'latest_run.json').exists():source=json.loads((Path(root)/'latest_run.json').read_text()).get('asof')
    date=str(source) if source and re.fullmatch(r'\d{4}-\d{2}-\d{2}',str(source)) else saved.astimezone(ZoneInfo('Asia/Kolkata')).strftime('%Y-%m-%d')
    sha=hashlib.sha256(json.dumps(data,sort_keys=True,indent=2).encode()).hexdigest()
    if not sha.startswith(match.group(2)):raise ValueError('Dated snapshot key differs from original manifest hash')
    useful={'financial_wide.parquet','prospective_observations.parquet','paired_mature_observations.parquet',
      'master_data/master_inputs.parquet','master_data/master_input_sources.parquet'}
    index={'format':'V13-dated-data-index-1','source_asof':source,'archive_saved_at_utc':saved.isoformat(),
      'snapshot':'v13/state/snapshots/'+key+'.json','manifest_sha256':sha,
      'shared_object_pool':True,'full_daily_data_copy_created':False,
      'minimum_retention_days':730,'tables':[r for r in data['files'] if r['path'] in useful],
      'restart_bundle':data.get('derived_state_bundle')}
    return 'v13/date_index/'+date.replace('-','/')+'/'+key+'.json',index

def encoding_index(logical,physical,folder):
    import pyarrow as pa
    import pyarrow.parquet as pq
    now=datetime.now(timezone.utc).isoformat()
    rows=[{'logical_path':a['remote'],'original_sha256':a['sha256'],'original_bytes':a['bytes'],
      'stored_sha256':b['sha256'],'stored_bytes':b['bytes'],'codec':'zstd','recorded_at_utc':now}
      for a,b in zip(logical,physical) if a!=b]
    if not rows:return None
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);path=folder/'encoding-index.parquet'
    pq.write_table(pa.Table.from_pylist(rows),path,compression='zstd',compression_level=9)
    sha=hashlib.sha256(path.read_bytes()).hexdigest();target=folder/(sha+'.parquet');path.replace(target)
    day=datetime.now(ZoneInfo('Asia/Kolkata')).strftime('%Y/%m/%d')
    return 'v13/storage_index/'+day+'/'+sha+'.parquet',target
