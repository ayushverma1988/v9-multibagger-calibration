"""Create a portable dated ZIP of verified tables, CSVs and research records."""
import argparse,hashlib,io,json,re,zipfile
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
from v13_worker_artifact import read_manifest,verified_direct

def compact_parquet(raw):
    table=pq.read_table(io.BytesIO(raw));output=io.BytesIO()
    pq.write_table(table,output,compression='zstd',compression_level=9)
    encoded=output.getvalue()
    pd.testing.assert_frame_equal(table.to_pandas(),pq.read_table(io.BytesIO(encoded)).to_pandas(),check_exact=True,check_dtype=True)
    return encoded

def run(artifact,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    manifest,source,manifest_raw=read_manifest(artifact)
    with zipfile.ZipFile(artifact) as z:
        latest=json.loads(verified_direct(z,manifest,'v13_state/latest_run.json'))
        date=str(latest['asof'])
        if not re.fullmatch(r'\d{4}-\d{2}-\d{2}',date):raise ValueError('Source export has no valid date')
        archive=out/f'V13_Dated_Data_{date}.zip';rows=[]
        with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as target:
            for name in z.namelist():
                if not name.startswith('v13_state/') or not name.endswith(('.parquet','.csv','.json','.md')):continue
                raw=verified_direct(z,manifest,name);relative=name[len('v13_state/'):]
                encoded=compact_parquet(raw) if name.endswith('.parquet') else raw
                key='snapshot_asof/'+date+'/'+relative
                target.writestr(key,encoded,compress_type=zipfile.ZIP_STORED if name.endswith('.parquet') else zipfile.ZIP_DEFLATED)
                rows.append({'file':key,'source_bytes':len(raw),'exported_bytes':len(encoded),
                  'source_sha256':hashlib.sha256(raw).hexdigest(),'exported_sha256':hashlib.sha256(encoded).hexdigest(),
                  'numeric_values_and_dates_unchanged':True,'storage':'PARQUET_ZSTD_9' if name.endswith('.parquet') else 'ZIP_DEFLATE_9'})
            target.writestr('source_manifest.json',manifest_raw)
            target.writestr('index.json',json.dumps({'snapshot_asof':date,'source_manifest_sha256':source['manifest_sha256'],'files':rows},indent=2))
            target.writestr('README.md',f'''# V13 dated data export\n\nSnapshot source date: {date}.\n\nTables use Parquet with lossless Zstandard compression. Existing CSV, JSON and Markdown records are included in the ZIP. The snapshot date identifies the saved view; each table retains its own original observation dates, issuer identities, missing values and numeric precision. CSV files can be extracted for Excel.\n\nEvery included input was checked against the original V13 checkpoint manifest. Recompressed Parquet values and types were compared exactly. Model binaries and the large emergency-recovery container are excluded from this portable data export; they remain in the working V13 archive. Source-document bytes and all historical vintages remain in the compressed shared archive, rather than being copied into this export.\n\nSee index.json for file hashes and source_manifest.json for the complete original checkpoint reference list.\n''')
    result={'source_asof':date,'source_manifest_sha256':source['manifest_sha256'],'source_manifest_kind':source['manifest_kind'],
      'source_worker_run':latest.get('workflow_run'),'verified_exported_files':len(rows),
      'original_table_and_record_bytes':sum(r['source_bytes'] for r in rows),'zip_bytes':archive.stat().st_size,
      'zip_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'zip_file':archive.name,
      'numeric_values_missing_values_dates_and_types_unchanged':True,'is_complete_model_runtime_backup':False}
    (out/'export_receipt.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--artifact',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();run(a.artifact,a.out)
