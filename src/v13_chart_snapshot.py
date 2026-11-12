"""Read-only chart postprocessor for any completed V13 worker snapshot."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
import zipfile

import pandas as pd

from v13_chart_interpretation import export
from v13_worker_artifact import read_manifest, verified_direct, extract_delta_histories


def read_snapshot(artifact):
    manifest, receipt, _ = read_manifest(artifact)
    with zipfile.ZipFile(artifact) as z:
        source_raw = verified_direct(z, manifest, 'v13_state/redevelopment/latest.json')
        source = json.loads(source_raw)
        replay = source['prediction_replay']
        if hashlib.sha256(verified_direct(z, manifest, 'v13_state/active_heads.joblib')).hexdigest() != replay['model_sha256']:
            raise ValueError('Snapshot model identity mismatch')
    frame = pd.DataFrame(source['records'])
    if frame.empty or not {'date','isin','symbol','close','p6','p12','loss12'}.issubset(frame):
        raise ValueError('No complete scored snapshot')
    frame['date'] = pd.to_datetime(frame.date)
    if (not frame.date.eq(pd.Timestamp(source['source_date'])).all() or
        frame.duplicated(['date','isin']).any() or frame['isin'].isna().any()):
        raise ValueError('Mixed source dates or ambiguous issuer identity')
    replay = {**replay, 'source_json_sha256':hashlib.sha256(source_raw).hexdigest()}
    return frame, manifest, receipt, replay


def verify_forecast_export(frame, result):
    """Check exact saved values despite CSV's untyped all-null rank column."""
    protected = [c for c in ['isin','p6','p12','loss12','joint12','research_selected','research_rank'] if c in frame]
    left = frame[protected].sort_values('isin').reset_index(drop=True).copy()
    right = result[protected].sort_values('isin').reset_index(drop=True).copy()
    if 'research_rank' in protected:
        # JSON nulls form an object column when no name is selected; CSV
        # infers float64 NaNs. Canonicalise only this nullable numeric field.
        # Values, null positions and every other protected dtype stay checked.
        for table in (left, right):
            table['research_rank'] = pd.to_numeric(table['research_rank'], errors='raise').astype('Float64')
    pd.testing.assert_frame_equal(left, right, check_exact=True)


def run(artifact, output='chart_snapshot', restore_root=None):
    from v13_checkpoint import VerifiedObjectDownloader, materialize_one
    frame, manifest, receipt, replay = read_snapshot(artifact)
    out = Path(output);out.mkdir(parents=True, exist_ok=True)
    restore = Path(restore_root) if restore_root is not None else out
    wanted = {'history/securities/'+hashlib.sha256(s.encode()).hexdigest()+'.parquet'
              for s in frame.symbol}
    items = [r for r in manifest['files'] if r['path'] in wanted]
    if len(items) != len(wanted):
        raise ValueError('Scored history objects missing from exact snapshot')
    recovered = extract_delta_histories(artifact, manifest, items, restore)
    missing = []
    for item in items:
        path = restore/item['path']
        if not (path.is_file() and path.stat().st_size == item['bytes']
                and hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256']):
            missing.append(item)
    if missing:
        download = VerifiedObjectDownloader(missing,restore/'objects',os.environ['HF_ARCHIVE_TOKEN'])
        with ThreadPoolExecutor(max_workers=16) as pool:
            list(pool.map(lambda item:materialize_one(item,restore,download),missing))
    summary = export(frame,out,replay,history=restore/'history'/'securities')
    result = pd.read_csv(out/'chart_interpretation'/'all_states.csv', float_precision='round_trip')
    verify_forecast_export(frame, result)
    summary.update({'executed_at_utc':datetime.now(timezone.utc).isoformat(),
        'history_files_hash_verified':len(items), 'source_snapshot':receipt['snapshot'],
        'source_manifest_sha256':receipt['manifest_sha256'],
        'source_manifest_kind':receipt['manifest_kind'],
        'source_remote_publication_confirmed':receipt.get('remote_publication_confirmed',True),
        'history_files_recovered_from_verified_delta':recovered,
        'forecast_values_and_selection_unchanged':True,
        'archive_or_frozen_forecasts_written':False,
        'out_of_sample_performance_proven':False})
    (out/'chart_interpretation'/'snapshot_validation.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)
    return summary


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--artifact', required=True)
    p.add_argument('--output', default='chart_snapshot')
    p.add_argument('--restore-root')
    a=p.parse_args()
    run(a.artifact,a.output,a.restore_root)
