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


def read_snapshot(artifact):
    with zipfile.ZipFile(artifact) as z:
        receipt = json.loads(z.read('archive_receipt.json'))
        with tarfile.open(fileobj=io.BytesIO(z.read('v13_state.tar.gz')), mode='r:gz') as t:
            manifest_raw = t.extractfile('v13_checkpoint_manifest.json').read()
        if hashlib.sha256(manifest_raw).hexdigest() != receipt['manifest_sha256']:
            raise ValueError('Snapshot manifest hash mismatch')
        source_raw = z.read('v13_state/redevelopment/latest.json')
        source = json.loads(source_raw)
        replay = source['prediction_replay']
        if hashlib.sha256(z.read('v13_state/active_heads.joblib')).hexdigest() != replay['model_sha256']:
            raise ValueError('Snapshot model identity mismatch')
    frame = pd.DataFrame(source['records'])
    if frame.empty or not {'date','isin','symbol','close','p6','p12','loss12'}.issubset(frame):
        raise ValueError('No complete scored snapshot')
    frame['date'] = pd.to_datetime(frame.date)
    if (not frame.date.eq(pd.Timestamp(source['source_date'])).all() or
        frame.duplicated(['date','isin']).any() or frame['isin'].isna().any()):
        raise ValueError('Mixed source dates or ambiguous issuer identity')
    replay = {**replay, 'source_json_sha256':hashlib.sha256(source_raw).hexdigest()}
    return frame, json.loads(manifest_raw), receipt, replay


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
    download = VerifiedObjectDownloader(items,restore/'objects',os.environ['HF_ARCHIVE_TOKEN'])
    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(lambda item:materialize_one(item,restore,download),items))
    summary = export(frame,out,replay,history=restore/'history'/'securities')
    result = pd.read_csv(out/'chart_interpretation'/'all_states.csv', float_precision='round_trip')
    protected = [c for c in ['isin','p6','p12','loss12','joint12','research_selected','research_rank'] if c in frame]
    pd.testing.assert_frame_equal(frame[protected].sort_values('isin').reset_index(drop=True),
        result[protected].sort_values('isin').reset_index(drop=True), check_exact=True)
    summary.update({'executed_at_utc':datetime.now(timezone.utc).isoformat(),
        'history_files_hash_verified':len(items), 'source_snapshot':receipt['snapshot'],
        'source_manifest_sha256':receipt['manifest_sha256'],
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
