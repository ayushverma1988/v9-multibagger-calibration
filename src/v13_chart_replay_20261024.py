"""Read-only chart replay of the exact 23 October cohort and saved histories."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import io
import json
import os
import tarfile
import zipfile

import pandas as pd

from v13_checkpoint import VerifiedObjectDownloader, materialize_one
from v13_chart_interpretation import export


def run(worker='worker.zip', rerun='rerun.zip', output='chart_replay'):
    root = Path(output);root.mkdir(exist_ok=True)
    with zipfile.ZipFile(worker) as z:
        receipt = json.loads(z.read('archive_receipt.json'))
        with tarfile.open(fileobj=io.BytesIO(z.read('v13_state.tar.gz')),mode='r:gz') as t:
            manifest_raw = t.extractfile('v13_checkpoint_manifest.json').read()
        assert hashlib.sha256(manifest_raw).hexdigest() == receipt['manifest_sha256']
        assert receipt['manifest_sha256'] == '7c118d97b7c83956358519b5b12d90c2871e22175d63a3409fa1724c86b68d8b'
        manifest = json.loads(manifest_raw)
    with zipfile.ZipFile(rerun) as z:
        raw = z.read('All_Scored_Shares_2026-10-24.csv')
        frame = pd.read_csv(io.BytesIO(raw), float_precision='round_trip')
    assert len(frame) == 666 and frame.date.eq('2026-10-23').all()
    assert not frame.duplicated(['date','isin']).any()
    wanted = {'history/securities/' + hashlib.sha256(s.encode()).hexdigest() + '.parquet'
              for s in frame.symbol}
    items = [r for r in manifest['files'] if r['path'] in wanted]
    assert len(items) == len(wanted)
    download = VerifiedObjectDownloader(items,root/'objects',os.environ['HF_ARCHIVE_TOKEN'])
    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(lambda item:materialize_one(item,root,download),items))
    summary = export(frame,root,{'source_run':40096314555,
        'model_sha256':'c9208f89c46d043ee1d74e44dee1c85b9721b7a094e40a0fabf2c5506c246571',
        'source_csv_sha256':hashlib.sha256(raw).hexdigest()})
    current = pd.read_csv(root/'chart_interpretation'/'all_states.csv', float_precision='round_trip')
    protected = ['isin','p6','p12','loss12','joint12','research_selected','research_rank']
    pd.testing.assert_frame_equal(frame[protected].sort_values('isin').reset_index(drop=True),
        current[protected].sort_values('isin').reset_index(drop=True), check_exact=True)
    selected = current[current.research_selected.fillna(False)].copy()
    selected.sort_values('research_rank').to_csv(root/'chart_interpretation'/'previous_top20_chart_review.csv',index=False)
    # Small reproducible price samples for inspecting the derived chart rules.
    sample_symbols = set(selected.symbol) | set(current[current.chart_candidate].symbol)
    sample_dir = root/'chart_interpretation'/'price_samples';sample_dir.mkdir(exist_ok=True)
    for symbol in sorted(sample_symbols):
        key = hashlib.sha256(symbol.encode()).hexdigest()
        b = pd.read_parquet(root/'history'/'securities'/(key+'.parquet'))
        b = b[pd.to_datetime(b.date).le('2026-10-23')].sort_values('date')
        b.tail(320).to_csv(sample_dir/(symbol+'.csv'),index=False)
    summary.update({'executed_at_utc':datetime.now(timezone.utc).isoformat(),
        'history_files_hash_verified':len(items),'source_snapshot':receipt['snapshot'],
        'source_manifest_sha256':receipt['manifest_sha256'],
        'history_manifest_entries':items,'forecast_values_and_selection_unchanged':True,
        'archive_or_frozen_forecasts_written':False,'out_of_sample_performance_proven':False})
    (root/'chart_interpretation'/'replay_validation.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k!='history_manifest_entries'},indent=2),flush=True)


if __name__ == '__main__':
    run()
