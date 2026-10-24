"""Execute a requested read-only rerun of the latest complete saved input cohort."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, io, json, zipfile
import joblib, numpy as np, pandas as pd
from v13_paired_forecasts import scored
from v13_research_selection import screen, RSI_MIN
from v13_early_potential import select as early_select

def run(artifact, outdir):
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(artifact) as z:
        selection = json.loads(z.read('v13_state/research_selection/latest.json'))
        report = json.loads(z.read('v13_state/latest_run.json'))
        receipt = selection['prediction_replay']
        key = receipt['input_parquet_sha256']
        raw = z.read('v13_state/prediction_replay/' + key + '/inputs.parquet')
        assert hashlib.sha256(raw).hexdigest() == key
        frame = pd.read_parquet(io.BytesIO(raw))
        headraw = z.read('v13_state/active_heads.joblib')
        assert hashlib.sha256(headraw).hexdigest() == receipt['model_sha256']
        heads = joblib.load(io.BytesIO(headraw))
        rd = pd.DataFrame(json.loads(z.read('v13_state/redevelopment/latest.json'))['records'])
        previous_predictions = None
        prefix = 'v13_state/prediction_replay/' + key + '/'
        for name in z.namelist():
            if name.startswith(prefix) and name.endswith('.json'):
                obj = json.loads(z.read(name))
                if 'probabilities' in obj and obj.get('model_sha256') == receipt['model_sha256']:
                    previous_predictions = obj['probabilities']; break
    assert RSI_MIN == 50 and frame.date.nunique() == 1
    predictions = scored(heads, frame)
    selected, audit = screen(predictions)
    rows = selected[selected.research_selected].sort_values('research_rank').copy()
    assert rows.rsi14.ge(50).all()
    expected = {r['isin']: r for r in selection['records']}
    assert set(expected) == set(rows['isin'])
    largest = 0.
    for r in rows.to_dict('records'):
        for k in ('p6','p12','loss12','joint12'):
            largest = max(largest, abs(float(r[k]) - float(expected[r['isin']][k])))
    assert largest < 1e-12, largest
    if previous_predictions:
        for k in ('p6','p12','loss12','joint12'):
            diff = np.max(np.abs(predictions[k].to_numpy(float) - np.asarray(previous_predictions[k], float)))
            assert diff < 1e-12, (k, diff)
    assert set(rd['isin']) == set(selected['isin'])
    for k in ('p6','p12','loss12','joint12'):
        rd[k] = rd['isin'].map(selected.set_index('isin')[k])
    early, early_audit = early_select(rd)
    early_rows = early[early.early_eligible].sort_values('early_rank').copy()
    assert early_rows.rsi14.ge(50).all()
    for table in (selected, rows, early, early_rows):
        table['two_x_arithmetic_reference_INR'] = table.source_close_INR * 2
    columns = ['research_rank','symbol','isin','series','source_close_INR','two_x_arithmetic_reference_INR',
               'rsi14','distance_ma200','research_financial_tier','p6','p12','loss12','joint12']
    earlycols = ['early_rank','symbol','isin','source_close_INR','two_x_arithmetic_reference_INR',
                 'rsi14','early_pattern','early_score','early_volume_cue','early_filing_cue',
                 'early_sales_growth_cue','early_owner_profit_cue','p6','p12','loss12','financial_quality_status']
    rows[columns].to_csv(out/'Algorithm_Top20_2026-10-24.csv', index=False)
    early_rows[earlycols].to_csv(out/'Early_Uptrend_Queue_2026-10-24.csv', index=False)
    selected.to_csv(out/'All_Scored_Shares_2026-10-24.csv', index=False)
    selected.to_parquet(out/'all_scored.parquet', index=False)
    early.to_parquet(out/'all_early_states.parquet', index=False)
    summary = {'executed_at_utc':datetime.now(timezone.utc).isoformat(),
        'source_price_date':str(pd.Timestamp(frame.date.iloc[0]).date()),
        'information_cutoffs_utc':sorted(frame.information_cutoff_utc.dropna().astype(str).unique().tolist()),
        'source_workflow_run':40056471417,'code_pin':'9f08d9d818cd71eaa850be1a2a5e03fdd7e5ee5f',
        'input_sha256':key,'model_sha256':receipt['model_sha256'],
        'rows_scored':len(selected),'public_universe_rows':report['public_universe_rows'],
        'selection':audit,'early_selection':early_audit,
        'selected_financial_passes':int(rows.research_financial_tier.eq('PASS_AVAILABLE_SEVEN_CHECKS').sum()),
        'selected_incomplete_names':rows[rows.research_financial_tier.ne('PASS_AVAILABLE_SEVEN_CHECKS')].symbol.tolist(),
        'maximum_selected_probability_difference_from_saved_run':largest,
        'full_saved_probability_arrays_compared':previous_predictions is not None,
        'model_retrained':False,'formal_forecasts_or_checkpoints_modified':False,
        'probability_accuracy_validated':False,
        'records':json.loads(rows[columns].to_json(orient='records')),
        'early_records':json.loads(early_rows[earlycols].to_json(orient='records')),
        'requested_shares':json.loads(selected[selected.symbol.isin(['SINGERIND','MIDWESTLTD','SOUTHBANK'])][[
            'symbol','source_close_INR','rsi14','p6','p12','loss12','research_selected',
            'research_rank','research_financial_tier','research_exclusion_reasons']].to_json(orient='records')),
        'financial_complete_rows':int(selected.financial_complete.sum()),
        'financial_quality_pass_rows':int(selected.financial_quality_pass.sum()),
        'master_data_summary':{k:report.get('master_data',{}).get(k) for k in [
            'issuers','required_input_cells','resolved_input_cells','remaining_missing_or_unresolved_cells',
            'current_usable_input_cells']}}
    (out/'Algorithm_Run_Summary_2026-10-24.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k not in ['records','early_records']}, indent=2), flush=True)
if __name__ == '__main__':
    run('latest_worker.zip','requested_rerun')
