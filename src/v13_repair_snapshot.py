"""Recover original public financial gaps and replay a dated model snapshot.

This review never saves private archive state, edits registries, trains heads,
or advances a stale market date. The regular single-writer collector separately
persists the same collection fixes on its next normal run.
"""
import argparse,hashlib,io,json,zipfile
from pathlib import Path
from datetime import datetime,timezone
import joblib,pandas as pd
from v13_collection import collect_financials,enrich
from v13_financial_evidence import enrich_evidence
from v13_universe_financial_audit import audit
from v13_financial_crosscheck import export as crosscheck
from v13_paired_forecasts import scored
from v13_prediction_replay import snapshot
from v13_research_selection import export as research_export
from v13_redevelopment import export as redevelopment_export,MARKET_FEATURES,BOTTOM_FEATURES
from v13_2x_report import export as two_x_export,_seed_classifications


def run(artifact,diagnostic,output,budget=1600):
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    started=datetime.now(timezone.utc).isoformat()
    with zipfile.ZipFile(artifact) as z:
        report=json.loads(z.read('v13_state/latest_run.json'));asof=report['asof']
        receipt=report['exact_prediction_replay']['champion'];key=receipt['input_parquet_sha256']
        raw=z.read('v13_state/prediction_replay/'+key+'/inputs.parquet')
        if hashlib.sha256(raw).hexdigest()!=key:raise ValueError('Original inference snapshot hash mismatch')
        original=pd.read_parquet(io.BytesIO(raw))
        allowed=['financial_wide.parquet','active_heads.joblib','identity/symbol_changes.json',
          'evidence/valuation_snapshot.json','evidence/valuation_research_hurdles.json',
          'evidence/market_coverage_funnel.json','catalyst_evidence/document_evidence.json',
          'early_potential/coverage_queue.json']
        for name in allowed:
            if 'v13_state/'+name not in z.namelist():continue
            p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(z.read('v13_state/'+name))
        old_rd=pd.DataFrame(json.loads(z.read('v13_state/redevelopment/latest.json'))['records'])
        old_queue=json.loads(z.read('v13_state/early_potential/coverage_queue.json'))
    if not (root/'active_heads.joblib').exists():
        seed=Path(__file__).resolve().parents[1]/'v13_seed'/'active_heads.joblib.b64'
        import base64
        (root/'active_heads.joblib').write_bytes(base64.b64decode(seed.read_text()))
    model_path=root/'active_heads.joblib'
    if hashlib.sha256(model_path.read_bytes()).hexdigest()!=receipt['model_sha256']:raise ValueError('Frozen model identity mismatch')
    with zipfile.ZipFile(diagnostic) as z:
        for name in ['catalog.parquet','universe.parquet']:
            (root/name).write_bytes(z.read(name))
    # Record successful original versions without assuming that all economic
    # periods or owner facts in those documents have already been extracted.
    wide=pd.read_parquet(root/'financial_wide.parquet')
    observed=wide[wide.source_url.notna()]
    accepted=sorted(set(observed.source_url+'|'+observed.available_at_utc))
    (root/'accepted_filings.json').write_text(json.dumps(accepted))
    (root/'history/securities').mkdir(parents=True,exist_ok=True)
    recovery=collect_financials(root,asof,budget)
    (root/'primary_recovery.json').write_text(json.dumps(recovery,indent=2))
    print('PRIMARY_RECOVERY',json.dumps({k:v for k,v in recovery.items() if k!='errors'}),flush=True)
    wide=pd.read_parquet(root/'financial_wide.parquet')
    financial_audit=audit(pd.read_parquet(root/'universe.parquet'),wide,asof,root)
    source_comparison=crosscheck(root,wide,asof)
    frame=enrich(original,wide);frame=enrich_evidence(frame,wide,root,asof)
    # Market features and the verified current cohort come from the exact
    # original snapshot. Missing current bars remain missing, without an overlay.
    current=scored(joblib.load(model_path),frame)
    replay=snapshot(frame,current,model_path,root,'corrected_research')
    research_export(current,root,replay)
    extra=[c for c in MARKET_FEATURES+BOTTOM_FEATURES if c in old_rd and c not in current]
    current=current.merge(old_rd[['isin',*extra]],on='isin',how='left',validate='1:1')
    redevelopment_export(current,root,replay)
    # No histories were downloaded in this read-only financial replay. Restore
    # the actual source collector's coverage queue, rather than infer all absent.
    (root/'early_potential/coverage_queue.json').write_text(json.dumps(old_queue,indent=2))
    classes=_seed_classifications(root/'two_x_research/classifications/verified_20261016','Controlled dated snapshot replay')
    result=two_x_export(root,replay,classification_dir=classes)
    before=original[['isin','symbol','date']].copy()
    # Input equality of raw market features is a stronger check than seeing a
    # plausible stock list. Only the dated financial enrichment is recomputed.
    market_cols=[c for c in original if c in ['date','symbol','isin','close','rsi14','log_ret63','distance_ma50','distance_ma200','market_breadth63','relative_strength63']]
    if not frame[market_cols].equals(original[market_cols]):raise ValueError('Financial repair altered original market inputs')
    summary={'review_started_at_utc':started,'review_completed_at_utc':datetime.now(timezone.utc).isoformat(),
      'source_asof':asof,'source_snapshot_sha256':hashlib.sha256(Path(artifact).read_bytes()).hexdigest(),
      'original_input_sha256':key,'corrected_prediction_replay':replay,
      'new_primary_fact_observations':recovery['new_facts'],'originals_attempted':recovery['requested'],
      'original_source_errors':len(recovery['errors']),'financial_coverage':financial_audit,
      'financial_crosscheck':source_comparison,'two_x_output':result,
      'market_cohort_rows':len(frame),'market_inputs_unchanged':True,
      'frozen_heads_or_prior_registries_changed':False,'private_archive_state_written':False,
      'full_market_rerun':False,'reason':'Latest source-aligned market cohort retained; missing current provider closes remain outside fresh inference.'}
    (root/'correction_run.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--artifact',required=True);p.add_argument('--diagnostic',required=True);p.add_argument('--out',required=True);p.add_argument('--budget',type=int,default=1600)
    a=p.parse_args();run(a.artifact,a.diagnostic,a.out,a.budget)
