"""Bounded Apollo data repair, exact-head reassessment and dual-screen proof."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile

import joblib
import numpy as np
import pandas as pd


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protected(root):
    root = Path(root)
    paths = list(root.glob('registries/*/*.json'))+list(root.glob('paired_cohorts/*.json'))
    paths += [root/p for p in ('active_heads.joblib','active_model.json','challenger/research_heads.joblib',
        'challenger/freeze.json','prospective_observations.parquet','champion_latest_watchlist.json','challenger_latest_watchlist.json')]
    return {str(p.relative_to(root)):digest(p) for p in paths if p.is_file()}


def run(root, out, publish=False):
    from v13_checkpoint import restore, save
    from v13_frozen_integrity import FrozenGuard
    from v13_archive_retention import download_artifact
    from v13_financial_recovery_seed import apply
    from v13_image_financial_reviews import collect
    from v12_dated_source_collection import sha
    from v12_5_features import features
    from v13_financial_evidence import balancing_quarters, financial_quality
    from v13_available_financials import assess, export as available_export
    from v13_model import probabilities, FEATURES
    from v13_prediction_replay import snapshot
    from v13_priority_research import export as priority_export
    root, out = Path(root), Path(out);out.mkdir(parents=True, exist_ok=True)
    if not restore(root):raise ValueError('No verified saved V13 state')
    guard=FrozenGuard(root);before_protected=protected(root)
    seed=Path('v13_evidence_seed')
    before=pd.read_parquet(root/'financial_wide.parquet')
    before_apollo=before[before.security_isin.eq('INE713T01028')].copy()
    before_apollo.to_json(out/'apollo_facts_before.json',orient='records',date_format='iso',double_precision=15)
    # Reuse the primary bytes actually collected earlier today. Their original
    # retrieval clocks are retained; this is not another source request/retry.
    capture=out/'primary_capture.zip'
    download_artifact(13915601609,'sha256:852d4d20728e83d61d0424b35eb786d0dd33d0f27cb5f82eeeb5cf93ff95a089',str(capture))
    cache=root/'filing_raw/reviewed_image_originals';cache.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(capture) as archive:
        for name,suffix in [('results','.pdf'),('catalogue','.json')]:
            proof=json.loads(archive.read(name+'_receipt.json'));raw=archive.read(name+suffix)
            if hashlib.sha256(raw).hexdigest()!=proof['sha256']:raise ValueError('Original primary capture changed')
            key=sha(proof['url'].encode());(cache/(key+'.bin')).write_bytes(raw)
            (cache/(key+'.json')).write_text(json.dumps({**proof,'requested_url':proof['url'],'status':200}))
            (out/(name+suffix)).write_bytes(raw)
            (out/(name+'_receipt.json')).write_text(json.dumps(proof,indent=2))
    capture.unlink()
    machine=apply(root,seed/'financial_primary_recovery_20261112.json','2026-11-12',marker_name='financial_primary_seed_recovery_20261112.json')
    if machine['errors'] or machine['same_original_numeric_conflicts']:raise ValueError('Current original financial recovery failed')
    image=collect(root,seed/'image_financial_reviews_20261112.json','2026-11-12')
    if not image['complete']:raise ValueError('Reviewed image recovery failed: '+json.dumps(image['errors']))
    wide=pd.read_parquet(root/'financial_wide.parquet')
    after_apollo=wide[wide.security_isin.eq('INE713T01028')].copy()
    after_apollo.to_parquet(out/'apollo_facts_after.parquet',index=False,compression='zstd')
    after_apollo.to_json(out/'apollo_facts_after.json',orient='records',date_format='iso',double_precision=15)
    # None of the existing economic observations may be replaced by this repair.
    keys=['source_sha256','period_kind','period_end','security_isin','reporting_mode','available_at_utc']
    restored=after_apollo.set_index(keys).loc[before_apollo.set_index(keys).index].reset_index()
    pd.testing.assert_frame_equal(before_apollo.sort_values(keys).reset_index(drop=True),
        restored[before_apollo.columns].sort_values(keys).reset_index(drop=True),check_dtype=False,check_exact=True)
    parent=out/'latest_parent.zip'
    download_artifact(13912504824,'sha256:33196362aff11a4b385d23d13aad5bb9fb86a22a3c4d9d4d0624becfe3998079',str(parent))
    with zipfile.ZipFile(parent) as archive:
        import io
        frame=pd.read_parquet(io.BytesIO(archive.read('latest_model_inputs_and_predictions.parquet')))
        expected={str(t):json.loads(archive.read(f'latest_RSI{t}.json')) for t in (70,50)}
    parent.unlink()
    model_path=root/'active_heads.joblib'
    if digest(model_path)!='c9208f89c46d043ee1d74e44dee1c85b9721b7a094e40a0fabf2c5506c246571':
        raise ValueError('The preserved model differs from the reviewed source run')
    heads=joblib.load(model_path);old=frame[frame['isin'].eq('INE713T01028')].copy()
    replay=probabilities(heads,old)
    for key in ('p6','p12','joint12','loss12'):
        if not np.allclose(replay[key],old[key],rtol=0,atol=1e-12):raise ValueError('Original Apollo replay differs')
    now=pd.Timestamp.now(tz='UTC');known=balancing_quarters(after_apollo[after_apollo.available.le(now)])
    financial,_=features(known,pd.Timestamp('2026-11-12'))
    eps=[r for r in json.loads((root/'statement_eps/accepted.json').read_text()) if r['isin']=='INE713T01028']
    latest=max(eps,key=lambda r:r['available_at_utc'])
    financial['reported_diluted_EPS_yoy_same_document']=latest['reported_diluted_EPS_yoy_same_document']
    current=old.copy()
    for key,value in financial.items():current[key]=value
    quality=financial_quality(financial)
    current['financial_complete']=quality['complete'];current['financial_quality_pass']=quality['quality_pass']
    current['financial_unknown_fields']='|'.join(quality['unknown_fields']);current['financial_quality_status']=quality['status']
    values=probabilities(heads,current)
    changes={k:{'before':None if pd.isna(old.iloc[0].get(k)) else float(old.iloc[0][k]),'after':None if pd.isna(v) else float(v)}
        for k,v in financial.items() if k in old and not np.isclose(float(old.iloc[0][k]),v,equal_nan=True)}
    assessment=assess(known,pd.Timestamp('2026-11-12'),'NONFINANCIAL')
    result={'price_source_asof':'2026-11-11','financial_reassessment_at_utc':now.isoformat(),
        'scope':'CURRENT_RESEARCH_REASSESSMENT_NOT_A_REWRITTEN_SOURCE_DAY_FORECAST',
        'source_close_INR':float(old.iloc[0].source_close_INR),'model_sha256':digest(model_path),
        'before':{k:float(old.iloc[0][k]) for k in ('p6','p12','joint12','loss12')},
        'after':{k:float(v[0]) for k,v in values.items()},'changed_financial_inputs':changes,
        'financial_quality':quality,'financial_tier':assessment['tier'],'current_safety_failures':assessment['current_safety_failures'],
        'machine_recovery':machine,'image_recovery':image,'original_fact_rows_preserved':len(before_apollo),
        'apollo_fact_rows_after':len(after_apollo),'model_retrained':False,'new_formal_forecast_registered':False}
    (out/'apollo_corrected_feature_vector.parquet').write_bytes(_parquet_bytes(current[FEATURES]))
    correction=root/'data_corrections';correction.mkdir(exist_ok=True)
    (correction/'apollo_20261112.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    integrity=guard.check(pd.read_parquet(root/'prospective_observations.parquet'))
    if protected(root)!=before_protected:raise ValueError('A protected original changed')
    result['protected_files_unchanged']=True;result['frozen_integrity']=integrity
    if publish:
        result['checkpoint']=save(root)
        if not result['checkpoint'].get('remote_publication_confirmed'):raise ValueError('Corrected state not durably confirmed')
        fresh=out.parent/'fresh_reloaded_state'
        if not restore(fresh):raise ValueError('Independent corrected-state restart failed')
        if digest(fresh/'financial_wide.parquet')!=digest(root/'financial_wide.parquet') or digest(fresh/'statement_eps/accepted.json')!=digest(root/'statement_eps/accepted.json'):
            raise ValueError('Corrected financial observations did not survive a restart')
        if protected(fresh)!=before_protected:raise ValueError('Restart changed a protected original')
        result['corrected_facts_independently_reloaded']=True
    (out/'apollo_correction_result.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print('APOLLO_CORRECTION_JSON '+json.dumps({k:v for k,v in result.items() if k not in ('machine_recovery','image_recovery')},separators=(',',':')),flush=True)
    # The two-list contract is also exercised on the original same-price,
    # same-cutoff broad research snapshot. Newly reviewed facts do not leak
    # backwards into its 11-November information clock.
    view=out/'dual_rsi_state';view.mkdir(exist_ok=True)
    receipt=snapshot(frame,frame,model_path,view,'champion')
    available_export(frame,wide,view,receipt)
    summary=priority_export(frame,wide,view,receipt)
    for threshold in (70,50):
        actual=json.loads((view/f'priority_research/latest_RSI{threshold}.json').read_text())
        old_records=expected[str(threshold)]['records']
        if [r['isin'] for r in actual['records']]!=[r['isin'] for r in old_records]:raise ValueError('The unchanged source-day list differs')
        for new,prior in zip(actual['records'],old_records):
            for key in ('p6','p12','joint12','loss12','priority_score'):
                if not np.isclose(new[key],prior[key],rtol=0,atol=1e-12):raise ValueError('The unchanged source-day score differs')
    (out/'dual_rsi_contract_result.json').write_text(json.dumps(summary,indent=2))
    print('DUAL_RSI_RESULT_JSON '+json.dumps(summary,separators=(',',':')),flush=True)
    return result


def _parquet_bytes(frame):
    import io
    stream=io.BytesIO();frame.to_parquet(stream,index=False,compression='zstd');return stream.getvalue()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--state',required=True);p.add_argument('--out',required=True);p.add_argument('--publish',action='store_true')
    args=p.parse_args();run(args.state,args.out,args.publish)
