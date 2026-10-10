"""Immutable V12.1 prospective research registration and mature outcomes.

The entry must be a verified actual first trade after registration. Future
observations remain pending; missing evidence is UNKNOWN, never a loss/zero.
"""
from __future__ import annotations
import argparse,hashlib,json,os
from datetime import datetime,timezone
from pathlib import Path
import pandas as pd
import numpy as np
from v12_four_year_model import COST_PER_SIDE,sha_file

def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(value):return hashlib.sha256(canonical(value)).hexdigest()

def register(package,predictions,output,now=None):
    root=Path(package);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((root/'freeze_manifest.json').read_text())
    for file in ['frozen_model.joblib','frozen_recipe.json','current_predictions_and_gate_audit.parquet']:
        if sha_file(root/file)!=manifest['files_sha256'][file]:raise ValueError('Frozen artifact changed')
    p=pd.read_parquet(predictions)
    if sha_file(predictions)!=manifest['files_sha256']['current_predictions_and_gate_audit.parquet']:raise ValueError('Registration must use exact saved predictions')
    clock=pd.Timestamp(now or datetime.now(timezone.utc))
    if clock.tzinfo is None:raise ValueError('Registration clock requires timezone')
    clock=clock.tz_convert('UTC')
    if clock<pd.Timestamp(manifest['frozen_at_utc']):raise ValueError('Backdated registration')
    data_date=pd.to_datetime(p.date).max()
    if data_date.tz_localize('Asia/Kolkata')+pd.Timedelta(hours=15,minutes=30)>clock:raise ValueError('Future unfinished quote snapshot')
    if (clock.tz_convert('Asia/Kolkata').tz_localize(None).normalize()-data_date).days>4:raise ValueError('Snapshot too old for new registration')
    if p.duplicated(['isin','date']).any():raise ValueError('Duplicate forecast identities')
    columns=['p_hit2_6','p_hit2_12','p_loss30_12'];a=p[columns].to_numpy(float)
    if not np.isfinite(a).all() or (a<0).any() or (a>1).any() or p.p_hit2_6.gt(p.p_hit2_12).any():raise ValueError('Invalid frozen probabilities')
    source_checked=p[p.current_quality_status.eq('CURRENT_SOURCE_CHECKS_PASS')].sort_values(['p_hit2_12','symbol'],ascending=[False,True]).head(10)
    selected=set(source_checked['isin'])
    records=[]
    for _,r in p.sort_values('isin').iterrows():
        records.append({'isin':r['isin'],'symbol':r['symbol'],'venue':'NSE','data_asof_date':str(pd.Timestamp(r['date']).date()),'decision_at_utc':clock.isoformat(),**{c:float(r[c]) for c in columns},'source_gate_status':str(r['current_quality_status']),'research_top10_selected':r['isin'] in selected,'threshold_pass':bool(r['high_conviction']),'trading_approved':False,'entry_status':'PENDING_ACTUAL_VERIFIED_FIRST_TRADE','outcome6_status':'PENDING','outcome12_status':'PENDING'})
    protocol={'adjudication_code_sha256':sha_file(Path(__file__)),'core_source_sha256':sha_file(Path(__file__).with_name('v12_four_year_model.py')),'entry':'VERIFIED_FIRST_ACTUAL_NSE_TRADE_AFTER_REGISTRATION','entry_proof_required':True,'calendar_months':[6,12],'minimum_observed_horizon_rows':'months*16','maximum_observation_gap_days':10,'maximum_unverified_daily_log_jump':'log(1.4)','maximum_deadline_end_gap_days':5,'full_horizon_required_even_after_early_hit':True,'target':'CLOSING_ADJUSTED_TOTAL_RETURN_PROXY_2X_NET','cost_per_side':COST_PER_SIDE,'missing_selected_outcomes':'UNKNOWN_kept_in_denominator','high_only_touch_is_not_success':True,'production_approval':False}
    registration={'schema':'V12_PROSPECTIVE_RESEARCH_2','registered_at_utc':clock.isoformat(),'original_prediction_created_at_utc':manifest['frozen_at_utc'],'version':manifest['version'],'model_sha256':manifest['files_sha256']['frozen_model.joblib'],'recipe_sha256':manifest['files_sha256']['frozen_recipe.json'],'prediction_sha256':sha_file(predictions),'protocol':protocol,'protocol_sha256':digest(protocol),'records':records,'six_twelve_month_deadlines':'ACTUAL_ENTRY_DATE_PLUS_CALENDAR_MONTHS','cost_per_side':COST_PER_SIDE,'target':'CLOSING_ADJUSTED_TOTAL_RETURN_PROXY_2X_NET','unseen_matured_outcomes':0,'production_approved':False}
    reg_hash=digest(registration)
    # O_EXCL prevents accidental overwrites, including a second registration.
    with (out/'registration.json').open('x') as f:json.dump(registration,f,indent=2,allow_nan=False)
    receipt={'registration_sha256':reg_hash,'protocol_sha256':registration['protocol_sha256'],'model_sha256':registration['model_sha256'],'recipe_sha256':registration['recipe_sha256'],'predictions':len(records),'research_top10':len(selected),'threshold_passes':sum(r['threshold_pass'] for r in records),'registered_at_utc':clock.isoformat(),'unseen_matured_outcomes':0,'outcome_status':'PENDING_FUTURE_MARKET_OBSERVATIONS'}
    with (out/'public_registration_receipt.json').open('x') as f:json.dump(receipt,f,indent=2)
    pd.DataFrame(records).to_csv(out/'prospective_prediction_registry_PRIVATE.csv',index=False)
    return receipt

def validate_registry(root):
    root=Path(root);reg=json.loads((root/'registration.json').read_text());receipt=json.loads((root/'public_registration_receipt.json').read_text())
    if digest(reg)!=receipt['registration_sha256']:raise ValueError('Registration altered')
    if reg.get('schema')!='V12_PROSPECTIVE_RESEARCH_2' or digest(reg['protocol'])!=reg['protocol_sha256']:raise ValueError('Unsupported/changed outcome protocol')
    if sha_file(Path(__file__))!=reg['protocol']['adjudication_code_sha256'] or sha_file(Path(__file__).with_name('v12_four_year_model.py'))!=reg['protocol']['core_source_sha256']:raise ValueError('Frozen adjudication code changed; use registered revision')
    return reg,receipt

def adjudicate_one(record,bars,months,observed_at):
    if months not in [6,12]:raise ValueError('Unsupported outcome horizon')
    clock=pd.Timestamp(observed_at)
    if clock.tzinfo is None:raise ValueError('Outcome clock requires timezone')
    decision=pd.Timestamp(record['decision_at_utc']);base={'isin':record['isin'],'symbol':record['symbol'],'months':months,'outcome':None}
    required={'date','open','high','low','close','adj_close','isin','venue','currency','source_sha256','first_retrieved_utc','first_trade_after_decision_verified','entry_proof_sha256'}
    if not required.issubset(bars):return {**base,'status':'UNKNOWN_MISSING_SOURCE_SCHEMA'}
    d=bars.copy();d['date']=pd.to_datetime(d.date).dt.normalize();d=d.sort_values('date')
    if d.date.duplicated().any():raise ValueError('Duplicate prospective source dates')
    if not d['isin'].eq(record['isin']).all():raise ValueError('Outcome security identity mismatch')
    if not d.venue.eq('NSE').all() or not d.currency.eq('INR').all():raise ValueError('Outcome venue/currency mismatch')
    seen=pd.to_datetime(d.first_retrieved_utc,utc=True,format='mixed',errors='coerce')
    if seen.isna().any() or seen.gt(clock).any():raise ValueError('Future/unclocked observation')
    close_clock=d.date.dt.tz_localize('Asia/Kolkata')+pd.Timedelta(hours=15,minutes=30)
    if close_clock.gt(seen).any():raise ValueError('Observation retrieved before its completed session')
    open_clock=d.date.dt.tz_localize('Asia/Kolkata')+pd.Timedelta(hours=9,minutes=15)
    future=d[open_clock.gt(decision)].copy()
    if future.empty:return {**base,'status':'PENDING_ENTRY'}
    entry=future.iloc[0]
    if entry['first_trade_after_decision_verified'] is not True and entry['first_trade_after_decision_verified']!=True:return {**base,'status':'UNKNOWN_FIRST_ACTUAL_TRADE_NOT_VERIFIED'}
    if not isinstance(entry['entry_proof_sha256'],str) or len(entry['entry_proof_sha256'])!=64:return {**base,'status':'UNKNOWN_FIRST_TRADE_PROOF_MISSING'}
    deadline=entry.date+pd.DateOffset(months=months);base.update({'entry_date':str(entry.date.date()),'deadline':str(deadline.date())})
    # Require completion of deadline session; an early hit never matures the label.
    if clock<deadline.tz_localize('Asia/Kolkata')+pd.Timedelta(hours=15,minutes=30):return {**base,'status':'PENDING_HORIZON'}
    f=future[future.date.le(deadline)]
    if d.date.max()<deadline or len(f)<months*16 or (deadline-f.date.max()).days>5 or f.date.diff().dt.days.max()>10:return {**base,'status':'UNKNOWN_INCOMPLETE_HORIZON'}
    numbers=f[['open','high','low','close','adj_close']].to_numpy(float)
    if not np.isfinite(numbers).all() or (numbers<=0).any() or not f.low.le(f[['open','close']].min(axis=1)).all() or not f.high.ge(f[['open','close']].max(axis=1)).all():return {**base,'status':'UNKNOWN_INVALID_PRICES'}
    if np.log(f.adj_close).diff().abs().max()>np.log(1.4):return {**base,'status':'UNKNOWN_UNVERIFIED_PRICE_DISCONTINUITY'}
    entry_adj=float(entry.open*entry.adj_close/entry.close);net=f.adj_close.to_numpy(float)/entry_adj*(1-COST_PER_SIDE)/(1+COST_PER_SIDE)
    lows=f.low.to_numpy(float)*f.adj_close.to_numpy(float)/f.close.to_numpy(float)/entry_adj*(1-COST_PER_SIDE)/(1+COST_PER_SIDE)
    hits=np.flatnonzero(net>=2);losses=np.flatnonzero(lows<=.7)
    risk=int(bool(len(losses)) and (not len(hits) or losses[0]<=hits[0]))
    return {**base,'status':'MATURED_RESEARCH_PROXY','outcome':int(bool(len(hits))),'loss30_before_hit2':risk,'endpoint_return':float(net[-1]-1),'peak_return':float(net.max()-1),'independent_corporate_actions_verified':False}

def append_evaluation(registry,bars,observed_at):
    root=Path(registry);reg,receipt=validate_registry(root)
    frame=pd.read_parquet(bars);groups={k:g for k,g in frame.groupby('isin')};rows=[]
    for record in reg['records']:
        for months in [6,12]:rows.append(adjudicate_one(record,groups.get(record['isin'],pd.DataFrame()),months,observed_at))
    previous='0'*64;path=root/'evaluation_chain.jsonl'
    if path.exists():
        for line in path.read_text().splitlines():
            item=json.loads(line);payload=item['payload']
            if payload['previous_sha256']!=previous or digest(payload)!=item['sha256']:raise ValueError('Evaluation chain altered')
            previous=item['sha256']
    payload={'observed_at_utc':pd.Timestamp(observed_at).isoformat(),'registration_sha256':receipt['registration_sha256'],'previous_sha256':previous,'source_file_sha256':sha_file(bars),'rows':rows}
    item={'payload':payload,'sha256':digest(payload)}
    with path.open('a') as f:f.write(json.dumps(item,allow_nan=False)+'\n')
    return {'hash':item['sha256'],'status_counts':pd.Series([r['status'] for r in rows]).value_counts().to_dict(),'matured_outcomes':sum(r['outcome'] is not None for r in rows)}

def main():
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['register','evaluate'],required=True);p.add_argument('--output',required=True);p.add_argument('--package');p.add_argument('--predictions');p.add_argument('--bars');p.add_argument('--observed-at');a=p.parse_args()
    result=register(a.package,a.predictions,a.output) if a.kind=='register' else append_evaluation(a.output,a.bars,a.observed_at or datetime.now(timezone.utc).isoformat());print(json.dumps(result,indent=2))
if __name__=='__main__':main()
