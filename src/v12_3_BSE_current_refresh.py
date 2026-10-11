"""Current BSE research inference with overlap-checked free-source history.

Refresh the whole public reference cohort, never only model-selected names.
Reject unexplained historical revisions rather than splicing adjustment bases.
No model fitting, historical result replacement or trading approval occurs.
"""
from __future__ import annotations
import argparse,json,hashlib,os
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote,urlencode
import numpy as np,pandas as pd,joblib
from v12_dated_source_collection import Store,sha
from v12_public_history import bounds
from v12_bse_transfer import build_transfer_panel
from v12_3_selective_evaluation import choose,VARIANTS
import v12_dated_enrichment_experiment as learner
import v12_coherent_hazard_model as coherent


def merge_recent(raw,receipt,reference,old,asof):
    if sha(raw)!=receipt['sha256']:raise ValueError('Recent response hash mismatch')
    j=json.loads(raw);results=j.get('chart',{}).get('result')
    if not isinstance(results,list) or len(results)!=1:raise ValueError('Missing recent chart')
    c=results[0];m=c.get('meta',{})
    norm=lambda x:''.join(v for v in str(x).lower() if v.isalnum())
    if m.get('symbol')!=reference['provider_symbol'] or m.get('exchangeName') not in ('BSE','BOM') or m.get('currency')!='INR' or m.get('instrumentType')!='EQUITY' or m.get('exchangeTimezoneName')!='Asia/Kolkata' or norm(m.get('longName'))!=norm(reference['provider_long_name']):
        raise ValueError('Recent BSE identity/venue/currency mismatch')
    q=c.get('indicators',{}).get('quote',[]);a=c.get('indicators',{}).get('adjclose',[]);t=c.get('timestamp',[])
    if len(q)!=1 or len(a)!=1:raise ValueError('Ambiguous recent prices')
    arrays={k:q[0].get(k,[]) for k in ['open','high','low','close','volume']};arrays['adj_close']=a[0].get('adjclose',[])
    if any(len(v)!=len(t) for v in arrays.values()):raise ValueError('Recent price arrays misaligned')
    d=pd.DataFrame(arrays);d['date']=pd.to_datetime(t,unit='s',utc=True).tz_convert('Asia/Kolkata').tz_localize(None).normalize()
    if d.date.duplicated().any():raise ValueError('Duplicate recent source date')
    lo,hi=bounds(asof);d=d[d.date.between(lo,hi)].copy()
    for k in arrays:d[k]=pd.to_numeric(d[k],errors='coerce')
    valid=np.isfinite(d[list(arrays)].to_numpy(float)).all(axis=1)&d[['open','high','low','close','adj_close']].gt(0).all(axis=1)&d.volume.ge(0)&d.low.le(d[['open','close']].min(axis=1))&d.high.ge(d[['open','close']].max(axis=1))
    d=d[valid].sort_values('date');old=old.copy();old['date']=pd.to_datetime(old.date)
    if d.empty or not d.date.max()==hi:raise ValueError('Requested current session unavailable')
    seen=pd.Timestamp(receipt['first_retrieved_utc'])
    if seen.tzinfo is None or hi.tz_localize('Asia/Kolkata')+pd.Timedelta(hours=15,minutes=30)>seen:raise ValueError('Unfinished current session')
    if old.empty or old.date.duplicated().any() or not old['isin'].eq(reference['isin']).all() or not old.symbol.eq(reference['provider_symbol']).all():raise ValueError('Historical identity conflict')
    if not old.date.between(lo,hi).all():raise ValueError('History outside four-year boundary')
    common=old.merge(d,on='date',suffixes=('_old','_new'),validate='one_to_one')
    if len(common)<3:raise ValueError('Fewer than three overlapping sessions')
    for k in ['open','high','low','close','adj_close']:
        if not (common[k+'_old']-common[k+'_new']).abs().le(.011).all():raise ValueError('Unreconciled historical price/adjustment revision: '+k)
    if not common.volume_old.eq(common.volume_new).all():raise ValueError('Unreconciled historical volume revision')
    new=d[d.date.gt(old.date.max())].copy();new['symbol']=reference['provider_symbol'];new['isin']=reference['isin'];new['source_sha256']=receipt['sha256']
    merged=pd.concat([old,new],ignore_index=True).sort_values('date')
    if merged.date.duplicated().any() or merged.date.max()!=hi:raise ValueError('Current merge incomplete')
    return merged,{'overlapping_sessions_checked':len(common),'new_bars':len(new),'latest_bar':str(hi.date()),'recent_source_sha256':receipt['sha256'],'provider_ISIN_binding_independently_verified':False,'corporate_actions_independently_verified':False}


def run(reference,history,nse_scores,model,output,asof):
    out=Path(output)
    if (out/'summary.json').exists() or any((out/'securities').glob('*.parquet')):
        raise FileExistsError('Use a fresh output directory; never reuse previous normalized histories')
    out.mkdir(parents=True,exist_ok=True);(out/'securities').mkdir(exist_ok=True)
    with (out/'run_started.json').open('x') as f:
        json.dump({'started_at_utc':datetime.now(timezone.utc).isoformat(),'asof':asof,
                   'code_sha256':sha(Path(__file__).read_bytes()),'production_approved':False},f,indent=2)
    ref=pd.read_parquet(reference);ref.to_parquet(out/'public_request_universe.parquet',index=False)
    if ref.duplicated('isin').any():raise ValueError('Ambiguous public universe')
    store=Store(out/'raw');audit=[];errors=[]
    def fetch(r):
        try:
            oldpath=Path(history)/'securities'/(sha(r['isin'].encode())+'.parquet')
            if not oldpath.exists():raise ValueError('Four-year history unavailable')
            url='https://query1.finance.yahoo.com/v8/finance/chart/'+quote(r['provider_symbol'],safe='')+'?'+urlencode({'interval':'1d','range':'1mo','events':'div,splits'})
            raw,receipt=store.get(url);bars,info=merge_recent(raw,receipt,r,pd.read_parquet(oldpath),asof)
            bars.to_parquet(out/'securities'/oldpath.name,index=False)
            return {'isin':r['isin'],'symbol':r['provider_symbol'],'name':r['provider_long_name'],'first_retrieved_utc':receipt['first_retrieved_utc'],**info},None
        except Exception as e:return None,{'isin':r['isin'],'symbol':r['provider_symbol'],'error':str(e)[:240]}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i,(a,e) in enumerate(pool.map(fetch,ref.to_dict('records')),1):
            if a:audit.append(a)
            if e:errors.append(e)
            if i%100==0:print('BSE refreshed',i,'usable',len(audit),'errors',len(errors),flush=True)
    pd.DataFrame(audit).to_parquet(out/'history_refresh_audit.parquet',index=False)
    (out/'errors.json').write_text(json.dumps(errors,indent=2))
    nse=pd.read_parquet(nse_scores);nse=nse[nse.date.eq(pd.Timestamp(asof))]
    x=build_transfer_panel(out,nse,asof)
    for c in learner.EXTRA:x[c]=np.nan
    packages=joblib.load(model);early=learner.score(packages['hit2_6'],x)
    x['p_hit2_6']=early;x['p_hit2_12']=coherent.union_probability(early,learner.score(packages[coherent.LATE],x));x['p_loss30_12']=learner.score(packages[coherent.LOSS],x)
    x['approved_for_trading']=False;x.to_parquet(out/'current_BSE_RESEARCH_scores.parquet',index=False)
    picks=[];counts={}
    for v in VARIANTS:
        counts[v]={}
        for h in (6,12):
            p=choose(x,v,h).copy();counts[v][str(h)]=len(p);p['variant']=v;p['months']=h;picks.append(p)
    pd.concat(picks,ignore_index=True).to_parquet(out/'current_BSE_RESEARCH_comparisons.parquet',index=False)
    summary={'asof':asof,'created_at_utc':datetime.now(timezone.utc).isoformat(),'public_universe':len(ref),'histories_refreshed':len(audit),'source_errors_or_conflicts':len(errors),'feature_eligible_current_rows':len(x),'selection_counts':counts,'current_source':'Yahoo .BO recent chart with overlap checks','guaranteed_live_latency':False,'full_BSE_financial_catalyst_inputs_verified':False,'new_model_fit':False,'historical_results_changed':False,'production_approved':False,'model_sha256':sha(Path(model).read_bytes()),'scoring_code_sha256':sha(Path(__file__).read_bytes())}
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reference',required=True);p.add_argument('--history',required=True);p.add_argument('--nse-scores',required=True);p.add_argument('--model',required=True);p.add_argument('--output',required=True);p.add_argument('--asof',default='2026-10-09');a=p.parse_args()
    run(a.reference,a.history,a.nse_scores,a.model,a.output,a.asof)
