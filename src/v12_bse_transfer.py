"""BSE-only research transfer using strictly earlier NSE training labels.

Never score historical outcomes with today's weights. Retrospective venue
transfer is distinct from prospective independent validation. Membership and
actions remain incompletely verified, therefore no production promotion.
"""
from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss
import v12_four_year_model as core
import v12_coherent_hazard_model as coherent
from v12_public_history import bounds

def parse_bse_chart(raw,reference,asof,receipt):
    j=json.loads(raw);results=j.get('chart',{}).get('result')
    if not isinstance(results,list) or len(results)!=1:raise ValueError('Missing BSE chart')
    c=results[0];m=c.get('meta',{});symbol=reference['provider_symbol']
    if m.get('symbol')!=symbol or m.get('exchangeName') not in ('BSE','BOM') or m.get('currency')!='INR' or m.get('instrumentType')!='EQUITY' or m.get('exchangeTimezoneName')!='Asia/Kolkata':raise ValueError('Wrong BSE identity/venue/currency')
    def norm(s):return ''.join(x for x in str(s).lower() if x.isalnum())
    if norm(m.get('longName'))!=norm(reference['provider_long_name']):raise ValueError('Provider issuer name changed')
    quotes=c.get('indicators',{}).get('quote',[]);adj=c.get('indicators',{}).get('adjclose',[]);stamps=c.get('timestamp',[])
    if len(quotes)!=1 or len(adj)!=1:raise ValueError('Ambiguous prices')
    arrays={k:quotes[0].get(k,[]) for k in ['open','high','low','close','volume']};arrays['adj_close']=adj[0].get('adjclose',[])
    if any(len(v)!=len(stamps) for v in arrays.values()):raise ValueError('Misaligned daily prices')
    d=pd.DataFrame(arrays);d['date']=pd.to_datetime(stamps,unit='s',utc=True).tz_convert('Asia/Kolkata').tz_localize(None).normalize()
    if d.date.duplicated().any():raise ValueError('Duplicate BSE dates')
    lo,hi=bounds(asof);d=d[d.date.between(lo,hi)].copy()
    for k in arrays:d[k]=pd.to_numeric(d[k],errors='coerce')
    valid=np.isfinite(d[list(arrays)].to_numpy(float)).all(axis=1)&d[['open','high','low','close','adj_close']].gt(0).all(axis=1)&d.volume.ge(0)&d.low.le(d[['open','close']].min(axis=1))&d.high.ge(d[['open','close']].max(axis=1))
    invalid=int((~valid).sum());d=d[valid].sort_values('date').reset_index(drop=True)
    if len(d)<127:raise ValueError('Insufficient valid BSE history')
    seen=pd.Timestamp(receipt['first_retrieved_utc'])
    if seen.tzinfo is None or d.date.max().tz_localize('Asia/Kolkata')+pd.Timedelta(hours=15,minutes=30)>seen:raise ValueError('Future unfinished session')
    d['symbol']=symbol;d['isin']=reference['isin'];d['source_sha256']=hashlib.sha256(raw).hexdigest()
    return d,{'isin':reference['isin'],'symbol':symbol,'provider_name':m.get('longName'),'valid_bars':len(d),'invalid_bars_excluded':invalid,'earliest_bar':str(d.date.min().date()),'latest_bar':str(d.date.max().date()),'source_sha256':d.source_sha256.iloc[0],'first_retrieved_utc':receipt['first_retrieved_utc'],'split_events':len(c.get('events',{}).get('splits',{})),'dividend_events':len(c.get('events',{}).get('dividends',{})),'ISIN_in_provider_response':False,'independent_corporate_actions_verified':False}

def build_transfer_panel(history,nse_panel,asof):
    nse=coherent.prepare(nse_panel);dates=set(pd.to_datetime(nse.date).unique());records=[]
    # Stock-relative predictors use the same dated NSE reference distribution.
    for path in sorted((Path(history)/'securities').glob('*.parquet')):
        d=core.make_security_features(pd.read_parquet(path),asof)
        for t in d.index[d.date.isin(dates)&d.feature_eligible]:
            r=d.loc[t].to_dict()
            r['outcome_entry_date']=str(d.iloc[t+1].date.date()) if t+1<len(d) else None
            for months in (6,12):
                r['calendar_deadline_'+str(months)]=str((d.iloc[t+1].date+pd.DateOffset(months=months)).date()) if t+1<len(d) else None
                result=core.forward_outcome(d,t,months,asof)
                for field in ['hit2','endpoint_return','peak_return','loss30','hit2_before_loss30','mature_date']:
                    r[field+'_'+str(months)]=result[field] if result else None
            records.append(r)
    if not records:raise ValueError('No BSE transfer rows')
    x=pd.DataFrame(records)
    for date,g in x.groupby('date'):
        base=nse[nse.date.eq(date)]
        if len(base)<20:raise ValueError('Insufficient dated NSE market reference')
        for src,dst in [('log_ret20','relative_strength20'),('log_ret63','relative_strength63')]:
            a=np.sort(base[src].to_numpy(float));b=g[src].to_numpy(float)
            rank=(np.searchsorted(a,b,side='left')+np.searchsorted(a,b,side='right')+1)/(2*len(a))
            x.loc[g.index,dst]=np.clip(rank,1/len(a),1)
        x.loc[g.index,'market_breadth63']=base.log_ret63.gt(0).mean()
        x.loc[g.index,'market_median_ret63']=base.log_ret63.median()
        x.loc[g.index,'market_dispersion63']=base.log_ret63.std()
    for col in ['mature_date_6','mature_date_12']:x[col]=pd.to_datetime(x[col])
    if not np.isfinite(x[list(core.FEATURES)].to_numpy(float)).all():raise ValueError('Invalid BSE features')
    if x.duplicated(['date','isin']).any():raise ValueError('Duplicate BSE security dates')
    return coherent.prepare(x)

def evaluate(history,nse_path,output,asof):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    nse=coherent.prepare(pd.read_parquet(nse_path));x=build_transfer_panel(history,nse,asof)
    x.to_parquet(out/'BSE_four_year_panel_and_outcomes.parquet',index=False)
    all_scores=[];metrics=[];splits=[]
    for date in sorted(x.date.unique()):
        if pd.Timestamp(date)>=pd.Timestamp(asof):continue
        test=x[x.date.eq(date)].copy();parts=core.chronological_parts(nse,date,'hit2_6')
        if parts is None:continue
        early=core.fit_head(*parts,'hit2_6');packages=coherent.fit_packages(nse,date)
        predictions=coherent.probabilities(packages[0],test) if packages else {'p_hit2_6':core.score(early,test)}
        for c,v in predictions.items():test[c]=v
        for h in [6,12]:
            col='p_hit2_'+str(h);target='hit2_'+str(h)
            if col not in test:continue
            # Do not remove unknown outcomes before ranking the fixed portfolio.
            pool=test[test.technical_pass].sort_values([col,'symbol'],ascending=[False,True]);picks=pool.head(10)
            matured=test['mature_date_'+str(h)].lt(pd.Timestamp(asof))&test[target].isin([0,1])
            scored=test[matured];baseline=float(parts[1].hit2_6.mean()) if h==6 else float(packages[0]['hit2_6']['past_calibration_event_rate']+(1-packages[0]['hit2_6']['past_calibration_event_rate'])*packages[0][coherent.LATE]['past_calibration_event_rate'])
            metrics.append({'date':str(pd.Timestamp(date).date()),'months':h,'eligible_pool':len(pool),'selected':len(picks),'selected_known':int(picks[target].isin([0,1]).sum()),'selected_hits':int(picks[target].eq(1).sum()),'fully_matured_selection':bool(len(picks)>0 and picks[target].isin([0,1]).all() and picks['mature_date_'+str(h)].lt(pd.Timestamp(asof)).all()),'all_matured_rows':len(scored),'brier':brier_score_loss(scored[target],scored[col]) if len(scored) else None,'baseline_NSE_calibration_prior':baseline})
        for target,parts0 in (packages[1].items() if packages else [('hit2_6',parts)]):
            train,cal=parts0;h='6' if target=='hit2_6' else '12'
            splits.append({'date':str(pd.Timestamp(date).date()),'target':target,'train_latest_label_maturity':str(train['mature_date_'+h].max().date()),'first_calibration_decision':str(cal.date.min().date()),'cal_latest_label_maturity':str(cal['mature_date_'+h].max().date()),'test_decision':str(pd.Timestamp(date).date()),'BSE_labels_in_training':False})
        all_scores.append(test);print('transfer',str(pd.Timestamp(date).date()),len(test),flush=True)
    scores=pd.concat(all_scores,ignore_index=True) if all_scores else pd.DataFrame();scores.to_parquet(out/'BSE_chronological_transfer_scores.parquet',index=False)
    pd.DataFrame(metrics).to_csv(out/'BSE_fold_metrics.csv',index=False);pd.DataFrame(splits).to_csv(out/'NSE_training_clocks_for_BSE_transfer.csv',index=False)
    summary={'panel_rows':len(x),'distinct_current_reference_ISINs':int(x['isin'].nunique()),'dates':int(x.date.nunique()),'outcome_horizons':{},'scope':'RETROSPECTIVE_BSE_REFERENCE_TRANSFER_RESEARCH','unseen_independent_outcomes':0,'historical_BSE_only_membership_verified':False,'survivor_bias':True,'corporate_actions_independently_verified':False,'financial_catalyst_gates_validated_on_BSE':False,'production_approved':False}
    for h in [6,12]:
        ms=[r for r in metrics if r['months']==h and r['fully_matured_selection']];selected=sum(r['selected'] for r in ms);hits=sum(r['selected_hits'] for r in ms)
        summary['outcome_horizons'][str(h)]={'fully_matured_selection_dates':len(ms),'selected_observations':selected,'hits':hits,'hit_fraction':hits/selected if selected else None,'dates_with_incomplete_selections':sum(r['months']==h and not r['fully_matured_selection'] for r in metrics)}
    (out/'BSE_transfer_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)

def summarize_saved(history,output,asof):
    """Repair/report from saved scores without fitting or changing selections."""
    out=Path(output);scores=pd.read_parquet(out/'BSE_chronological_transfer_scores.parquet');panel=pd.read_parquet(out/'BSE_four_year_panel_and_outcomes.parquet')
    deadlines=[]
    for path in sorted((Path(history)/'securities').glob('*.parquet')):
        d=pd.read_parquet(path);d=d[d.volume.gt(0)].sort_values('date');d['date']=pd.to_datetime(d.date)
        for date in panel.loc[panel['isin'].eq(d['isin'].iloc[0]),'date']:
            later=d[d.date.gt(date)]
            if later.empty:continue
            entry=later.date.iloc[0]
            deadlines.append({'isin':d['isin'].iloc[0],'date':date,'outcome_entry_date':entry,'calendar_deadline_6':entry+pd.DateOffset(months=6),'calendar_deadline_12':entry+pd.DateOffset(months=12),'entry_gap_days':(entry-date).days})
    scores=scores.merge(pd.DataFrame(deadlines),on=['isin','date'],how='left',validate='one_to_one')
    scores.to_parquet(out/'BSE_transfer_scores_with_deadline_audit.parquet',index=False)
    folds=[]
    for date,g in scores.groupby('date'):
        for h in [6,12]:
            c='p_hit2_'+str(h)
            if c not in g or g[c].isna().all():continue
            picks=g[g.technical_pass].sort_values([c,'symbol'],ascending=[False,True]).head(10)
            elapsed=len(picks)>0 and picks['calendar_deadline_'+str(h)].notna().all() and picks['calendar_deadline_'+str(h)].le(pd.Timestamp(asof)).all() and picks.entry_gap_days.le(10).all()
            known=picks['hit2_'+str(h)].isin([0,1])
            folds.append({'date':str(date.date()),'months':h,'calendar_horizons_elapsed':bool(elapsed),'selected':len(picks),'selected_known':int(known.sum()),'selected_unknown':int((~known).sum()),'selected_hits':int(picks['hit2_'+str(h)].eq(1).sum())})
    summary={'scope':'RETROSPECTIVE_CURRENT_BSE_EXCLUSIVE_REFERENCE_TRANSFER','panel_rows':len(panel),'panel_ISINs':int(panel['isin'].nunique()),'unseen_outcomes':0,'production_approved':False,'financial_catalyst_gates_validated':False,'historical_membership_and_corporate_actions_verified':False,'outcome_horizons':{}}
    for h in [6,12]:
        z=[r for r in folds if r['months']==h and r['calendar_horizons_elapsed']];n=sum(r['selected'] for r in z);known=sum(r['selected_known'] for r in z);hits=sum(r['selected_hits'] for r in z)
        summary['outcome_horizons'][str(h)]={'calendar_followup_dates':len(z),'selected_observations':n,'selected_known':known,'selected_unknown':n-known,'confirmed_hits':hits,'confirmed_hit_fraction_all_selected':hits/n if n else None,'hit_fraction_known_only_diagnostic':hits/known if known else None,'fully_known_dates':sum(r['selected_unknown']==0 for r in z)}
    pd.DataFrame(folds).to_csv(out/'BSE_fixed_selection_outcome_completeness.csv',index=False)
    (out/'BSE_transfer_summary.json').write_text(json.dumps(summary,indent=2));return summary

def main():
    p=argparse.ArgumentParser();p.add_argument('--history',required=True);p.add_argument('--nse-panel',required=True);p.add_argument('--output',required=True);p.add_argument('--asof',default='2026-10-09');a=p.parse_args();evaluate(a.history,a.nse_panel,a.output,a.asof)
if __name__=='__main__':main()
