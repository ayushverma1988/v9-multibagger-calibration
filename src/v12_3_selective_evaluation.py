"""Fixed development comparisons; no tuning or automatic model promotion."""
from __future__ import annotations
import argparse,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss
import v12_four_year_model as core
import v12_coherent_hazard_model as coherent
import v12_dated_enrichment_experiment as learner
from v12_3_point_in_time import FIELDS

VARIANTS = ('momentum_top10','quality_momentum_top10','enriched_top10','enriched_selective_max5')


def quality_mask(d, strict=False):
    finite = pd.Series(np.isfinite(d[list(FIELDS)].to_numpy(float)).all(axis=1),index=d.index)
    return (finite & d.quarter_sales_yoy.ge(.10 if strict else 0)
            & d.quarter_pat_yoy.ge(.15 if strict else 0)
            & d.quarter_pat_margin.ge(.08 if strict else 0)
            & d.annual_CFO_PAT.ge(.6) & d.dated_debt_equity.between(0,1))


def choose(d, variant, months):
    prob = 'p_hit2_'+str(months)
    pool = d[d.technical_pass & d.close.between(20,2000)].copy()
    if variant == 'momentum_top10':
        return pool.sort_values(['relative_strength63','symbol'],ascending=[False,True]).head(10)
    if variant == 'quality_momentum_top10':
        return pool[quality_mask(pool)].sort_values(['relative_strength63','symbol'],ascending=[False,True]).head(10)
    if variant == 'enriched_top10':
        return pool.sort_values([prob,'symbol'],ascending=[False,True]).head(10)
    if variant != 'enriched_selective_max5': raise ValueError('Unregistered variant')
    # Both horizons and downside must be estimable; no six-month-only shortcut.
    needed = ['p_hit2_6','p_hit2_12','p_loss30_12']
    if not set(needed).issubset(pool): return pool.iloc[:0]
    mask = (quality_mask(pool,True) & pool.rsi14.between(70,90)
            & pool.p_hit2_6.ge(.10) & pool.p_hit2_12.ge(.25)
            & pool.p_loss30_12.le(.45))
    return pool[mask].sort_values([prob,'symbol'],ascending=[False,True]).head(5)


def train_scores(panel_path,matrix_path,output,asof,bse_panel=None,protocol_path=None):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    if protocol_path is None:raise ValueError('Predeclared protocol required')
    protocol=json.loads(Path(protocol_path).read_text())
    if protocol.get('version')!='V12.3-selective-reliability' or protocol.get('variants')!=list(VARIANTS):raise ValueError('Unsupported experiment protocol')
    receipt={'run_started_at_utc':datetime.now(timezone.utc).isoformat(),'protocol_sha256':core.sha_file(protocol_path),
             'panel_sha256':core.sha_file(panel_path),'training_matrix_sha256':core.sha_file(matrix_path),
             'source_hashes':{n:core.sha_file(Path(__file__).with_name(n)) for n in ['v12_3_selective_evaluation.py','v12_3_point_in_time.py','v12_dated_enrichment_experiment.py','v12_coherent_hazard_model.py','v12_four_year_model.py']},
             'BSE_panel_sha256':core.sha_file(bse_panel) if bse_panel else None,'production_approved':False}
    with (out/'model_run_receipt.json').open('x') as f:json.dump(receipt,f,indent=2)
    panel=coherent.prepare(pd.read_parquet(panel_path));m=pd.read_parquet(matrix_path);m['date']=pd.to_datetime(m.date)
    data=panel.merge(m[['date','symbol','isin',*learner.EXTRA]],on=['date','symbol','isin'],validate='one_to_one')
    frames=[];clocks=[];packages={};bse_frames=[]
    bse=coherent.prepare(pd.read_parquet(bse_panel)) if bse_panel else None
    if bse is not None:
        for col in learner.EXTRA:bse[col]=np.nan
    for date in sorted(data.date.unique()):
        test=data[data.date.eq(date)].copy();pred={};bpred={}
        bt=bse[bse.date.eq(date)].copy() if bse is not None else None
        for target in ['hit2_6',coherent.LATE,coherent.LOSS]:
            parts=core.chronological_parts(data,date,target)
            if parts is None:continue
            train,cal=parts;p=learner.fit(train,cal,target);pred[target]=learner.score(p,test)
            if bt is not None and len(bt):bpred[target]=learner.score(p,bt)
            h=6 if target=='hit2_6' else 12
            clocks.append({'date':str(pd.Timestamp(date).date()),'target':target,'train_rows':len(train),'calibration_rows':len(cal),
                           'train_latest_maturity':str(train['mature_date_'+str(h)].max().date()),'first_calibration_date':str(cal.date.min().date()),
                           'cal_latest_maturity':str(cal['mature_date_'+str(h)].max().date()),'features_used':p['features']})
            if target=='hit2_6': test['prior6']=float(cal.hit2_6.mean())
            if target==coherent.LATE:
                union=data[data.date.between(cal.date.min(),cal.date.max()) & data.mature_date_12.lt(date) & data.hit2_12.isin([0,1])]
                test['prior12']=float(union.hit2_12.mean())
            if date==pd.Timestamp(asof):packages[target]=p
        if 'hit2_6' not in pred:continue
        test['p_hit2_6']=pred['hit2_6']
        if coherent.LATE in pred:test['p_hit2_12']=coherent.union_probability(pred['hit2_6'],pred[coherent.LATE])
        if coherent.LOSS in pred:test['p_loss30_12']=pred[coherent.LOSS]
        frames.append(test)
        if 'hit2_6' in bpred:
            bt['p_hit2_6']=bpred['hit2_6'];bt['prior6']=test.prior6.iloc[0]
            if coherent.LATE in bpred:
                bt['p_hit2_12']=coherent.union_probability(bpred['hit2_6'],bpred[coherent.LATE]);bt['prior12']=test.prior12.iloc[0]
            if coherent.LOSS in bpred:bt['p_loss30_12']=bpred[coherent.LOSS]
            bse_frames.append(bt)
        print('scored',str(pd.Timestamp(date).date()),len(test),flush=True)
    scored=pd.concat(frames,ignore_index=True)
    scored.to_parquet(out/'chronological_development_scores.parquet',index=False)
    if bse_frames:pd.concat(bse_frames,ignore_index=True).to_parquet(out/'BSE_transfer_development_scores.parquet',index=False)
    (out/'training_clocks.json').write_text(json.dumps(clocks,indent=2))
    import joblib
    joblib.dump(packages,out/'UNPROMOTED_research_heads.joblib')
    return scored


def extended_outcome(history,date,months,asof):
    """Independent arithmetic on immutable history; not an independent assessor."""
    h=history[history.volume.gt(0)].copy();h['date']=pd.to_datetime(h.date)
    future=h[h.date.gt(pd.Timestamp(date))]
    if future.empty:return {'status':'UNKNOWN_NO_ENTRY'}
    entry=future.iloc[0];deadline=entry.date+pd.DateOffset(months=months)
    base={'entry_date':str(entry.date.date()),'deadline':str(deadline.date())}
    if deadline>pd.Timestamp(asof):return {**base,'status':'PENDING_HORIZON'}
    f=future[future.date.le(deadline)]
    if h.date.max()<deadline or len(f)<months*16 or (deadline-f.date.max()).days>5 or f.date.diff().dt.days.max()>10:
        return {**base,'status':'UNKNOWN_INCOMPLETE_HISTORY'}
    nums=f[['open','high','low','close','adj_close','volume']].to_numpy(float)
    if not np.isfinite(nums).all() or (nums<=0).any() or not f.low.le(f[['open','close']].min(axis=1)).all() or not f.high.ge(f[['open','close']].max(axis=1)).all():
        return {**base,'status':'UNKNOWN_INVALID_PRICE_VOLUME'}
    if np.log(f.adj_close).diff().abs().max()>np.log(1.4):return {**base,'status':'UNKNOWN_CORPORATE_ACTION_OR_JUMP'}
    adjusted_entry=float(entry.open*entry.adj_close/entry.close)
    net=f.adj_close.to_numpy(float)/adjusted_entry*(1-core.COST_PER_SIDE)/(1+core.COST_PER_SIDE)
    lows=f.low.to_numpy(float)*f.adj_close.to_numpy(float)/f.close.to_numpy(float)/adjusted_entry*(1-core.COST_PER_SIDE)/(1+core.COST_PER_SIDE)
    hit=np.flatnonzero(net>=2);loss=np.flatnonzero(lows<=.7)
    wealth=np.r_[1.,net];drawdown=float(np.min(wealth/np.maximum.accumulate(wealth))-1)
    return {**base,'status':'KNOWN_RESEARCH_PROXY','hit2':int(len(hit)>0),'endpoint_return':float(net[-1]-1),
            'endpoint_loss':int(net[-1]<1),'max_closing_drawdown':drawdown,
            'loss30_before_hit2':int(len(loss)>0 and (len(hit)==0 or loss[0]<=hit[0])),
            'days_to_hit2':int((f.iloc[hit[0]].date-entry.date).days) if len(hit) else None,
            'target_exit_return_proxy':float(net[hit[0]]-1) if len(hit) else float(net[-1]-1),
            'execution_verified':False}


def block_interval(folds,months):
    """Descriptive moving-block bootstrap; too few blocks returns UNKNOWN."""
    f=folds.sort_values('date');a=f[['hits','selected']].to_numpy(float);n=len(a)
    if n<2*months or a[:,1].sum()==0:return None
    rng=np.random.default_rng(31);values=[]
    for _ in range(2000):
        starts=rng.integers(0,n-months+1,size=int(np.ceil(n/months)))
        b=np.concatenate([a[s:s+months] for s in starts])[:n]
        if b[:,1].sum():values.append(b[:,0].sum()/b[:,1].sum())
    return np.quantile(values,[.025,.975]).tolist() if values else None


def evaluate(scored,history_dir,output,asof,conflicts_path=None):
    out=Path(output);out.mkdir(parents=True,exist_ok=True);hist={};calendar=set()
    conflicts=pd.read_parquet(conflicts_path) if conflicts_path else None
    if conflicts is not None:
        conflicts=conflicts[~conflicts['within_0_011_INR'].eq(True)].copy();conflicts['date']=pd.to_datetime(conflicts.date)
    for path in sorted((Path(history_dir)/'securities').glob('*.parquet')):
        h=pd.read_parquet(path);h['date']=pd.to_datetime(h.date);hist[str(h['isin'].iloc[0])]=h
        calendar.update(h.loc[h.volume.gt(0),'date'].tolist())
    calendar=sorted(calendar);folds=[];records=[];curr=[];cache={}
    for date,g in scored.groupby('date',sort=True):
        for months in [6,12]:
            prob='p_hit2_'+str(months)
            if prob not in g or not g[prob].notna().all():continue
            next_dates=[x for x in calendar if x>date]
            sufficient=bool(next_dates and next_dates[0]+pd.DateOffset(months=months)<=pd.Timestamp(asof))
            for variant in VARIANTS:
                picks=choose(g,variant,months)
                if date==pd.Timestamp(asof):
                    for r in picks.to_dict('records'):curr.append({k:r.get(k) for k in ['date','symbol','isin','close','p_hit2_6','p_hit2_12','p_loss30_12',*FIELDS]}|{'variant':variant,'months':months,'approved_for_trading':False})
                if not sufficient:continue
                results=[]
                for r in picks.to_dict('records'):
                    key=(r['isin'],date,months)
                    if key not in cache:
                        cache[key]=extended_outcome(hist[r['isin']],date,months,asof) if r['isin'] in hist else {'status':'UNKNOWN_HISTORY_NOT_FOUND'}
                    o=dict(cache[key])
                    if conflicts is not None and 'deadline' in o:
                        affected=conflicts[conflicts['isin'].eq(r['isin']) & conflicts.date.between(date-pd.Timedelta(days=100),pd.Timestamp(o['deadline']))]
                        if len(affected):o={'status':'UNKNOWN_SECOND_SOURCE_PRICE_CONFLICT','entry_date':o['entry_date'],'deadline':o['deadline'],'conflict_dates':','.join(affected.date.dt.strftime('%Y-%m-%d'))}
                    if o.get('status')=='KNOWN_RESEARCH_PROXY' and pd.notna(r['hit2_'+str(months)]) and o['hit2']!=int(r['hit2_'+str(months)]):
                        raise ValueError('Independent target arithmetic disagrees with original labels')
                    results.append(o);records.append({'date':str(date.date()),'variant':variant,'months':months,'symbol':r['symbol'],'isin':r['isin'],'predicted_probability':r[prob],**o})
                known=[r for r in results if r['status']=='KNOWN_RESEARCH_PROXY']
                slots=5 if variant=='enriched_selective_max5' else 10
                complete=len(known)==len(results)
                folds.append({'date':str(date.date()),'variant':variant,'months':months,'selected':len(picks),'known':len(known),'unknown':len(results)-len(known),
                              'hits':sum(r['hit2'] for r in known),'abstained':len(picks)==0,
                              'invested_slot_fraction':len(picks)/slots,
                              'fixed_slot_cohort_endpoint_return':sum(r['endpoint_return'] for r in known)/slots if complete else None,
                              'fixed_slot_cohort_target_exit_return_proxy':sum(r['target_exit_return_proxy'] for r in known)/slots if complete else None,
                              'mean_endpoint_return_known_only':float(np.mean([r['endpoint_return'] for r in known])) if known else None})
    f=pd.DataFrame(folds);r=pd.DataFrame(records)
    f.to_csv(out/'fixed_variant_folds.csv',index=False);r.to_parquet(out/'selected_outcomes.parquet',index=False)
    pd.DataFrame(curr).to_parquet(out/'current_RESEARCH_comparisons.parquet',index=False)
    summary={'development_only':True,'full_historical_universe':False,'future_unseen_outcomes':0,'production_approved':False,'variants':[],'probability_metrics':{}}
    for (variant,months),g in f.groupby(['variant','months']):
        n=int(g.selected.sum());hits=int(g.hits.sum());rr=r[r.variant.eq(variant)&r.months.eq(months)] if len(r) else pd.DataFrame()
        k=rr[rr.status.eq('KNOWN_RESEARCH_PROXY')] if len(rr) else rr
        stat={'variant':variant,'months':int(months),'calendar_followup_dates':len(g),'selection_dates':int(g.selected.gt(0).sum()),'abstention_dates':int(g.abstained.sum()),
              'selected_observations':n,'known_outcomes':int(g.known.sum()),'unknown_outcomes':int(g.unknown.sum()),'hits':hits,'confirmed_hits_all_selected':hits/n if n else None,
              'distinct_selected_issuers':rr['isin'].nunique() if len(rr) else 0,'descriptive_block_interval':block_interval(g,months),
              'precision_gate_pass':bool(n and hits/n>=.30),'enough_selected_dates':bool(g.selected.gt(0).sum()>=6),
              'returns_scope':'known selected observations, overlapping cohorts; not a nonoverlapping portfolio CAGR'}
        stat['mean_invested_slot_fraction']=float(g.invested_slot_fraction.mean())
        stat['cohorts_with_known_returns_including_cash']=int(g.fixed_slot_cohort_endpoint_return.notna().sum())
        stat['fixed_slot_cohort_endpoint_return_mean']=float(g.fixed_slot_cohort_endpoint_return.mean()) if g.fixed_slot_cohort_endpoint_return.notna().any() else None
        stat['cash_assumption']='unused slots and proceeds held as zero-interest cash; unresolved held positions make cohort return UNKNOWN'
        for name in ['endpoint_return','endpoint_loss','max_closing_drawdown','loss30_before_hit2','days_to_hit2','target_exit_return_proxy']:
            stat[name+'_mean_known_only']=float(k[name].mean()) if len(k) and k[name].notna().any() else None
            if name in ['endpoint_return','max_closing_drawdown','days_to_hit2']:
                stat[name+'_median_known_only']=float(k[name].median()) if len(k) and k[name].notna().any() else None
        summary['variants'].append(stat)
    for h in [6,12]:
        k=scored[scored['hit2_'+str(h)].isin([0,1]) & scored['p_hit2_'+str(h)].notna() & scored['mature_date_'+str(h)].le(pd.Timestamp(asof))]
        if len(k):
            b=brier_score_loss(k['hit2_'+str(h)],k['p_hit2_'+str(h)]);prior=brier_score_loss(k['hit2_'+str(h)],k['prior'+str(h)])
            summary['probability_metrics'][str(h)]={'rows':len(k),'brier':b,'past_prior_brier':prior,'brier_skill':1-b/prior,'reliability':core.reliability(k,'hit2_'+str(h),'p_hit2_'+str(h))}
    (out/'development_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False));print(json.dumps(summary,indent=2),flush=True)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--panel',required=True);p.add_argument('--matrix',required=True);p.add_argument('--history',required=True);p.add_argument('--output',required=True);p.add_argument('--asof',default='2026-10-09');p.add_argument('--scores');p.add_argument('--bse-panel');p.add_argument('--conflicts');p.add_argument('--protocol');a=p.parse_args()
    scores=pd.read_parquet(a.scores) if a.scores else train_scores(a.panel,a.matrix,a.output,a.asof,a.bse_panel,a.protocol)
    evaluate(scores,a.history,a.output,a.asof,a.conflicts)
