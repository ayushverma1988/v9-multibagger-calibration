"""Matched current-policy development tests; never promote examined history."""
import argparse,hashlib,json
from pathlib import Path
import joblib,numpy as np,pandas as pd
from sklearn.metrics import average_precision_score,brier_score_loss,log_loss
import v13_model as model
import v13_redevelopment as rd
from v13_available_financials import QUALIFIED,TURNAROUND,DISCOVERY

VERSION='V13_CURRENT_POLICY_MATCHED_EVALUATION_20261103'
MARKET=[x for x in model.FEATURES if x in model.EXTRA or x in ['log_ret20','log_ret63',
 'momentum_acceleration','rsi14','distance_ma20','distance_ma50','distance_ma200','off_high63',
 'volatility20','volatility63','volume_confirmation','up_down_volume63','trend_consistency63',
 'drawdown63','log_liquidity63','relative_strength63','relative_strength20','market_breadth63',
 'market_median_ret63','market_dispersion63']]
MARKET=[x for x in MARKET if x!='earnings_market_confirmation']
CATALYST=[x for x in model.FEATURES if any(s in x for s in ['order','capacity','commission','utilisation','funding','incremental_profit'])]

def eligible(d,primary=True):
    valid=np.isfinite(d.close)&d.close.gt(0)&d.distance_ma50.gt(0)&d.log_ret63.le(np.log(2.5))&d.rsi14.gt(70)
    valid &= d.feature_eligible.fillna(False).astype(bool)
    valid &= d.available_financial_tier.isin(QUALIFIED|{TURNAROUND,DISCOVERY})
    if primary:valid &= d.historical_official_identity_match.fillna(False).astype(bool)
    return valid

def four_date_parts(data,date,target):
    h=6 if target=='hit2_6' else 12;m='mature_date_'+str(h)
    known=data[data.date.lt(date)&data[m].lt(date)&data[target].isin([0,1])]
    dates=sorted(known.date.unique())
    if len(dates)<5:return None
    times=dates[-4:]
    train=known[known.date.lt(times[0])&known[m].lt(times[0])]
    cal=known[known.date.isin(times)]
    if len(train)<30 or train.date.nunique()<2 or len(cal)<10:return None
    assert train[m].lt(cal.date.min()).all() and cal[m].lt(date).all()
    return train,cal

def fit(data,date,variant):
    heads={};clocks=[]
    names=MARKET if variant=='market_only' else [x for x in model.FEATURES if x not in CATALYST] if variant=='without_catalysts' else list(dict.fromkeys([*model.FEATURES,*rd.MARKET_FEATURES])) if variant=='breakout_features' else model.FEATURES
    for target in model.TARGETS:
        split=four_date_parts(data,date,target)
        if split is None:continue
        train,cal=split;dates=sorted(cal.date.unique())
        # Model blending sees only the first two calibration dates. The last
        # two dates are genuinely separate from that mixture choice.
        head=model.fit(train,cal[cal.date.lt(dates[-1])],target,feature_names=names)
        single=cal[cal.date.eq(dates[-1])]
        if variant=='single_date':
            head['calibration']=model.core.calibrate(model.raw_score(head,single),single[target],train[target])
        else:
            pooled=cal[cal.date.isin(dates[-2:])]
            head['calibration']=rd.shrink_calibration(model.raw_score(head,pooled),pooled[target],train[target])
        heads[target]=head
        clocks.append({'target':target,'train_rows':len(train),'train_maturity_max':str(train['mature_date_'+str(6 if target=='hit2_6' else 12)].max()),
          'blend_dates':[str(pd.Timestamp(x).date()) for x in dates[:2]],
          'calibration_dates':[str(pd.Timestamp(x).date()) for x in (dates[-1:] if variant=='single_date' else dates[-2:])],
          'test_date':str(pd.Timestamp(date).date()),'calibration':head['calibration']})
    return heads,clocks

def probability_metrics(d,target,prob,maturity,asof):
    k=d[d[maturity].le(pd.Timestamp(asof))&d[target].isin([0,1])&np.isfinite(d[prob])]
    if k.empty:return {'rows':0,'events':0,'brier':None,'log_loss':None,'average_precision':None,'bins':[]}
    p=k[prob].to_numpy(float);y=k[target].to_numpy(int);bins=[]
    assignment=np.minimum((p*10).astype(int),9)
    for b in range(10):
        mask=assignment==b
        if mask.any():bins.append({'lower':b/10,'upper':(b+1)/10,'rows':int(mask.sum()),'predicted':float(p[mask].mean()),'observed':float(y[mask].mean())})
    assert sum(x['rows'] for x in bins)==len(k)
    return {'rows':len(k),'events':int(y.sum()),'brier':float(brier_score_loss(y,p)),
      'log_loss':float(log_loss(y,np.clip(p,1e-8,1-1e-8),labels=[0,1])),
      'average_precision':float(average_precision_score(y,p)) if y.sum() else None,
      'mean_predicted':float(p.mean()),'observed':float(y.mean()),'bins':bins}

def ranked(d,mask,recipe,limit):
    pool=d[mask&d.loss12.le(.60 if recipe=='six_month_risk60' else .95)].copy()
    dma=pool.distance_ma200.abs().le(.10+1e-12).fillna(False)
    if recipe=='current_joint12':
        pool['score']=.75*pool.joint12.rank(pct=True)+.10*pool.available_financial_evidence_score+.05*dma
    else:pool['score']=.70*pool.p6.rank(pct=True)+.25*pool.joint12.rank(pct=True)+.05*dma
    return pool.sort_values(['score','isin'],ascending=[False,True]).head(limit)

def blocked_interval(differences,block=6):
    a=np.asarray(differences,float);a=a[np.isfinite(a)]
    if len(a)<block:return {'date_blocks':len(a),'lower':None,'upper':None}
    rng=np.random.default_rng(20261103);values=[]
    for _ in range(1000):
        starts=rng.integers(0,len(a),size=int(np.ceil(len(a)/block)))
        sample=np.concatenate([a[(s+np.arange(block))%len(a)] for s in starts])[:len(a)]
        values.append(sample.mean())
    return {'date_blocks':len(a),'block_length_months':block,'lower':float(np.quantile(values,.025)),'upper':float(np.quantile(values,.975))}

def stable_context(data,reference=None,reference_date=None):
    d=data.copy();sizes=d.groupby('date').size()
    if reference is None:
        candidates=sizes[sizes.ge(100)]
        if candidates.empty:return d,[],None
        reference_date=candidates.index.min();reference=d.loc[d.date.eq(reference_date),'isin'].tolist()
    pairs=[('log_ret63','relative_strength63'),('log_ret20','relative_strength20'),
      ('momentum126','relative_strength126'),('momentum252_skip21','relative_strength252_skip21')]
    for date,g in d.groupby('date',sort=True):
        subset=g[g['isin'].isin(reference)] if date>=reference_date else g.iloc[:0]
        for source,target in pairs:
            values=np.sort(subset[source].dropna().to_numpy(float)) if source in subset else np.array([])
            if len(values)<20:d.loc[g.index,target]=np.nan
            else:
                observed=g[source].to_numpy(float)
                ranks=np.searchsorted(values,observed,side='right')/len(values)
                ranks[~np.isfinite(observed)]=np.nan
                d.loc[g.index,target]=ranks
        for target,value in [('market_breadth63',subset.log_ret63.gt(0).mean()),('market_median_ret63',subset.log_ret63.median()),('market_dispersion63',subset.log_ret63.std())]:
            d.loc[g.index,target]=value if len(subset)>=20 else np.nan
    d['trend_regime_interaction']=d.relative_strength63*(d.market_breadth63-.5)
    if 'quarter_pat_yoy' in d:d['earnings_market_confirmation']=d.quarter_pat_yoy*d.relative_strength63
    return d,reference,reference_date

def run(matrix,current,out,asof,max_dates=12,variants=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    source=Path(matrix);data=model.prepare(pd.read_parquet(source));fresh=pd.read_parquet(current)
    for c in ['date','mature_date_6','mature_date_12']:data[c]=pd.to_datetime(data[c])
    available_dates=sorted(data.loc[data.mature_date_6.le(pd.Timestamp(asof)),'date'].unique())
    dates=available_dates[-max_dates:] if max_dates else available_dates
    variants=variants or ['single_date','pooled_calibration','market_only','without_catalysts','breakout_features','stable_context']
    stable,reference,reference_date=stable_context(data)
    protocol={'version':VERSION,'source_asof':asof,'matrix_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
      'RSI_rule':'strict >70; simple rolling 14 sessions','price_band':None,'DMA_band_optional':True,
      'historical_policy':'matched available-financial numeric/safety rules plus exact per-row primary session match',
      'historical_results_previously_examined':True,'unseen_holdout':False,'promotion_approved':False,
      'current_listing_survivor_bias':True,'historical_adjustment_vintage_limit':True,
      'historical_corporate_action_intervals_independently_verified':False,
      'model_selection_and_policy_selection_separately_reported':True,
      'matched_selection_counts':[5,20,25],'group_caps_replaced_by_matched_global_K_for_experiment':True,
      'variants':variants,'stable_reference_count':len(reference),'stable_reference_date':str(reference_date),
      'dates_before_reference_use_unknown_context':True,'dates':[str(pd.Timestamp(x).date()) for x in dates]}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    scores=[];selections=[];clocks=[];blocked=[]
    for date in dates:
        test=data[data.date.eq(date)].copy()
        for variant in variants:
            training=stable if variant=='stable_context' else data
            heads,clock=fit(training,date,variant)
            if not set(model.TARGETS).issubset(heads):blocked.append({'date':str(date),'variant':variant,'reason':'INSUFFICIENT_MATURE_CONDITIONAL_HEADS'});continue
            d=(stable[stable.date.eq(date)] if variant=='stable_context' else test).copy()
            for k,v in model.probabilities(heads,d).items():d[k]=v
            d['variant']=variant;d['policy_eligible']=eligible(d)
            scores.append(d);clocks.append({'variant':variant,'date':str(date),'heads':clock})
            for recipe in ['current_joint12','six_month','six_month_risk60']:
                for limit in [5,20,25]:
                    chosen=ranked(d,d.policy_eligible,recipe,limit)
                    h6=chosen[chosen.mature_date_6.le(pd.Timestamp(asof))&chosen.hit2_6.isin([0,1])]
                    h12=chosen[chosen.mature_date_12.le(pd.Timestamp(asof))&chosen.hit2_12.isin([0,1])]
                    ret=h6.get('endpoint_return_6',pd.Series(dtype=float)).dropna()
                    selections.append({'date':str(pd.Timestamp(date).date()),'variant':variant,'recipe':recipe,'K':limit,
                      'eligible':int(d.policy_eligible.sum()),'selected':len(chosen),'known6':len(h6),'hits6':int(h6.hit2_6.sum()),
                      'precision6':float(h6.hit2_6.mean()) if len(h6) else None,'known12':len(h12),'hits12':int(h12.hit2_12.sum()),
                      'loss_events6':int(h6.loss30_6.sum()) if 'loss30_6' in h6 else None,
                      'median_return6':float(ret.median()) if len(ret) else None,'mean_return6':float(ret.mean()) if len(ret) else None,
                      'selected_isins':chosen['isin'].tolist()})
            print('Matched evaluation',str(pd.Timestamp(date).date()),variant,len(d),'eligible',int(d.policy_eligible.sum()),flush=True)
    all_scores=pd.concat(scores,ignore_index=True) if scores else pd.DataFrame()
    summary={**protocol,'blocked':blocked,'models':{},'selection_results':selections,'training_clocks':clocks}
    for variant,g in all_scores.groupby('variant') if len(all_scores) else []:
        targets=[('p6','hit2_6','mature_date_6'),('p12','hit2_12','mature_date_12'),('loss12',model.coherent.LOSS,'mature_date_12')]
        primary=g[g.historical_official_identity_match.fillna(False).astype(bool)]
        summary['models'][variant]={
          'primary_scored_rows':{k:probability_metrics(primary,t,k,m,asof) for k,t,m in targets},
          'eligible_rows':{k:probability_metrics(g[g.policy_eligible],t,k,m,asof) for k,t,m in targets}}
    comparisons=[];sel=pd.DataFrame(selections)
    if not sel.empty:
        baseline=sel[sel.variant.eq('single_date')&sel.recipe.eq('current_joint12')&sel.K.eq(25)].set_index('date')
        for (variant,recipe),g in sel[sel.K.eq(25)].groupby(['variant','recipe']):
            paired=g.set_index('date').join(baseline[['precision6']],rsuffix='_baseline',how='inner')
            diff=(paired.precision6-paired.precision6_baseline).dropna()
            comparisons.append({'variant':variant,'recipe':recipe,'matched_dates':len(diff),
              'mean_monthly_precision6_difference':float(diff.mean()) if len(diff) else None,
              'positive_months':int(diff.gt(0).sum()),'interval':blocked_interval(diff),
              'independent_validation':False})
    summary['paired_comparisons']=comparisons
    summary['decision']='RETAIN_FIXED_CHAMPION_EXAMINED_DEVELOPMENT_ONLY'
    summary['accuracy_claim']=None
    summary['historical_feature_nonmissing']={x:int(data[x].notna().sum()) for x in model.FEATURES if x in data}
    if len(all_scores):all_scores.to_parquet(out/'historical_scores.parquet',index=False,compression='zstd',compression_level=9)
    # Fit experimental candidates at today's date; they remain independent of
    # both original frozen model files and all registered cohorts.
    for variant in variants:
        heads,clock=fit(stable if variant=='stable_context' else data,pd.Timestamp(asof),variant)
        if not set(model.TARGETS).issubset(heads):continue
        joblib.dump(heads,out/(variant+'_experimental_heads.joblib'))
        d=stable_context(fresh,reference,reference_date)[0] if variant=='stable_context' and reference_date is not None else fresh.copy()
        for k,v in model.probabilities(heads,d).items():d[k]=v
        d['historical_official_identity_match']=True
        for recipe in ['current_joint12','six_month','six_month_risk60']:
            chosen=ranked(d,eligible(d,primary=False),recipe,25)
            cols=[x for x in ['isin','symbol','series','source_close_INR','rsi14','p6','p12','joint12','loss12','available_financial_tier','score'] if x in chosen]
            chosen[cols].to_json(out/(variant+'_'+recipe+'_current.json'),orient='records',indent=2)
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--matrix',required=True);p.add_argument('--current',required=True);p.add_argument('--out',required=True);p.add_argument('--asof',required=True);p.add_argument('--max-dates',type=int,default=12);a=p.parse_args()
    run(a.matrix,a.current,a.out,a.asof,a.max_dates)
