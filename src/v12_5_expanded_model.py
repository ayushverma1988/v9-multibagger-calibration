"""One fixed chronological expansion, compared on the same profit-qualified pool.

Separately fits the joint 2x-before-30%-loss target. No independent-marginal
multiplication, post-hoc probability inflation, or promotion on this archive.
"""
from __future__ import annotations
import argparse,json
from datetime import datetime,timezone
from pathlib import Path
import joblib,numpy as np,pandas as pd
from scipy.special import expit,logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss,average_precision_score
import v12_four_year_model as core
import v12_coherent_hazard_model as coherent
import v12_dated_enrichment_experiment as prior
from v12_5_features import FINANCIAL,CATALYST,MARKET,BASE_FIELDS

JOINT='hit2_before_loss30_12'
FEATURES=list(dict.fromkeys([*core.FEATURES,*prior.EXTRA,*FINANCIAL,*MARKET,*CATALYST]))

def transform(p,d):
    x=d[p['features']].to_numpy(float);missing=~np.isfinite(x)
    return np.c_[np.clip(np.where(missing,p['median'],x),p['lo'],p['hi']),missing[:,p['missing_positions']].astype(float)]

def raw_score(p,d):
    x=transform(p,d);return .7*p['linear'].predict_proba(x)[:,1]+.3*p['nonlinear'].predict_proba(x)[:,1]

def score(p,d):
    c=p['calibration'];raw=np.clip(raw_score(p,d),1e-5,1-1e-5)
    return expit(c['slope']*logit(raw)+c['intercept'])

def joint_consistent(joint,upside,loss):
    return (joint<=upside)&(joint+loss<=1+1e-12)

def fit(train,cal,target):
    features=[c for c in FEATURES if np.isfinite(train[c]).any()]
    x=train[features].to_numpy(float);clean=np.where(np.isfinite(x),x,np.nan)
    p={'features':features,'median':np.nanmedian(clean,axis=0),'lo':np.nanquantile(clean,.005,axis=0),
       'hi':np.nanquantile(clean,.995,axis=0),'missing_positions':[i for i,c in enumerate(features) if c not in core.FEATURES]}
    x=transform(p,train);y=train[target].astype(int);w=core.weights(train)
    p['linear']=make_pipeline(StandardScaler(),LogisticRegression(C=.03,max_iter=1000,random_state=31))
    p['linear'].fit(x,y,logisticregression__sample_weight=w)
    p['nonlinear']=HistGradientBoostingClassifier(max_iter=60,max_leaf_nodes=7,min_samples_leaf=100,
        l2_regularization=10,learning_rate=.05,early_stopping=False,random_state=31).fit(x,y,sample_weight=w)
    p['calibration']=core.calibrate(raw_score(p,cal),cal[target],train[target]);return p

def profit_gate(d):
    finite=pd.Series(np.isfinite(d[list(BASE_FIELDS)].to_numpy(float)).all(axis=1),index=d.index)
    return (finite & d.four_profitable_quarters.eq(1) & d.quarter_sales_yoy.gt(0)
       & d.quarter_pat_yoy.gt(0) & d.quarter_pat_margin.gt(0) & d.annual_CFO_PAT.ge(.6)
       & d.dated_debt_equity.between(0,1) & d.close.between(20,2000)
       & d.distance_ma50.gt(0) & d.log_ret63.le(np.log(1.75)))

def run(matrix_path,baseline_path,output,asof='2026-10-09',bse_matrix=None):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    targets=['hit2_6',coherent.LATE,coherent.LOSS,JOINT]
    protocol={'version':'V12.5-expanded-evidence-development','registered_at_utc':datetime.now(timezone.utc).isoformat(),
        'features':FEATURES,'targets':targets,'input_sha256':core.sha_file(matrix_path),
        'baseline_sha256':core.sha_file(baseline_path),'code_sha256':core.sha_file(__file__),
        'learner':'unchanged_70pct_LR_30pct_HGB_fixed_hyperparameters','top_k':5,
        'same_profit_pool_for_all_rankers':True,'historical_period_already_examined':True,
        'unseen_outcomes':0,'production_approved':False,'current_sector_valuation_used_in_training':False,
        'verified_recurring_EPS_unavailable_not_substituted':True,
        'BSE_input_sha256':core.sha_file(bse_matrix) if bse_matrix else None}
    with (out/'protocol.json').open('x') as f:json.dump(protocol,f,indent=2)
    data=coherent.prepare(pd.read_parquet(matrix_path));baseline=pd.read_parquet(baseline_path)
    bse=coherent.prepare(pd.read_parquet(bse_matrix)) if bse_matrix else None
    parts=[];clocks=[];heads={};bse_parts=[]
    for date,test in data.groupby('date',sort=True):
        d=test.copy();pred={};bp={}
        bd=bse[bse.date.eq(date)].copy() if bse is not None else None
        for target in targets:
            split=core.chronological_parts(data,date,target)
            if split is None:continue
            train,cal=split;p=fit(train,cal,target);pred[target]=score(p,d)
            if bd is not None and len(bd):bp[target]=score(p,bd)
            h=6 if target=='hit2_6' else 12
            clocks.append({'date':str(date.date()),'target':target,'train_rows':len(train),'cal_rows':len(cal),
                'train_maturity_max':str(train['mature_date_'+str(h)].max()),'cal_start':str(cal.date.min()),
                'cal_maturity_max':str(cal['mature_date_'+str(h)].max()),'features_used':p['features']})
            if date==pd.Timestamp(asof):heads[target]=p
        if 'hit2_6' not in pred:continue
        d['expanded_p6']=pred['hit2_6']
        d['expanded_p12']=coherent.union_probability(pred['hit2_6'],pred[coherent.LATE]) if coherent.LATE in pred else np.nan
        d['expanded_loss12']=pred.get(coherent.LOSS,np.nan);d['expanded_joint12']=pred.get(JOINT,np.nan)
        # Independent joint head is not projected/clipped to inflate confidence.
        # Flag incoherence for research inspection and block that joint ranking.
        d['joint_coherent']=joint_consistent(d.expanded_joint12,d.expanded_p12,d.expanded_loss12)
        if 'hit2_6' in bp:
            bd['expanded_p6']=bp['hit2_6']
            bd['expanded_p12']=coherent.union_probability(bp['hit2_6'],bp[coherent.LATE]) if coherent.LATE in bp else np.nan
            bd['expanded_loss12']=bp.get(coherent.LOSS,np.nan)
            bd['expanded_joint12']=bp.get(JOINT,np.nan)
            bd['profit_gate']=profit_gate(bd);bse_parts.append(bd)
        parts.append(d);print('expanded model',str(date.date()),len(d),flush=True)
    scores=pd.concat(parts,ignore_index=True)
    old=baseline[['date','isin','p_hit2_6','p_hit2_12']]
    scores=scores.merge(old,on=['date','isin'],how='left',validate='1:1')
    scores['profit_gate']=profit_gate(scores)
    scores.to_parquet(out/'scores.parquet',index=False)
    joblib.dump(heads,out/'UNPROMOTED_heads.joblib')
    (out/'training_clocks.json').write_text(json.dumps(clocks,indent=2))
    summaries=[];current=[];selected=[]
    for h in [6,12]:
        rankers={'old_upside':'p_hit2_'+str(h),'expanded_upside':'expanded_p'+str(h),'momentum':'relative_strength63'}
        if h==12:rankers['expanded_joint']='expanded_joint12'
        for variant,col in rankers.items():
            stat={'variant':variant,'months':h,'dates':0,'selection_dates':0,'selected':0,'known':0,'hits':0,'joint_wins':0,'joint_known':0}
            for date,d in scores.groupby('date',sort=True):
                # The same horizon-scoreable calendar dates and financial pool.
                if not d['p_hit2_'+str(h)].notna().all() or not d['expanded_p'+str(h)].notna().all():continue
                pool=d[d.profit_gate & d[col].notna()]
                if variant=='expanded_joint':pool=pool[pool.joint_coherent]
                picks=pool.sort_values([col,'isin'],ascending=[False,True]).head(5)
                if date==pd.Timestamp(asof):
                    current.extend(picks.assign(variant=variant,months=h,approved_for_trading=False).to_dict('records'));continue
                matured=d['mature_date_'+str(h)].le(pd.Timestamp(asof))&d['hit2_'+str(h)].isin([0,1])
                if not matured.any():continue
                stat['dates']+=1;stat['selection_dates']+=int(len(picks)>0);stat['selected']+=len(picks)
                known=picks[picks['mature_date_'+str(h)].le(pd.Timestamp(asof))&picks['hit2_'+str(h)].isin([0,1])]
                stat['known']+=len(known);stat['hits']+=int(known['hit2_'+str(h)].sum())
                if h==12:
                    stat['joint_known']+=int(known[JOINT].isin([0,1]).sum());stat['joint_wins']+=int(known[JOINT].eq(1).sum())
                for r in picks.to_dict('records'):
                    selected.append({'date':date,'variant':variant,'months':h,'isin':r['isin'],'symbol':r['symbol'],'score':r[col],
                        'hit2':r['hit2_'+str(h)] if pd.notna(r['mature_date_'+str(h)]) and r['mature_date_'+str(h)]<=pd.Timestamp(asof) else None})
            stat['unknown']=stat['selected']-stat['known'];stat['precision']=stat['hits']/stat['selected'] if stat['selected'] else None;summaries.append(stat)
    metrics={}
    for h in [6,12]:
        d=scores[scores['mature_date_'+str(h)].le(pd.Timestamp(asof))&scores['hit2_'+str(h)].isin([0,1])&scores['expanded_p'+str(h)].notna()&scores['p_hit2_'+str(h)].notna()]
        if len(d):metrics[str(h)]={'rows':len(d),'expanded_brier':brier_score_loss(d['hit2_'+str(h)],d['expanded_p'+str(h)]),
            'old_brier':brier_score_loss(d['hit2_'+str(h)],d['p_hit2_'+str(h)]),'expanded_PR_AUC':average_precision_score(d['hit2_'+str(h)],d['expanded_p'+str(h)])}
    risk_metrics={}
    for target,col in [(coherent.LOSS,'expanded_loss12'),(JOINT,'expanded_joint12')]:
        z=scores[scores.mature_date_12.le(pd.Timestamp(asof)) & scores[target].isin([0,1]) & scores[col].notna()]
        if len(z):risk_metrics[target]={'rows':len(z),'mean_predicted':float(z[col].mean()),'observed_rate':float(z[target].mean()),'brier':brier_score_loss(z[target],z[col])}
    now=scores[scores.date.eq(pd.Timestamp(asof))]
    summary={'development_results':summaries,'probability_metrics':metrics,'current_scored':len(now),
        'current_profit_gate':int(now.profit_gate.sum()),'current_joint_incoherence':int((~now.joint_coherent).sum()),
        'current_profit_gate_and_loss45':int((now.profit_gate & now.expanded_loss12.le(.45)).sum()),
        'risk_probability_metrics':risk_metrics,
        'original_freeze_changed':False,'new_unseen_outcomes':0,'production_approved':False}
    if bse_parts:
        bs=pd.concat(bse_parts,ignore_index=True);bs.to_parquet(out/'BSE_transfer_scores.parquet',index=False)
        bm={}
        for h in [6,12]:
            z=bs[bs['mature_date_'+str(h)].le(pd.Timestamp(asof)) & bs['hit2_'+str(h)].isin([0,1]) & bs['expanded_p'+str(h)].notna()]
            if len(z):bm[str(h)]={'rows':len(z),'expanded_brier':brier_score_loss(z['hit2_'+str(h)],z['expanded_p'+str(h)]),'base_rate':float(z['hit2_'+str(h)].mean())}
        summary['BSE_transfer']={'scored_rows':len(bs),'current_rows':int(bs.date.eq(pd.Timestamp(asof)).sum()),'profit_gate_rows':int(bs.profit_gate.sum()),'probability_metrics':bm,'independent_validation':False,'financial_coverage_complete':False}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    pd.DataFrame(current).to_json(out/'current_research.json',orient='records',date_format='iso',indent=2)
    pd.DataFrame(selected).to_parquet(out/'selected_outcomes.parquet',index=False)
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--matrix',required=True);p.add_argument('--baseline',required=True);p.add_argument('--output',required=True);p.add_argument('--bse-matrix');a=p.parse_args();run(a.matrix,a.baseline,a.output,bse_matrix=a.bse_matrix)
