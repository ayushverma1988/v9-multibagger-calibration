"""Chronological, coherent 6/12-month research probabilities and soft risk rank.

Four conditional heads enforce nested horizons and mutually exclusive first
events by construction. Conditional outcomes never enter the predictor list.
Mixture weights are learned on earlier calibration dates, not forecast dates.
"""
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
import v12_5_expanded_model as old
from v13_features import EXTRA
from v13_price_policy import valid_price

FEATURES=list(dict.fromkeys([*old.FEATURES,*EXTRA]))
HIT_LOSS='loss_given_hit12';NOHIT_LOSS='loss_given_nohit12'
TARGETS=['hit2_6',coherent.LATE,HIT_LOSS,NOHIT_LOSS]
RISK_CEILING=.95;RISK_WEIGHT=.05;TOP_K=20

def prepare(d):
    d=coherent.prepare(d)
    d[HIT_LOSS]=d[coherent.LOSS].where(d.hit2_12.eq(1))
    d[NOHIT_LOSS]=d[coherent.LOSS].where(d.hit2_12.eq(0))
    for c in FEATURES:
        if c not in d:d[c]=np.nan
    return d

def parts(data,date,target):
    h=6 if target=='hit2_6' else 12; maturity='mature_date_'+str(h)
    known=data[data.date.lt(date)&data[maturity].lt(date)&data[target].isin([0,1])]
    dates=sorted(known.date.unique())
    if len(dates)<3:return None
    cal_dates=dates[-3:];train=known[known.date.lt(cal_dates[0])&known[maturity].lt(cal_dates[0])]
    cal=known[known.date.isin(cal_dates)]
    if len(train)<30 or train.date.nunique()<2 or len(cal)<10:return None
    assert train[maturity].lt(cal.date.min()).all() and cal[maturity].lt(date).all()
    return train,cal

def transform(p,d):
    x=d[p['features']].to_numpy(float);missing=~np.isfinite(x)
    return np.c_[np.clip(np.where(missing,p['median'],x),p['lo'],p['hi']),missing.astype(float)]

def raw_score(p,d):
    if p['method']=='shrunk_past_event_rate':return np.repeat(p['constant'],len(d))
    x=transform(p,d);w=p['linear_weight']
    return w*p['linear'].predict_proba(x)[:,1]+(1-w)*p['nonlinear'].predict_proba(x)[:,1]

def score(p,d):
    raw=np.clip(raw_score(p,d),1e-5,1-1e-5);c=p['calibration']
    return expit(c['slope']*logit(raw)+c['intercept'])

def fit(train,cal,target,feature_names=None):
    h=6 if target=='hit2_6' else 12
    p={'target':target,'train_rows':len(train),'cal_rows':len(cal),
       'train_maturity_max':str(train['mature_date_'+str(h)].max()),'cal_start':str(cal.date.min()),
       'cal_maturity_max':str(cal['mature_date_'+str(h)].max())}
    y=train[target].astype(int)
    if len(train)<500 or min(y.sum(),y.eq(0).sum())<10:
        # Conditional classes can be rare. Report an explicit baseline instead
        # of training an unstable tree or manufacturing certainty.
        p.update(method='shrunk_past_event_rate',constant=float((y.sum()+1)/(len(y)+2)),features=[])
        p['calibration']=core.calibrate(raw_score(p,cal),cal[target],y);return p
    # Optional catalogue is used only by separately evaluated research heads.
    # Existing callers and frozen serialized heads retain their exact inputs.
    features=[c for c in (FEATURES if feature_names is None else feature_names) if c in train and np.isfinite(train[c]).any()]
    clean=train[features].replace([np.inf,-np.inf],np.nan).to_numpy(float)
    p.update(method='learned_mixture',features=features,median=np.nanmedian(clean,axis=0),
       lo=np.nanquantile(clean,.005,axis=0),hi=np.nanquantile(clean,.995,axis=0))
    x=transform(p,train);w=core.weights(train)
    p['linear']=make_pipeline(StandardScaler(),LogisticRegression(C=.03,max_iter=1000,random_state=31))
    p['linear'].fit(x,y,logisticregression__sample_weight=w)
    p['nonlinear']=HistGradientBoostingClassifier(max_iter=60,max_leaf_nodes=7,min_samples_leaf=100,
       l2_regularization=10,learning_rate=.05,early_stopping=False,random_state=31).fit(x,y,sample_weight=w)
    # Select blend using first two calibration dates; fit calibration on the
    # final date. This avoids fitting both knobs to the same held-out rows.
    blend=cal[cal.date.lt(cal.date.max())];final=cal[cal.date.eq(cal.date.max())]
    a=p['linear'].predict_proba(transform(p,blend))[:,1];b=p['nonlinear'].predict_proba(transform(p,blend))[:,1]
    choices=[(brier_score_loss(blend[target],v*a+(1-v)*b),abs(v-.75),v) for v in (0,.25,.5,.75,1)]
    p['linear_weight']=float(min(choices)[2]);p['nonlinear_weight']=1-p['linear_weight']
    p['calibration']=core.calibrate(raw_score(p,final),final[target],y)
    p['feature_weights']=dict(zip([*features,*[c+'__missing' for c in features]],p['linear'].named_steps['logisticregression'].coef_[0].tolist()))
    return p

def probabilities(heads,d):
    out={}
    if 'hit2_6' not in heads:return out
    p6=score(heads['hit2_6'],d);out['p6']=p6
    if coherent.LATE not in heads:return out
    p12=coherent.union_probability(p6,score(heads[coherent.LATE],d));out['p12']=p12
    if HIT_LOSS not in heads or NOHIT_LOSS not in heads:return out
    a=score(heads[HIT_LOSS],d);b=score(heads[NOHIT_LOSS],d)
    out.update(loss12=p12*a+(1-p12)*b,joint12=p12*(1-a),loss_given_hit12=a,loss_given_nohit12=b)
    return out

def gates(d):
    known_bad=(d.four_profitable_quarters.eq(0)|d.quarter_pat_margin.le(0)|d.annual_CFO_PAT.lt(.4)
      |d.dated_debt_equity.gt(1.5)|d.annual_EBIT_proxy_interest_coverage.lt(1.5))
    market=valid_price(d.close)&d.distance_ma50.gt(0)&d.log_ret63.le(np.log(2.5))
    financial=np.isfinite(d[list(old.BASE_FIELDS)].to_numpy(float)).all(axis=1)
    verified=market & ~known_bad & financial & d.four_profitable_quarters.eq(1)&d.quarter_sales_yoy.ge(0)&d.quarter_pat_yoy.gt(0)
    discovery=market&~known_bad&~verified
    return verified,discovery,known_bad

def ranked(d,kind):
    v,q,b=gates(d);pool=d[v if kind=='verified' else q].copy()
    pool=pool[pool.p12.notna()&pool.loss12.le(RISK_CEILING)]
    pool['ranking_score']=(1-RISK_WEIGHT)*pool.p12.rank(pct=True)+RISK_WEIGHT*(-pool.loss12).rank(pct=True)
    pool['financial_tier']=kind;pool['accuracy_validated']=False
    return pool.sort_values(['ranking_score','isin'],ascending=[False,True]).head(TOP_K)

def run(matrix,output,asof,only_current=False):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    protocol={'version':'V13-NSE-coherent-soft-risk','registered_utc':datetime.now(timezone.utc).isoformat(),
      'asof':asof,'matrix_sha256':core.sha_file(matrix),'code_sha256':core.sha_file(__file__),
      'features':FEATURES,'targets':TARGETS,'risk_ceiling':RISK_CEILING,'risk_rank_weight':RISK_WEIGHT,
      'upside_rank_weight':1-RISK_WEIGHT,'top_k_per_tier':TOP_K,'blend_grid':[0,.25,.5,.75,1],
      'historical_results_previously_examined':True,'unseen_mature_outcomes':0,
      'legacy_V10_numeric_weights_reused':False,'loss_label':'30pct adverse price move before 2x close or deadline',
      'target':'next-session-open to adjusted closing total-return proxy net of 0.5pct per side',
      'production_accuracy_approved':False,'BSE_required':False}
    with (out/'protocol.json').open('x') as f:json.dump(protocol,f,indent=2)
    data=prepare(pd.read_parquet(matrix));scores=[];clocks=[];current_heads={}
    for date,test in data.groupby('date',sort=True):
        if only_current and date!=pd.Timestamp(asof):continue
        heads={}
        for target in TARGETS:
            split=parts(data,date,target)
            if split is None:continue
            p=fit(*split,target);heads[target]=p
            clocks.append({**{k:p[k] for k in ['target','method','train_rows','cal_rows','train_maturity_max','cal_start','cal_maturity_max']},'date':str(date.date()),'linear_weight':p.get('linear_weight')})
        predictions=probabilities(heads,test)
        if not predictions:continue
        d=test.copy()
        for name,values in predictions.items():d[name]=values
        scores.append(d)
        if date==pd.Timestamp(asof):current_heads=heads
        print('V13 scored',str(date.date()),len(d),list(predictions),flush=True)
    s=pd.concat(scores,ignore_index=True)
    for c in ('p12','loss12','joint12'):
        if c not in s:s[c]=np.nan
    s.to_parquet(out/'scores.parquet',index=False);joblib.dump(current_heads,out/'research_heads.joblib')
    (out/'training_clocks.json').write_text(json.dumps(clocks,indent=2))
    weights={t:{'method':p['method'],'linear_weight':p.get('linear_weight'),'nonlinear_weight':p.get('nonlinear_weight'),
       'standardized_linear_coefficients':p.get('feature_weights',{}),'features_used':p['features']} for t,p in current_heads.items()}
    (out/'learned_weights.json').write_text(json.dumps(weights,indent=2))
    summaries=[];selected=[];metrics={}
    for h in (6,12):
        target='hit2_'+str(h);prob='p'+str(h)
        eligible=s[s['mature_date_'+str(h)].le(pd.Timestamp(asof))&s[target].isin([0,1])&s[prob].notna()]
        if len(eligible):metrics[str(h)]={'rows':len(eligible),'base_rate':float(eligible[target].mean()),'brier':brier_score_loss(eligible[target],eligible[prob]),'PR_AUC':average_precision_score(eligible[target],eligible[prob])}
        for tier in ('verified','discovery'):
            dates=0;hits=known=total=0;all_hits=0
            for date,d in s.groupby('date',sort=True):
                if date==pd.Timestamp(asof) or d[prob].isna().any():continue
                matured=d['mature_date_'+str(h)].le(pd.Timestamp(asof))&d[target].isin([0,1])
                if not matured.any():continue
                picks=ranked(d,tier);k=picks[picks['mature_date_'+str(h)].le(pd.Timestamp(asof))&picks[target].isin([0,1])]
                dates+=1;total+=len(picks);known+=len(k);hits+=int(k[target].sum());all_hits+=int(d.loc[matured,target].sum())
                for r in picks.to_dict('records'):selected.append({'date':date,'isin':r['isin'],'symbol':r['symbol'],'tier':tier,'months':h,'known':r['mature_date_'+str(h)]<=pd.Timestamp(asof) if pd.notna(r['mature_date_'+str(h)]) else False,'hit2':r[target]})
            summaries.append({'months':h,'tier':tier,'dates':dates,'selected':total,'known':known,'unknown':total-known,'hits':hits,'precision_known':hits/known if known else None,'all_market_hits_on_dates':all_hits,'market_hit_capture':hits/all_hits if all_hits else None})
    now=s[s.date.eq(pd.Timestamp(asof))].copy();v,q,b=gates(now)
    current=pd.concat([ranked(now,'verified'),ranked(now,'discovery')],ignore_index=True)
    current.to_json(out/'current_watchlist.json',orient='records',date_format='iso',indent=2)
    pd.DataFrame(selected).to_parquet(out/'selected_outcomes.parquet',index=False)
    z=s[s.mature_date_12.le(pd.Timestamp(asof))&s[coherent.LOSS].isin([0,1])&s.loss12.notna()]
    summary={'current_scored':len(now),'verified_pool':int(v.sum()),'discovery_pool':int(q.sum()),'known_bad_excluded':int(b.sum()),
      'current_selected':len(current),'historical':summaries,'probability_metrics':metrics,
      'risk_metrics':{'rows':len(z),'brier':brier_score_loss(z[coherent.LOSS],z.loss12),'predicted_mean':float(z.loss12.mean()),'observed_rate':float(z[coherent.LOSS].mean())} if len(z) else None,
      'coherence_failures':int(((s.p6>s.p12+1e-12)|(s.joint12>s.p12+1e-12)|(s.joint12+s.loss12>1+1e-12)).sum()),
      'unseen_mature_outcomes':0,'production_accuracy_approved':False}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False));print(json.dumps(summary,indent=2),flush=True)
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--matrix',required=True);p.add_argument('--output',required=True);p.add_argument('--asof',required=True);p.add_argument('--only-current',action='store_true');a=p.parse_args();run(a.matrix,a.output,a.asof,a.only_current)
