"""One registered dated-input experiment; never overwrite the V12.1 freeze.

Optional source fields remain UNKNOWN in the source matrix. Learner-only
training medians and explicit missingness flags do not pass financial gates.
Absent-in-training features are dropped for that fold. No parameter search.
"""
from __future__ import annotations
import argparse,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import brier_score_loss
import v12_four_year_model as core
import v12_coherent_hazard_model as coherent

EXTRA=('quarter_sales_yoy','quarter_pat_yoy','quarter_pat_margin','annual_CFO_PAT','dated_debt_equity','order_disclosure_count90','capacity_disclosure_count90')

def inputs(package,frame):
    raw=frame[package['features']].to_numpy(float);missing=~np.isfinite(raw)
    filled=np.where(missing,package['medians'],raw)
    return np.concatenate([np.clip(filled,package['lower'],package['upper']),missing[:,package['missing_indicator_positions']].astype(float)],axis=1)

def raw_score(p,frame):
    x=inputs(p,frame);return .7*p['linear'].predict_proba(x)[:,1]+.3*p['nonlinear'].predict_proba(x)[:,1]

def score(p,frame):
    raw=np.clip(raw_score(p,frame),1e-5,1-1e-5);c=p['calibration'];return expit(c['slope']*logit(raw)+c['intercept'])

def fit(train,cal,target):
    features=[c for c in [*core.FEATURES,*EXTRA] if np.isfinite(train[c].to_numpy(float)).any()]
    a=train[features].to_numpy(float);missing=~np.isfinite(a);clean=np.where(missing,np.nan,a)
    medians=np.nanmedian(clean,axis=0);lower,upper=np.nanquantile(clean,[.005,.995],axis=0)
    p={'features':features,'medians':medians,'lower':lower,'upper':upper,'missing_indicator_positions':[i for i,c in enumerate(features) if c in EXTRA]}
    x=inputs(p,train);w=core.weights(train);y=train[target].astype(int)
    p['linear']=make_pipeline(StandardScaler(),LogisticRegression(C=.03,max_iter=1000,random_state=31));p['linear'].fit(x,y,logisticregression__sample_weight=w)
    p['nonlinear']=HistGradientBoostingClassifier(max_iter=60,max_leaf_nodes=7,min_samples_leaf=100,l2_regularization=10,learning_rate=.05,early_stopping=False,random_state=31).fit(x,y,sample_weight=w)
    p['calibration']=core.calibrate(raw_score(p,cal),cal[target],train[target]);return p

def run(panel_path,matrix_path,output,asof):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    recipe={'version':'V12.2-DATED-INPUT-EXPERIMENT-20261010','registered_at_utc':datetime.now(timezone.utc).isoformat(),'parent':'V12.1','added_features':list(EXTRA),'algorithm':'unchanged_70pct_LR_30pct_HGB','source_financials':'exact-period original NSE XBRL; dated max publication/dissemination/revision clocks','event_counts':'observed exchange metadata disclosures; not verified earnings catalysts','missing_features':'training-only medians plus missing indicators; all-missing-in-training fields dropped; source UNKNOWN not changed','four_year_boundary_enforced':True,'same_chronological_maturity_and_calibration_rules':True,'same_target_and_selection_technical_gate':True,'no_hyperparameter_search':True,'predeclared_variants':1,'precision_gate':.30,'production_approval_requires_full_source_gates_and_unseen_validation':True,'V12_1_weights_modified':False,'market_panel_sha256':core.sha_file(panel_path),'input_matrix_sha256':core.sha_file(matrix_path)}
    with (out/'registered_recipe.json').open('x') as f:json.dump(recipe,f,indent=2)
    panel=coherent.prepare(pd.read_parquet(panel_path));matrix=pd.read_parquet(matrix_path);matrix['date']=pd.to_datetime(matrix.date)
    for c in EXTRA:
        if c not in matrix:matrix[c]=np.nan
    data=panel.merge(matrix[['date','symbol',*EXTRA]],on=['date','symbol'],how='left',validate='one_to_one')
    frames=[];folds=[];clocks=[]
    for date in sorted(data.date.unique()):
        if date>=pd.Timestamp(asof):continue
        parts=core.chronological_parts(data,date,'hit2_6')
        if parts is None:continue
        test=data[data.date.eq(date)].copy();early=fit(*parts,'hit2_6');test['p_hit2_6']=score(early,test)
        late_parts=core.chronological_parts(data,date,coherent.LATE)
        if late_parts is not None:
            late=fit(*late_parts,coherent.LATE);test['p_hit2_12']=coherent.union_probability(test.p_hit2_6,score(late,test))
        for target,spl in [('hit2_6',parts),*([(coherent.LATE,late_parts)] if late_parts is not None else [])]:
            train,cal=spl;h='6' if target=='hit2_6' else '12';clocks.append({'date':str(pd.Timestamp(date).date()),'target':target,'train_latest_maturity':str(train['mature_date_'+h].max().date()),'first_calibration_date':str(cal.date.min().date()),'cal_latest_maturity':str(cal['mature_date_'+h].max().date())})
        for h in [6,12]:
            c='p_hit2_'+str(h);target='hit2_'+str(h)
            if c not in test:continue
            picks=test[test.technical_pass].sort_values([c,'symbol'],ascending=[False,True]).head(10)
            known=test[test['mature_date_'+str(h)].le(pd.Timestamp(asof))&test[target].isin([0,1])]
            matured=len(picks)>0 and picks['mature_date_'+str(h)].notna().all() and picks['mature_date_'+str(h)].le(pd.Timestamp(asof)).all() and picks[target].isin([0,1]).all()
            folds.append({'date':str(pd.Timestamp(date).date()),'months':h,'fully_matured':bool(matured),'selected':len(picks),'hits':int(picks[target].eq(1).sum()),'known_rows':len(known),'brier':brier_score_loss(known[target],known[c]) if len(known) else None})
        frames.append(test);print('dated candidate',str(pd.Timestamp(date).date()),len(test),flush=True)
    pd.concat(frames,ignore_index=True).to_parquet(out/'dated_candidate_chronological_scores.parquet',index=False);pd.DataFrame(folds).to_csv(out/'dated_candidate_fold_metrics.csv',index=False);pd.DataFrame(clocks).to_csv(out/'dated_candidate_training_clock_audit.csv',index=False)
    summary={'version':recipe['version'],'outcomes':{},'prospective_unseen_outcomes':0,'complete_financial_catalyst_gate_coverage':False,'original_freeze_changed':False,'production_approved':False}
    for h in [6,12]:
        ms=[r for r in folds if r['months']==h and r['fully_matured']];n=sum(r['selected'] for r in ms);hits=sum(r['hits'] for r in ms)
        summary['outcomes'][str(h)]={'fully_matured_dates':len(ms),'selected_observations':n,'confirmed_hits':hits,'selected_hit_fraction':hits/n if n else None,'precision_30pct_gate_pass':bool(n and hits/n>=.30)}
    (out/'dated_candidate_evaluation.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--panel',required=True);p.add_argument('--matrix',required=True);p.add_argument('--output',required=True);p.add_argument('--asof',default='2026-10-09');a=p.parse_args();run(a.panel,a.matrix,a.output,a.asof)
if __name__=='__main__':main()
