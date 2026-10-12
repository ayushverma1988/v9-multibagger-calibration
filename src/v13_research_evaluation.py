"""Predeclared small experiments and complete-cohort prospective diagnostics.

All experiments are development diagnostics, never unseen promotion evidence.
The simple baseline is a maturity-purged, smoothed prior event rate.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss,average_precision_score
from v12_public_history import bounds

GROUPS={
 'market':['relative_strength63','log_ret63','volume_confirmation','volatility63'],
 'market_financial':['relative_strength63','log_ret63','volume_confirmation','quarter_sales_yoy','sales_growth_acceleration','quarter_pat_yoy','annual_CFO_PAT','dated_debt_equity'],
 'market_financial_catalyst':['relative_strength63','log_ret63','volume_confirmation','quarter_sales_yoy','sales_growth_acceleration','quarter_pat_yoy','annual_CFO_PAT','dated_debt_equity','commissioned_document_count90','utilisation_disclosed_pct','disclosed_order_value_due12_to_sales']}

def prior_rates(data,date):
    result={}
    low,_=bounds(date)
    for h in (6,12):
        q=data[data.date.ge(low)&data.date.lt(pd.Timestamp(date))&data['mature_date_'+str(h)].lt(pd.Timestamp(date))&data['hit2_'+str(h)].isin([0,1])]
        result['baseline_p'+str(h)]=float((q['hit2_'+str(h)].sum()+1)/(len(q)+2)) if len(q) else None
    return result

def calibration(y,p):
    rows=[]
    for a,b in zip([0,.05,.1,.2,.4,.6,.8],[.05,.1,.2,.4,.6,.8,1.00001]):
        k=(p>=a)&(p<b)
        if k.any():rows.append({'lower':a,'upper':min(b,1),'rows':int(k.sum()),'predicted':float(p[k].mean()),'observed':float(y[k].mean())})
    return rows

def metrics(q,ycol,pcol):
    z=q[q[ycol].isin([0,1])&q[pcol].notna()]
    if z.empty:return {'known':0,'brier':None,'PR_AUC':None,'calibration':[]}
    y=z[ycol].to_numpy(float);p=z[pcol].to_numpy(float)
    return {'known':len(z),'observed_rate':float(y.mean()),'brier':float(brier_score_loss(y,p)),
      'PR_AUC':float(average_precision_score(y,p)) if y.sum() else None,'calibration':calibration(y,p)}

def prospective_report(observations):
    result={'scope':'ALL_REGISTERED_SELECTED_AND_UNSELECTED','accuracy_approved':False,'groups':[]}
    if observations.empty:return result
    d=observations.copy()
    for c in ('financial_tier','market_regime','liquidity_tier'):
        if c not in d:d[c]='UNKNOWN'
        d[c]=d[c].fillna('UNKNOWN')
    for h in (6,12):
        ycol='verified_hit2_'+str(h);mature='six_month_mature' if h==6 else 'twelve_month_mature'
        for axis in ('financial_tier','market_regime','liquidity_tier'):
            for (role,value),q in d.groupby(['model_role',axis],dropna=False):
                known=q[q[mature].eq(True)&q.independent_outcome_verified.eq(True)]
                if ycol not in known:known=known.assign(**{ycol:np.nan})
                picked=q.selected_verified.fillna(False)|q.selected_discovery.fillna(False)
                selected_known=known[known.selected_verified.fillna(False)|known.selected_discovery.fillna(False)]
                result['groups'].append({'months':h,'model_role':role,'axis':axis,'group':str(value),
                  'registered':len(q),'selected':int(picked.sum()),'unknown_or_immature':len(q)-len(known),
                  'selected_precision':float(selected_known[ycol].mean()) if selected_known[ycol].isin([0,1]).any() else None,
                  **metrics(known,ycol,'p'+str(h))})
    result['confirmed_early_2x']=int(d.get('confirmed_early_2x',pd.Series(False,index=d.index)).fillna(False).sum())
    result['confirmed_risk_events']=int(d.get('confirmed_risk_event',pd.Series(False,index=d.index)).fillna(False).sum())
    return result

def experiments(data,asof,output):
    """Three fixed feature groups, three latest matured monthly holdouts/horizon.

    Fit/imputation only uses labels mature BEFORE the test decision. No tuning
    or automatic winner deployment. Missing catalyst observations can block a
    group; they never become zero commissionings or fabricated profitability.
    """
    d=data.copy();low,_=bounds(asof);d=d[d.date.ge(low)]
    report={'asof':asof,'four_year_start':str(low.date()),'historical_results_previously_examined':True,
      'promotion_evidence':False,'groups_predeclared':GROUPS,'folds':[]}
    for h in (6,12):
        y='hit2_'+str(h);m='mature_date_'+str(h)
        if y not in d or m not in d:continue
        candidates=sorted(d.loc[d[m].le(pd.Timestamp(asof))&d[y].isin([0,1]),'date'].unique())[-3:]
        for date in candidates:
            train=d[d.date.lt(date)&d[m].lt(date)&d[y].isin([0,1])]
            test=d[d.date.eq(date)&d[m].le(pd.Timestamp(asof))&d[y].isin([0,1])]
            if train.empty or test.empty:continue
            p=float((train[y].sum()+1)/(len(train)+2))
            base=test.assign(prediction=p)
            report['folds'].append({'months':h,'date':str(pd.Timestamp(date).date()),'group':'prior_event_rate','train_rows':len(train),**metrics(base,y,'prediction')})
            for name,cols in GROUPS.items():
                usable=[c for c in cols if c in train and train[c].notna().sum()>=100]
                # A catalyst experiment needs measured execution features.
                if name.endswith('catalyst') and not any(c in usable for c in cols[-3:]):
                    report['folds'].append({'months':h,'date':str(pd.Timestamp(date).date()),'group':name,'status':'BLOCKED_MISSING_QUANTIFIED_CATALYSTS'});continue
                if len(train)<500 or train[y].sum()<10 or not usable:continue
                pipe=make_pipeline(SimpleImputer(strategy='median',add_indicator=True),StandardScaler(),LogisticRegression(C=.03,max_iter=1000,random_state=31))
                pipe.fit(train[usable],train[y].astype(int));scored=test.assign(prediction=pipe.predict_proba(test[usable])[:,1])
                report['folds'].append({'months':h,'date':str(pd.Timestamp(date).date()),'group':name,'features':usable,'train_rows':len(train),
                  'train_maturity_max':str(train[m].max()),**metrics(scored,y,'prediction')})
    Path(output).write_text(json.dumps(report,indent=2,allow_nan=False));return report
