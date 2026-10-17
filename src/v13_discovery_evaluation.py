"""Matched, maturity-purged six-month ablations; no operational promotion."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import brier_score_loss,average_precision_score
import v13_model as model
import v13_discovery_diagnostics as dd
import v13_historical_discovery as hd
from v13_financial_evidence import REQUIRED,financial_quality

def eligible(d):
    valid=d.feature_eligible.fillna(False).astype(bool) if 'feature_eligible' in d else pd.Series(False,index=d.index)
    valid &= d.close.between(20,2000)&d.distance_ma50.gt(0)&d.log_ret63.le(np.log(2.5))&d.rsi14.gt(70)
    failed=pd.Series([financial_quality({k:r.get(k,np.nan) for k in REQUIRED})['status']=='KNOWN_FINANCIAL_FAILURE' for r in d.to_dict('records')],index=d.index)
    return valid&~failed

def run(matrix,output,asof):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    source=Path(matrix);d=model.prepare(pd.read_parquet(source));asof=pd.Timestamp(asof)
    variants={'baseline':model.FEATURES,'breakout':list(dict.fromkeys(model.FEATURES+dd.FEATURES)),
      'catalyst':list(dict.fromkeys(model.FEATURES+hd.FEATURES)),
      'combined':list(dict.fromkeys(model.FEATURES+dd.FEATURES+hd.FEATURES))}
    protocol={'primary_target':'hit2_6','secondary_target':'hit2_12_only_when_mature','matrix_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
      'asof':str(asof.date()),'variants':variants,'selection':'same eligible pool; top20 by p6; strict RSI>70',
      'unseen_holdout':False,'deployment_approved':False,'prospective_heads_changed':False,
      'current_listing_survivor_bias':True,'historical_price_vintage_limit':True,'historical_sector_size_unverified':True,
      'model_parameters':'unchanged model.fit defaults; trained feature weights; no fitted selector weights',
      'calibration_experiment':False,'code_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('v13_*discovery*.py')}}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2));outputs=[];clocks=[];blocked=[]
    for date in sorted(d.loc[d.mature_date_6.le(asof),'date'].unique()):
        split=model.parts(d,date,'hit2_6')
        if split is None:blocked.append(str(pd.Timestamp(date).date()));continue
        train,cal=split;test=d[d.date.eq(date)&d.mature_date_6.le(asof)&d.hit2_6.isin([0,1])].copy();pool=eligible(test)
        for name,features in variants.items():
            head=model.fit(train,cal,'hit2_6',feature_names=features)
            s=test.copy();s['p6']=model.score(head,test);s['variant']=name;s['selected']=False
            picks=s[pool].sort_values(['p6','isin'],ascending=[False,True]).head(20);s.loc[picks.index,'selected']=True;outputs.append(s)
            clocks.append({'date':str(pd.Timestamp(date).date()),'variant':name,**{k:head[k] for k in ['train_rows','cal_rows','train_maturity_max','cal_start','cal_maturity_max','features']},'feature_weights':head.get('feature_weights',{})})
        print('Evaluated',str(pd.Timestamp(date).date()),len(test),'eligible',int(pool.sum()),flush=True)
    if not outputs:raise ValueError('No maturity-purged folds')
    scores=pd.concat(outputs,ignore_index=True);scores.to_parquet(out/'scores.parquet',index=False)
    metrics={};folds=[]
    for name,g in scores.groupby('variant'):
        p=g[g.selected];m12=p[p.mature_date_12.le(asof)&p.hit2_12.isin([0,1])]
        metrics[name]={'rows':len(g),'decision_dates':g.date.nunique(),'selected':len(p),'hits6':int(p.hit2_6.sum()),'precision6':float(p.hit2_6.mean()) if len(p) else None,
          'brier6':float(brier_score_loss(g.hit2_6,g.p6)),'pr_auc6':float(average_precision_score(g.hit2_6,g.p6)),
          'loss30_6_events':int(p.loss30_6.sum()),'median_endpoint_return6':float(p.endpoint_return_6.median()) if len(p) else None,
          'known12':len(m12),'hits12':int(m12.hit2_12.sum()),'mean_predicted6':float(g.p6.mean()),'observed_rate6':float(g.hit2_6.mean())}
        for date,z in g.groupby('date'):
            pick=z[z.selected];folds.append(dict(variant=name,date=str(date.date()),rows=len(z),selected=len(pick),hits6=int(pick.hit2_6.sum()),risk6=int(pick.loss30_6.sum()),brier6=float(brier_score_loss(z.hit2_6,z.p6))))
    base=scores[scores.variant.eq('baseline')].set_index(['date','isin']);paired={}
    for name in variants:
        if name=='baseline':continue
        x=scores[scores.variant.eq(name)].set_index(['date','isin'])
        if not base.index.equals(x.index):raise ValueError('Unpaired stock/date keys')
        gains=((base.p6-base.hit2_6)**2-(x.p6-x.hit2_6)**2).groupby(level='date').mean()
        paired[name]={'matched_rows':len(x),'positive_gain_months':int(gains.gt(0).sum()),'total_months':len(gains),'monthly_brier_gains':{str(k.date()):float(v) for k,v in gains.items()}}
    summary={**protocol,'results':metrics,'paired':paired,'folds':folds,'blocked_early_dates':blocked,'training_clocks':clocks,
      'catalyst_finite_rows':{k:int(d[k].notna().sum()) for k in hd.FEATURES},'promotion_decision':'NOT_PROMOTED_DEVELOPMENT_ONLY'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False));return summary
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--matrix',required=True);p.add_argument('--output',required=True);p.add_argument('--asof',required=True)
    a=p.parse_args();run(a.matrix,a.output,a.asof)
