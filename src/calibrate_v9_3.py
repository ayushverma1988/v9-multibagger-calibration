from __future__ import annotations
import argparse, json, math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

import calibrate_v9_2 as base


def add_return_label(data, daily, cfg):
    out=data.copy(); calendar=np.array(sorted(daily.date.drop_duplicates().to_numpy(dtype='datetime64[ns]')))
    lookup={d:i for i,d in enumerate(calendar)}; maps={}
    for sym,g in daily.groupby('symbol',sort=False):
        g=g.sort_values('date'); dates=g.date.to_numpy(dtype='datetime64[ns]')
        maps[sym]=(dates,g.adj_close.to_numpy(float),np.array([lookup[d] for d in dates],int))
    h=int(cfg['label_days']['y6']); lo,hi=map(float,cfg.get('return_target_clip',[-.95,3.0]))
    vals=[]; model=[]
    for r in out.itertuples(index=False):
        dates,p,cp=maps[r.symbol]; d=np.datetime64(pd.Timestamp(r.date).to_datetime64())
        ci=np.searchsorted(calendar,d); i=np.searchsorted(dates,d)
        if ci>=len(calendar) or i>=len(dates) or ci+h>=len(calendar) or calendar[ci]!=d or dates[i]!=d:
            x=np.nan
        elif cp[-1] < ci+h:
            x=-1.0
        else:
            j=np.searchsorted(cp,ci+h,side='right')-1; p0=float(r.adj_close)
            x=float(p[j]/p0-1) if j>i and np.isfinite(p[j]) and p0>0 else np.nan
        vals.append(x); model.append(float(np.clip(x,lo,hi)) if np.isfinite(x) else np.nan)
    out['ret6']=vals; out['ret6_model']=model
    return out


def return_models(seed):
    lin=Pipeline([('impute',SimpleImputer(strategy='median')),('scale',RobustScaler()),('reg',Ridge(alpha=5.0))])
    gbm=Pipeline([('impute',SimpleImputer(strategy='median')),('reg',HistGradientBoostingRegressor(
        max_depth=3,learning_rate=.05,max_iter=180,min_samples_leaf=60,l2_regularization=1.0,random_state=seed))])
    return lin,gbm


def predict_return(model,train,test):
    tr=train.dropna(subset=['ret6_model'])
    if len(tr)<250: return np.full(len(test),np.nan)
    model.fit(tr[base.MODEL_FEATURES],tr.ret6_model.astype(float))
    return model.predict(test[base.MODEL_FEATURES])


def walk_forward_return(data,oos_dates,cfg):
    seed=int(cfg['random_seed']); rows=[]
    for td in sorted(pd.to_datetime(oos_dates)):
        test=data[data.date==td].copy()
        start=td-pd.DateOffset(years=int(cfg['rolling_train_years']))
        train=data[(data.date<td)&(data.date>=start)].copy()
        if 'y6_mature_date' in train:
            train.loc[pd.to_datetime(train.y6_mature_date)>td,['ret6','ret6_model']]=np.nan
        if train.ret6_model.notna().sum()<800: continue
        m1,m2=return_models(seed+202); a=predict_return(m1,train,test); b=predict_return(m2,train,test)
        z=test[['date','symbol']].copy(); z['er_raw']=np.nanmean(np.vstack([a,b]).T,axis=1)
        z['return_model_dispersion']=np.nanstd(np.vstack([a,b]).T,axis=1); rows.append(z)
    return pd.concat(rows,ignore_index=True) if rows else pd.DataFrame()


def fit_return_calibrator(d,cfg):
    q=d.dropna(subset=['er_raw','ret6_model'])
    if len(q)<int(cfg.get('return_calibration_min_rows',1500)): return None
    m=Ridge(alpha=float(cfg.get('return_calibration_alpha',10.0))).fit(q[['er_raw']],q.ret6_model)
    return m


def forward_return_calibration(oos,cfg):
    out=oos.copy(); out['er_cal']=np.nan; lo,hi=map(float,cfg.get('return_prediction_clip',[-.75,2.0]))
    for td in sorted(pd.to_datetime(out.date.unique())):
        m=fit_return_calibrator(out[out.date<td],cfg); idx=out.index[(out.date==td)&out.er_raw.notna()]
        if m is not None and len(idx):
            out.loc[idx,'er_cal']=np.clip(m.predict(out.loc[idx,['er_raw']]),lo,hi)
    return out


def add_components(df):
    parts=[]
    for _,g in df.groupby('date',sort=False):
        g=g.copy(); g['comp_upside']=g.p_cal.rank(pct=True)
        g['comp_safety']=g.p_dd30_cal.rank(pct=True,ascending=False)
        g['comp_expected_return']=g.er_cal.rank(pct=True)
        cs=[]
        for c in ['model_dispersion','dd_model_dispersion','return_model_dispersion']:
            if c in g: cs.append(g[c].rank(pct=True,ascending=False))
        g['comp_consensus']=pd.concat(cs,axis=1).mean(axis=1) if cs else .5; parts.append(g)
    return pd.concat(parts).sort_index()


def weight_grid(cfg):
    step=float(cfg.get('selection_weight_step',.1)); n=int(round(1/step))
    mins=[int(round(float(cfg.get(k,v))/step)) for k,v in [
        ('min_upside_weight',.2),('min_safety_weight',.1),('min_expected_return_weight',.2)]]
    maxc=int(round(float(cfg.get('max_consensus_weight',.3))/step)); seen=set()
    for w in [(1.,0.,0.,0.),(.35,.25,.30,.10)]:
        seen.add(tuple(round(x,6) for x in w)); yield w
    for a in range(n+1):
        for b in range(n+1-a):
            for c in range(n+1-a-b):
                d=n-a-b-c
                if a<mins[0] or b<mins[1] or c<mins[2] or d>maxc: continue
                w=(a*step,b*step,c*step,d*step); k=tuple(round(x,6) for x in w)
                if k not in seen: seen.add(k); yield w


def score(g,w):
    return w[0]*g.comp_upside+w[1]*g.comp_safety+w[2]*g.comp_expected_return+w[3]*g.comp_consensus


def fold_stats(g,col,k,cfg):
    q=g.dropna(subset=[col,'y6','ret6','dd30_6m']).sort_values(col,ascending=False)
    if len(q)<max(k,20): return None
    s=q.head(k); br=float(q.y6.mean()); pr=float(s.y6.mean()); lift=pr/br if br>0 else np.nan
    lo,hi=map(float,cfg.get('return_eval_clip',[-1.,3.])); qr=np.clip(q.ret6.to_numpy(float),lo,hi); sr=np.clip(s.ret6.to_numpy(float),lo,hi)
    bret=float(qr.mean()); sret=float(sr.mean()); bdd=float(q.dd30_6m.mean()); sdd=float(s.dd30_6m.mean())
    ls=math.log1p(min(max(lift,0),10))/math.log(11) if np.isfinite(lift) else 0
    uq=.5*float(np.clip(pr/.20,0,1))+.5*ls
    rs=float(np.clip((sret-bret)/.50,-1,1)); ds=float(np.clip((bdd-sdd)/.30,-1,1))
    uw=cfg.get('selection_utility_weights',{'upside':.40,'return':.45,'downside':.15})
    util=float(uw['upside'])*uq+float(uw['return'])*rs+float(uw['downside'])*ds
    return dict(n=len(q),precision_2x=pr,lift_2x=lift,mean_ret_6m=sret,median_ret_6m=float(np.median(sr)),
                positive_return_rate=float(np.mean(sr>0)),dd30_rate=sdd,base_rate_2x=br,base_mean_ret_6m=bret,
                base_dd30_rate=bdd,utility=util)


def eval_weights(hist,w,cfg):
    rows=[]; k=int(cfg.get('selection_k',10))
    for td,g in hist.groupby('date'):
        g=g.copy(); g['_s']=score(g,w); r=fold_stats(g,'_s',k,cfg)
        if r: r['date']=td; rows.append(r)
    if len(rows)<int(cfg.get('selection_min_folds',8)): return None
    t=pd.DataFrame(rows); u=t.utility.to_numpy(); q1,q3=np.quantile(u,[.25,.75]); iqr=float(q3-q1); bad=float(np.mean(u<0))
    obj=float(np.median(u))-float(cfg.get('selection_iqr_penalty',.35))*iqr-float(cfg.get('selection_bad_fold_penalty',.15))*bad
    return dict(objective=obj,median_utility=float(np.median(u)),mean_utility=float(np.mean(u)),utility_iqr=iqr,
                bad_fold_rate=bad,folds=len(t),median_precision_2x=float(t.precision_2x.median()),
                median_lift_2x=float(t.lift_2x.replace([np.inf,-np.inf],np.nan).median()),
                median_mean_ret_6m=float(t.mean_ret_6m.median()),median_dd30_rate=float(t.dd30_rate.median()))


def optimize(hist,cfg):
    req=['p_cal','p_dd30_cal','er_cal','y6','ret6','dd30_6m']; d=hist.dropna(subset=req).copy()
    if d.date.nunique()<int(cfg.get('selection_min_folds',8)): return None,pd.DataFrame()
    if 'comp_upside' not in d: d=add_components(d)
    rows=[]
    for w in weight_grid(cfg):
        m=eval_weights(d,w,cfg)
        if m: rows.append(dict(w_upside=w[0],w_safety=w[1],w_expected_return=w[2],w_consensus=w[3],**m))
    if not rows: return None,pd.DataFrame()
    t=pd.DataFrame(rows).sort_values(['objective','median_utility','median_mean_ret_6m'],ascending=False).reset_index(drop=True)
    r=t.iloc[0]; return (float(r.w_upside),float(r.w_safety),float(r.w_expected_return),float(r.w_consensus)),t


def forward_selection(oos,cfg):
    out=add_components(oos); out['selection_score']=np.nan
    for c in ['weight_upside','weight_safety','weight_expected_return','weight_consensus','selection_train_objective']: out[c]=np.nan
    for td in sorted(pd.to_datetime(out.date.unique())):
        w,t=optimize(out[out.date<td],cfg); idx=out.index[out.date==td]
        if w is None: continue
        out.loc[idx,'selection_score']=score(out.loc[idx],w)
        out.loc[idx,['weight_upside','weight_safety','weight_expected_return','weight_consensus']]=list(w)
        if len(t): out.loc[idx,'selection_train_objective']=float(t.iloc[0].objective)
    return out


def selection_metrics(oos,cfg):
    k=int(cfg.get('selection_k',10)); valid=set(oos.loc[oos.selection_score.notna(),'date'].unique()); rows=[]
    for name,col in [('V9.3','selection_score'),('V9.2_pcal','p_cal')]:
        for td,g in oos[oos.date.isin(valid)].groupby('date'):
            r=fold_stats(g,col,k,cfg)
            if r: r.update(date=td,strategy=name,k=k); rows.append(r)
    return pd.DataFrame(rows)


def current_return_predictions(data,daily,cfg):
    last=pd.Timestamp(daily.date.max()); cur=daily[daily.date==last].copy()
    cur=cur[cur.isin.notna() & cur.isin.astype(str).str.startswith('INE')]
    cur=cur[(cur.close>=cfg['price_min'])&(cur.close<=cfg['price_max'])]
    cur=cur[(cur.avg_turnover_63>=cfg['min_avg_turnover_63d'])&(cur.history_days>=cfg['min_history_days'])]
    cur['breadth_120']=float(np.nanmean(cur.ret_120.to_numpy()>0)); cur['market_median_ret120']=float(np.nanmedian(cur.ret_120))
    for c in base.RANKABLE: cur[f'{c}_rank']=cur[c].rank(pct=True)
    start=last-pd.DateOffset(years=int(cfg['rolling_train_years'])); tr=data[(data.date>=start)&data.ret6_model.notna()]
    a,b=return_models(int(cfg['random_seed'])+202); p1=predict_return(a,tr,cur); p2=predict_return(b,tr,cur)
    z=cur[['date','symbol']].copy(); z['er_raw']=np.nanmean(np.vstack([p1,p2]).T,axis=1)
    z['return_model_dispersion']=np.nanstd(np.vstack([p1,p2]).T,axis=1); return z


def summarize(m,name):
    d=m[m.strategy==name]
    if d.empty: return {}
    return dict(folds=int(d.date.nunique()),median_precision_2x=float(d.precision_2x.median()),
        median_lift_2x=float(d.lift_2x.replace([np.inf,-np.inf],np.nan).median()),
        median_mean_ret_6m=float(d.mean_ret_6m.median()),median_median_ret_6m=float(d.median_ret_6m.median()),
        median_positive_return_rate=float(d.positive_return_rate.median()),median_dd30_rate=float(d.dd30_rate.median()),
        mean_utility=float(d.utility.mean()),median_utility=float(d.utility.median()))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default='config_v9_3.json'); ap.add_argument('--output',default='outputs_v9_3'); ap.add_argument('--legacy-dir',default=None)
    a=ap.parse_args(); cfg=json.load(open(a.config)); out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    base.cfg_h24=int(cfg['label_days']['y24']); end=pd.Timestamp.today().year
    daily=base.load_market(int(cfg['start_year']),end,a.legacy_dir); daily=base.add_features(daily)
    snap=base.build_snapshots(daily,cfg); data=add_return_label(base.add_labels(snap,daily,cfg),daily,cfg); data.to_parquet(out/'snapshot_dataset.parquet',index=False)

    oos=base.walk_forward(data,cfg)
    if oos.empty: raise RuntimeError('No OOS predictions')
    labels=data[['date','symbol','ret6','ret6_model']]; oos=oos.merge(labels,on=['date','symbol'],how='left')
    ro=walk_forward_return(data,oos.date.unique(),cfg); oos=oos.merge(ro,on=['date','symbol'],how='left')
    best,ct=base.evaluate_calibrators(oos,'y6','p_raw'); bestdd,dt=base.evaluate_calibrators(oos,'dd30_6m','p_dd30_raw')
    oos=base.apply_forward_calibration(oos,best,'y6','p_raw','p_cal')
    oos=base.apply_forward_calibration(oos,bestdd,'dd30_6m','p_dd30_raw','p_dd30_cal')
    oos=forward_return_calibration(oos,cfg); oos['asymmetry']=oos.p_cal/np.maximum(oos.p_dd30_cal,.01); oos=forward_selection(oos,cfg)
    oos.to_parquet(out/'oos_predictions.parquet',index=False); ct.to_csv(out/'calibrator_comparison.csv',index=False); dt.to_csv(out/'downside_calibrator_comparison.csv',index=False)
    top=base.top_metrics(oos); top.to_csv(out/'topk_by_fold.csv',index=False); sm=selection_metrics(oos,cfg); sm.to_csv(out/'selection_metrics_by_fold.csv',index=False)

    current=base.fit_current(data,daily,oos,best,bestdd,cfg); current['p_dd30_cal']=current.p_dd30
    cr=current_return_predictions(data,daily,cfg); current=current.merge(cr,on=['date','symbol'],how='left')
    rcal=fit_return_calibrator(oos,cfg); lo,hi=map(float,cfg.get('return_prediction_clip',[-.75,2.0]))
    current['er_cal']=np.clip(rcal.predict(current[['er_raw']]),lo,hi) if rcal is not None else current.er_raw
    current['expected_6m_price_proxy']=current.close*(1+current.er_cal); current=add_components(current)
    w,search=optimize(oos,cfg); w=w or (1.,0.,0.,0.); current['selection_score']=score(current,w)
    current['selection_rank']=current.selection_score.rank(method='first',ascending=False).astype(int)
    for c,v in zip(['weight_upside','weight_safety','weight_expected_return','weight_consensus'],w): current[c]=v
    current=current.sort_values('selection_score',ascending=False).reset_index(drop=True)
    current.to_csv(out/'current_selection.csv',index=False); current.head(50).to_csv(out/'current_top50.csv',index=False); search.to_csv(out/'selection_weight_search.csv',index=False)

    e=oos.dropna(subset=['y6','p_cal']); y=e.y6.astype(int).to_numpy(); p=e.p_cal.to_numpy(); slope,inter=base.calibration_slope_intercept(y,p)
    re=oos.dropna(subset=['ret6_model','er_cal']); rho=spearmanr(re.ret6_model,re.er_cal,nan_policy='omit') if len(re) else None
    summary={'model':cfg['model_name'],'pipeline_version':cfg['pipeline_version'],'data_start':str(daily.date.min().date()),'data_end':str(daily.date.max().date()),
      'oos_rows_calibrated':len(e),'oos_positive_6m':int(y.sum()),'base_rate_6m':float(y.mean()),'best_calibrator':best,
      'brier':float(base.brier_score_loss(y,p)),'logloss':float(base.log_loss(y,np.clip(p,base.EPS,1-base.EPS))),'pr_auc':float(base.average_precision_score(y,p)),
      'calibration_slope':slope,'calibration_intercept':inter,'best_downside_calibrator':bestdd,
      'return_model_target':'126-session split/bonus-normalized return; dividends excluded; disappearance before horizon=-100%',
      'return_mae':float(mean_absolute_error(re.ret6_model,re.er_cal)) if len(re) else None,
      'return_rmse':float(mean_squared_error(re.ret6_model,re.er_cal)**.5) if len(re) else None,
      'return_spearman':float(rho.statistic) if rho is not None and np.isfinite(rho.statistic) else None,
      'selection_k':int(cfg.get('selection_k',10)),'production_weights':dict(zip(['upside_2x','downside_safety','expected_return','model_consensus'],map(float,w))),
      'selection_optimizer':{'objective':'median fold utility - IQR penalty - bad-fold penalty','leakage_control':'each OOS fold uses prior folds only','do_no_harm_baseline_in_search':True},
      'selection_history':{'V9.3':summarize(sm,'V9.3'),'V9.2_pcal':summarize(sm,'V9.2_pcal')}}
    json.dump(summary,open(out/'summary.json','w'),indent=2); print(json.dumps(summary,indent=2))
    print(current.head(25)[['selection_rank','symbol','close','p_cal','p_dd30','er_cal','selection_score']].to_string(index=False))


if __name__=='__main__': main()
