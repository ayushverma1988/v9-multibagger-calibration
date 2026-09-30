from __future__ import annotations
import argparse, json, math, warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from scipy.special import logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from lifelines import CoxPHFitter

warnings.filterwarnings('ignore')

REPO = 'tejhq/indian-markets'
EPS = 1e-6

FEATURES = [
    'ret_20','ret_60','ret_120','ret_252','mom_accel','vol_accel','turnover_accel',
    'off_high_252','above_low_252','trend_consistency_60','volatility_60',
    'drawdown_126','log_turnover_63','breadth_120','market_median_ret120'
]
RANKABLE = [
    'ret_20','ret_60','ret_120','ret_252','mom_accel','vol_accel','turnover_accel',
    'off_high_252','above_low_252','trend_consistency_60','volatility_60',
    'drawdown_126','log_turnover_63'
]
MODEL_FEATURES = FEATURES + [f'{c}_rank' for c in RANKABLE]


def download_optional(filename: str):
    try:
        return hf_hub_download(REPO, filename, repo_type='dataset')
    except Exception:
        return None


def _heuristic_adjust_legacy(df: pd.DataFrame) -> pd.Series:
    canon=np.array([0.1,0.2,0.25,1/3,0.5,2.0,3.0,4.0,5.0,10.0])
    out=np.empty(len(df),float)
    for _,idx in df.groupby('symbol',sort=False).groups.items():
        inds=np.asarray(list(idx),int); g=df.loc[inds].sort_values('date')
        inds=g.index.to_numpy(); close=g['close'].to_numpy(float); fac=np.ones(len(g),float)
        ratios=close[1:]/np.maximum(close[:-1],1e-12)
        for j,r in enumerate(ratios,1):
            if not np.isfinite(r) or (0.6<=r<=1.7): continue
            k=int(np.argmin(np.abs(np.log(canon)-np.log(max(r,1e-8)))))
            c=float(canon[k])
            if abs(math.log(r/c)) <= math.log(1.22):
                fac[:j]*=c
        out[inds]=close*fac
    return pd.Series(out,index=df.index)


def load_actions(start_year:int,end_year:int)->pd.DataFrame:
    frames=[]
    for y in range(start_year,end_year+1):
        p=download_optional(f'actions/nse_{y}.parquet')
        if p:
            frames.append(pd.read_parquet(p))
    if not frames:
        return pd.DataFrame()
    a=pd.concat(frames,ignore_index=True)
    a['ex_date']=pd.to_datetime(a['ex_date'])
    return a


def apply_split_bonus_adjustment(df: pd.DataFrame, start_year:int,end_year:int) -> pd.Series:
    a=load_actions(start_year,end_year)
    if a.empty:
        print('WARNING: corporate actions unavailable; using raw close.')
        return df['close'].astype(float).copy()
    a=a[a['type'].isin(['split','bonus'])].copy()
    def fac(r):
        try:
            if r['type']=='split' and pd.notna(r.get('face_value_from')) and pd.notna(r.get('face_value_to')) and float(r['face_value_from'])>0:
                return float(r['face_value_to'])/float(r['face_value_from'])
            if r['type']=='bonus' and pd.notna(r.get('ratio_num')) and pd.notna(r.get('ratio_den')):
                n,d=float(r['ratio_num']),float(r['ratio_den'])
                return d/(n+d) if n+d>0 else 1.0
        except Exception:
            pass
        return 1.0
    a['factor']=a.apply(fac,axis=1)
    a=a[(a['factor']>0)&(a['factor']<1.01)]
    amap={k:g[['ex_date','factor']].sort_values('ex_date') for k,g in a.groupby('symbol')}
    out=np.empty(len(df),dtype=float)
    for sym, idx in df.groupby('symbol',sort=False).groups.items():
        inds=np.asarray(list(idx),dtype=int)
        dates=df.loc[inds,'date'].to_numpy(dtype='datetime64[ns]')
        close=df.loc[inds,'close'].to_numpy(float)
        factors=np.ones(len(inds),dtype=float)
        g=amap.get(sym)
        if g is not None:
            ev_dates=g['ex_date'].to_numpy(dtype='datetime64[ns]')
            ev_fac=g['factor'].to_numpy(float)
            for d,f in zip(ev_dates,ev_fac):
                pos=np.searchsorted(dates,d,side='left')
                if pos>0:
                    factors[:pos]*=f
        out[inds]=close*factors
    return pd.Series(out,index=df.index)


def add_security_key(
    df: pd.DataFrame,
    continuity_low: float = 0.67,
    continuity_high: float = 1.50,
) -> pd.DataFrame:
    """Create a PIT-safe market identity with conditional ticker stitching.

    Known ISIN is the primary identity, but a ticker rename is stitched only
    when the split/bonus-normalized price is continuous across the transition.
    Large scale breaks start a new segment even when the ISIN string is the
    same. This avoids carrying momentum/labels through restructurings or bad
    corporate-action normalization. Missing ISIN rows use contemporaneous
    symbol only; no future identity is backfilled.
    """
    x = df.copy()
    x["symbol"] = x["symbol"].astype(str).str.upper().str.strip()
    isin = x.get("isin", pd.Series(index=x.index, dtype=object))
    norm = isin.astype(str).str.upper().str.strip()
    valid = isin.notna() & ~norm.isin(["", "NAN", "NONE", "<NA>"])

    x["security_key"] = "SYM:" + x["symbol"]
    if not valid.any():
        return x

    x.loc[valid, "_isin_norm"] = norm[valid]
    x.loc[valid, "_seg"] = 0

    for iv, idx in x.loc[valid].groupby("_isin_norm", sort=False).groups.items():
        g = x.loc[list(idx)].sort_values(["date", "symbol"]).copy()
        seg = 0
        prev_symbol = None
        prev_adj = np.nan
        seg_values = {}

        for ridx, row in g.iterrows():
            sym = str(row["symbol"])
            cur_adj = float(row["adj_close"]) if "adj_close" in row and np.isfinite(row["adj_close"]) else np.nan

            if prev_symbol is not None and sym != prev_symbol:
                ratio = (
                    cur_adj / prev_adj
                    if np.isfinite(cur_adj) and np.isfinite(prev_adj) and abs(prev_adj) > 1e-12
                    else np.nan
                )
                continuous = (
                    np.isfinite(ratio)
                    and float(continuity_low) <= ratio <= float(continuity_high)
                )
                if not continuous:
                    seg += 1

            seg_values[ridx] = seg
            prev_symbol = sym
            prev_adj = cur_adj

        for ridx, s in seg_values.items():
            x.at[ridx, "_seg"] = int(s)

    x.loc[valid, "security_key"] = (
        "ISIN:"
        + x.loc[valid, "_isin_norm"].astype(str)
        + "#"
        + x.loc[valid, "_seg"].astype(int).astype(str)
    )
    return x.drop(columns=["_isin_norm", "_seg"], errors="ignore")


def load_market(start_year:int,end_year:int,legacy_dir:str|None=None)->pd.DataFrame:
    frames=[]
    if legacy_dir and int(start_year)<2010:
        ld=Path(legacy_dir); legacy=[]
        for y in range(int(start_year),min(2009,end_year)+1):
            fp=ld/f'nse_{y}.parquet'
            if fp.exists(): legacy.append(pd.read_parquet(fp))
        if legacy:
            old=pd.concat(legacy,ignore_index=True)
            old['date']=pd.to_datetime(old['date'])
            old=old[old['series'].isin(['EQ','BE','BZ'])].copy()
            old=old.sort_values(['symbol','date']).reset_index(drop=True)
            # Pre-2010 archive lacks a complete corporate-action tree; use only the
            # conservative split-like jump heuristic for continuity.
            old['adj_close']=_heuristic_adjust_legacy(old)
            keep=['date','symbol','series','isin','close','volume','turnover','adj_close']
            for c in keep:
                if c not in old: old[c]=np.nan
            frames.append(old[keep])

    hf_start=max(2010,int(start_year))
    z=[]
    for y in range(hf_start,end_year+1):
        pth=hf_hub_download(REPO,f'nse/year={y}/nse_{y}.parquet',repo_type='dataset')
        x=pd.read_parquet(pth,columns=['date','symbol','series','isin','close','volume','turnover'])
        z.append(x)
    newer=pd.concat(z,ignore_index=True)
    newer['date']=pd.to_datetime(newer['date'])
    newer=newer[newer['series'].isin(['EQ','BE','BZ'])].copy()
    # Company equities only. INE = Indian corporate security; INF fund/ETF rows
    # are removed. Missing historical ISINs are retained rather than creating
    # survivorship bias.
    company_mask=newer['isin'].isna() | newer['isin'].astype(str).str.startswith('INE')
    newer=newer[company_mask].copy()
    newer=newer.sort_values(['symbol','date']).reset_index(drop=True)
    print('Constructing split/bonus-only normalized price series (dividends excluded).')
    newer['adj_close']=apply_split_bonus_adjustment(newer,hf_start,end_year)
    frames.append(newer[['date','symbol','series','isin','close','volume','turnover','adj_close']])

    out=pd.concat(frames,ignore_index=True).sort_values(['symbol','date']).drop_duplicates(['date','symbol','series']).reset_index(drop=True)
    # Apply the same company-only rule wherever an ISIN is known.
    mask=out['isin'].isna() | out['isin'].astype(str).str.startswith('INE')
    out = out[mask].reset_index(drop=True)
    return add_security_key(out)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    pieces=[]
    group_col = 'security_key' if 'security_key' in df.columns else 'symbol'
    for _,g in df.groupby(group_col,sort=False):
        g=g.sort_values('date').copy()
        p=g['adj_close'].astype(float)
        v=g['volume'].astype(float)
        t=g['turnover'].astype(float)
        r1=p.pct_change()
        g['ret_20']=p/p.shift(20)-1
        g['ret_60']=p/p.shift(60)-1
        g['ret_120']=p/p.shift(120)-1
        g['ret_252']=p/p.shift(252)-1
        g['mom_accel']=g['ret_20']-g['ret_60']/3.0
        g['vol_accel']=v.rolling(20,min_periods=15).mean()/v.rolling(60,min_periods=40).mean()-1
        g['turnover_accel']=t.rolling(20,min_periods=15).mean()/t.rolling(60,min_periods=40).mean()-1
        hi252=p.rolling(252,min_periods=126).max(); lo252=p.rolling(252,min_periods=126).min()
        g['off_high_252']=p/hi252-1
        g['above_low_252']=p/lo252-1
        g['trend_consistency_60']=(r1>0).rolling(60,min_periods=40).mean()
        g['volatility_60']=r1.rolling(60,min_periods=40).std()*math.sqrt(252)
        hi126=p.rolling(126,min_periods=63).max()
        g['drawdown_126']=p/hi126-1
        g['avg_turnover_63']=t.rolling(63,min_periods=40).mean()
        g['log_turnover_63']=np.log1p(g['avg_turnover_63'])
        g['history_days']=np.arange(1,len(g)+1)
        g['med3_adj_close']=p.rolling(3,min_periods=3).median()
        pieces.append(g)
    return pd.concat(pieces,ignore_index=True).sort_values(['date','symbol']).reset_index(drop=True)


def snapshot_dates(df:pd.DataFrame, months:List[int])->List[pd.Timestamp]:
    d=df[['date']].drop_duplicates().sort_values('date').copy()
    d['year']=d['date'].dt.year; d['month']=d['date'].dt.month
    return sorted(pd.Timestamp(x) for x in d[d['month'].isin(months)].groupby(['year','month'])['date'].max().tolist())


def build_snapshots(df:pd.DataFrame,cfg:dict)->pd.DataFrame:
    snaps=snapshot_dates(df,cfg['snapshot_months']); snapset=set(snaps)
    s=df[df['date'].isin(snapset)].copy()
    s=s[(s['close']>=cfg['price_min'])&(s['close']<=cfg['price_max'])]
    s=s[(s['avg_turnover_63']>=cfg['min_avg_turnover_63d'])&(s['history_days']>=cfg['min_history_days'])]
    regime=s.groupby('date').agg(
        breadth_120=('ret_120',lambda x: float(np.nanmean(np.asarray(x)>0))),
        market_median_ret120=('ret_120','median')
    ).reset_index()
    s=s.merge(regime,on='date',how='left')
    for c in RANKABLE:
        s[f'{c}_rank']=s.groupby('date')[c].rank(pct=True,method='average')
    return s.sort_values(['date','symbol']).reset_index(drop=True)


def add_labels(snap:pd.DataFrame, daily:pd.DataFrame,cfg:dict)->pd.DataFrame:
    calendar=np.array(sorted(daily['date'].drop_duplicates().to_numpy(dtype='datetime64[ns]')))
    cal_lookup={d:i for i,d in enumerate(calendar)}
    maps={}
    key_col = 'security_key' if 'security_key' in daily.columns else 'symbol'
    for security_key,g in daily.groupby(key_col,sort=False):
        g=g.sort_values('date')
        dates=g['date'].to_numpy(dtype='datetime64[ns]')
        maps[security_key]=(dates,
                   g['adj_close'].to_numpy(float),
                   g['turnover'].to_numpy(float),
                   np.array([cal_lookup[d] for d in dates],dtype=int))
    out=[]
    max_h=max(int(v) for v in cfg['label_days'].values())
    min_hit_turnover=float(cfg['min_avg_turnover_63d'])

    for row in snap.itertuples(index=False):
        row_key = getattr(row, 'security_key', row.symbol)
        dates,p,turn,calpos=maps[row_key]
        d=np.datetime64(row.date.to_datetime64())
        i=np.searchsorted(dates,d,side='left'); ci=np.searchsorted(calendar,d,side='left')
        if i>=len(dates) or dates[i]!=d or ci>=len(calendar) or calendar[ci]!=d: continue
        p0=float(row.adj_close); rec=row._asdict()

        # Find the first accessible 2x event. A hit requires three consecutive
        # MARKET sessions with an observed close, median split/bonus-normalized
        # close >= 2x, and average traded value >= the eligibility threshold.
        end_ci=min(ci+max_h,len(calendar)-1)
        j=np.searchsorted(calpos,end_ci,side='right')
        start=i+1
        first_hit_offset=None
        if j-start>=3:
            cp=calpos[start:j]; pp=p[start:j]; tt=turn[start:j]
            consecutive=(cp[1:-1]==cp[:-2]+1) & (cp[2:]==cp[:-2]+2)
            med=np.array([np.nanmedian(pp[k:k+3]) for k in range(len(pp)-2)])
            avt=np.array([np.nanmean(tt[k:k+3]) for k in range(len(tt)-2)])
            hit=np.where(consecutive & np.isfinite(med) & (med>=2*p0) & np.isfinite(avt) & (avt>=min_hit_turnover))[0]
            if len(hit):
                first_hit_offset=int(cp[int(hit[0])+2]-ci)

        for key,h0 in cfg['label_days'].items():
            h=int(h0); mature=(ci+h < len(calendar))
            rec[f'{key}_mature_date']=pd.Timestamp(calendar[ci+h]) if mature else pd.NaT
            rec[key]=float(first_hit_offset is not None and first_hit_offset<=h) if mature else np.nan

        rec['days_to_2x']=float(first_hit_offset) if first_hit_offset is not None else np.nan

        h6=int(cfg['label_days']['y6'])
        rec['dd30_6m_mature_date']=pd.Timestamp(calendar[ci+h6]) if ci+h6 < len(calendar) else pd.NaT
        if ci+h6 < len(calendar):
            end6=ci+h6
            j6=np.searchsorted(calpos,end6,side='right')
            fut=p[i+1:j6]
            disappeared=(calpos[-1] < end6)
            rec['dd30_6m']=float((len(fut)>0 and np.nanmin(fut)<=0.7*p0) or disappeared)
        else:
            rec['dd30_6m']=np.nan
        out.append(rec)
    return pd.DataFrame(out)


def make_models(seed:int):
    prep=Pipeline([('impute',SimpleImputer(strategy='median')),('scale',RobustScaler())])
    elastic=Pipeline([
        ('prep',prep),
        ('clf',LogisticRegression(penalty='elasticnet',solver='saga',C=0.25,l1_ratio=0.2,
                                  class_weight='balanced',max_iter=2500,random_state=seed))
    ])
    gbm=Pipeline([
        ('impute',SimpleImputer(strategy='median')),
        ('clf',HistGradientBoostingClassifier(max_depth=3,learning_rate=0.05,max_iter=180,
                                              min_samples_leaf=60,l2_regularization=1.0,class_weight='balanced',
                                              random_state=seed))
    ])
    return elastic,gbm


def structural_raw(train:pd.DataFrame,test:pd.DataFrame,cfg:dict)->np.ndarray:
    score_train=np.zeros(len(train)); score_test=np.zeros(len(test))
    for c,w in cfg['structural_weights'].items():
        rc=f'{c}_rank'
        if rc not in train.columns: continue
        score_train += float(w)*(train[rc].fillna(0.5).to_numpy()-0.5)
        score_test += float(w)*(test[rc].fillna(0.5).to_numpy()-0.5)
    lr=LogisticRegression(C=0.2,class_weight='balanced',max_iter=1000)
    lr.fit(score_train.reshape(-1,1),train['y6'].astype(int))
    return lr.predict_proba(score_test.reshape(-1,1))[:,1]


def safe_predict_model(model,train,test,target='y6'):
    tr=train.dropna(subset=[target]).copy()
    if len(tr)<100 or tr[target].nunique()<2:
        return np.full(len(test),np.nan)
    model.fit(tr[MODEL_FEATURES],tr[target].astype(int))
    return model.predict_proba(test[MODEL_FEATURES])[:,1]


def survival_raw(train,test,seed):
    mature=train.dropna(subset=['y24']).copy()
    if len(mature)<500 or mature['y24'].sum()<30:
        return np.full(len(test),np.nan)
    X=mature[MODEL_FEATURES].replace([np.inf,-np.inf],np.nan)
    imp=SimpleImputer(strategy='median')
    scale=RobustScaler()
    Xi=imp.fit_transform(X)
    Xs=scale.fit_transform(Xi)
    Xt=scale.transform(imp.transform(test[MODEL_FEATURES].replace([np.inf,-np.inf],np.nan)))
    cols=[f'x{i}' for i in range(Xs.shape[1])]
    surv=pd.DataFrame(Xs,columns=cols)
    event=mature['y24'].astype(int).to_numpy()
    dur=np.where(event==1,
                 mature['days_to_2x'].fillna(cfg_h24 if 'cfg_h24' in globals() else 504).to_numpy(float),
                 float(cfg_h24 if 'cfg_h24' in globals() else 504))
    dur=np.clip(dur,1,504)
    surv['duration']=dur; surv['event']=event
    try:
        cph=CoxPHFitter(penalizer=0.10,l1_ratio=0.0)
        cph.fit(surv,duration_col='duration',event_col='event',show_progress=False)
        sf=cph.predict_survival_function(pd.DataFrame(Xt,columns=cols),times=[126.0])
        return np.clip(1.0-sf.iloc[0].to_numpy(float),EPS,1-EPS)
    except Exception as e:
        print('Survival model fallback:',type(e).__name__,str(e)[:120],flush=True)
        return np.full(len(test),np.nan)


def walk_forward(data:pd.DataFrame,cfg:dict)->pd.DataFrame:
    dates=sorted(pd.to_datetime(data['date'].unique()))
    first=pd.Timestamp(f"{cfg['first_oos_year']}-06-30"); oos=[]; seed=int(cfg['random_seed'])
    for td in dates:
        if td<first: continue
        test=data[data['date']==td].copy()
        if test.empty or test['y6'].isna().all(): continue
        start=td-pd.DateOffset(years=int(cfg['rolling_train_years']))
        train=data[(data['date']<td)&(data['date']>=start)].copy()
        for target,mcol in [('y6','y6_mature_date'),('y12','y12_mature_date'),('y24','y24_mature_date'),('dd30_6m','dd30_6m_mature_date')]:
            if mcol in train.columns:
                train.loc[pd.to_datetime(train[mcol])>td,target]=np.nan
        train=train[train['y6'].notna()].copy()
        if len(train)<800 or train['y6'].sum()<20: continue

        elastic,gbm=make_models(seed)
        p_struct=structural_raw(train,test,cfg)
        p_el=safe_predict_model(elastic,train,test,'y6')
        p_gbm=safe_predict_model(gbm,train,test,'y6')
        p_h=survival_raw(train,test,seed)
        arr=np.vstack([p_struct,p_el,p_gbm,p_h]).T
        p_ens=np.nanmean(arr,axis=1)

        dd_elastic,dd_gbm=make_models(seed+101)
        p_dd_el=safe_predict_model(dd_elastic,train,test,'dd30_6m')
        p_dd_g=safe_predict_model(dd_gbm,train,test,'dd30_6m')
        dd_arr=np.vstack([p_dd_el,p_dd_g]).T
        p_dd_raw=np.nanmean(dd_arr,axis=1)

        base_cols=['date','symbol','isin','close','adj_close','y6','y12','y24','dd30_6m']
        if 'security_key' in test.columns:
            base_cols.insert(3,'security_key')
        z=test[base_cols].copy()
        z['p_struct']=p_struct; z['p_elastic']=p_el; z['p_gbm']=p_gbm; z['p_survival']=p_h
        z['p_raw']=p_ens; z['model_dispersion']=np.nanstd(arr,axis=1)
        z['p_dd30_raw']=p_dd_raw; z['dd_model_dispersion']=np.nanstd(dd_arr,axis=1)
        oos.append(z)
        print(f"OOS {td.date()} train={len(train):,} test={len(test):,} positives={int(test['y6'].fillna(0).sum())}",flush=True)
    return pd.concat(oos,ignore_index=True) if oos else pd.DataFrame()


class BetaCalibrator:
    def __init__(self): self.lr=LogisticRegression(C=10,max_iter=1000)
    def _x(self,p):
        p=np.clip(np.asarray(p),EPS,1-EPS)
        return np.c_[np.log(p),-np.log1p(-p)]
    def fit(self,p,y): self.lr.fit(self._x(p),y); return self
    def predict(self,p): return self.lr.predict_proba(self._x(p))[:,1]

class PlattCalibrator:
    def __init__(self): self.lr=LogisticRegression(C=10,max_iter=1000)
    def fit(self,p,y):
        p=np.clip(np.asarray(p),EPS,1-EPS); self.lr.fit(logit(p).reshape(-1,1),y); return self
    def predict(self,p):
        p=np.clip(np.asarray(p),EPS,1-EPS); return self.lr.predict_proba(logit(p).reshape(-1,1))[:,1]

class IsoCalibrator:
    def __init__(self): self.iso=IsotonicRegression(out_of_bounds='clip')
    def fit(self,p,y): self.iso.fit(p,y); return self
    def predict(self,p): return np.clip(self.iso.predict(p),EPS,1-EPS)


def metric_pack(y,p):
    y=np.asarray(y).astype(int); p=np.clip(np.asarray(p,float),EPS,1-EPS)
    return {'brier':float(brier_score_loss(y,p)),'logloss':float(log_loss(y,p))}


def _fit_cal(name,p,y):
    if name=='platt': return PlattCalibrator().fit(p,y)
    if name=='beta': return BetaCalibrator().fit(p,y)
    if name=='isotonic': return IsoCalibrator().fit(p,y)
    return None

def choose_calibrator_temporal(prior:pd.DataFrame,target='y6',pcol='p_raw'):
    d=prior.dropna(subset=[target,pcol]).copy()
    dates=sorted(pd.to_datetime(d['date'].unique()))
    if len(dates)<8: return None, pd.DataFrame()
    cut=max(4,int(math.floor(len(dates)*0.70)))
    if cut>=len(dates)-1: return None, pd.DataFrame()
    dev=d[d['date'].isin(dates[:cut])]; val=d[d['date'].isin(dates[cut:])]
    if len(dev)<1200 or dev[target].sum()<35 or val[target].sum()<15: return None, pd.DataFrame()
    rows=[]
    for name in ['none','platt','beta','isotonic']:
        try:
            if name=='none': pred=val[pcol].to_numpy()
            else: pred=_fit_cal(name,dev[pcol].to_numpy(),dev[target].astype(int).to_numpy()).predict(val[pcol].to_numpy())
            met=metric_pack(val[target].astype(int),pred); met['method']=name; rows.append(met)
        except Exception:
            pass
    tab=pd.DataFrame(rows).sort_values(['brier','logloss']) if rows else pd.DataFrame()
    return (str(tab.iloc[0].method) if len(tab) else None),tab

def evaluate_calibrators(oos:pd.DataFrame,target='y6',pcol='p_raw')->Tuple[str,pd.DataFrame]:
    # Descriptive challenger table only. Production fold probabilities use
    # choose_calibrator_temporal() on PRIOR folds, so this table cannot leak
    # method choice into OOS predictions.
    methods={'none':None,'platt':PlattCalibrator,'beta':BetaCalibrator,'isotonic':IsoCalibrator}
    dates=sorted(pd.to_datetime(oos['date'].unique())); rec=[]
    for m,Cls in methods.items():
        preds=[]; ys=[]
        for td in dates:
            prior=oos[oos['date']<td].dropna(subset=[target,pcol])
            cur=oos[oos['date']==td].dropna(subset=[target,pcol])
            if len(prior)<1500 or prior[target].sum()<40 or cur.empty: continue
            if m=='none': pc=cur[pcol].to_numpy()
            else: pc=Cls().fit(prior[pcol].to_numpy(),prior[target].astype(int).to_numpy()).predict(cur[pcol].to_numpy())
            preds.extend(pc); ys.extend(cur[target].astype(int))
        if len(ys):
            met=metric_pack(ys,preds); met.update(method=m,n=len(ys),positives=int(np.sum(ys))); rec.append(met)
    tab=pd.DataFrame(rec).sort_values(['brier','logloss']) if rec else pd.DataFrame()
    # Current-production method is chosen on a final chronological validation
    # slice of all OOS history, then refit on all OOS predictions.
    best,_=choose_calibrator_temporal(oos,target,pcol)
    return (best or 'none'),tab


def fit_final_calibrator(oos,best,target='y6',pcol='p_raw'):
    d=oos.dropna(subset=[target,pcol])
    if best=='platt': return PlattCalibrator().fit(d[pcol],d[target].astype(int))
    if best=='beta': return BetaCalibrator().fit(d[pcol],d[target].astype(int))
    if best=='isotonic': return IsoCalibrator().fit(d[pcol],d[target].astype(int))
    return None


def apply_forward_calibration(oos,best_unused=None,target='y6',pcol='p_raw',outcol='p_cal'):
    out=oos.copy(); out[outcol]=np.nan; out[f'{outcol}_method']=''
    dates=sorted(pd.to_datetime(out['date'].unique()))
    for td in dates:
        prior=out[out['date']<td].dropna(subset=[target,pcol])
        idx=out.index[(out['date']==td)&out[target].notna()&out[pcol].notna()]
        if len(idx)==0: continue
        method,_=choose_calibrator_temporal(prior,target,pcol)
        if method is None: continue
        raw=out.loc[idx,pcol].to_numpy()
        if method=='none': pred=raw
        else: pred=_fit_cal(method,prior[pcol].to_numpy(),prior[target].astype(int).to_numpy()).predict(raw)
        out.loc[idx,outcol]=pred; out.loc[idx,f'{outcol}_method']=method
    return out


def calibration_slope_intercept(y,p):
    p=np.clip(np.asarray(p),EPS,1-EPS); y=np.asarray(y).astype(int)
    lr=LogisticRegression(C=1e6,max_iter=1000).fit(logit(p).reshape(-1,1),y)
    return float(lr.coef_[0,0]),float(lr.intercept_[0])


def top_metrics(oos,pcol='p_cal'):
    rows=[]
    for td,g in oos.dropna(subset=['y6',pcol]).groupby('date'):
        g=g.sort_values(pcol,ascending=False); base=float(g.y6.mean())
        for k in [5,10,20]:
            q=g.head(k); prec=float(q.y6.mean()) if len(q) else np.nan
            rows.append({'date':td,'k':k,'base_rate':base,'precision':prec,'lift':prec/base if base>0 else np.nan})
    return pd.DataFrame(rows)


def bootstrap_calibration(oos,current_raw,cal_name,B=500,seed=1,target='y6',pcol='p_raw'):
    rng=np.random.default_rng(seed); dates=np.array(sorted(oos['date'].unique())); vals=[]
    for _ in range(B):
        sample_dates=rng.choice(dates,size=len(dates),replace=True)
        samp=pd.concat([oos[oos['date']==d] for d in sample_dates],ignore_index=True).dropna(subset=[target,pcol])
        if samp[target].sum()<20: continue
        try:
            if cal_name=='beta': pp=BetaCalibrator().fit(samp[pcol],samp[target].astype(int)).predict(current_raw)
            elif cal_name=='platt': pp=PlattCalibrator().fit(samp[pcol],samp[target].astype(int)).predict(current_raw)
            elif cal_name=='isotonic': pp=IsoCalibrator().fit(samp[pcol],samp[target].astype(int)).predict(current_raw)
            else: pp=current_raw
            vals.append(pp)
        except Exception: pass
    if not vals: return None
    return np.quantile(np.vstack(vals),[.05,.5,.95],axis=0)


def fit_current(data,daily,oos,best,best_dd,cfg):
    last=pd.Timestamp(daily['date'].max()); cur=daily[daily['date']==last].copy()
    cur=cur[cur['isin'].notna() & cur['isin'].astype(str).str.startswith('INE')].copy()
    cur=cur[(cur['close']>=cfg['price_min'])&(cur['close']<=cfg['price_max'])]
    cur=cur[(cur['avg_turnover_63']>=cfg['min_avg_turnover_63d'])&(cur['history_days']>=cfg['min_history_days'])]
    cur['breadth_120']=float(np.nanmean(cur['ret_120'].to_numpy()>0))
    cur['market_median_ret120']=float(np.nanmedian(cur['ret_120']))
    for c in RANKABLE: cur[f'{c}_rank']=cur[c].rank(pct=True,method='average')
    train_start=last-pd.DateOffset(years=int(cfg['rolling_train_years']))
    train=data[(data['date']>=train_start)&data['y6'].notna()].copy(); seed=int(cfg['random_seed'])

    elastic,gbm=make_models(seed)
    p_struct=structural_raw(train,cur,cfg)
    p_el=safe_predict_model(elastic,train,cur,'y6')
    p_g=safe_predict_model(gbm,train,cur,'y6')
    p_h=survival_raw(train,cur,seed)
    arr=np.vstack([p_struct,p_el,p_g,p_h]).T; raw=np.nanmean(arr,axis=1)
    cal=fit_final_calibrator(oos,best,'y6','p_raw')
    pcal=raw if cal is None else cal.predict(raw)
    boot=bootstrap_calibration(oos,raw,best,int(cfg['bootstrap_blocks']),seed,'y6','p_raw')

    dd_el,dd_gbm=make_models(seed+101)
    pdd1=safe_predict_model(dd_el,train,cur,'dd30_6m'); pdd2=safe_predict_model(dd_gbm,train,cur,'dd30_6m')
    ddarr=np.vstack([pdd1,pdd2]).T; dd_raw=np.nanmean(ddarr,axis=1)
    ddcal=fit_final_calibrator(oos,best_dd,'dd30_6m','p_dd30_raw')
    pdd=dd_raw if ddcal is None else ddcal.predict(dd_raw)

    out=cur[['date','symbol','isin','close','adj_close','avg_turnover_63']].copy()
    out['target_2x']=out['close']*2; out['p_raw']=raw; out['p_cal']=pcal
    out['p_dd30']=pdd; out['asymmetry']=pcal/np.maximum(pdd,0.01)
    out['model_dispersion']=np.nanstd(arr,axis=1); out['dd_model_dispersion']=np.nanstd(ddarr,axis=1)
    if boot is not None:
        out['p05']=boot[0]; out['p50']=boot[1]; out['p95']=boot[2]; out['p_conservative']=boot[0]
    return out.sort_values(['p_cal','asymmetry'],ascending=False).reset_index(drop=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',default='config.json')
    ap.add_argument('--output',default='outputs')
    ap.add_argument('--legacy-dir',default=None)
    args=ap.parse_args(); cfg=json.load(open(args.config)); outdir=Path(args.output); outdir.mkdir(parents=True,exist_ok=True)
    global cfg_h24
    cfg_h24=int(cfg['label_days']['y24'])
    end_year=pd.Timestamp.today().year
    print('Loading market data...',flush=True)
    daily=load_market(int(cfg['start_year']),end_year,args.legacy_dir)
    print('Rows',len(daily),'symbols',daily.symbol.nunique(),'range',daily.date.min(),daily.date.max(),flush=True)
    print('Computing features...',flush=True); daily=add_features(daily)
    print('Building snapshots/labels...',flush=True); snap=build_snapshots(daily,cfg); data=add_labels(snap,daily,cfg)
    data.to_parquet(outdir/'snapshot_dataset.parquet',index=False)
    print('Walk-forward...',flush=True); oos=walk_forward(data,cfg)
    if oos.empty: raise RuntimeError('No OOS predictions generated')

    best,cal_tab=evaluate_calibrators(oos,'y6','p_raw')
    best_dd,cal_dd_tab=evaluate_calibrators(oos,'dd30_6m','p_dd30_raw')
    oos=apply_forward_calibration(oos,best,'y6','p_raw','p_cal')
    oos=apply_forward_calibration(oos,best_dd,'dd30_6m','p_dd30_raw','p_dd30_cal')
    oos['asymmetry']=oos['p_cal']/np.maximum(oos['p_dd30_cal'],0.01)
    oos.to_parquet(outdir/'oos_predictions.parquet',index=False)
    cal_tab.to_csv(outdir/'calibrator_comparison.csv',index=False)
    cal_dd_tab.to_csv(outdir/'downside_calibrator_comparison.csv',index=False)
    top=top_metrics(oos); top.to_csv(outdir/'topk_by_fold.csv',index=False)

    eval_oos=oos.dropna(subset=['y6','p_cal'])
    y=eval_oos.y6.astype(int).to_numpy(); p=eval_oos.p_cal.to_numpy()
    slope,intercept=calibration_slope_intercept(y,p)
    dd_eval=oos.dropna(subset=['dd30_6m','p_dd30_cal'])
    overall={
        'model':cfg['model_name'],'data_start':str(daily.date.min().date()),'data_end':str(daily.date.max().date()),
        'snapshots':int(data.date.nunique()),'oos_folds_total':int(oos.date.nunique()),'oos_folds_calibrated':int(eval_oos.date.nunique()),
        'oos_rows_calibrated':int(len(eval_oos)),'oos_positive_6m':int(y.sum()),'base_rate_6m':float(y.mean()),
        'best_calibrator':best,'brier':float(brier_score_loss(y,p)),
        'logloss':float(log_loss(y,np.clip(p,EPS,1-EPS))),
        'pr_auc':float(average_precision_score(y,p)),
        'calibration_slope':slope,'calibration_intercept':intercept,
        'best_downside_calibrator':best_dd
    }
    if len(dd_eval):
        overall['dd30_brier']=float(brier_score_loss(dd_eval.dd30_6m.astype(int),dd_eval.p_dd30_cal))
    for k in [5,10,20]:
        q=top[top.k==k]
        overall[f'precision_at_{k}_median']=float(q.precision.median())
        overall[f'lift_at_{k}_median']=float(q.lift.median())
    json.dump(overall,open(outdir/'summary.json','w'),indent=2)

    current=fit_current(data,daily,oos,best,best_dd,cfg)
    current.to_csv(outdir/'current_predictions.csv',index=False)
    current.head(50).to_csv(outdir/'current_top50.csv',index=False)
    print(json.dumps(overall,indent=2),flush=True)
    cols=['symbol','close','target_2x','p_cal','p_conservative','p95','p_dd30','asymmetry','model_dispersion'] if 'p_conservative' in current else ['symbol','close','target_2x','p_cal','p_dd30','asymmetry','model_dispersion']
    print(current.head(20)[cols].to_string(index=False),flush=True)

if __name__=='__main__':
    main()
