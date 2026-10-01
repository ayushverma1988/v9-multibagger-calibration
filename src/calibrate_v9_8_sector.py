from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

FEATURES=[
    "baseline_rank",
    "sector_rel_ret20","sector_rel_ret60","sector_rel_ret120",
    "sector_rel_vol60","sector_rel_trend60",
    "sector_breadth120","sector_dispersion120",
    "sector_size",
]


def spec_from_row(r):
    return {
        "pool_n":int(r.pool_n),"risk_drop":float(r.risk_drop),
        "w_safety":float(r.w_safety),"w_consensus":float(r.w_consensus),
        "baseline":bool(r.fell_back_to_v92),
    }


def survivors(g,spec,k):
    q=g.dropna(subset=["p_cal","p_dd30_cal","model_dispersion"]).copy()
    if len(q)<k: return pd.DataFrame()
    if spec["baseline"]:
        q["selection_score"]=q["p_cal"]
        return q.sort_values(["selection_score","p_cal","model_dispersion"],ascending=[False,False,True])
    p=q.sort_values(["p_cal","model_dispersion"],ascending=[False,True]).head(min(spec["pool_n"],len(q))).copy()
    if len(p)<k:return pd.DataFrame()
    if spec["risk_drop"]>0:
        c=float(p["p_dd30_cal"].quantile(1-spec["risk_drop"]))
        p=p[p["p_dd30_cal"]<=c].copy()
    if len(p)<k:return pd.DataFrame()
    p["comp_alpha"]=p["p_cal"].rank(pct=True,method="average")
    p["comp_safety"]=p["p_dd30_cal"].rank(pct=True,method="average",ascending=False)
    p["comp_consensus"]=p["model_dispersion"].rank(pct=True,method="average",ascending=False)
    wa=max(0,1-spec["w_safety"]-spec["w_consensus"])
    p["selection_score"]=wa*p["comp_alpha"]+spec["w_safety"]*p["comp_safety"]+spec["w_consensus"]*p["comp_consensus"]
    return p.sort_values(["selection_score","p_cal","model_dispersion"],ascending=[False,False,True])


def latest_sector_map(snap,events):
    s=snap.copy()
    s["date"]=pd.to_datetime(s["date"])
    if "isin" not in s.columns:s["isin"]=np.nan
    s["security_key"]=np.where(
        s["isin"].notna() & ~s["isin"].astype(str).str.upper().isin(["NAN","NONE","<NA>"]),
        "ISIN:"+s["isin"].astype(str).str.upper().str.strip(),
        "SYM:"+s["symbol"].astype(str).str.upper().str.strip(),
    )
    e=events[["published_ts","security_key","industry_raw"]].copy()
    e["published_ts"]=pd.to_datetime(e["published_ts"],utc=True,errors="coerce")
    e=e.dropna(subset=["published_ts","security_key","industry_raw"])
    e["event_date"]=e["published_ts"].dt.tz_convert("Asia/Kolkata").dt.tz_localize(None).dt.normalize()
    e["industry_raw"]=e["industry_raw"].astype(str).str.strip()
    e=e[~e["industry_raw"].isin(["","nan","None"])]
    out=[]
    for td,g in s.groupby("date",sort=True):
        q=e[e["event_date"]<pd.Timestamp(td).normalize()].sort_values("event_date")
        latest=q.drop_duplicates("security_key",keep="last")[["security_key","industry_raw","event_date"]]
        z=g.merge(latest,on="security_key",how="left")
        z["industry_age_days"]=(pd.Timestamp(td).normalize()-pd.to_datetime(z["event_date"])).dt.days
        out.append(z.drop(columns=["event_date"],errors="ignore"))
    return pd.concat(out,ignore_index=True)


def add_sector_features(snap,events):
    x=latest_sector_map(snap,events)
    rows=[]
    for td,g in x.groupby("date",sort=True):
        h=g.copy()
        valid=h["industry_raw"].notna()
        h["sector_size"]=0.0
        for c in ["sector_rel_ret20","sector_rel_ret60","sector_rel_ret120","sector_rel_vol60","sector_rel_trend60","sector_breadth120","sector_dispersion120"]:
            h[c]=np.nan
        for sec,idx in h[valid].groupby("industry_raw").groups.items():
            z=h.loc[idx]
            if len(z)<5: continue
            med20=float(np.nanmedian(z["ret_20"]))
            med60=float(np.nanmedian(z["ret_60"]))
            med120=float(np.nanmedian(z["ret_120"]))
            medv=float(np.nanmedian(z["volatility_60"]))
            medt=float(np.nanmedian(z["trend_consistency_60"]))
            breadth=float(np.nanmean(z["ret_120"].to_numpy(float)>0))
            disp=float(np.nanstd(z["ret_120"]))
            h.loc[idx,"sector_size"]=len(z)
            h.loc[idx,"sector_rel_ret20"]=h.loc[idx,"ret_20"]-med20
            h.loc[idx,"sector_rel_ret60"]=h.loc[idx,"ret_60"]-med60
            h.loc[idx,"sector_rel_ret120"]=h.loc[idx,"ret_120"]-med120
            h.loc[idx,"sector_rel_vol60"]=medv-h.loc[idx,"volatility_60"]
            h.loc[idx,"sector_rel_trend60"]=h.loc[idx,"trend_consistency_60"]-medt
            h.loc[idx,"sector_breadth120"]=breadth
            h.loc[idx,"sector_dispersion120"]=disp
        rows.append(h)
    return pd.concat(rows,ignore_index=True)


def fit_meta(train):
    m=Pipeline([
        ("impute",SimpleImputer(strategy="median",add_indicator=True)),
        ("scale",RobustScaler()),
        ("clf",LogisticRegression(C=.10,class_weight="balanced",max_iter=3000,random_state=20261001)),
    ])
    m.fit(train[FEATURES],train["y6"].astype(int))
    return m


def agg(t):
    if t.empty:return None
    lift=np.clip(t["lift_2x"].replace([np.inf,-np.inf],np.nan).dropna(),0,10)
    return {
        "folds":int(t["date"].nunique()),
        "mean_precision_2x":float(t["precision_2x"].mean()),
        "median_precision_2x":float(t["precision_2x"].median()),
        "mean_capped_lift_2x":float(lift.mean()) if len(lift) else np.nan,
        "hit_fold_rate":float(t["hit"].mean()),
        "mean_dd30_rate":float(t["dd30_rate"].mean()),
    }


def retain(a,b,r=.95):
    for f in ["mean_precision_2x","median_precision_2x","mean_capped_lift_2x","hit_fold_rate"]:
        if not(np.isfinite(a.get(f,np.nan)) and np.isfinite(b.get(f,np.nan))):return False
        if b[f]>0 and a[f]+1e-12<r*b[f]:return False
    return True


def util(a,b):
    rr=[a[f]/b[f] if b[f]>0 else 0 for f in ["mean_precision_2x","mean_capped_lift_2x","hit_fold_rate"]]
    dd=(b["mean_dd30_rate"]-a["mean_dd30_rate"])/b["mean_dd30_rate"] if b["mean_dd30_rate"]>0 else 0
    return .45*rr[0]+.35*rr[1]+.20*rr[2]+.10*dd


def eval_weight(pools,w,k):
    rows=[]
    for td,g in pools.groupby("date"):
        q=g.dropna(subset=["y6","selection_score"])
        if len(q)<k or q["p_sector_meta"].notna().sum()<k:continue
        q=q.copy()
        q["baseline_rank"]=q["selection_score"].rank(pct=True,method="average")
        q["sector_rank"]=q["p_sector_meta"].rank(pct=True,method="average").fillna(.5)
        q["score"]=(1-w)*q["baseline_rank"]+w*q["sector_rank"]
        s=q.sort_values(["score","baseline_rank","p_cal"],ascending=[False,False,False]).head(k)
        br=float(q["y6"].mean());pr=float(s["y6"].mean())
        rows.append({"date":pd.Timestamp(td),"precision_2x":pr,"lift_2x":pr/br if br>0 else np.nan,"hit":float(pr>0),"dd30_rate":float(s["dd30_6m"].mean())})
    return pd.DataFrame(rows)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--events",required=True)
    ap.add_argument("--snapshot",required=True)
    ap.add_argument("--oos",required=True)
    ap.add_argument("--chosen931",required=True)
    ap.add_argument("--config",required=True)
    ap.add_argument("--output",required=True)
    args=ap.parse_args()

    cfg=json.load(open(args.config));k=int(cfg.get("selection_k",10))
    snap=pd.read_parquet(args.snapshot);snap["date"]=pd.to_datetime(snap["date"])
    events=pd.read_parquet(args.events)
    oos=pd.read_parquet(args.oos);oos["date"]=pd.to_datetime(oos["date"])
    chosen=pd.read_csv(args.chosen931);chosen["date"]=pd.to_datetime(chosen["date"])
    cmap={pd.Timestamp(r.date):r for r in chosen.itertuples(index=False)}

    sf=add_sector_features(snap,events)
    cols=["date","symbol","industry_raw","industry_age_days"]+FEATURES[1:]
    m=oos.merge(sf[cols],on=["date","symbol"],how="left")
    mature=snap[["date","symbol","y6_mature_date"]].drop_duplicates()
    mature["y6_mature_date"]=pd.to_datetime(mature["y6_mature_date"],errors="coerce")
    m=m.merge(mature,on=["date","symbol"],how="left")

    pools=[]
    for td,r in cmap.items():
        p=survivors(m[m["date"]==td],spec_from_row(r),k)
        if p.empty:continue
        p["baseline_rank"]=p["selection_score"].rank(pct=True,method="average")
        pools.append(p)
    pools=pd.concat(pools,ignore_index=True);pools["p_sector_meta"]=np.nan

    for td in sorted(pools["date"].unique()):
        prior=pools[(pools["date"]<td)&pools["y6"].notna()].copy()
        prior=prior[pd.to_datetime(prior["y6_mature_date"],errors="coerce")<pd.Timestamp(td)]
        idx=pools.index[pools["date"]==td]
        if prior["date"].nunique()<8 or len(prior)<500 or int(prior["y6"].sum())<20 or prior["y6"].nunique()<2:continue
        model=fit_meta(prior)
        pools.loc[idx,"p_sector_meta"]=model.predict_proba(pools.loc[idx,FEATURES])[:,1]

    weights=[0,.02,.05,.10]; decisions=[]; sels=[]
    for td in sorted(pools["date"].unique()):
        hist=pools[pools["date"]<td]
        base=agg(eval_weight(hist,0,k));bw=0;bu=1.0
        if base and base["folds"]>=4:
            for w in weights[1:]:
                a=agg(eval_weight(hist,w,k))
                if not a or a["folds"]<4 or not retain(a,base):continue
                u=util(a,base)
                if u>=1.02 and u>bu:bw=w;bu=u
        cur=pools[pools["date"]==td].copy()
        if cur["p_sector_meta"].notna().sum()<k:bw=0
        cur["sector_rank"]=cur["p_sector_meta"].rank(pct=True,method="average").fillna(.5)
        cur["score"]=(1-bw)*cur["baseline_rank"]+bw*cur["sector_rank"]
        s=cur.sort_values(["score","baseline_rank","p_cal"],ascending=[False,False,False]).head(k)
        for rank,(_,r) in enumerate(s.iterrows(),1):
            sels.append({"date":td,"symbol":r["symbol"],"rank":rank,"weight":bw,"y6":r["y6"],"dd30_6m":r["dd30_6m"]})
        decisions.append({"date":td,"weight":bw,"prior_policy_folds":base["folds"] if base else 0,"train_utility":bu})

    sel=pd.DataFrame(sels);dec=pd.DataFrame(decisions)
    sr=[];br=[]
    for td,g in sel.groupby("date"):
        q=pools[pools["date"]==td].dropna(subset=["y6"])
        if len(g)!=k or q.empty:continue
        base_rate=float(q["y6"].mean());pr=float(g["y6"].mean())
        sr.append({"date":td,"precision_2x":pr,"lift_2x":pr/base_rate if base_rate>0 else np.nan,"hit":float(pr>0),"dd30_rate":float(g["dd30_6m"].mean())})
        b=q.sort_values(["selection_score","p_cal","model_dispersion"],ascending=[False,False,True]).head(k);bp=float(b["y6"].mean())
        br.append({"date":td,"precision_2x":bp,"lift_2x":bp/base_rate if base_rate>0 else np.nan,"hit":float(bp>0),"dd30_rate":float(b["dd30_6m"].mean())})
    sr=pd.DataFrame(sr);br=pd.DataFrame(br)
    a=agg(sr);b=agg(br)

    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    sf.to_parquet(out/"sector_features_by_snapshot.parquet",index=False)
    dec.to_csv(out/"sector_weight_decisions.csv",index=False)
    sr.to_csv(out/"sector_metrics.csv",index=False)
    br.to_csv(out/"baseline_metrics.csv",index=False)
    summary={
        "model":"V9.8 PIT sector-relative meta-overlay",
        "sector_source":"latest strictly-prior NSE announcement industry_raw",
        "coverage_fraction":float(sf["industry_raw"].notna().mean()),
        "V9.8":a,"V9.4.1_same_folds":b,
        "retention_passed":bool(a and b and retain(a,b)),
        "active_sector_folds":int((dec["weight"]>0).sum()) if len(dec) else 0,
        "production_gate":bool(a and b and retain(a,b) and util(a,b)>=1.02 and int((dec["weight"]>0).sum())>=4),
        "utility_ratio":float(util(a,b)) if a and b else None,
    }
    json.dump(summary,open(out/"summary.json","w"),indent=2,default=str)
    print(json.dumps(summary,indent=2,default=str))

if __name__=="__main__":
    main()
