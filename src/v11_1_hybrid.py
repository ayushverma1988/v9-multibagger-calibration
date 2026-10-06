from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


EPS=1e-6

BASE_FEATURES=[
    "p100_logit",
    "catalyst_score",
    "business_inflection_score",
    "promoter_conviction_score",
    "accumulation_score",
    "ownership_accumulation_score",
    "early_stage_score",
    "priced_in_penalty",
    "risk_score",
    "evt_commissioning_180",
    "evt_capacity_180",
    "evt_order_180",
    "evt_new_product_180",
    "evt_future_product_180",
    "evt_approval_180",
    "evt_customer_180",
    "evt_pledge_improvement_180",
    "evt_pledge_risk_180",
    "catalyst_x_early",
    "promoter_x_accumulation",
    "catalyst_x_accumulation",
    "business_x_catalyst",
]


def _num(s,index=None,default=np.nan):
    if s is None:
        return pd.Series(default,index=index,dtype=float)
    return pd.to_numeric(s,errors="coerce")


def engineer(df:pd.DataFrame)->pd.DataFrame:
    x=df.copy()
    p=_num(x.get("p100_cal"),x.index)
    if p.isna().all():
        p=_num(x.get("p_cal"),x.index)
    p=p.clip(EPS,1-EPS)
    x["p100_anchor"]=p
    x["p100_logit"]=logit(p)

    for c in [
        "catalyst_score","business_inflection_score","promoter_conviction_score",
        "accumulation_score","ownership_accumulation_score","early_stage_score",
        "priced_in_penalty","risk_score","evt_commissioning_180","evt_capacity_180",
        "evt_order_180","evt_new_product_180","evt_future_product_180",
        "evt_approval_180","evt_customer_180","evt_pledge_improvement_180",
        "evt_pledge_risk_180",
    ]:
        if c not in x:
            x[c]=0.0
        x[c]=_num(x[c],x.index,0.0)

    x["catalyst_x_early"]=x["catalyst_score"]*x["early_stage_score"]
    x["promoter_x_accumulation"]=x["promoter_conviction_score"]*x["accumulation_score"]
    x["catalyst_x_accumulation"]=x["catalyst_score"]*x["accumulation_score"]
    x["business_x_catalyst"]=x["business_inflection_score"]*x["catalyst_score"]
    return x


def make_model(cfg:dict)->Pipeline:
    return Pipeline([
        ("impute",SimpleImputer(strategy="median")),
        ("scale",RobustScaler()),
        ("clf",LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            C=float(cfg["meta_C"]),
            l1_ratio=float(cfg["meta_l1_ratio"]),
            class_weight="balanced",
            max_iter=3000,
            random_state=17,
        )),
    ])


def percentile(s:pd.Series)->pd.Series:
    x=_num(s,s.index)
    if x.notna().sum()<2:
        return pd.Series(0.5,index=s.index,dtype=float)
    return x.rank(pct=True,method="average").fillna(0.5)


def candidate_mask(g:pd.DataFrame,cfg:dict)->pd.Series:
    pool=cfg["main_candidate_pool"]
    pctl=percentile(g["p100_anchor"])
    dctl=percentile(g.get("discovery_score",pd.Series(0.0,index=g.index)))
    mask=pctl>=float(pool["v10_top_percentile"])
    if bool(pool.get("allow_strong_transformation",True)):
        mask=mask | (dctl>=float(pool["strong_transformation_percentile"]))
    return mask


def select_control(g:pd.DataFrame,k:int)->pd.DataFrame:
    q=g.copy()
    if "selected_v941" in q and q["selected_v941"].fillna(False).sum()>=k:
        return q[q["selected_v941"].fillna(False)].sort_values(
            ["selection_rank_v941","p100_anchor"],ascending=[True,False]
        ).head(k)
    return q.sort_values("p100_anchor",ascending=False).head(k)


def fit_predict_meta(train:pd.DataFrame,test:pd.DataFrame,cfg:dict)->np.ndarray:
    tr=train.dropna(subset=["y6"]).copy()
    if len(tr)<800 or tr["y6"].sum()<20 or tr["y6"].nunique()<2:
        return test["p100_anchor"].to_numpy(float)
    model=make_model(cfg)
    model.fit(tr[BASE_FEATURES],tr["y6"].astype(int))
    return model.predict_proba(test[BASE_FEATURES])[:,1]


def score_fold(train:pd.DataFrame,test:pd.DataFrame,cfg:dict)->pd.DataFrame:
    z=test.copy()
    meta=fit_predict_meta(train,z,cfg)
    z["meta_probability"]=meta
    anchor_rank=percentile(z["p100_anchor"])
    meta_rank=percentile(z["meta_probability"])
    z["hybrid_score_raw"]=(
        float(cfg["anchor_weight"])*anchor_rank
        + float(cfg["meta_weight"])*meta_rank
        - float(cfg["risk_penalty"])*z["risk_score"].clip(0,1)
    )
    z["v10_percentile"]=anchor_rank
    z["meta_percentile"]=meta_rank
    z["hybrid_candidate"]=candidate_mask(z,cfg)
    return z


def walk_forward(hist:pd.DataFrame,cfg:dict)->tuple[pd.DataFrame,pd.DataFrame]:
    dates=sorted(pd.to_datetime(hist["date"].dropna().unique()))
    rows=[]
    scored=[]
    k=int(cfg["main_k"])
    min_prior=int(cfg["min_prior_folds"])
    roll_years=int(cfg["rolling_train_years"])

    for i,td in enumerate(dates):
        if i<min_prior:
            continue
        test=hist[hist["date"].eq(td)].copy()
        if test.empty or test["y6"].notna().sum()<k:
            continue
        start=pd.Timestamp(td)-pd.DateOffset(years=roll_years)
        train=hist[(hist["date"]<td)&(hist["date"]>=start)&hist["y6"].notna()].copy()
        if train.empty:
            continue

        z=score_fold(train,test,cfg)
        scored.append(z)
        pool=z[z["hybrid_candidate"]].copy()
        if len(pool)<k:
            pool=z.copy()
        sel=pool.sort_values(["hybrid_score_raw","p100_anchor"],ascending=False).head(k)
        ctl=select_control(z,k)

        base=float(z["y6"].mean())
        prec=float(sel["y6"].mean())
        cprec=float(ctl["y6"].mean())
        dd=float(sel["dd30_6m"].mean()) if "dd30_6m" in sel and sel["dd30_6m"].notna().any() else np.nan
        cdd=float(ctl["dd30_6m"].mean()) if "dd30_6m" in ctl and ctl["dd30_6m"].notna().any() else np.nan
        rows.append({
            "date":pd.Timestamp(td),
            "train_rows":int(len(train)),
            "candidate_pool":int(len(pool)),
            "hybrid_precision_100":prec,
            "hybrid_lift_100":prec/base if base>0 else np.nan,
            "hybrid_dd30_rate":dd,
            "control_precision_100":cprec,
            "control_lift_100":cprec/base if base>0 else np.nan,
            "control_dd30_rate":cdd,
        })

    return (
        pd.concat(scored,ignore_index=True) if scored else pd.DataFrame(),
        pd.DataFrame(rows),
    )


def radar(current:pd.DataFrame,main_symbols:set[str],cfg:dict)->pd.DataFrame:
    r=current.copy()
    rcfg=cfg["radar"]
    m=(
        (
            (r["catalyst_score"]>=float(rcfg["min_catalyst_score"]))
            | (r["ownership_accumulation_score"]>=float(rcfg["min_ownership_accumulation_score"]))
        )
        & (r["priced_in_penalty"]<=float(rcfg["max_priced_in_penalty"]))
        & (r["v10_percentile"]<float(rcfg["max_v10_percentile"]))
        & (~r["symbol"].astype(str).isin(main_symbols))
    )
    q=r[m].copy()
    q["radar_score"]=(
        0.35*q["catalyst_score"]
        +0.20*q["business_inflection_score"]
        +0.20*q["ownership_accumulation_score"]
        +0.20*q["early_stage_score"]
        +0.05*q["technical_confirmation_score"]
        -0.10*q["risk_score"]
    )
    return q.sort_values(["radar_score","catalyst_score"],ascending=False).head(int(cfg["radar_k"]))


def summary_metrics(folds:pd.DataFrame,cfg:dict)->dict:
    if folds.empty:
        return {"folds":0}
    out={
        "folds":int(len(folds)),
        "mean_hybrid_precision_100":float(folds["hybrid_precision_100"].mean()),
        "mean_control_precision_100":float(folds["control_precision_100"].mean()),
        "mean_hybrid_lift_100":float(folds["hybrid_lift_100"].mean()),
        "mean_control_lift_100":float(folds["control_lift_100"].mean()),
        "hybrid_hit_fold_rate":float((folds["hybrid_precision_100"]>0).mean()),
        "control_hit_fold_rate":float((folds["control_precision_100"]>0).mean()),
        "mean_hybrid_dd30_rate":float(folds["hybrid_dd30_rate"].mean()),
        "mean_control_dd30_rate":float(folds["control_dd30_rate"].mean()),
    }
    out["precision_ratio"]=out["mean_hybrid_precision_100"]/out["mean_control_precision_100"] if out["mean_control_precision_100"]>0 else None
    out["lift_ratio"]=out["mean_hybrid_lift_100"]/out["mean_control_lift_100"] if out["mean_control_lift_100"]>0 else None
    out["dd_delta"]=out["mean_hybrid_dd30_rate"]-out["mean_control_dd30_rate"]

    recent=folds[pd.to_datetime(folds["date"])>=pd.Timestamp("2023-01-01")]
    if len(recent):
        hp=float(recent["hybrid_precision_100"].mean())
        cp=float(recent["control_precision_100"].mean())
        out["recent_folds"]=int(len(recent))
        out["recent_hybrid_precision_100"]=hp
        out["recent_control_precision_100"]=cp
        out["recent_precision_ratio"]=hp/cp if cp>0 else None
    else:
        out["recent_folds"]=0
        out["recent_precision_ratio"]=None

    a=cfg["acceptance"]
    gates={
        "precision_ratio":out["precision_ratio"] is not None and out["precision_ratio"]>=float(a["precision_ratio_min"]),
        "lift_ratio":out["lift_ratio"] is not None and out["lift_ratio"]>=float(a["lift_ratio_min"]),
        "dd_delta":out["dd_delta"]<=float(a["dd_delta_max"]),
        "recent_precision_ratio":out["recent_precision_ratio"] is not None and out["recent_precision_ratio"]>=float(a["recent_precision_ratio_min"]),
    }
    out["acceptance_gates"]=gates
    out["all_gates_pass"]=all(gates.values())
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",required=True)
    ap.add_argument("--historical-scored",required=True)
    ap.add_argument("--current-scored",required=True)
    ap.add_argument("--output",required=True)
    args=ap.parse_args()

    cfg=json.load(open(args.config))
    outdir=Path(args.output); outdir.mkdir(parents=True,exist_ok=True)

    hist=engineer(pd.read_parquet(args.historical_scored))
    hist["date"]=pd.to_datetime(hist["date"])
    current=engineer(pd.read_csv(args.current_scored))
    current["date"]=pd.to_datetime(current["date"])

    scored_oos,folds=walk_forward(hist,cfg)
    folds.to_csv(outdir/"fold_metrics.csv",index=False)
    if len(scored_oos):
        scored_oos.to_parquet(outdir/"hybrid_oos_scored.parquet",index=False)

    mature=hist[hist["y6"].notna()].copy()
    latest=pd.Timestamp(current["date"].max())
    train_start=latest-pd.DateOffset(years=int(cfg["rolling_train_years"]))
    train=mature[(mature["date"]<latest)&(mature["date"]>=train_start)].copy()
    cur=score_fold(train,current,cfg)

    pool=cur[cur["hybrid_candidate"]].copy()
    if len(pool)<int(cfg["main_k"]):
        pool=cur.copy()
    main=pool.sort_values(["hybrid_score_raw","p100_anchor"],ascending=False).head(int(cfg["main_k"])).copy()
    main["hybrid_rank"]=np.arange(1,len(main)+1)
    main.to_csv(outdir/"current_hybrid_top10.csv",index=False)

    rd=radar(cur,set(main["symbol"].astype(str)),cfg)
    rd["radar_rank"]=np.arange(1,len(rd)+1)
    rd.to_csv(outdir/"current_early_discovery_radar.csv",index=False)
    cur.sort_values("hybrid_score_raw",ascending=False).to_csv(outdir/"current_universe_hybrid_scored.csv",index=False)

    metrics=summary_metrics(folds,cfg)
    summary={
        "model":cfg["model_name"],
        "objective":"V10.2 anchored transformation discovery",
        "design":"temporal elastic-net stack + protected V10.2 anchor + separate early-discovery radar",
        "current_date":str(latest.date()),
        "current_main_selected":int(len(main)),
        "current_radar_selected":int(len(rd)),
        "historical_metrics":metrics,
        "config":cfg,
        "important_note":"V11.1 was designed after observing V11.0 failure. Historical evidence is therefore research evidence, not an untouched final holdout; prospective shadow tracking is required before replacing V10.2."
    }
    json.dump(summary,open(outdir/"summary.json","w"),indent=2,default=str)
    print(json.dumps(summary,indent=2,default=str))

    cols=[c for c in [
        "symbol","close","p100_anchor","meta_probability","hybrid_score_raw",
        "catalyst_score","business_inflection_score","ownership_accumulation_score",
        "early_stage_score","priced_in_penalty","risk_score","hybrid_rank"
    ] if c in main]
    print("\nV11.1 HYBRID TOP-10")
    print(main[cols].to_string(index=False))
    if len(rd):
        rcols=[c for c in [
            "symbol","close","radar_score","p100_anchor","catalyst_score",
            "ownership_accumulation_score","early_stage_score","priced_in_penalty","radar_rank"
        ] if c in rd]
        print("\nEARLY DISCOVERY RADAR")
        print(rd[rcols].to_string(index=False))


if __name__=="__main__":
    main()
