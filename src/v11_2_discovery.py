from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

import v11_1_hybrid as h

TRANSFORM_FEATURES=[
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


def make_transform_model(cfg:dict)->Pipeline:
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
            random_state=23,
        )),
    ])


def pct(s:pd.Series)->pd.Series:
    x=pd.to_numeric(s,errors="coerce")
    if x.notna().sum()<2:
        return pd.Series(0.5,index=s.index,dtype=float)
    return x.rank(pct=True,method="average").fillna(0.5)


def transform_gate(x:pd.DataFrame,cfg:dict)->pd.Series:
    g=cfg["gate"]
    catalyst=x["catalyst_score"]>=float(g["min_catalyst_score"])
    business=(
        (x["business_inflection_score"]>=float(g["min_business_with_catalyst"]))
        & (x["catalyst_score"]>=float(g["min_secondary_catalyst"]))
    )
    ownership=(
        (x["promoter_conviction_score"]>=float(g["min_promoter_score"]))
        & (x["accumulation_score"]>=float(g["min_accumulation_score"]))
    )
    not_priced=x["priced_in_penalty"]<=float(g["max_priced_in_penalty"])
    v10_ok=x["v10_percentile"]>=float(g["min_v10_percentile"])
    return (catalyst|business|ownership) & not_priced & v10_ok


def fit_transform_prob(train:pd.DataFrame,test:pd.DataFrame,cfg:dict)->np.ndarray:
    tr=train.dropna(subset=["y6"]).copy()
    if len(tr)<800 or tr["y6"].sum()<20 or tr["y6"].nunique()<2:
        # neutral fallback, never substitute V10 probability into transformation model
        return np.full(len(test),0.5)
    m=make_transform_model(cfg)
    m.fit(tr[TRANSFORM_FEATURES],tr["y6"].astype(int))
    return m.predict_proba(test[TRANSFORM_FEATURES])[:,1]


def score(train:pd.DataFrame,test:pd.DataFrame,cfg:dict)->pd.DataFrame:
    z=test.copy()
    z["v10_percentile"]=pct(z["p100_anchor"])
    z["transform_probability"]=fit_transform_prob(train,z,cfg)
    z["transform_percentile"]=pct(z["transform_probability"])
    z["transformation_gate"]=transform_gate(z,cfg)
    z["discovery_hybrid_score"]=(
        float(cfg["v10_confirmation_weight"])*z["v10_percentile"]
        + float(cfg["transformation_weight"])*z["transform_percentile"]
        - float(cfg["risk_penalty"])*z["risk_score"].clip(0,1)
    )
    return z


def control_select(g:pd.DataFrame,k:int)->pd.DataFrame:
    if "selected_v941" in g and g["selected_v941"].fillna(False).sum()>=k:
        return g[g["selected_v941"].fillna(False)].sort_values(
            ["selection_rank_v941","p100_anchor"],ascending=[True,False]
        ).head(k)
    return g.sort_values("p100_anchor",ascending=False).head(k)


def select_discovery(g:pd.DataFrame,k:int)->pd.DataFrame:
    q=g[g["transformation_gate"]].sort_values(
        ["discovery_hybrid_score","transform_probability","p100_anchor"],
        ascending=False
    )
    return q.head(k)


def walk_forward(hist:pd.DataFrame,cfg:dict):
    dates=sorted(pd.to_datetime(hist["date"].unique()))
    k=int(cfg["main_k"])
    min_prior=int(cfg["min_prior_folds"])
    years=int(cfg["rolling_train_years"])
    rows=[]
    scored=[]
    for i,td in enumerate(dates):
        if i<min_prior:
            continue
        test=hist[hist["date"].eq(td)].copy()
        if test.empty or test["y6"].notna().sum()<k:
            continue
        start=pd.Timestamp(td)-pd.DateOffset(years=years)
        train=hist[(hist["date"]<td)&(hist["date"]>=start)&hist["y6"].notna()].copy()
        if train.empty:
            continue
        z=score(train,test,cfg)
        scored.append(z)
        sel=select_discovery(z,k)
        if len(sel)<k:
            continue
        ctl=control_select(z,k)
        base=float(z["y6"].mean())
        hp=float(sel["y6"].mean())
        cp=float(ctl["y6"].mean())
        hdd=float(sel["dd30_6m"].mean()) if sel["dd30_6m"].notna().any() else np.nan
        cdd=float(ctl["dd30_6m"].mean()) if ctl["dd30_6m"].notna().any() else np.nan
        rows.append({
            "date":pd.Timestamp(td),
            "eligible":int(z["transformation_gate"].sum()),
            "hybrid_precision_100":hp,
            "control_precision_100":cp,
            "hybrid_lift_100":hp/base if base>0 else np.nan,
            "control_lift_100":cp/base if base>0 else np.nan,
            "hybrid_dd30_rate":hdd,
            "control_dd30_rate":cdd,
            "hybrid_early_stage":float(sel["early_stage_score"].mean()),
            "control_early_stage":float(ctl["early_stage_score"].mean()),
            "hybrid_priced_in":float(sel["priced_in_penalty"].mean()),
            "control_priced_in":float(ctl["priced_in_penalty"].mean()),
            "hybrid_ret120":float(pd.to_numeric(sel["ret_120"],errors="coerce").mean()),
            "control_ret120":float(pd.to_numeric(ctl["ret_120"],errors="coerce").mean()),
            "hybrid_catalyst_rate":float((sel["catalyst_score"]>0).mean()),
            "control_catalyst_rate":float((ctl["catalyst_score"]>0).mean()),
        })
    return (
        pd.concat(scored,ignore_index=True) if scored else pd.DataFrame(),
        pd.DataFrame(rows)
    )


def summarize(f:pd.DataFrame,cfg:dict)->dict:
    if f.empty:
        return {"folds":0,"all_gates_pass":False}
    out={
        "folds":int(len(f)),
        "mean_hybrid_precision_100":float(f["hybrid_precision_100"].mean()),
        "mean_control_precision_100":float(f["control_precision_100"].mean()),
        "mean_hybrid_lift_100":float(f["hybrid_lift_100"].mean()),
        "mean_control_lift_100":float(f["control_lift_100"].mean()),
        "hybrid_hit_fold_rate":float((f["hybrid_precision_100"]>0).mean()),
        "control_hit_fold_rate":float((f["control_precision_100"]>0).mean()),
        "mean_hybrid_dd30_rate":float(f["hybrid_dd30_rate"].mean()),
        "mean_control_dd30_rate":float(f["control_dd30_rate"].mean()),
        "mean_hybrid_early_stage":float(f["hybrid_early_stage"].mean()),
        "mean_control_early_stage":float(f["control_early_stage"].mean()),
        "mean_hybrid_priced_in":float(f["hybrid_priced_in"].mean()),
        "mean_control_priced_in":float(f["control_priced_in"].mean()),
        "mean_hybrid_ret120":float(f["hybrid_ret120"].mean()),
        "mean_control_ret120":float(f["control_ret120"].mean()),
        "mean_hybrid_catalyst_rate":float(f["hybrid_catalyst_rate"].mean()),
        "mean_control_catalyst_rate":float(f["control_catalyst_rate"].mean()),
    }
    out["precision_ratio"]=out["mean_hybrid_precision_100"]/out["mean_control_precision_100"] if out["mean_control_precision_100"]>0 else None
    out["lift_ratio"]=out["mean_hybrid_lift_100"]/out["mean_control_lift_100"] if out["mean_control_lift_100"]>0 else None
    out["dd_delta"]=out["mean_hybrid_dd30_rate"]-out["mean_control_dd30_rate"]
    recent=f[pd.to_datetime(f["date"])>=pd.Timestamp("2023-01-01")]
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
        "early_stage":out["mean_hybrid_early_stage"]>=float(a["early_stage_score_min"]),
        "transformation_presence":out["mean_hybrid_catalyst_rate"]>=float(a["selected_transformation_rate_min"]),
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
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)

    hist=h.engineer(pd.read_parquet(args.historical_scored))
    hist["date"]=pd.to_datetime(hist["date"])
    cur=h.engineer(pd.read_csv(args.current_scored))
    cur["date"]=pd.to_datetime(cur["date"])

    scored,folds=walk_forward(hist,cfg)
    folds.to_csv(out/"fold_metrics.csv",index=False)
    if len(scored):
        scored.to_parquet(out/"historical_scored_v11_2.parquet",index=False)

    latest=pd.Timestamp(cur["date"].max())
    start=latest-pd.DateOffset(years=int(cfg["rolling_train_years"]))
    train=hist[(hist["date"]<latest)&(hist["date"]>=start)&hist["y6"].notna()].copy()
    current=score(train,cur,cfg)
    selected=select_discovery(current,int(cfg["main_k"])).copy()
    selected["discovery_rank"]=np.arange(1,len(selected)+1)
    selected.to_csv(out/"current_discovery_top10.csv",index=False)
    current.sort_values("discovery_hybrid_score",ascending=False).to_csv(out/"current_universe_scored.csv",index=False)

    metrics=summarize(folds,cfg)
    summary={
        "model":cfg["model_name"],
        "objective":"transformation discovery before obvious rerating",
        "architecture":"60% independent transformation meta / 40% V10.2 confirmation",
        "current_date":str(latest.date()),
        "current_eligible":int(current["transformation_gate"].sum()),
        "current_selected":int(len(selected)),
        "historical_metrics":metrics,
        "config":cfg,
        "important_note":"Historical evidence is development evidence because V11.2 follows V11.0/V11.1. Prospective validation is mandatory before production replacement."
    }
    json.dump(summary,open(out/"summary.json","w"),indent=2,default=str)
    print(json.dumps(summary,indent=2,default=str))

    cols=[c for c in [
        "symbol","close","discovery_rank","discovery_hybrid_score","transform_probability",
        "p100_anchor","catalyst_score","business_inflection_score","promoter_conviction_score",
        "accumulation_score","ownership_accumulation_score","early_stage_score",
        "priced_in_penalty","risk_score"
    ] if c in selected]
    if len(selected):
        print(selected[cols].to_string(index=False))


if __name__=="__main__":
    main()
