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

MATURITY_MONTHS={"y6":7,"y12":13,"y24":25}


def pct(s:pd.Series)->pd.Series:
    x=pd.to_numeric(s,errors="coerce")
    if x.notna().sum()<2:
        return pd.Series(0.5,index=s.index,dtype=float)
    return x.rank(pct=True,method="average").fillna(0.5)


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
            random_state=31,
        )),
    ])


def mature_train(hist:pd.DataFrame,td:pd.Timestamp,target:str,cfg:dict)->pd.DataFrame:
    years=int(cfg["rolling_train_years"])
    start=td-pd.DateOffset(years=years)
    latest_snapshot=td-pd.DateOffset(months=MATURITY_MONTHS[target])
    tr=hist[
        (hist["date"]<td)
        & (hist["date"]>=start)
        & (hist["date"]<=latest_snapshot)
        & hist[target].notna()
    ].copy()
    return tr


def fit_prob(train:pd.DataFrame,test:pd.DataFrame,target:str,cfg:dict)->np.ndarray:
    tr=train.dropna(subset=[target]).copy()
    if len(tr)<500 or tr[target].sum()<15 or tr[target].nunique()<2:
        return np.full(len(test),0.5)
    m=make_model(cfg)
    m.fit(tr[TRANSFORM_FEATURES],tr[target].astype(int))
    return m.predict_proba(test[TRANSFORM_FEATURES])[:,1]


def transformation_gate(x:pd.DataFrame,cfg:dict)->pd.Series:
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


def score_at(hist:pd.DataFrame,test:pd.DataFrame,td:pd.Timestamp,cfg:dict)->pd.DataFrame:
    z=test.copy()
    z["v10_percentile"]=pct(z["p100_anchor"])

    w=cfg["horizon_weights"]
    rank_parts=[]
    weight_parts=[]
    for target in ["y6","y12","y24"]:
        tr=mature_train(hist,td,target,cfg)
        pred=fit_prob(tr,z,target,cfg)
        z[f"transform_p_{target}"]=pred
        rp=pct(pd.Series(pred,index=z.index))
        z[f"transform_pct_{target}"]=rp
        # If the model falls back to neutral because history is insufficient,
        # do not give that horizon artificial influence.
        active=not np.allclose(pred,0.5)
        if active:
            rank_parts.append(float(w[target])*rp.to_numpy(float))
            weight_parts.append(float(w[target]))

    if weight_parts:
        z["transform_multi_percentile"]=np.sum(np.vstack(rank_parts),axis=0)/sum(weight_parts)
    else:
        z["transform_multi_percentile"]=0.5

    z["transformation_gate"]=transformation_gate(z,cfg)
    z["multi_horizon_score"]=(
        float(cfg["transformation_weight"])*z["transform_multi_percentile"]
        + float(cfg["v10_confirmation_weight"])*z["v10_percentile"]
        - float(cfg["risk_penalty"])*z["risk_score"].clip(0,1)
    )
    return z


def control_select(g:pd.DataFrame,k:int)->pd.DataFrame:
    if "selected_v941" in g and g["selected_v941"].fillna(False).sum()>=k:
        return g[g["selected_v941"].fillna(False)].sort_values(
            ["selection_rank_v941","p100_anchor"],ascending=[True,False]
        ).head(k)
    return g.sort_values("p100_anchor",ascending=False).head(k)


def discovery_select(g:pd.DataFrame,k:int)->pd.DataFrame:
    q=g[g["transformation_gate"]].sort_values(
        ["multi_horizon_score","transform_multi_percentile","p100_anchor"],
        ascending=False
    )
    return q.head(k)


def horizon_metric(sel:pd.DataFrame,ctl:pd.DataFrame,universe:pd.DataFrame,target:str)->dict:
    s=sel.dropna(subset=[target])
    c=ctl.dropna(subset=[target])
    u=universe.dropna(subset=[target])
    if len(s)==0 or len(c)==0 or len(u)==0:
        return {}
    sp=float(s[target].mean())
    cp=float(c[target].mean())
    br=float(u[target].mean())
    return {
        f"{target}_hybrid_precision":sp,
        f"{target}_control_precision":cp,
        f"{target}_hybrid_lift":sp/br if br>0 else np.nan,
        f"{target}_control_lift":cp/br if br>0 else np.nan,
        f"{target}_hybrid_n":int(len(s)),
        f"{target}_control_n":int(len(c)),
    }


def walk_forward(hist:pd.DataFrame,cfg:dict):
    dates=sorted(pd.to_datetime(hist["date"].dropna().unique()))
    k=int(cfg["main_k"])
    min_prior=int(cfg["min_prior_folds"])
    rows=[]
    scored=[]
    for i,td in enumerate(dates):
        if i<min_prior:
            continue
        test=hist[hist["date"].eq(td)].copy()
        if test.empty:
            continue
        z=score_at(hist,test,pd.Timestamp(td),cfg)
        scored.append(z)
        sel=discovery_select(z,k)
        if len(sel)<k:
            continue
        ctl=control_select(z,k)

        row={
            "date":pd.Timestamp(td),
            "eligible":int(z["transformation_gate"].sum()),
            "hybrid_dd30_rate":float(sel["dd30_6m"].mean()) if "dd30_6m" in sel and sel["dd30_6m"].notna().any() else np.nan,
            "control_dd30_rate":float(ctl["dd30_6m"].mean()) if "dd30_6m" in ctl and ctl["dd30_6m"].notna().any() else np.nan,
            "hybrid_early_stage":float(sel["early_stage_score"].mean()),
            "control_early_stage":float(ctl["early_stage_score"].mean()),
            "hybrid_priced_in":float(sel["priced_in_penalty"].mean()),
            "control_priced_in":float(ctl["priced_in_penalty"].mean()),
            "hybrid_ret120":float(pd.to_numeric(sel["ret_120"],errors="coerce").mean()),
            "control_ret120":float(pd.to_numeric(ctl["ret_120"],errors="coerce").mean()),
            "hybrid_catalyst_rate":float((sel["catalyst_score"]>0).mean()),
            "control_catalyst_rate":float((ctl["catalyst_score"]>0).mean()),
        }
        for target in ["y6","y12","y24"]:
            row.update(horizon_metric(sel,ctl,z,target))
        rows.append(row)

    return (
        pd.concat(scored,ignore_index=True) if scored else pd.DataFrame(),
        pd.DataFrame(rows),
    )


def mean_col(f:pd.DataFrame,c:str):
    return float(pd.to_numeric(f[c],errors="coerce").dropna().mean()) if c in f and pd.to_numeric(f[c],errors="coerce").notna().any() else None


def ratio(a,b):
    return a/b if a is not None and b is not None and b>0 else None


def summarize(f:pd.DataFrame,cfg:dict)->dict:
    if f.empty:
        return {"folds":0,"all_gates_pass":False}
    out={
        "folds":int(len(f)),
        "mean_hybrid_dd30_rate":mean_col(f,"hybrid_dd30_rate"),
        "mean_control_dd30_rate":mean_col(f,"control_dd30_rate"),
        "mean_hybrid_early_stage":mean_col(f,"hybrid_early_stage"),
        "mean_control_early_stage":mean_col(f,"control_early_stage"),
        "mean_hybrid_priced_in":mean_col(f,"hybrid_priced_in"),
        "mean_control_priced_in":mean_col(f,"control_priced_in"),
        "mean_hybrid_ret120":mean_col(f,"hybrid_ret120"),
        "mean_control_ret120":mean_col(f,"control_ret120"),
        "mean_hybrid_catalyst_rate":mean_col(f,"hybrid_catalyst_rate"),
        "mean_control_catalyst_rate":mean_col(f,"control_catalyst_rate"),
    }
    for target in ["y6","y12","y24"]:
        hp=mean_col(f,f"{target}_hybrid_precision")
        cp=mean_col(f,f"{target}_control_precision")
        hl=mean_col(f,f"{target}_hybrid_lift")
        cl=mean_col(f,f"{target}_control_lift")
        out[f"{target}_mean_hybrid_precision"]=hp
        out[f"{target}_mean_control_precision"]=cp
        out[f"{target}_precision_ratio"]=ratio(hp,cp)
        out[f"{target}_mean_hybrid_lift"]=hl
        out[f"{target}_mean_control_lift"]=cl
        out[f"{target}_lift_ratio"]=ratio(hl,cl)
        col=f"{target}_hybrid_precision"
        out[f"{target}_evaluated_folds"]=int(pd.to_numeric(f[col],errors="coerce").notna().sum()) if col in f else 0

    out["dd_delta"]=(
        out["mean_hybrid_dd30_rate"]-out["mean_control_dd30_rate"]
        if out["mean_hybrid_dd30_rate"] is not None and out["mean_control_dd30_rate"] is not None else None
    )
    out["priced_in_improvement"]=(
        out["mean_control_priced_in"]-out["mean_hybrid_priced_in"]
        if out["mean_control_priced_in"] is not None and out["mean_hybrid_priced_in"] is not None else None
    )

    a=cfg["acceptance"]
    gates={
        "y6_precision":out["y6_precision_ratio"] is not None and out["y6_precision_ratio"]>=float(a["y6_precision_ratio_min"]),
        "y12_precision":out["y12_precision_ratio"] is not None and out["y12_precision_ratio"]>=float(a["y12_precision_ratio_min"]),
        "y24_precision":out["y24_precision_ratio"] is not None and out["y24_precision_ratio"]>=float(a["y24_precision_ratio_min"]),
        "dd_delta":out["dd_delta"] is not None and out["dd_delta"]<=float(a["dd_delta_max"]),
        "transformation_presence":out["mean_hybrid_catalyst_rate"] is not None and out["mean_hybrid_catalyst_rate"]>=float(a["selected_transformation_rate_min"]),
        "early_stage":out["mean_hybrid_early_stage"] is not None and out["mean_hybrid_early_stage"]>=float(a["early_stage_score_min"]),
    }
    out["acceptance_gates"]=gates
    out["all_gates_pass"]=all(gates.values())
    return out


def radar(current:pd.DataFrame,main_symbols:set[str],cfg:dict)->pd.DataFrame:
    r=current.copy()
    rcfg=cfg["radar"]
    m=(
        (r["transform_multi_percentile"]>=float(rcfg["min_transform_percentile"]))
        & (r["v10_percentile"]<float(rcfg["max_v10_percentile"]))
        & (r["priced_in_penalty"]<=float(rcfg["max_priced_in_penalty"]))
        & (~r["symbol"].astype(str).isin(main_symbols))
    )
    q=r[m].copy()
    q["radar_score"]=(
        0.45*q["transform_multi_percentile"]
        +0.20*q["catalyst_score"]
        +0.15*q["ownership_accumulation_score"]
        +0.20*q["early_stage_score"]
        -0.10*q["risk_score"]
    )
    return q.sort_values(["radar_score","transform_multi_percentile"],ascending=False).head(int(cfg["radar_k"]))


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
    for target in ["y6","y12","y24"]:
        if target not in hist:
            raise RuntimeError(f"Required historical label missing: {target}")

    current=h.engineer(pd.read_csv(args.current_scored))
    current["date"]=pd.to_datetime(current["date"])

    scored,folds=walk_forward(hist,cfg)
    folds.to_csv(out/"fold_metrics.csv",index=False)
    if len(scored):
        scored.to_parquet(out/"historical_multi_horizon_scored.parquet",index=False)

    latest=pd.Timestamp(current["date"].max())
    cur=score_at(hist,current,latest,cfg)
    main=discovery_select(cur,int(cfg["main_k"])).copy()
    main["discovery_rank"]=np.arange(1,len(main)+1)
    main.to_csv(out/"current_multi_horizon_top10.csv",index=False)

    rd=radar(cur,set(main["symbol"].astype(str)),cfg)
    rd["radar_rank"]=np.arange(1,len(rd)+1)
    rd.to_csv(out/"current_early_discovery_radar.csv",index=False)
    cur.sort_values("multi_horizon_score",ascending=False).to_csv(out/"current_universe_scored.csv",index=False)

    metrics=summarize(folds,cfg)
    summary={
        "model":cfg["model_name"],
        "objective":"discover transformation stories before obvious rerating",
        "architecture":"70% multi-horizon transformation evidence (20% 6m / 35% 12m / 45% 24m) + 30% V10.2 confirmation",
        "current_date":str(latest.date()),
        "current_eligible":int(cur["transformation_gate"].sum()),
        "current_selected":int(len(main)),
        "current_radar_selected":int(len(rd)),
        "historical_metrics":metrics,
        "config":cfg,
        "methodology_note":"12/24m training uses conservative maturity cutoffs at each fold to avoid forward-label leakage. This is development evidence; prospective shadow validation remains mandatory."
    }
    json.dump(summary,open(out/"summary.json","w"),indent=2,default=str)
    print(json.dumps(summary,indent=2,default=str))

    if len(main):
        cols=[c for c in [
            "symbol","close","discovery_rank","multi_horizon_score","transform_multi_percentile",
            "transform_p_y6","transform_p_y12","transform_p_y24","p100_anchor","v10_percentile",
            "catalyst_score","business_inflection_score","promoter_conviction_score",
            "accumulation_score","ownership_accumulation_score","early_stage_score",
            "priced_in_penalty","risk_score"
        ] if c in main]
        print("\nV11.3 MULTI-HORIZON TOP-10")
        print(main[cols].to_string(index=False))


if __name__=="__main__":
    main()
