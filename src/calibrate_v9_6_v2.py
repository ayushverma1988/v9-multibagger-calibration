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


META = [
    "baseline_score_rank",
    "alpha_rank",
    "safety_rank",
    "consensus_rank",
    "excess_ret20",
    "excess_ret60",
    "excess_ret120",
    "mom_accel",
    "turnover_accel",
    "off_high_252",
    "trend_consistency_60",
    "regime_breadth_20",
    "regime_breadth_60",
    "regime_breadth_120",
    "regime_median_ret20",
    "regime_median_ret60",
    "regime_median_ret120",
    "regime_median_vol60",
    "regime_median_drawdown126",
    "regime_dispersion_ret120",
    "alpha_x_breadth120",
    "alpha_x_market120",
    "safety_x_volatility",
]


def spec_from_row(r):
    return {
        "pool_n": int(r.pool_n),
        "risk_drop": float(r.risk_drop),
        "w_safety": float(r.w_safety),
        "w_consensus": float(r.w_consensus),
        "baseline": bool(r.fell_back_to_v92),
    }


def survivor_pool(g: pd.DataFrame, spec: dict, k: int) -> pd.DataFrame:
    q=g.dropna(subset=["p_cal","p_dd30_cal","model_dispersion"]).copy()
    if len(q)<k:
        return pd.DataFrame()

    if spec.get("baseline",False):
        q["selection_score"]=q["p_cal"]
        q["comp_alpha"]=q["p_cal"].rank(pct=True,method="average")
        q["comp_safety"]=q["p_dd30_cal"].rank(pct=True,method="average",ascending=False)
        q["comp_consensus"]=q["model_dispersion"].rank(pct=True,method="average",ascending=False)
        return q.sort_values(["selection_score","p_cal","model_dispersion"],ascending=[False,False,True])

    pool=q.sort_values(["p_cal","model_dispersion"],ascending=[False,True]).head(min(int(spec["pool_n"]),len(q))).copy()
    if len(pool)<k:
        return pd.DataFrame()
    rd=float(spec["risk_drop"])
    if rd>0:
        cutoff=float(pool["p_dd30_cal"].quantile(1-rd))
        pool=pool[pool["p_dd30_cal"]<=cutoff].copy()
    if len(pool)<k:
        return pd.DataFrame()

    pool["comp_alpha"]=pool["p_cal"].rank(pct=True,method="average")
    pool["comp_safety"]=pool["p_dd30_cal"].rank(pct=True,method="average",ascending=False)
    pool["comp_consensus"]=pool["model_dispersion"].rank(pct=True,method="average",ascending=False)
    ws=float(spec["w_safety"]); wc=float(spec["w_consensus"]); wa=max(0.0,1.0-ws-wc)
    pool["selection_score"]=wa*pool["comp_alpha"]+ws*pool["comp_safety"]+wc*pool["comp_consensus"]
    return pool.sort_values(["selection_score","p_cal","model_dispersion"],ascending=[False,False,True])


def regime_table(snapshot: pd.DataFrame) -> pd.DataFrame:
    s=snapshot.copy()
    s["date"]=pd.to_datetime(s["date"])
    rows=[]
    for td,g in s.groupby("date",sort=True):
        rows.append({
            "date":pd.Timestamp(td),
            "regime_breadth_20":float(np.nanmean(g["ret_20"].to_numpy(float)>0)),
            "regime_breadth_60":float(np.nanmean(g["ret_60"].to_numpy(float)>0)),
            "regime_breadth_120":float(np.nanmean(g["ret_120"].to_numpy(float)>0)),
            "regime_median_ret20":float(np.nanmedian(g["ret_20"])),
            "regime_median_ret60":float(np.nanmedian(g["ret_60"])),
            "regime_median_ret120":float(np.nanmedian(g["ret_120"])),
            "regime_median_vol60":float(np.nanmedian(g["volatility_60"])),
            "regime_median_drawdown126":float(np.nanmedian(g["drawdown_126"])),
            "regime_dispersion_ret120":float(np.nanstd(g["ret_120"])),
        })
    return pd.DataFrame(rows)


def meta_features(pool: pd.DataFrame) -> pd.DataFrame:
    x=pool.copy()
    x["baseline_score_rank"]=x["selection_score"].rank(pct=True,method="average")
    x["alpha_rank"]=x["comp_alpha"]
    x["safety_rank"]=x["comp_safety"]
    x["consensus_rank"]=x["comp_consensus"]
    x["excess_ret20"]=x["ret_20"]-x["regime_median_ret20"]
    x["excess_ret60"]=x["ret_60"]-x["regime_median_ret60"]
    x["excess_ret120"]=x["ret_120"]-x["regime_median_ret120"]
    x["alpha_x_breadth120"]=x["alpha_rank"]*(x["regime_breadth_120"]-0.5)
    x["alpha_x_market120"]=x["alpha_rank"]*x["regime_median_ret120"]
    x["safety_x_volatility"]=x["safety_rank"]*x["regime_median_vol60"]
    return x


def fit_meta(train: pd.DataFrame):
    m=Pipeline([
        ("impute",SimpleImputer(strategy="median")),
        ("scale",RobustScaler()),
        ("clf",LogisticRegression(
            C=0.15,penalty="l2",class_weight="balanced",
            max_iter=2000,random_state=20261001,
        )),
    ])
    m.fit(train[META],train["y6"].astype(int))
    return m


def aggregate(t):
    if t.empty:
        return None
    lift=np.clip(t["lift_2x"].replace([np.inf,-np.inf],np.nan).dropna(),0,10)
    return {
        "folds":int(t["date"].nunique()),
        "mean_precision_2x":float(t["precision_2x"].mean()),
        "median_precision_2x":float(t["precision_2x"].median()),
        "mean_capped_lift_2x":float(lift.mean()) if len(lift) else np.nan,
        "hit_fold_rate":float(t["hit"].mean()),
        "mean_dd30_rate":float(t["dd30_rate"].mean()),
    }


def retention(a,b,cfg):
    req=float(cfg.get("regime_v2_alpha_retention",0.95))
    for f in ["mean_precision_2x","median_precision_2x","mean_capped_lift_2x","hit_fold_rate"]:
        av=a.get(f,np.nan); bv=b.get(f,np.nan)
        if not(np.isfinite(av) and np.isfinite(bv)):
            return False
        if bv>0 and av+1e-12<req*bv:
            return False
    return True


def utility(a,b):
    rs=[]
    for f in ["mean_precision_2x","mean_capped_lift_2x","hit_fold_rate"]:
        rs.append(a[f]/b[f] if b[f]>0 else 0.0)
    dd=(b["mean_dd30_rate"]-a["mean_dd30_rate"])/b["mean_dd30_rate"] if b["mean_dd30_rate"]>0 else 0.0
    return 0.45*rs[0]+0.35*rs[1]+0.20*rs[2]+0.10*dd


def evaluate_weight(pred: pd.DataFrame, w: float, cfg: dict) -> pd.DataFrame:
    rows=[]; k=int(cfg.get("selection_k",10))
    for td,g in pred.groupby("date"):
        if g["p_regime_meta"].notna().sum()<k or g["y6"].notna().sum()<k:
            continue
        q=g.dropna(subset=["y6","dd30_6m","selection_score"]).copy()
        if len(q)<k:
            continue
        q["baseline_rank"]=q["selection_score"].rank(pct=True,method="average")
        q["meta_rank"]=q["p_regime_meta"].rank(pct=True,method="average").fillna(0.5)
        q["score_v2"]=(1-float(w))*q["baseline_rank"]+float(w)*q["meta_rank"]
        s=q.sort_values(["score_v2","baseline_rank","p_cal"],ascending=[False,False,False]).head(k)
        base_rate=float(q["y6"].mean())
        pr=float(s["y6"].mean())
        rows.append({
            "date":pd.Timestamp(td),"precision_2x":pr,
            "lift_2x":pr/base_rate if base_rate>0 else np.nan,
            "hit":float(pr>0),"dd30_rate":float(s["dd30_6m"].mean()),
        })
    return pd.DataFrame(rows)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--snapshot",required=True)
    ap.add_argument("--oos",required=True)
    ap.add_argument("--chosen931",required=True)
    ap.add_argument("--config",required=True)
    ap.add_argument("--output",required=True)
    args=ap.parse_args()

    cfg=json.load(open(args.config))
    cfg.setdefault("regime_v2_alpha_retention",0.95)
    cfg.setdefault("regime_v2_min_prior_folds",8)
    cfg.setdefault("regime_v2_min_train_rows",500)
    cfg.setdefault("regime_v2_min_train_positives",20)
    cfg.setdefault("regime_v2_min_policy_folds",4)
    cfg.setdefault("regime_v2_min_utility_gain",0.02)
    cfg.setdefault("regime_v2_max_weight",0.10)

    snap=pd.read_parquet(args.snapshot)
    snap["date"]=pd.to_datetime(snap["date"])
    oos=pd.read_parquet(args.oos)
    oos["date"]=pd.to_datetime(oos["date"])
    chosen=pd.read_csv(args.chosen931)
    chosen["date"]=pd.to_datetime(chosen["date"])
    cmap={pd.Timestamp(r.date):r for r in chosen.itertuples(index=False)}

    extra=["date","symbol","ret_20","ret_60","ret_120","mom_accel","turnover_accel",
           "off_high_252","trend_consistency_60","volatility_60","drawdown_126",
           "y6_mature_date"]
    extra=[c for c in extra if c in snap.columns]
    base=oos.merge(snap[extra],on=["date","symbol"],how="left",suffixes=("","_snap"))
    reg=regime_table(snap)
    base=base.merge(reg,on="date",how="left")

    # Construct the frozen V9.4.1 risk-survivor pool for every fold.
    pools=[]
    for td,r in cmap.items():
        g=base[base["date"]==td].copy()
        p=survivor_pool(g,spec_from_row(r),int(cfg.get("selection_k",10)))
        if p.empty:
            continue
        p=meta_features(p)
        pools.append(p)
    allp=pd.concat(pools,ignore_index=True) if pools else pd.DataFrame()
    allp["p_regime_meta"]=np.nan

    # Generate strictly forward meta predictions; labels must be mature at td.
    min_folds=int(cfg["regime_v2_min_prior_folds"])
    for td in sorted(allp["date"].unique()):
        current=allp[allp["date"]==td].copy()
        prior=allp[allp["date"]<td].copy()
        if "y6_mature_date" in prior.columns:
            prior=prior[pd.to_datetime(prior["y6_mature_date"],errors="coerce")<pd.Timestamp(td)]
        prior=prior.dropna(subset=["y6"]).copy()
        if (
            prior["date"].nunique()<min_folds
            or len(prior)<int(cfg["regime_v2_min_train_rows"])
            or int(prior["y6"].sum())<int(cfg["regime_v2_min_train_positives"])
            or prior["y6"].nunique()<2
        ):
            continue
        m=fit_meta(prior)
        idx=allp.index[allp["date"]==td]
        allp.loc[idx,"p_regime_meta"]=m.predict_proba(allp.loc[idx,META])[:,1]

    # Choose only a small regime bonus from strictly prior meta-predicted folds.
    weights=[0.0,0.02,0.05,0.10]
    decisions=[]; final_rows=[]
    min_policy=int(cfg["regime_v2_min_policy_folds"])
    min_gain=float(cfg["regime_v2_min_utility_gain"])

    for td in sorted(allp["date"].unique()):
        hist=allp[allp["date"]<td].copy()
        baseline=aggregate(evaluate_weight(hist,0.0,cfg))
        best_w=0.0; best_u=1.0
        if baseline and baseline["folds"]>=min_policy:
            for w in weights[1:]:
                a=aggregate(evaluate_weight(hist,w,cfg))
                if not a or a["folds"]<min_policy or not retention(a,baseline,cfg):
                    continue
                u=utility(a,baseline)
                if u>=1.0+min_gain and u>best_u:
                    best_w=w; best_u=u

        cur=allp[allp["date"]==td].copy()
        if cur["p_regime_meta"].notna().sum()<int(cfg.get("selection_k",10)):
            best_w=0.0
        cur["baseline_rank"]=cur["selection_score"].rank(pct=True,method="average")
        cur["meta_rank"]=cur["p_regime_meta"].rank(pct=True,method="average").fillna(0.5)
        cur["score_v2"]=(1-best_w)*cur["baseline_rank"]+best_w*cur["meta_rank"]
        sel=cur.sort_values(["score_v2","baseline_rank","p_cal"],ascending=[False,False,False]).head(int(cfg.get("selection_k",10)))
        for rank,(_,r) in enumerate(sel.iterrows(),1):
            final_rows.append({
                "date":td,"symbol":r["symbol"],"rank":rank,
                "weight":best_w,"y6":r["y6"],"dd30_6m":r["dd30_6m"],
            })
        decisions.append({
            "date":td,"weight":best_w,
            "prior_policy_folds":baseline["folds"] if baseline else 0,
            "train_utility":best_u,
        })

    final=pd.DataFrame(final_rows)
    comp=[]
    for td,g in final.groupby("date"):
        q=allp[allp["date"]==td].dropna(subset=["y6"])
        if len(g)!=int(cfg.get("selection_k",10)) or q.empty:
            continue
        br=float(q["y6"].mean())
        pr=float(g["y6"].mean())
        comp.append({
            "date":td,"precision_2x":pr,
            "lift_2x":pr/br if br>0 else np.nan,
            "hit":float(pr>0),"dd30_rate":float(g["dd30_6m"].mean()),
            "weight":float(g["weight"].iloc[0]),
        })
    comp=pd.DataFrame(comp)

    # Frozen baseline comparison over the same folds.
    base_rows=[]
    for td in sorted(final["date"].unique()) if len(final) else []:
        q=allp[allp["date"]==td].dropna(subset=["y6"]).copy()
        if q.empty:
            continue
        s=q.sort_values(["selection_score","p_cal","model_dispersion"],ascending=[False,False,True]).head(int(cfg.get("selection_k",10)))
        br=float(q["y6"].mean()); pr=float(s["y6"].mean())
        base_rows.append({
            "date":td,"precision_2x":pr,"lift_2x":pr/br if br>0 else np.nan,
            "hit":float(pr>0),"dd30_rate":float(s["dd30_6m"].mean()),
        })
    base_comp=pd.DataFrame(base_rows)

    outdir=Path(args.output); outdir.mkdir(parents=True,exist_ok=True)
    allp.to_parquet(outdir/"regime_meta_predictions.parquet",index=False)
    pd.DataFrame(decisions).to_csv(outdir/"regime_v2_decisions.csv",index=False)
    final.to_csv(outdir/"regime_v2_selected.csv",index=False)
    comp.to_csv(outdir/"regime_v2_metrics.csv",index=False)
    base_comp.to_csv(outdir/"baseline_metrics.csv",index=False)

    a=aggregate(comp); b=aggregate(base_comp)
    summary={
        "model":"V9.6B-v2 continuous regime meta-overlay",
        "method":"fixed V9.4.1 survivor pool + prior-only ridge meta-model + max 10% regime bonus",
        "leakage_policy":"meta training uses only prior rows whose y6 maturity date is strictly before the decision fold; bonus weight selected only from prior meta-predicted folds",
        "V9.6B_v2":a,
        "V9.4.1_same_folds":b,
        "retention_passed":bool(a and b and retention(a,b,cfg)),
        "active_regime_folds":int((pd.DataFrame(decisions)["weight"]>0).sum()) if decisions else 0,
        "production_gate":bool(
            a and b and retention(a,b,cfg)
            and utility(a,b)>=1.0+min_gain
            and int((pd.DataFrame(decisions)["weight"]>0).sum())>=4
        ),
    }
    if a and b:
        summary["utility_ratio"]=float(utility(a,b))
        summary["precision_retention"]=float(a["mean_precision_2x"]/b["mean_precision_2x"]) if b["mean_precision_2x"]>0 else None
        summary["lift_retention"]=float(a["mean_capped_lift_2x"]/b["mean_capped_lift_2x"]) if b["mean_capped_lift_2x"]>0 else None
    json.dump(summary,open(outdir/"summary.json","w"),indent=2,default=str)
    print(json.dumps(summary,indent=2,default=str))


if __name__=="__main__":
    main()
