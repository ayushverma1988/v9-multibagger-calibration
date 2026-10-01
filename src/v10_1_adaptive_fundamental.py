from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

import calibrate_v9_3_1 as v931

BASE_FEATURES=[
    "baseline_rank","p_cal","p_dd30_cal","model_dispersion",
    "p_fund_cal","fund_rank",
    "breadth_120","market_median_ret120",
    "fund_x_alpha","fund_x_safety","fund_x_breadth","fund_x_market",
]


def ensure_features(x:pd.DataFrame)->pd.DataFrame:
    z=x.copy()
    for c in ["breadth_120","market_median_ret120"]:
        if c not in z.columns:z[c]=np.nan
    z["fund_rank"]=z.groupby("date")["p_fund_cal"].rank(pct=True,method="average")
    z["baseline_rank"]=z.groupby("date")["selection_score"].rank(pct=True,method="average")
    safety=1.0-z["p_dd30_cal"].clip(0,1)
    z["fund_x_alpha"]=z["fund_rank"]*z["baseline_rank"]
    z["fund_x_safety"]=z["fund_rank"]*safety
    z["fund_x_breadth"]=z["fund_rank"]*(z["breadth_120"]-0.5)
    z["fund_x_market"]=z["fund_rank"]*z["market_median_ret120"]
    return z


def spec_from_row(r):
    return {
        "name":r.config,
        "pool_n":int(r.pool_n),
        "risk_drop":float(r.risk_drop),
        "w_safety":float(r.w_safety),
        "w_consensus":float(r.w_consensus),
        "baseline":bool(r.fell_back_to_v92),
    }


def survivor_pool(g,spec,cfg):
    return v931.select_topk(
        g,
        spec,
        {**cfg,"selection_k":max(int(cfg.get("selection_k",10)),int(spec.get("pool_n",100)))}
    )


def build_survivors(oos,chosen,cfg):
    # Reconstruct the risk-survivor set, not merely top-10. This mirrors the
    # validated selector while allowing a small adaptive fundamental re-ranking.
    rows=[]
    k=int(cfg.get("selection_k",10))
    for r in chosen.itertuples(index=False):
        td=pd.Timestamp(r.date)
        g=oos[oos["date"]==td].copy()
        if g.empty:continue
        spec=spec_from_row(r)
        q=g.dropna(subset=["p_cal","p_dd30_cal","model_dispersion"]).copy()
        if len(q)<k:continue

        if spec["baseline"]:
            q["selection_score"]=q["p_cal"]
            surv=q.sort_values(["selection_score","p_cal","model_dispersion"],ascending=[False,False,True])
        else:
            pool=q.sort_values(["p_cal","model_dispersion"],ascending=[False,True]).head(min(spec["pool_n"],len(q))).copy()
            if len(pool)<k:continue
            if spec["risk_drop"]>0:
                cutoff=float(pool["p_dd30_cal"].quantile(1.0-spec["risk_drop"]))
                pool=pool[pool["p_dd30_cal"]<=cutoff].copy()
            if len(pool)<k:continue
            pool["comp_alpha"]=pool["p_cal"].rank(pct=True,method="average")
            pool["comp_safety"]=pool["p_dd30_cal"].rank(pct=True,method="average",ascending=False)
            pool["comp_consensus"]=pool["model_dispersion"].rank(pct=True,method="average",ascending=False)
            wa=max(0.0,1.0-spec["w_safety"]-spec["w_consensus"])
            pool["selection_score"]=(
                wa*pool["comp_alpha"]+
                spec["w_safety"]*pool["comp_safety"]+
                spec["w_consensus"]*pool["comp_consensus"]
            )
            surv=pool.sort_values(["selection_score","p_cal","model_dispersion"],ascending=[False,False,True])
        rows.append(surv)
    x=pd.concat(rows,ignore_index=True) if rows else pd.DataFrame()
    return ensure_features(x) if len(x) else x


def fit_meta(train):
    m=Pipeline([
        ("impute",SimpleImputer(strategy="median",add_indicator=True)),
        ("scale",RobustScaler()),
        ("clf",LogisticRegression(
            C=0.10,class_weight="balanced",max_iter=3000,random_state=20261001
        ))
    ])
    m.fit(train[BASE_FEATURES],train["y6"].astype(int))
    return m


def score_fold(g,w,k):
    q=g.dropna(subset=["y6","selection_score"]).copy()
    if len(q)<k:return None
    q["base_rank"]=q["selection_score"].rank(pct=True,method="average")
    q["meta_rank"]=q["p_v101_meta"].rank(pct=True,method="average").fillna(0.5)
    q["score_v101"]=(1-w)*q["base_rank"]+w*q["meta_rank"]
    s=q.sort_values(["score_v101","base_rank","p_cal"],ascending=[False,False,False]).head(k)
    br=float(q["y6"].mean()); pr=float(s["y6"].mean())
    return {
        "precision_2x":pr,
        "lift_2x":pr/br if br>0 else np.nan,
        "hit":float(pr>0),
        "dd30_rate":float(s["dd30_6m"].mean()),
    },s


def aggregate(rows):
    if not rows:return None
    t=pd.DataFrame(rows)
    lift=np.clip(t["lift_2x"].replace([np.inf,-np.inf],np.nan).dropna(),0,10)
    return {
        "folds":int(len(t)),
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


def utility(a,b):
    rr=[a[f]/b[f] if b[f]>0 else 0 for f in ["mean_precision_2x","mean_capped_lift_2x","hit_fold_rate"]]
    dd=(b["mean_dd30_rate"]-a["mean_dd30_rate"])/b["mean_dd30_rate"] if b["mean_dd30_rate"]>0 else 0
    return .45*rr[0]+.35*rr[1]+.20*rr[2]+.10*dd


def nested_prior_predictions(prior,min_train_folds=3,min_rows=1000,min_pos=35):
    q=prior.dropna(subset=["y6","p_fund_cal"]).copy()
    q["date"]=pd.to_datetime(q["date"])
    q["p_v101_meta"]=np.nan
    dates=sorted(q["date"].unique())
    for i,td in enumerate(dates):
        tr=q[q["date"]<td].copy()
        te_idx=q.index[q["date"]==td]
        if tr["date"].nunique()<min_train_folds or len(tr)<min_rows or int(tr["y6"].sum())<min_pos or tr["y6"].nunique()<2:
            continue
        m=fit_meta(tr)
        q.loc[te_idx,"p_v101_meta"]=m.predict_proba(q.loc[te_idx,BASE_FEATURES])[:,1]
    return q


def choose_weight(prior,cfg):
    k=int(cfg.get("selection_k",10))
    pred=nested_prior_predictions(prior)
    dates=sorted(pred.loc[pred["p_v101_meta"].notna(),"date"].unique())
    if len(dates)<3:
        return 0.0,pd.DataFrame()

    candidates=[]
    base_rows=[]
    by_weight={w:[] for w in [0.0,.02,.05,.10]}
    for td in dates:
        g=pred[pred["date"]==td].copy()
        for w in by_weight:
            res=score_fold(g,w,k)
            if res is not None:by_weight[w].append(res[0])
    b=aggregate(by_weight[0.0])
    if not b:return 0.0,pd.DataFrame()

    bestu=1.0;bestw=0.0
    for w,rows in by_weight.items():
        a=aggregate(rows)
        if not a:continue
        ok=True if w==0 else retain(a,b,.95)
        u=utility(a,b)
        candidates.append({"weight":w,**a,"retention_passed":ok,"utility":u,"utility_gain":u-1.0})
        if w>0 and ok and u>=1.02 and u>bestu:
            bestw=w;bestu=u
    return bestw,pd.DataFrame(candidates)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--oos",required=True)
    ap.add_argument("--chosen931",required=True)
    ap.add_argument("--config",required=True)
    ap.add_argument("--output",required=True)
    args=ap.parse_args()

    cfg=json.load(open(args.config));k=int(cfg.get("selection_k",10))
    oos=pd.read_parquet(args.oos);oos["date"]=pd.to_datetime(oos["date"])
    chosen=pd.read_csv(args.chosen931);chosen["date"]=pd.to_datetime(chosen["date"])

    surv=build_survivors(oos,chosen,cfg)
    if "y6_mature_date" in surv.columns:
        surv["y6_mature_date"]=pd.to_datetime(surv["y6_mature_date"],errors="coerce")
    else:
        surv["y6_mature_date"]=pd.NaT

    decisions=[];selected=[]
    all_dates=sorted(surv["date"].unique())
    for td in all_dates:
        prior=surv[(surv["date"]<td)&surv["y6"].notna()&surv["p_fund_cal"].notna()].copy()
        prior=prior[(prior["y6_mature_date"].isna()) | (prior["y6_mature_date"]<pd.Timestamp(td))]
        w,search=choose_weight(prior,cfg)

        cur=surv[surv["date"]==td].copy()
        cur["p_v101_meta"]=np.nan
        train=prior.dropna(subset=["p_fund_cal"]).copy()
        if train["date"].nunique()>=3 and len(train)>=1000 and int(train["y6"].sum())>=35 and train["y6"].nunique()>=2:
            m=fit_meta(train)
            cur["p_v101_meta"]=m.predict_proba(cur[BASE_FEATURES])[:,1]
        else:
            w=0.0

        if cur["p_v101_meta"].notna().sum()<k:w=0.0
        res=score_fold(cur,w,k)
        if res is None:
            continue
        _,s=res
        for rank,(_,r) in enumerate(s.iterrows(),1):
            selected.append({"date":td,"symbol":r["symbol"],"rank":rank,"weight":w,"y6":r["y6"],"dd30_6m":r["dd30_6m"]})
        decisions.append({
            "date":td,"weight":w,
            "prior_fund_folds":int(prior["date"].nunique()),
            "nested_validation_folds":int(search["folds"].max()) if len(search) and "folds" in search else 0,
            "best_nested_utility":float(search["utility"].max()) if len(search) else 1.0,
        })

    sel=pd.DataFrame(selected);dec=pd.DataFrame(decisions)
    vrows=[];brows=[]
    for td,g in sel.groupby("date"):
        q=surv[surv["date"]==td].dropna(subset=["y6"]).copy()
        if len(g)!=k or q.empty:continue
        br=float(q["y6"].mean());pr=float(g["y6"].mean())
        vrows.append({"date":td,"precision_2x":pr,"lift_2x":pr/br if br>0 else np.nan,"hit":float(pr>0),"dd30_rate":float(g["dd30_6m"].mean())})
        bs=q.sort_values(["selection_score","p_cal","model_dispersion"],ascending=[False,False,True]).head(k)
        bp=float(bs["y6"].mean())
        brows.append({"date":td,"precision_2x":bp,"lift_2x":bp/br if br>0 else np.nan,"hit":float(bp>0),"dd30_rate":float(bs["dd30_6m"].mean())})
    a=aggregate(vrows);b=aggregate(brows)

    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    dec.to_csv(out/"adaptive_weight_decisions.csv",index=False)
    sel.to_csv(out/"adaptive_selected_rows.csv",index=False)
    pd.DataFrame(vrows).to_csv(out/"adaptive_metrics.csv",index=False)
    pd.DataFrame(brows).to_csv(out/"baseline_metrics.csv",index=False)

    active=int((dec["weight"]>0).sum()) if len(dec) else 0
    summary={
        "model":"V10.1 adaptive fundamental integration",
        "method":"prior-only regularized technical/fundamental interaction meta-model; max 10% reranking weight",
        "fixed_linear_blend_replaced":True,
        "V10.1":a,
        "V10_baseline_same_folds":b,
        "retention_passed":bool(a and b and retain(a,b,.95)),
        "active_adaptive_folds":active,
        "production_gate":bool(a and b and retain(a,b,.95) and utility(a,b)>=1.02 and active>=4),
        "utility_ratio":float(utility(a,b)) if a and b else None,
        "fallback":"0% fundamental influence / frozen validated baseline whenever prior evidence is insufficient",
    }
    json.dump(summary,open(out/"summary.json","w"),indent=2,default=str)
    print(json.dumps(summary,indent=2,default=str))


if __name__=="__main__":
    main()
