from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

import calibrate_v9_3_1 as v931


BASE_PATH = [
    "ret_20","ret_60","ret_120","ret_252","mom_accel",
    "vol_accel","turnover_accel","off_high_252","above_low_252",
    "trend_consistency_60","volatility_60","drawdown_126",
]
DERIVED = [
    "accel_60_120","accel_120_252","breakout_proximity",
    "recovery_position","trend_momentum","turnover_breakout",
    "drawdown_recovery_pressure",
]
ALL_PATH = BASE_PATH + DERIVED


def add_path_features(df: pd.DataFrame) -> pd.DataFrame:
    x=df.copy()
    x["accel_60_120"]=x["ret_60"]-0.5*x["ret_120"]
    x["accel_120_252"]=x["ret_120"]-(120.0/252.0)*x["ret_252"]
    x["breakout_proximity"]=(1.0+x["off_high_252"]).clip(0,1.5)
    x["recovery_position"]=np.log1p(x["above_low_252"].clip(lower=-0.99))
    x["trend_momentum"]=x["trend_consistency_60"]*x["ret_60"]
    x["turnover_breakout"]=x["turnover_accel"]*x["breakout_proximity"]
    x["drawdown_recovery_pressure"]=(
        x["ret_20"] - x["drawdown_126"].abs()
    )
    return x


def make_model(seed: int):
    return Pipeline([
        ("impute",SimpleImputer(strategy="median")),
        ("clf",HistGradientBoostingClassifier(
            max_depth=3,learning_rate=0.04,max_iter=160,
            min_samples_leaf=80,l2_regularization=1.5,
            class_weight="balanced",random_state=seed,
        )),
    ])


def forward_path(snapshot: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    x=add_path_features(snapshot)
    x["date"]=pd.to_datetime(x["date"])
    rows=[]
    min_train=int(cfg.get("path_min_train_rows",800))
    min_pos=int(cfg.get("path_min_train_positives",20))
    years=int(cfg.get("path_rolling_years",8))
    seed=int(cfg.get("random_seed",20260928))+9800

    for td in sorted(x["date"].unique()):
        test=x[(x["date"]==td)&x["y6"].notna()].copy()
        if test.empty:
            continue
        start=pd.Timestamp(td)-pd.DateOffset(years=years)
        train=x[
            (x["date"]<td)&(x["date"]>=start)&x["y6"].notna()
        ].copy()
        if len(train)<min_train or int(train["y6"].sum())<min_pos:
            continue
        m=make_model(seed)
        m.fit(train[ALL_PATH],train["y6"].astype(int))
        p=m.predict_proba(test[ALL_PATH])[:,1]
        z=test[["date","symbol","y6","dd30_6m"]].copy()
        z["p_path_raw"]=p
        z["path_rank"]=pd.Series(p,index=z.index).rank(pct=True,method="average").to_numpy()
        rows.append(z)
    return pd.concat(rows,ignore_index=True) if rows else pd.DataFrame()


def spec_from_row(r):
    return {
        "name":r.config,
        "pool_n":int(r.pool_n),
        "risk_drop":float(r.risk_drop),
        "w_safety":float(r.w_safety),
        "w_consensus":float(r.w_consensus),
        "baseline":bool(r.fell_back_to_v92),
    }


def survivors(g: pd.DataFrame, spec: dict, k: int) -> pd.DataFrame:
    q=g.dropna(subset=["p_cal","p_dd30_cal","model_dispersion"]).copy()
    if len(q)<k:
        return pd.DataFrame()
    if spec.get("baseline",False):
        q["selection_score"]=q["p_cal"]
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
    ws=float(spec["w_safety"]); wc=float(spec["w_consensus"]); wa=max(0,1-ws-wc)
    pool["selection_score"]=wa*pool["comp_alpha"]+ws*pool["comp_safety"]+wc*pool["comp_consensus"]
    return pool.sort_values(["selection_score","p_cal","model_dispersion"],ascending=[False,False,True])


def select(g, spec, bonus, cfg):
    k=int(cfg.get("selection_k",10))
    s=survivors(g,spec,k)
    if len(s)<k:
        return pd.DataFrame()
    if bonus<=0 or "path_rank" not in s.columns or s["path_rank"].notna().sum()<k:
        return s.head(k).copy()
    q=s.copy()
    q["path_rank_neutral"]=q["path_rank"].fillna(0.5)
    q["score_v98"]=q["selection_score"]+float(bonus)*q["path_rank_neutral"]
    return q.sort_values(["score_v98","selection_score","p_cal"],ascending=[False,False,False]).head(k).copy()


def fold_metrics(hist, chosen_map, bonus, cfg):
    rows=[]
    k=int(cfg.get("selection_k",10))
    for td,r in chosen_map.items():
        g=hist[hist["date"]==td].copy()
        if g.empty:
            continue
        valid=g.dropna(subset=["y6","dd30_6m","p_cal","p_dd30_cal","model_dispersion"])
        if len(valid)<max(k,20):
            continue
        sel=select(valid,spec_from_row(r),bonus,cfg)
        if len(sel)!=k:
            continue
        br=float(valid["y6"].mean())
        pr=float(sel["y6"].mean())
        rows.append({
            "date":td,"precision_2x":pr,
            "lift_2x":pr/br if br>0 else np.nan,
            "hit":float(pr>0),
            "dd30_rate":float(sel["dd30_6m"].mean()),
        })
    return pd.DataFrame(rows)


def agg(t):
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


def retain(a,b,cfg):
    r=float(cfg.get("path_alpha_retention",0.95))
    for f in ["mean_precision_2x","median_precision_2x","mean_capped_lift_2x","hit_fold_rate"]:
        if not(np.isfinite(a.get(f,np.nan)) and np.isfinite(b.get(f,np.nan))):
            return False
        if a[f]+1e-12<r*b[f]:
            return False
    return True


def utility(a,b):
    ratios=[]
    for f in ["mean_precision_2x","mean_capped_lift_2x","hit_fold_rate"]:
        ratios.append(a[f]/b[f] if b[f]>0 else 0)
    dd=(b["mean_dd30_rate"]-a["mean_dd30_rate"])/b["mean_dd30_rate"] if b["mean_dd30_rate"]>0 else 0
    return 0.45*ratios[0]+0.35*ratios[1]+0.20*ratios[2]+0.10*dd


def forward_select(hist, chosen, cfg):
    hist=hist.copy(); hist["date"]=pd.to_datetime(hist["date"])
    chosen=chosen.copy(); chosen["date"]=pd.to_datetime(chosen["date"])
    cmap={pd.Timestamp(r.date):r for r in chosen.itertuples(index=False)}
    bonuses=[0.0,0.02,0.05,0.10]
    min_folds=int(cfg.get("path_min_prior_folds",8))
    min_gain=float(cfg.get("path_min_utility_gain",0.02))
    decisions=[]; selected=[]

    for td in sorted(cmap):
        g=hist[hist["date"]==td].copy()
        prior_map={d:r for d,r in cmap.items() if d<td}
        prior=hist[hist["date"]<td].copy()
        base=agg(fold_metrics(prior,prior_map,0.0,cfg))
        best=0.0; best_u=1.0
        if base and base["folds"]>=min_folds:
            for b in bonuses[1:]:
                a=agg(fold_metrics(prior,prior_map,b,cfg))
                if not a or a["folds"]<min_folds or not retain(a,base,cfg):
                    continue
                u=utility(a,base)
                if u>=1.0+min_gain and u>best_u:
                    best=b; best_u=u
        sel=select(g,spec_from_row(cmap[td]),best,cfg)
        if len(sel)!=int(cfg.get("selection_k",10)):
            best=0.0
            sel=select(g,spec_from_row(cmap[td]),0.0,cfg)
        for rank,idx in enumerate(sel.index.tolist(),1):
            selected.append({"date":td,"symbol":g.loc[idx,"symbol"],"rank":rank,"bonus":best})
        decisions.append({"date":td,"bonus":best,"prior_folds":base["folds"] if base else 0,"train_utility":best_u})
    return pd.DataFrame(selected),pd.DataFrame(decisions)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--snapshot",required=True)
    ap.add_argument("--oos",required=True)
    ap.add_argument("--chosen931",required=True)
    ap.add_argument("--config",required=True)
    ap.add_argument("--output",required=True)
    args=ap.parse_args()

    cfg=json.load(open(args.config))
    cfg.setdefault("path_alpha_retention",0.95)
    cfg.setdefault("path_min_prior_folds",8)
    cfg.setdefault("path_min_utility_gain",0.02)

    snap=pd.read_parquet(args.snapshot)
    oos=pd.read_parquet(args.oos)
    oos["date"]=pd.to_datetime(oos["date"])
    path=forward_path(snap,cfg)
    merged=oos.merge(path[["date","symbol","p_path_raw","path_rank"]],on=["date","symbol"],how="left")

    chosen=pd.read_csv(args.chosen931)
    selected,decisions=forward_select(merged,chosen,cfg)

    outdir=Path(args.output); outdir.mkdir(parents=True,exist_ok=True)
    path.to_parquet(outdir/"path_predictions.parquet",index=False)
    decisions.to_csv(outdir/"path_decisions_by_fold.csv",index=False)
    selected.to_csv(outdir/"path_selected_rows.csv",index=False)

    diag=[]
    cmap={pd.Timestamp(r.date):r for r in chosen.assign(date=pd.to_datetime(chosen["date"])).itertuples(index=False)}
    base=agg(fold_metrics(merged,cmap,0.0,cfg))
    for b in [0.0,0.02,0.05,0.10]:
        a=agg(fold_metrics(merged,cmap,b,cfg))
        if a:
            diag.append({"bonus":b,**a,"retention_passed":True if b==0 else retain(a,base,cfg),"utility":utility(a,base)})
    diag=pd.DataFrame(diag)
    diag.to_csv(outdir/"path_bonus_diagnostic.csv",index=False)

    summary={
        "model":"V9.8 path-shape challenger",
        "leakage_policy":"path model trained only on strictly prior mature folds; bonus chosen only from strictly prior fold outcomes",
        "derived_features":DERIVED,
        "diagnostic":diag.to_dict(orient="records"),
        "forward_active_folds":int((decisions["bonus"]>0).sum()) if len(decisions) else 0,
        "production_status":"challenger only until >=95% alpha retention and forward utility gate pass",
    }
    json.dump(summary,open(outdir/"summary.json","w"),indent=2,default=str)
    print(json.dumps(summary,indent=2,default=str))


if __name__=="__main__":
    main()
