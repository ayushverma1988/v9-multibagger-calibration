from __future__ import annotations

import argparse, json, math
from pathlib import Path

import numpy as np
import pandas as pd

import calibrate_v9_3_1 as v931


CAPITALS=[1e5,5e5,1e6,2.5e6,5e6,1e7]
TURNOVER_PARTICIPATION_LIMITS=[0.005,0.01,0.02,0.05]


def spec_from_row(r):
    return {
        "name":r.config,
        "pool_n":int(r.pool_n),
        "risk_drop":float(r.risk_drop),
        "w_safety":float(r.w_safety),
        "w_consensus":float(r.w_consensus),
        "baseline":bool(r.fell_back_to_v92),
    }


def selected_baseline(g,spec,cfg):
    return v931.select_topk(g,spec,cfg)


def turnover_from_row(df):
    if "avg_turnover_63" in df.columns:
        return pd.to_numeric(df["avg_turnover_63"],errors="coerce")
    if "log_turnover_63" in df.columns:
        return np.expm1(pd.to_numeric(df["log_turnover_63"],errors="coerce"))
    return pd.Series(np.nan,index=df.index)


def slippage_bps(participation):
    # Explicit stress assumption, not an observed execution model:
    # 10 bps base + square-root impact scaled so 1% participation ~= 35 bps.
    p=np.maximum(np.asarray(participation,float),0.0)
    return 10.0 + 25.0*np.sqrt(p/0.01)


def execution_capacity(oos,cfg):
    q=oos[oos["selected_v941"]==True].copy()
    q["turnover_63"]=turnover_from_row(q)
    q=q[np.isfinite(q["turnover_63"]) & (q["turnover_63"]>0)].copy()
    k=int(cfg.get("selection_k",10))
    rows=[]
    for capital in CAPITALS:
        position=capital/k
        part=position/q["turnover_63"]
        slip=slippage_bps(part)
        rows.append({
            "portfolio_capital_inr":capital,
            "position_size_inr":position,
            "median_participation":float(np.nanmedian(part)),
            "p90_participation":float(np.nanquantile(part,.90)),
            "median_slippage_bps_assumption":float(np.nanmedian(slip)),
            "p90_slippage_bps_assumption":float(np.nanquantile(slip,.90)),
            "fraction_over_1pct_adv":float(np.nanmean(part>0.01)),
            "fraction_over_2pct_adv":float(np.nanmean(part>0.02)),
            "fraction_over_5pct_adv":float(np.nanmean(part>0.05)),
        })
    caps=pd.DataFrame(rows)

    limits=[]
    for lim in TURNOVER_PARTICIPATION_LIMITS:
        max_position=q["turnover_63"]*lim
        limits.append({
            "participation_limit":lim,
            "median_max_position_inr":float(max_position.median()),
            "p10_max_position_inr":float(max_position.quantile(.10)),
            "median_equal_weight_portfolio_capacity_inr":float(max_position.median()*k),
            "p10_equal_weight_portfolio_capacity_inr":float(max_position.quantile(.10)*k),
        })
    return caps,pd.DataFrame(limits)


def perturbation_stability(oos,chosen,cfg,n_sims=500,seed=20261001):
    rng=np.random.default_rng(seed)
    chosen=chosen.copy();chosen["date"]=pd.to_datetime(chosen["date"])
    cmap={pd.Timestamp(r.date):r for r in chosen.itertuples(index=False)}
    rows=[]
    for td,r in cmap.items():
        g=oos[oos["date"]==td].copy()
        if g.empty:continue
        spec=spec_from_row(r)
        base=selected_baseline(g,spec,cfg)
        if base.empty:continue
        base_set=set(base["symbol"].astype(str))
        if not base_set:continue

        js=[]; top1=[]; turnover=[]
        for _ in range(n_sims):
            z=g.copy()
            # Small independent score perturbations approximate estimation /
            # stale-data uncertainty without changing labels or retraining.
            z["p_cal"]=np.clip(
                pd.to_numeric(z["p_cal"],errors="coerce").to_numpy(float)
                * np.exp(rng.normal(0,0.01,len(z))),0,1
            )
            z["p_dd30_cal"]=np.clip(
                pd.to_numeric(z["p_dd30_cal"],errors="coerce").to_numpy(float)
                * np.exp(rng.normal(0,0.01,len(z))),0,1
            )
            z["model_dispersion"]=np.maximum(
                0,pd.to_numeric(z["model_dispersion"],errors="coerce").to_numpy(float)
                * np.exp(rng.normal(0,0.01,len(z)))
            )
            s=selected_baseline(z,spec,cfg)
            ss=set(s["symbol"].astype(str))
            if not ss:continue
            js.append(len(base_set&ss)/len(base_set|ss))
            top1.append(float(s.iloc[0]["symbol"]==base.iloc[0]["symbol"]))
            turnover.append(len(base_set-ss)/max(1,len(base_set)))

        rows.append({
            "date":td,
            "simulations":len(js),
            "mean_jaccard":float(np.mean(js)) if js else np.nan,
            "p05_jaccard":float(np.quantile(js,.05)) if js else np.nan,
            "top1_stability":float(np.mean(top1)) if top1 else np.nan,
            "mean_names_replaced_fraction":float(np.mean(turnover)) if turnover else np.nan,
        })
    return pd.DataFrame(rows)


def score_margin_stability(oos,cfg):
    rows=[];k=int(cfg.get("selection_k",10))
    for td,g in oos.groupby("date"):
        q=g.dropna(subset=["p_cal"]).sort_values("p_cal",ascending=False)
        if len(q)<k+5:continue
        p10=float(q.iloc[k-1]["p_cal"])
        p11=float(q.iloc[k]["p_cal"])
        rows.append({
            "date":pd.Timestamp(td),
            "p10":p10,"p11":p11,
            "absolute_margin":p10-p11,
            "relative_margin":(p10-p11)/max(abs(p10),1e-9),
        })
    return pd.DataFrame(rows)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--oos",required=True)
    ap.add_argument("--chosen931",required=True)
    ap.add_argument("--config",required=True)
    ap.add_argument("--output",required=True)
    args=ap.parse_args()

    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    oos=pd.read_parquet(args.oos);oos["date"]=pd.to_datetime(oos["date"])
    chosen=pd.read_csv(args.chosen931)
    cfg=json.load(open(args.config))

    caps,limits=execution_capacity(oos,cfg)
    pert=perturbation_stability(oos,chosen,cfg)
    margins=score_margin_stability(oos,cfg)

    caps.to_csv(out/"execution_capacity.csv",index=False)
    limits.to_csv(out/"capacity_by_participation_limit.csv",index=False)
    pert.to_csv(out/"selection_perturbation_stability.csv",index=False)
    margins.to_csv(out/"score_margin_stability.csv",index=False)

    summary={
        "model":"V9.9 stage-2 execution and stability",
        "execution_model_note":"slippage numbers are explicit stress assumptions, not broker fills",
        "selection_stability":{
            "folds":int(len(pert)),
            "mean_jaccard":float(pert["mean_jaccard"].mean()) if len(pert) else None,
            "worst_p05_jaccard":float(pert["p05_jaccard"].min()) if len(pert) else None,
            "mean_top1_stability":float(pert["top1_stability"].mean()) if len(pert) else None,
            "mean_names_replaced_fraction":float(pert["mean_names_replaced_fraction"].mean()) if len(pert) else None,
        },
        "score_margin":{
            "median_relative_top10_boundary_margin":float(margins["relative_margin"].median()) if len(margins) else None,
            "p10_relative_top10_boundary_margin":float(margins["relative_margin"].quantile(.10)) if len(margins) else None,
        },
        "capacity":caps.to_dict(orient="records"),
        "participation_limits":limits.to_dict(orient="records"),
        "production_change":False,
    }
    json.dump(summary,open(out/"summary.json","w"),indent=2,default=str)
    print(json.dumps(summary,indent=2,default=str))


if __name__=="__main__":
    main()
