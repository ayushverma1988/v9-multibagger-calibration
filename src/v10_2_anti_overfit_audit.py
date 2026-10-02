from __future__ import annotations

import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd


def safe_mean(s):
    x=pd.to_numeric(s,errors="coerce").replace([np.inf,-np.inf],np.nan).dropna()
    return float(x.mean()) if len(x) else np.nan


def capped_lift(s):
    x=pd.to_numeric(s,errors="coerce").replace([np.inf,-np.inf],np.nan).dropna()
    return float(np.clip(x,0,10).mean()) if len(x) else np.nan


def agg(t):
    return {
        "folds":int(len(t)),
        "mean_precision_2x":safe_mean(t["precision_2x"]),
        "median_precision_2x":float(pd.to_numeric(t["precision_2x"],errors="coerce").median()),
        "mean_capped_lift_2x":capped_lift(t["lift_2x"]),
        "hit_fold_rate":safe_mean(t["hit"]),
        "mean_dd30_rate":safe_mean(t["dd30_rate"]),
    }


def paired_bootstrap(c,b,n=20000,seed=20261002):
    x=c.merge(b,on="date",suffixes=("_cand","_base"))
    rng=np.random.default_rng(seed)
    fields=["precision_2x","hit","dd30_rate"]
    out={}
    for f in fields:
        d=(pd.to_numeric(x[f+"_cand"],errors="coerce")-pd.to_numeric(x[f+"_base"],errors="coerce")).to_numpy(float)
        d=d[np.isfinite(d)]
        if not len(d):continue
        means=np.array([rng.choice(d,size=len(d),replace=True).mean() for _ in range(n)])
        out[f+"_delta"]={
            "mean":float(np.mean(means)),
            "p05":float(np.quantile(means,.05)),
            "p50":float(np.quantile(means,.50)),
            "p95":float(np.quantile(means,.95)),
        }
    return out


def loo_ratios(c,b):
    dates=sorted(set(c["date"])&set(b["date"]))
    rows=[]
    for td in dates:
        ca=agg(c[c["date"]!=td]);ba=agg(b[b["date"]!=td])
        rows.append({
            "left_out":str(pd.Timestamp(td).date()),
            "precision_ratio":ca["mean_precision_2x"]/ba["mean_precision_2x"] if ba["mean_precision_2x"]>0 else np.nan,
            "lift_ratio":ca["mean_capped_lift_2x"]/ba["mean_capped_lift_2x"] if ba["mean_capped_lift_2x"]>0 else np.nan,
            "hit_ratio":ca["hit_fold_rate"]/ba["hit_fold_rate"] if ba["hit_fold_rate"]>0 else np.nan,
            "dd_delta":ca["mean_dd30_rate"]-ba["mean_dd30_rate"],
        })
    return pd.DataFrame(rows)


def recent_holdout(c,b,start="2023-01-01"):
    s=pd.Timestamp(start)
    ca=agg(c[c["date"]>=s]);ba=agg(b[b["date"]>=s])
    return {
        "start":str(s.date()),
        "candidate":ca,
        "baseline":ba,
        "precision_ratio":ca["mean_precision_2x"]/ba["mean_precision_2x"] if ba["mean_precision_2x"]>0 else np.nan,
        "lift_ratio":ca["mean_capped_lift_2x"]/ba["mean_capped_lift_2x"] if ba["mean_capped_lift_2x"]>0 else np.nan,
        "hit_ratio":ca["hit_fold_rate"]/ba["hit_fold_rate"] if ba["hit_fold_rate"]>0 else np.nan,
        "dd_delta":ca["mean_dd30_rate"]-ba["mean_dd30_rate"],
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--candidate-metrics",required=True)
    ap.add_argument("--baseline-metrics",required=True)
    ap.add_argument("--candidate-stability",required=True)
    ap.add_argument("--output",required=True)
    ap.add_argument("--recent-start",default="2023-01-01")
    args=ap.parse_args()

    c=pd.read_csv(args.candidate_metrics);b=pd.read_csv(args.baseline_metrics);s=pd.read_csv(args.candidate_stability)
    for x in [c,b,s]:
        x["date"]=pd.to_datetime(x["date"])

    ca=agg(c);ba=agg(b)
    loo=loo_ratios(c,b)
    recent=recent_holdout(c,b,args.recent_start)
    boot=paired_bootstrap(c,b)

    full_alpha=(
        ca["mean_precision_2x"]>=.95*ba["mean_precision_2x"]
        and ca["median_precision_2x"]>=.95*ba["median_precision_2x"]
        and ca["mean_capped_lift_2x"]>=.95*ba["mean_capped_lift_2x"]
        and ca["hit_fold_rate"]>=.95*ba["hit_fold_rate"]
    )
    full_stability=(
        safe_mean(s["mean_jaccard"])>=.80
        and float(pd.to_numeric(s["p05_jaccard"],errors="coerce").min())>=.60
        and safe_mean(s["top1_stability"])>=.70
    )

    gates={
        "full_alpha_retention":bool(full_alpha),
        "full_stability":bool(full_stability),
        "dd_not_worse_3pp":bool(ca["mean_dd30_rate"]<=ba["mean_dd30_rate"]+.03),
        "loo_precision_ratio_min_90pct":bool(loo["precision_ratio"].min()>=.90),
        "loo_lift_ratio_min_90pct":bool(loo["lift_ratio"].min()>=.90),
        "loo_hit_ratio_min_90pct":bool(loo["hit_ratio"].min()>=.90),
        "loo_dd_delta_max_4pp":bool(loo["dd_delta"].max()<=.04),
        "recent_precision_ratio_min_80pct":bool(recent["precision_ratio"]>=.80),
        "recent_lift_ratio_min_80pct":bool(recent["lift_ratio"]>=.80),
        "recent_hit_ratio_min_80pct":bool(recent["hit_ratio"]>=.80),
        "recent_dd_delta_max_5pp":bool(recent["dd_delta"]<=.05),
    }

    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    loo.to_csv(out/"leave_one_fold_out.csv",index=False)
    summary={
        "model":"V10.2 anti-overfit acceptance audit",
        "candidate":ca,"baseline":ba,
        "stability":{
            "mean_jaccard":safe_mean(s["mean_jaccard"]),
            "worst_p05_jaccard":float(pd.to_numeric(s["p05_jaccard"],errors="coerce").min()),
            "mean_top1_stability":safe_mean(s["top1_stability"]),
        },
        "recent_holdout":recent,
        "paired_bootstrap":boot,
        "loo":{
            "min_precision_ratio":float(loo["precision_ratio"].min()),
            "min_lift_ratio":float(loo["lift_ratio"].min()),
            "min_hit_ratio":float(loo["hit_ratio"].min()),
            "max_dd_delta":float(loo["dd_delta"].max()),
        },
        "gates":gates,
        "acceptance_gate":bool(all(gates.values())),
        "note":"audit-only; no selector tuning or parameter choice is performed here",
    }
    json.dump(summary,open(out/"summary.json","w"),indent=2,default=str)
    print(json.dumps(summary,indent=2,default=str))

if __name__=="__main__":
    main()
