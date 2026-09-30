from __future__ import annotations

import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd


def fold_table(oos: pd.DataFrame, k: int = 10):
    rows=[]
    for td,g in oos.groupby("date",sort=True):
        q=g.dropna(subset=["y6","dd30_6m","p_cal"]).copy()
        s=q[q["selected_v941"]==True].copy() if "selected_v941" in q.columns else pd.DataFrame()
        if len(s)!=k or len(q)<k:
            continue
        br=float(q["y6"].mean())
        pr=float(s["y6"].mean())
        lift=pr/br if br>0 else np.nan
        rows.append({
            "date":pd.Timestamp(td),
            "precision_2x":pr,
            "capped_lift_2x":min(lift,10.0) if np.isfinite(lift) else np.nan,
            "hit":float(pr>0),
            "dd30_rate":float(s["dd30_6m"].mean()),
            "base_rate":br,
        })
    return pd.DataFrame(rows)


def summarize(t: pd.DataFrame):
    return {
        "folds":int(len(t)),
        "mean_precision_2x":float(t["precision_2x"].mean()),
        "median_precision_2x":float(t["precision_2x"].median()),
        "mean_capped_lift_2x":float(t["capped_lift_2x"].dropna().mean()),
        "hit_fold_rate":float(t["hit"].mean()),
        "mean_dd30_rate":float(t["dd30_rate"].mean()),
    }


def bootstrap(t: pd.DataFrame, n: int, seed: int):
    rng=np.random.default_rng(seed)
    vals=t[["precision_2x","capped_lift_2x","hit","dd30_rate"]].to_numpy(float)
    out=[]
    m=len(vals)
    for _ in range(n):
        z=vals[rng.integers(0,m,size=m)]
        out.append(np.nanmean(z,axis=0))
    a=np.asarray(out)
    names=["mean_precision_2x","mean_capped_lift_2x","hit_fold_rate","mean_dd30_rate"]
    res={}
    for i,nm in enumerate(names):
        res[nm]={
            "mean":float(np.mean(a[:,i])),
            "p05":float(np.quantile(a[:,i],.05)),
            "p50":float(np.quantile(a[:,i],.50)),
            "p95":float(np.quantile(a[:,i],.95)),
        }
    return res


def leave_one_fold_out(t: pd.DataFrame):
    full=summarize(t)
    rows=[]
    for i,r in t.reset_index(drop=True).iterrows():
        z=t.reset_index(drop=True).drop(index=i)
        s=summarize(z)
        rows.append({
            "removed_date":r["date"],
            **{f"{k}_lofo":v for k,v in s.items() if k!="folds"},
            "precision_delta":s["mean_precision_2x"]-full["mean_precision_2x"],
            "lift_delta":s["mean_capped_lift_2x"]-full["mean_capped_lift_2x"],
            "hit_delta":s["hit_fold_rate"]-full["hit_fold_rate"],
            "dd_delta":s["mean_dd30_rate"]-full["mean_dd30_rate"],
        })
    return pd.DataFrame(rows)


def liquidity_report(oos: pd.DataFrame):
    q=oos[oos.get("selected_v941",False)==True].copy()
    if "log_turnover_63" not in q.columns:
        return {}, pd.DataFrame()
    q["avg_turnover_63"]=np.exp(pd.to_numeric(q["log_turnover_63"],errors="coerce"))
    thresholds=[5e6,10e6,20e6,50e6,100e6]
    rows=[]
    for th in thresholds:
        low=q["avg_turnover_63"]<th
        rows.append({
            "turnover_floor":th,
            "selected_rows":int(len(q)),
            "below_floor_rows":int(low.sum()),
            "below_floor_fraction":float(low.mean()) if len(q) else np.nan,
            "precision_if_low_liquidity_forced_miss":float(
                np.where(low,0,q["y6"].fillna(0)).mean()
            ) if len(q) else np.nan,
            "dd30_if_low_liquidity_forced_bad":float(
                np.where(low,1,q["dd30_6m"].fillna(0)).mean()
            ) if len(q) else np.nan,
        })
    return {
        "selected_turnover_p10":float(q["avg_turnover_63"].quantile(.10)),
        "selected_turnover_p25":float(q["avg_turnover_63"].quantile(.25)),
        "selected_turnover_median":float(q["avg_turnover_63"].median()),
        "selected_turnover_p75":float(q["avg_turnover_63"].quantile(.75)),
    }, pd.DataFrame(rows)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--oos",required=True)
    ap.add_argument("--snapshot")
    ap.add_argument("--output",required=True)
    ap.add_argument("--bootstrap",type=int,default=10000)
    args=ap.parse_args()

    outdir=Path(args.output); outdir.mkdir(parents=True,exist_ok=True)
    oos=pd.read_parquet(args.oos)
    oos["date"]=pd.to_datetime(oos["date"])
    if args.snapshot and "log_turnover_63" not in oos.columns:
        snap=pd.read_parquet(args.snapshot)
        snap["date"]=pd.to_datetime(snap["date"])
        keep=["date","symbol"]+[x for x in ["log_turnover_63","turnover_accel"] if x in snap.columns]
        oos=oos.merge(snap[keep],on=["date","symbol"],how="left")

    ft=fold_table(oos)
    ft.to_csv(outdir/"fold_metrics.csv",index=False)
    lofo=leave_one_fold_out(ft)
    lofo.to_csv(outdir/"leave_one_fold_out.csv",index=False)
    liq,liq_table=liquidity_report(oos)
    liq_table.to_csv(outdir/"liquidity_stress.csv",index=False)

    summary={
        "model":"V9.9 robustness/liquidity diagnostic",
        "baseline":"V9.4.1 frozen selector",
        "fold_summary":summarize(ft),
        "bootstrap_90pct_interval":bootstrap(ft,args.bootstrap,20261001),
        "lofo":{
            "worst_precision_delta":float(lofo["precision_delta"].min()) if len(lofo) else None,
            "best_precision_delta":float(lofo["precision_delta"].max()) if len(lofo) else None,
            "worst_lift_delta":float(lofo["lift_delta"].min()) if len(lofo) else None,
            "max_dd_increase":float(lofo["dd_delta"].max()) if len(lofo) else None,
        },
        "liquidity":liq,
        "interpretation_policy":"diagnostic only; no production selector change from this run",
    }
    json.dump(summary,open(outdir/"summary.json","w"),indent=2,default=str)
    print(json.dumps(summary,indent=2,default=str))


if __name__=="__main__":
    main()
