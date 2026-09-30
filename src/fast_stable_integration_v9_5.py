from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import calibrate_v9_5 as v95
import fund_calibration_stable as fcal


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--oos", required=True)
    ap.add_argument("--fund-oos", required=True)
    ap.add_argument("--stable-preds", required=True)
    ap.add_argument("--chosen931", required=True)
    ap.add_argument("--current", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--config", default="config_v9_5.json")
    ap.add_argument("--output", default="outputs_v9_5_stable_fast")
    args=ap.parse_args()

    cfg=json.load(open(args.config))
    outdir=Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    oos=pd.read_parquet(args.oos)
    oos["date"]=pd.to_datetime(oos["date"])

    stable=pd.read_parquet(args.stable_preds)
    stable["date"]=pd.to_datetime(stable["date"])
    pcol="p_stab_p_fund_rank_platt_anchor_r8"
    stable=stable[["date","symbol",pcol]].rename(columns={pcol:"p_fund_cal_stable"})

    if "p_fund_cal" in oos.columns:
        oos=oos.drop(columns=["p_fund_cal"])
    oos=oos.merge(stable,on=["date","symbol"],how="left")
    oos=oos.rename(columns={"p_fund_cal_stable":"p_fund_cal"})

    chosen931=pd.read_csv(args.chosen931)
    chosen931["date"]=pd.to_datetime(chosen931["date"])
    oos2, chosen95=v95.forward_v95(oos,chosen931,cfg)
    comp=v95.comparison_by_fold(oos2,cfg)

    old_summary=json.load(open(args.summary))
    spec=old_summary["production_risk_config"]
    production_weight, search, baseline, gate=v95.optimize_weight(oos2,spec,cfg)

    fund_oos=pd.read_parquet(args.fund_oos)
    fund_oos["date"]=pd.to_datetime(fund_oos["date"])
    if "p_fund_cal" in fund_oos.columns:
        fund_oos=fund_oos.drop(columns=["p_fund_cal"])
    fund_oos=fund_oos.merge(stable,on=["date","symbol"],how="left")
    fund_oos=fund_oos.rename(columns={"p_fund_cal_stable":"p_fund_cal"})

    current=pd.read_csv(args.current)
    current["date"]=pd.to_datetime(current["date"])
    raw=current["p_fund_raw"].to_numpy(float)
    ok=np.isfinite(raw)
    current["p_fund_cal"]=np.nan
    if ok.any():
        current.loc[ok,"p_fund_cal"]=fcal.fit_current(
            fund_oos,
            raw[ok],
            cfg,
            target="y6",
            pcol="p_fund_raw",
        )

    if "p_dd30_cal" not in current.columns:
        current["p_dd30_cal"]=current["p_dd30"]

    cov=v95.candidate_coverage(current,spec,cfg)
    min_cov=float(cfg.get("fund_min_candidate_coverage",0.70))
    applied=(
        production_weight
        if production_weight>0
        and gate["passed"]
        and cov["coverage"]>=min_cov
        and cov["covered"]>=int(cfg.get("selection_k",10))
        else 0.0
    )

    sel=v95.select_blended(current,spec,applied,cfg)
    if sel.empty:
        applied=0.0
        sel=v95.select_blended(current,spec,0.0,cfg)

    current["selected_v95_stable"]=False
    current["selection_rank_v95_stable"]=np.nan
    current["selection_score_v95_stable"]=np.nan
    for rank,idx in enumerate(sel.index.tolist(),start=1):
        current.loc[idx,"selected_v95_stable"]=True
        current.loc[idx,"selection_rank_v95_stable"]=rank
        score=sel.loc[idx,"selection_score_v95"] if "selection_score_v95" in sel.columns else sel.loc[idx,"selection_score"]
        current.loc[idx,"selection_score_v95_stable"]=float(score)

    current["production_fund_weight_stable"]=applied
    current["production_fund_gate_stable"]=bool(gate["passed"])
    current["production_candidate_coverage_stable"]=cov["coverage"]

    s95=v95.summarize_strategy(comp,"V9.5")
    sbase=v95.summarize_strategy(comp,"V9.4.1_V9.3.1")
    retention={}
    for f in ["mean_precision_2x","mean_capped_lift_2x","hit_fold_rate","median_precision_2x"]:
        a=s95.get(f,np.nan); b=sbase.get(f,np.nan)
        retention[f]=float(a/b) if np.isfinite(a) and np.isfinite(b) and b>0 else None

    out={
        "calibration_policy":fcal.POLICY_NAME,
        "fundamental_gate":gate,
        "requested_weight":production_weight,
        "applied_current_weight":applied,
        "current_candidate_coverage":cov,
        "forward_history":{"V9.5":s95,"baseline":sbase,"alpha_retention_ratios":retention},
        "baseline_search":baseline,
        "production_risk_config":spec,
    }
    json.dump(out,open(outdir/"summary.json","w"),indent=2,default=str)
    search.to_csv(outdir/"fund_weight_search.csv",index=False)
    chosen95.to_csv(outdir/"chosen_v95_weight_by_fold.csv",index=False)
    comp.to_csv(outdir/"selection_metrics_by_fold.csv",index=False)
    current.to_csv(outdir/"current_selection.csv",index=False)
    current[current["selected_v95_stable"]].sort_values("selection_rank_v95_stable").head(50).to_csv(outdir/"current_top50.csv",index=False)

    print(json.dumps(out,indent=2,default=str))
    cols=["selection_rank_v95_stable","symbol","close","p_cal","p_fund_cal","p_dd30","production_fund_weight_stable","selection_score_v95_stable"]
    cols=[c for c in cols if c in current.columns]
    print(current[current["selected_v95_stable"]].sort_values("selection_rank_v95_stable")[cols].head(20).to_string(index=False))


if __name__=="__main__":
    main()
