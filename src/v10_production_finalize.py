from __future__ import annotations

import argparse, json, os, subprocess
from pathlib import Path
import numpy as np
import pandas as pd

import calibrate_v9_4 as v94


def git_blob(path: str) -> str:
    return subprocess.check_output(["git","hash-object",path], text=True).strip()


def verify_frozen_files(cfg: dict):
    mismatches=[]
    for path,expected in cfg["frozen_blob_shas"].items():
        actual=git_blob(path)
        if actual!=expected:
            mismatches.append({"path":path,"expected":expected,"actual":actual})
    if mismatches:
        raise RuntimeError("Frozen production source mismatch: "+json.dumps(mismatches,indent=2))
    return True


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model-output",required=True)
    ap.add_argument("--monitor-config",default="config_v10_production_monitor.json")
    ap.add_argument("--output",default="outputs_v10_production_finalized")
    args=ap.parse_args()

    cfg=json.load(open(args.monitor_config))
    verify_frozen_files(cfg)

    src=Path(args.model_output)
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    summary=json.load(open(src/"summary.json"))
    cur=pd.read_csv(src/"current_selection.csv")
    cur["date"]=pd.to_datetime(cur["date"])

    if str(summary.get("model"))!="V9.4.1 integrity-corrected challenger":
        raise RuntimeError(f"Unexpected model output: {summary.get('model')}")
    if len(cur)<100:
        raise RuntimeError(f"Current eligible universe unexpectedly small: {len(cur)}")

    spec=dict(cfg["frozen_selector"])
    shares=(
        float(cfg["frozen_ladder_shares"]["p100"]),
        float(cfg["frozen_ladder_shares"]["p50"]),
        float(cfg["frozen_ladder_shares"]["p25"]),
    )
    if shares!=(1.0,0.0,0.0):
        raise RuntimeError(f"Production ladder not frozen to P100-only: {shares}")

    selected=v94.select_ladder_topk(cur,spec,shares,10)
    if len(selected)!=10:
        raise RuntimeError(f"Frozen production selector returned {len(selected)} rows, expected 10")

    selected=selected.copy().reset_index(drop=True)
    selected["production_rank"]=np.arange(1,11)
    selected["production_model"]=cfg["production_model"]
    selected["production_selector"]=spec["name"]
    selected["production_share_p100"]=1.0
    selected["production_share_p50"]=0.0
    selected["production_share_p25"]=0.0

    # The selector score is a percentile composite, not a probability.
    if "selection_score_v94" in selected:
        selected["production_selection_score"]=selected["selection_score_v94"]

    # Information-only liquidity capacity estimates; never used for ranking.
    avt=pd.to_numeric(selected["avg_turnover_63"],errors="coerce")
    selected["capacity_1pct_adv_inr"]=0.01*avt
    selected["capacity_2_5pct_adv_inr"]=0.025*avt
    selected["capacity_5pct_adv_inr"]=0.05*avt

    selected.to_csv(out/"production_top10.csv",index=False)

    compact_cols=[c for c in [
        "date","production_rank","symbol","isin","close","adj_close","target_2x",
        "p100_cal","p_cal","p_dd30_cal","p_dd30","model_dispersion",
        "production_selection_score","avg_turnover_63",
        "capacity_1pct_adv_inr","capacity_2_5pct_adv_inr","capacity_5pct_adv_inr",
        "production_model","production_selector","production_share_p100",
        "production_share_p50","production_share_p25"
    ] if c in selected.columns]
    selected[compact_cols].to_csv(out/"production_top10_compact.csv",index=False)

    allcur=cur.copy()
    allcur["production_selected"]=False
    allcur["production_rank"]=np.nan
    allcur["production_selection_score"]=np.nan
    keycols=["date","symbol"]
    idx_map={(pd.Timestamp(r.date),str(r.symbol)):i for i,r in allcur.iterrows()}
    for r in selected.itertuples(index=False):
        key=(pd.Timestamp(r.date),str(r.symbol))
        i=idx_map.get(key)
        if i is not None:
            allcur.at[i,"production_selected"]=True
            allcur.at[i,"production_rank"]=int(r.production_rank)
            if hasattr(r,"production_selection_score"):
                allcur.at[i,"production_selection_score"]=float(r.production_selection_score)
    allcur.to_csv(out/"current_universe_frozen_selector.csv",index=False)

    fwd=summary["forward_history"]["V9.4.1"]
    p100=summary["probability_ladder"]["p100"]
    manifest={
        "production_model":cfg["production_model"],
        "production_rules_frozen":True,
        "data_start":summary.get("data_start"),
        "data_end":summary.get("data_end"),
        "eligible_current_rows":int(len(cur)),
        "selected_rows":int(len(selected)),
        "selector":spec,
        "ladder_shares":{"p100":1.0,"p50":0.0,"p25":0.0},
        "refresh_fit_policy":"Fixed algorithms/rules may refit on newly available historical data; no selector/feature/gate search is allowed in production finalization.",
        "historical_monitor_metrics":{
            "folds":int(fwd.get("folds",0)),
            "mean_precision_100":float(fwd.get("mean_precision_100",np.nan)),
            "hit_fold_rate_100":float(fwd.get("hit_fold_rate_100",np.nan)),
            "mean_capped_lift_100":float(fwd.get("mean_capped_lift_100",np.nan)),
            "mean_dd30_rate":float(fwd.get("mean_dd30_rate",np.nan)),
            "p100_brier":float(p100.get("brier",np.nan)),
            "p100_calibration_slope":float(p100.get("calibration_slope",np.nan)),
        },
        "frozen_blob_shas_verified":True,
        "anti_overfit":cfg["anti_overfit"],
    }
    json.dump(manifest,open(out/"production_manifest.json","w"),indent=2,default=str)
    print(json.dumps(manifest,indent=2,default=str))


if __name__=="__main__":
    main()
