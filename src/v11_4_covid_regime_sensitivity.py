"""Standalone V11.4 COVID regime sensitivity -- *never* silently omit failed folds.

Classification is established before looking at V11.4 outcomes:
* "market shock": 2020-02-20 to 2020-06-30 (crash/initial rebound)
* "pandemic extended": 2020-02-20 to 2021-12-31
* use actual frozen y6_mature_date six-month outcome completion, NOT simply
  the stock selection date. A selection before February 2020 whose six-month
  forward period crosses COVID is pandemic-affected.

Full-sample diagnostics always retained; COVID-excluded metrics are strictly
separate exploratory regime slices and DO NOT replace original hard 12-fold,
Jaccard 0.80/0.60 production approval checks.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd
import numpy as np

PERIODS={
 "covid_initial_crash_and_rebound":("2020-02-20","2020-06-30"),
 "covid_broad_pandemic":("2020-02-20","2021-12-31")
}
MIN_FOLDS=12
REQUIRED_MEAN_JACCARD=.80
REQUIRED_WORST_FOLD_P05=.60

def intersects(a,b,c,d):
    return pd.Timestamp(a)<=pd.Timestamp(d) and pd.Timestamp(b)>=pd.Timestamp(c)

def summarize(frame,stab):
    valid=frame[(frame["status"]=="tested") & frame["matured_selections"].eq(10)]
    if len(valid):
        precision=float(valid["six_month_double_hits"].sum()/valid["matured_selections"].sum())
        count=int(valid["matured_selections"].sum())
        hits=int(valid["six_month_double_hits"].sum())
    else:
        precision=None;count=0;hits=0
    s=stab[stab["date"].isin(valid["date"])]
    m=float(s["top10_jaccard"].mean()) if len(s) else None
    pp=s.groupby("date")["top10_jaccard"].quantile(.05) if len(s) else pd.Series(dtype=float)
    worst=float(pp.min()) if len(pp) else None
    return {
      "fully_clean_test_folds":len(valid),
      "selected":count,"doubled":hits,"six_month_precision":precision,
      "mean_retrained_Top10_Jaccard":m,"worst_fold_p05_Jaccard":worst,
      "twelve_fold_gate":len(valid)>=MIN_FOLDS,
      "mean_Jaccard_gate":bool(m is not None and m>=REQUIRED_MEAN_JACCARD),
      "worst_p05_gate":bool(worst is not None and worst>=REQUIRED_WORST_FOLD_P05),
      "production_approved":False
    }

def main():
    a=argparse.ArgumentParser()
    a.add_argument("--frozen-outcomes",required=True)
    a.add_argument("--fold-metrics",required=True)
    a.add_argument("--perturbations",required=True)
    a.add_argument("--output",required=True)
    args=a.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    hist=pd.read_parquet(args.frozen_outcomes,columns=["date","symbol","y6_mature_date"])
    hist["date"]=pd.to_datetime(hist["date"],errors="coerce").dt.normalize()
    hist["maturity"]=pd.to_datetime(hist["y6_mature_date"],utc=True,errors="coerce",format="mixed").dt.tz_convert(None)
    bounds=hist.groupby("date",as_index=False).agg(
       maturity_min=("maturity","min"),maturity_max=("maturity","max"),
       maturity_observed=("maturity","count"))
    folds=pd.read_csv(args.fold_metrics)
    folds["date"]=pd.to_datetime(folds["date"]).dt.normalize()
    stability=pd.read_csv(args.perturbations)
    stability["date"]=pd.to_datetime(stability["date"]).dt.normalize()
    if folds["date"].duplicated().any():raise SystemExit("Duplicated fold result")
    if folds["date"].nunique()!=18:raise SystemExit("Frozen 18fold history incomplete")
    if stability["seed"].nunique()!=12:raise SystemExit("Need 12 original independent perturbations")
    merged=folds.merge(bounds,on="date",how="left",validate="1:1")
    if merged["maturity_max"].isna().any():raise SystemExit("Missing original y6 maturity date")
    if (merged["maturity_max"]<merged["date"]).any():raise SystemExit("Outcome predates stock selection")
    allregimes={}
    for name,(lo,hi) in PERIODS.items():
        col=name+"_affected"
        merged[col]=merged.apply(
            lambda r:intersects(r["date"],r["maturity_max"],lo,hi),axis=1)
        excluded_dates=merged.loc[merged[col],"date"].tolist()
        unaffected=merged[~merged[col]]
        summary=summarize(unaffected,stability)
        summary["excluded_six_month_outcome_windows"]=[str(d.date()) for d in excluded_dates]
        summary["excluded_count"]=len(excluded_dates)
        summary["eligible_before_all_production_gates"]=False
        allregimes[name]=summary
    original=summarize(merged,stability)
    if original["fully_clean_test_folds"]<12:raise SystemExit("Original 12 fold gate not met")
    group=merged.copy()
    group["date"]=group["date"].dt.strftime("%Y-%m-%d")
    for c in ("maturity_min","maturity_max"):
        group[c]=pd.to_datetime(group[c]).dt.strftime("%Y-%m-%d")
    group.to_csv(out/"v11_4_historical_covid_impact_by_actual_label_horizon.csv",index=False)
    report={
      "scope":"COVID_CONDITIONAL_SENSITIVITY_ONLY_DO_NOT_SUPPRESS_FULL_RECORD",
      "COVID_regimes_predefined":PERIODS,
      "full_historical_diagnostic":original,
      "COVID_outcome_window_exclusion_sensitivities":allregimes,
      "2019_06_fold_not_covid_by_six_month_outcome":bool(not merged.loc[
          merged["date"].eq(pd.Timestamp("2019-06-28")),
          "covid_initial_crash_and_rebound_affected"].any()),
      "full_history_original_validation_gates_still_mandatory":True,
      "pandemic_assumption_can_be_changed_only_as_new_separately_labeled_sensitivity":True,
      "V11_4_standalone_only":True,
      "no_model_training_or_optimization_in_this_audit":True,
      "production_promoted":False}
    (out/"v11_4_regime_robustness_summary.json").write_text(json.dumps(report,indent=2,default=str))
    print(json.dumps(report,indent=2,default=str),flush=True)
    if not report["2019_06_fold_not_covid_by_six_month_outcome"]:
        raise SystemExit("June 2019 provenance unexpectedly overlaps COVID")
if __name__=="__main__":main()
