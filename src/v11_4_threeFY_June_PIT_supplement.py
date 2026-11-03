"""Date-correct June 2023/24/25 sources recycled from VERIFIED December originals.

A December source verification cannot automatically be used in preceding June.
Require each fiscal-year statement to have been originally published before
that June's 15:30 IST decision. Retain full June original security universe,
UNKNOWNs, actual source SHA256, mature six-month labels (research only).

No retraining, no six-month prices used as inputs, and no touch to 2025 Dec
holdout or Oct 2026 forward selections.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import pandas as pd
import numpy as np
from v11_4_strict_annual_numeric_features import fold_close

FOLDS={2023:("2023-06-30",(2021,2022,2023)),
       2024:("2024-06-28",(2022,2023,2024)),
       2025:("2025-06-30",(2023,2024,2025))}
MATURITY_EVAL=pd.Timestamp("2026-10-09T00:00:00Z")
def reopen_june_source(financial,market,year):
 fold,yrs=FOLDS[year]
 q=financial.copy()
 q["symbol"]=q["symbol"].astype(str).str.upper().str.strip()
 if q["symbol"].duplicated().any():raise ValueError("Original 3-FY fiscal source reuses company")
 known=fold_close(fold)
 good=pd.Series(True,index=q.index)
 checks={}
 for fy in yrs:
  pubfield=f"FY{fy}_published_utc"
  cols=[x for x in q.columns if x.lower() in (
   f"fy{fy}_sha256",f"fy{fy}_source_sha256")]
  if len(cols)!=1 or pubfield not in q:raise ValueError("Original immutable fiscal publication/hash absent")
  field=cols[0]
  dates=pd.to_datetime(q[pubfield],utc=True,errors="coerce",format="mixed")
  hashes=q[field].astype(str).str.fullmatch("[0-9a-f]{64}")
  eligible=dates.notna()&dates.le(known)&hashes
  good &=eligible
  checks[str(fy)]={"orig_source_before_June_1530_IST":int(eligible.sum()),
                   "late_or_invalid_financial_source":int((~eligible).sum())}
 # Retain June stock identities untouched to avoid survivorship filtering.
 m=market.copy()
 m["date"]=pd.to_datetime(m["date"],errors="raise").dt.normalize()
 m["symbol"]=m["symbol"].astype(str).str.upper().str.strip()
 m=m[m["date"].eq(pd.Timestamp(fold))].copy()
 if len(m)<1000 or m["symbol"].duplicated().any():raise ValueError("Original June frozen stock universe invalid")
 fields=["symbol"]+[f"FY{x}_{metric}_INR" for x in yrs for metric in ("revenue","PAT")]
 if not set(fields).issubset(q):raise ValueError("Original annual financial facts missing from 3FY source")
 for col in fields[1:]:
  q[col]=pd.to_numeric(q[col],errors="coerce")
 if q[fields[1:]].isna().any().any():raise ValueError("Original 3FY source missing verified numeric")
 selected=q.loc[good].copy()
 joined=m.merge(selected[fields],on="symbol",how="left",validate="1:1")
 if len(joined)!=len(m):raise ValueError("Original June stock identity changed")
 joined["verified_3FY_before_actual_June_fold_close"]=joined[fields[1]].notna()
 # Research labels are evaluated only after their actual complete 6-month maturity.
 mature=pd.to_datetime(joined["y6_mature_date"],utc=True,errors="coerce",format="mixed")
 eligible=joined["verified_3FY_before_actual_June_fold_close"]&\
  joined["integrity_y6_clean"].eq(True)&joined["y6"].isin([0,1])&\
  joined["close"].between(20,2000)&joined["avg_turnover_63"].gt(0)&\
  mature.notna()&mature.lt(MATURITY_EVAL)
 joined["source_eligible_mature_six_month_research_label"]=eligible
 report={
  "scope":"V11_4_ORIGINAL_JUNE_HISTORIC_THREE_FY_FACTS_STRICT_SOURCE_CLOCK",
  "historical_fold_IST":fold,"original_market_stock_universe":len(m),
  "original_Dec_full_3FY_company_numerics_available":len(q),
  "all_3FY_original_sources_preJune_filing_cutoff":int(good.sum()),
  "all_3FY_sources_with_original_June_stock_identity":int(joined["verified_3FY_before_actual_June_fold_close"].sum()),
  "June_3FY_actual_verified_numeric_coverage_pct":round(100*joined["verified_3FY_before_actual_June_fold_close"].mean(),2),
  "eligible_matured_6month_twoX_labels":int(eligible.sum()),
  "eligible_matured_actual_twoX_events":int(joined.loc[eligible,"y6"].astype(int).sum()),
  "fiscal_original_source_checks":checks,
  "asof_June_no_future_source":True,"original_2025_Dec_out_of_time_test_not_reused":True,
  "June_unique_half_year_decision_labels_not_labeled_as_independent_annual_financial_folds":True,
  "no_original_frozen_2026_V11_4_model_change":True,
  "new_financial_ML_training_or_calibration_done":False,
  "original_2025_Dec_test_evaluated_in_this_run":False}
 return joined,report
def main():
 p=argparse.ArgumentParser()
 for y in FOLDS:p.add_argument(f"--financial-{y}",required=True)
 p.add_argument("--snapshot",required=True);p.add_argument("--out",required=True)
 args=p.parse_args()
 m=pd.read_parquet(args.snapshot,columns=[
  "date","symbol","close","avg_turnover_63","y6","y6_mature_date","integrity_y6_clean"])
 root=Path(args.out);root.mkdir(parents=True,exist_ok=True)
 reports=[]
 for year in FOLDS:
  stock,report=reopen_june_source(pd.read_csv(getattr(args,f"financial_{year}")),m,year)
  stock.to_parquet(root/f"original_NSE_June_{year}_verified_3FY_asof_1530_IST_PRIVATE.parquet",index=False)
  reports.append(report)
 summary={
  "scope":"THREE_STRICT_3FY_ORIGINAL_FISCAL_JUNE_MID_YEAR_PIT_RESEARCH_FOLDS",
  "original_June_folds":len(reports),
  "all_3FY_preJune_source_unique_stock_matches":sum(z["all_3FY_sources_with_original_June_stock_identity"] for z in reports),
  "all_eligible_matured_research_labels":sum(z["eligible_matured_6month_twoX_labels"] for z in reports),
  "all_eligible_actual_6month_doublers":sum(z["eligible_matured_actual_twoX_events"] for z in reports),
  "already_evaluated_2025_Dec_holdout_untouched":True,
  "no_fit_calibration_or_holdout_rerank":True,
  "fold_reports":reports}
 (root/"June_2023_2024_2025_3FY_source_clock_mature_label_SUMMARY.json").write_text(json.dumps(summary,indent=2))
 print(json.dumps(summary,indent=2),flush=True)
if __name__=="__main__":main()
