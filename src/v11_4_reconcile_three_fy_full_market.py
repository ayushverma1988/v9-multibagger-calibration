"""Reconcile 12 independently parsed FY2021-FY2023 original NSE annual XBRL shards.

Strict 2023 15:30 IST publication clock, original stock universe identity,
original URL host, original content hash, 3 full fiscal annual periods. This
financial matrix is a data source, NOT a 3-year CAGR or production ML run.
"""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np
import pandas as pd
from v11_4_three_fy_source_inventory import candidates
from v11_4_strict_annual_numeric_features import fold_close
YEARS=(2021,2022,2023)
def reconcile(inventory,shard_dir,index,snapshot):
 original,inv_report=candidates(index,snapshot)
 if len(original["symbol"].unique())!=965:raise ValueError("Frozen original triple filing candidate count altered")
 groups=sorted(original["symbol"].unique())
 source=set(groups)
 triples=inventory.copy()
 if len(triples)!=3*len(groups) or set(triples["symbol"])!=source:
  raise ValueError("Historical original source inventory incomplete")
 root=Path(shard_dir)
 audits=list(root.rglob("three_FY_original_financial_numeric_pilot.json"))
 if len(audits)!=12:raise ValueError("Expected 12 independent NSE annual backfill shard audits")
 good=[];bad=[];seen=set();counts=[]
 for j,path in enumerate(audits):
  audit=json.loads(path.read_text())
  shard=int(audit["shard"])
  if shard in seen:raise ValueError("Duplicate source audit shard")
  if audit["shards"]!=12:raise ValueError("Unexpected independent partition count")
  seen.add(shard)
  expected=set(groups[shard::12])
  if audit["three_FY_companies_requested"]!=len(expected):
   raise ValueError("Unexpected original company count in partition")
  if audit["source_total_three_fy_companies"]!=965:
   raise ValueError("Original population count changed")
  ok_file=path.parent/"three_FY_FY2021_2023_strict_annual_numeric_RESEARCH_ONLY.csv"
  rejected_file=path.parent/"three_FY_FY2021_2023_original_source_rejections.csv"
  if not ok_file.is_file() or not rejected_file.is_file():raise ValueError("Missing original shard financial audit")
  ok=(pd.read_csv(ok_file) if audit["three_FY_source_complete_company_count"] else pd.DataFrame())
  re=(pd.read_csv(rejected_file) if audit["three_FY_source_rejected_company_count"] else pd.DataFrame())
  if len(ok)!=audit["three_FY_source_complete_company_count"] or len(re)!=audit["three_FY_source_rejected_company_count"]:
   raise ValueError("Source report not consistent with numeric and rejection CSVs")
  if len(ok)+len(re)!=len(expected):raise ValueError("Missing financial rejection or accepted symbol")
  actual=set(ok["symbol"]) if len(ok) else set()
  failed=set(re["symbol"]) if len(re) else set()
  if actual&failed or actual|failed!=expected:
   raise ValueError("Shard annual original company identities missing/duplicated")
  if len(ok):good.append(ok)
  if len(re):bad.append(re)
  counts.append(audit)
 if seen!=set(range(12)):raise ValueError("Original NSE annual shard missing")
 approved=pd.concat(good,ignore_index=True) if good else pd.DataFrame()
 rejected=pd.concat(bad,ignore_index=True) if bad else pd.DataFrame()
 if approved["symbol"].duplicated().any() or rejected["symbol"].duplicated().any():
  raise ValueError("Reused original fiscal source across independent shards")
 close=fold_close("2023-12-29")
 for fy in YEARS:
  rev=pd.to_numeric(approved[f"FY{fy}_revenue_INR"],errors="coerce")
  pat=pd.to_numeric(approved[f"FY{fy}_PAT_INR"],errors="coerce")
  if not rev.gt(0).all() or not pat.map(np.isfinite).all():
   raise ValueError("Nonpositive revenue or nonfinite profit in accepted original annual")
  if not approved[f"FY{fy}_sha256"].astype(str).str.fullmatch("[0-9a-f]{64}").all():
   raise ValueError("Invalid original NSE financial file SHA256")
  pub=pd.to_datetime(approved[f"FY{fy}_published_utc"],utc=True,errors="coerce")
  if pub.isna().any() or (pub>close).any():raise ValueError("Point-in-time annual publication leak")
  if not approved[f"FY{fy}_original_nse_xbrl_url"].str.match(r"^https://nsearchives\.nseindia\.com/[^?#]+\.xml$",case=False).all():
   raise ValueError("Nonoriginal original NSE XML host/URL")
  if not approved[f"FY{fy}_strict_fiscal_audit"].notna().all():
   raise ValueError("Missing fiscal-context annual provenance")
 for col,a,b in [
  ("FY2022_vs_2021_sales_yoy_fraction",2022,2021),
  ("FY2023_vs_2022_sales_yoy_fraction",2023,2022)]:
  actual=approved[f"FY{a}_revenue_INR"]/approved[f"FY{b}_revenue_INR"]-1
  if not np.allclose(actual,approved[col],rtol=1e-10,atol=1e-10):
   raise ValueError("Annual source YoY inconsistent")
 actual=np.sqrt(approved["FY2023_revenue_INR"]/approved["FY2021_revenue_INR"])-1
 if not np.allclose(actual,approved["FY2021_to_FY2023_sales_2yr_CAGR_fraction"],rtol=1e-10,atol=1e-10):
  raise ValueError("Two-year CAGR calculation inconsistent")
 if approved["actual_3yr_CAGR_calculated"].astype(str).str.lower().eq("true").any():
  raise ValueError("Two growth intervals incorrectly presented as 3-year CAGR")
 if approved["record_is_training_approved"].astype(str).str.lower().eq("true").any():
  raise ValueError("Unreviewed three-year numeric source promoted into old frozen model")
 _,universe_original=candidates(index,snapshot)
 universe_size=universe_original["original_stocks"]
 if len(approved)+len(rejected)!=965:
  raise ValueError("Full original triple annual population not accounted for")
 combined=pd.DataFrame({"symbol":sorted(set(snapshot.loc[
   pd.to_datetime(snapshot["date"]).eq(pd.Timestamp("2023-12-29")),"symbol"].str.upper()))})
 if len(combined)!=universe_size:raise ValueError("Original historical 2023 company universe changed")
 wanted=["symbol","source_filing_mode","FY2021_to_FY2023_sales_2yr_CAGR_fraction",
         "FY2022_vs_2021_sales_yoy_fraction","FY2023_vs_2022_sales_yoy_fraction",
         "FY2021_to_FY2023_profit_2yr_CAGR_fraction"]+[f"FY{y}_sha256" for y in YEARS]
 joined=combined.merge(approved[wanted],on="symbol",how="left",validate="1:1")
 if len(joined)!=universe_size:raise ValueError("Historical stock universe altered")
 joined["source_has_verified_exact_3_FY"]=joined["FY2021_sha256"].notna()
 coverage=len(approved)/universe_size
 report={
  "scope":"V11_4_3FY_ORIGINAL_NSE_2023_FOLD_COVERAGE_AUDIT_NO_PREDICTION_TRAINING",
  "historical_stock_universe_count":universe_size,
  "original_3_fy_link_company_count":965,
  "original_3_fy_fiscal_numeric_verified":len(approved),
  "original_3_fy_fiscal_numeric_rejected":len(rejected),
  "annual_3_FY_full_numeric_coverage_fraction":coverage,
  "annual_3_FY_full_numeric_coverage_percent":round(100*coverage,2),
  "original_FY2023_one_fold_70pct_threshold_met":bool(coverage>=.70),
  "twelve_original_year_end_folds_reconstructed":False,
  "all_18_original_folds_training_ready":False,
  "valid_two_year_revenue_CAGR_count":int(joined["FY2021_to_FY2023_sales_2yr_CAGR_fraction"].notna().sum()),
  "valid_two_year_profit_CAGR_count":int(joined["FY2021_to_FY2023_profit_2yr_CAGR_fraction"].notna().sum()),
  "true_3_year_CAGR_not_calculated_without_FY2020":True,
  "no_missing_company_dropped_from_original_fold":True,
  "no_imputed_revenue_PAT":True,
  "frozen_model_ranking_and_training_unchanged":True,
  "eligible_for_financial_enhanced_forecast":False
 }
 return approved,rejected,joined,report

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--inventory",required=True)
 p.add_argument("--shard-dir",required=True)
 p.add_argument("--annual-index",required=True)
 p.add_argument("--snapshot",required=True)
 p.add_argument("--out",required=True)
 a=p.parse_args()
 good,bad,joined,summary=reconcile(
  pd.read_csv(a.inventory,dtype=str).fillna(""),a.shard_dir,
  pd.read_csv(a.annual_index,dtype=str).fillna(""),
  pd.read_parquet(a.snapshot,columns=["date","symbol"]))
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 good.to_csv(out/"original_NSE_2021_2023_three_FY_STRICT_FINANCIAL_FACTS.csv",index=False)
 bad.to_csv(out/"original_NSE_2021_2023_strict_source_rejected_companies.csv",index=False)
 joined.to_parquet(out/"original_NSE_2023_frozen_universe_3FY_POINT_IN_TIME_RESEARCH.parquet",index=False)
 (out/"three_FY_full_numeric_recovery_2023_PIT_source_gate.json").write_text(json.dumps(summary,indent=2))
 print(json.dumps(summary,indent=2),flush=True)
if __name__=="__main__":main()
