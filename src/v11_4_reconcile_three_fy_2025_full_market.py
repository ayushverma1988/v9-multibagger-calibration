"""Reconcile complete original NSE 2025 fold three-year annual and integrated YTD.

One historical market-date only; 12 fiscal data shards + all original 1314
securities, 3 fiscal years FY2023/24/25. No 3Y CAGR, no future labels and
no alteration of frozen V11.4 or its source 22 predictors.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from v11_4_strict_annual_numeric_features import fold_close
FY=(2023,2024,2025)
def join(shards,index,snapshot):
 catalog=pd.read_csv(index,dtype=str).fillna("")
 expected=sorted(catalog["symbol"].astype(str).str.upper().unique())
 if len(expected)!=1029:raise ValueError("2025 original three FY source inventory changed")
 if len(catalog)!=3*len(expected):raise ValueError("Three-FY original NSE source catalog incomplete")
 items=list(Path(shards).rglob("three_FY_2025_strict_numeric_quality_summary.json"))
 if len(items)!=12:raise ValueError("Not exactly 12 original independent annual source partitions")
 all_good=[];all_bad=[];seen=set()
 for p in items:
  meta=json.loads(p.read_text());shard=int(meta["shard"])
  if shard in seen:raise ValueError("Duplicate 2025 fiscal original source partition")
  seen.add(shard)
  if meta["shards"]!=12 or meta["all_company_2025_three_FY_link_candidates"]!=1029:
   raise ValueError("2025 historic fiscal original partition shape inconsistent")
  wanted=set(expected[shard::12])
  if int(meta["sampled"])!=len(wanted):raise ValueError("Requested 2025 fiscal company shard wrong")
  good=p.parent/"three_FY2023_2025_original_NSE_integrated_STRICT_NUMERICS.csv"
  bad=p.parent/"three_FY2023_2025_original_NSE_strict_rejections.csv"
  if not good.is_file() or not bad.is_file():raise ValueError("Source result and rejection manifests missing")
  valid=pd.read_csv(good) if meta["three_year_annual_numeric_verified"] else pd.DataFrame()
  re=pd.read_csv(bad) if meta["three_year_original_source_rejected"] else pd.DataFrame()
  if len(valid)!=meta["three_year_annual_numeric_verified"] or len(re)!=meta["three_year_original_source_rejected"]:
   raise ValueError("2025 original source summary count incorrect")
  a=set(valid["symbol"]) if len(valid) else set()
  b=set(re["symbol"]) if len(re) else set()
  if a&b or a|b!=wanted:raise ValueError("Source/rejection companies do not exactly cover 2025 original shard")
  if len(valid):all_good.append(valid)
  if len(re):all_bad.append(re)
 if seen!=set(range(12)):raise ValueError("Missing 2025 independent source shard")
 good=pd.concat(all_good,ignore_index=True)
 rejected=pd.concat(all_bad,ignore_index=True) if all_bad else pd.DataFrame()
 if good["symbol"].duplicated().any() or (len(rejected) and rejected["symbol"].duplicated().any()):
  raise ValueError("Duplicate original FY2025 financial issuer")
 if len(good)+len(rejected)!=1029:raise ValueError("2025 fiscal source backfill omitted company")
 cutoff=fold_close("2025-12-31")
 for yr in FY:
  if not good[f"FY{yr}_source_SHA256"].astype(str).str.fullmatch("[0-9a-f]{64}").all():
   raise ValueError("Invalid FY source SHA256")
  pub=pd.to_datetime(good[f"FY{yr}_published_utc"],utc=True,errors="coerce")
  if pub.isna().any() or (pub>cutoff).any():raise ValueError("Future filing leaked into 2025 original annual research")
  if not good[f"FY{yr}_source_URL"].astype(str).str.match(r"^https://nsearchives\.nseindia\.com/[^?#]+\.xml$",case=False).all():
   raise ValueError("Third-party unaudited 2025 fiscal source sneaked in")
  if not pd.to_numeric(good[f"FY{yr}_revenue_INR"],errors="coerce").gt(0).all():
   raise ValueError("Nonpositive annual financial sales baseline")
  if not pd.to_numeric(good[f"FY{yr}_PAT_INR"],errors="coerce").map(np.isfinite).all():
   raise ValueError("Nonfinite annual PAT")
 for name in ("revenue","pat"):
  contexts=good["FY2025_annual_fact_context"].map(lambda x:json.loads(x).get(name,{}).get("context"))
  if not contexts.eq("FourD").all():raise ValueError("2025 quarter-only OneD used as FY2025 annual")
 cmp=np.sqrt(good["FY2025_revenue_INR"]/good["FY2023_revenue_INR"])-1
 if not np.allclose(cmp,good["revenue_2023_to_2025_2yr_CAGR"],rtol=1e-10,atol=1e-10):
  raise ValueError("Original 2-year revenue CAGR miscalculated")
 if good["valid_for_3year_sales_CAGR"].astype(str).str.lower().eq("true").any():
  raise ValueError("Two annual intervals falsely promoted to 3-year growth")
 if good["financial_training_approved"].astype(str).str.lower().eq("true").any():
  raise ValueError("Financials entered V11.4 without independent validation")
 hist=snapshot.copy();hist["date"]=pd.to_datetime(hist["date"]).dt.normalize()
 stocks=sorted(set(hist.loc[hist["date"].eq(pd.Timestamp("2025-12-31")),"symbol"].astype(str).str.upper().str.strip()))
 if len(stocks)!=1314:raise ValueError("Original 2025 1314 company source universe mutated")
 cols=["symbol","source_mode","revenue_2023_to_2025_2yr_CAGR","revenue_FY2024_yoy","revenue_FY2025_yoy",
       "profit_2023_to_2025_2yr_CAGR","profit_FY2025_yoy",
       "FY2023_source_SHA256","FY2024_source_SHA256","FY2025_source_SHA256"]
 frame=pd.DataFrame({"date":"2025-12-31","symbol":stocks}).merge(good[cols],on="symbol",how="left",validate="1:1")
 if len(frame)!=1314:raise ValueError("2025 original historical securities dropped")
 frame["verified_original_NSE_three_FY_2023_2025"]=frame["FY2023_source_SHA256"].notna()
 report={
  "scope":"V11_4_FY2025_THREE_FISCAL_YEAR_VERIFIED_ORIGINAL_FINANCIAL_SOURCE_NOT_TRAINING",
  "original_FY2025_historical_stock_universe":1314,
  "FY2023_FY2024_FY2025_source_eligible_companies":1029,
  "FY2023_FY2024_FY2025_three_year_company_numerics_verified":len(good),
  "FY2023_FY2024_FY2025_original_source_rejected":len(rejected),
  "three_fy_original_2025_universe_coverage_fraction":len(good)/1314,
  "three_fy_original_2025_universe_coverage_pct":round(100*len(good)/1314,2),
  "one_original_2025_fold_70pct_source_coverage_passed":bool(len(good)/1314>=.70),
  "FY2023_FY2025_two_year_revenue_CAGR_available":int(frame["revenue_2023_to_2025_2yr_CAGR"].notna().sum()),
  "FY2023_FY2025_two_year_profit_CAGR_available":int(frame["profit_2023_to_2025_2yr_CAGR"].notna().sum()),
  "integrated_2025_FourD_only_not_quarter_OneD":True,
  "2023_and_2025_two_fold_not_sufficient_for_calibration":True,
  "3FY_financial_predictive_model_trained":False,
  "all_original_company_rows_retained":True,
  "missing_data_kept_unknown":True,
  "original_V11_4_frozen_model_unchanged":True}
 return good,rejected,frame,report
def main():
 p=argparse.ArgumentParser()
 p.add_argument("--shards",required=True);p.add_argument("--index",required=True)
 p.add_argument("--snapshot",required=True);p.add_argument("--out",required=True)
 a=p.parse_args()
 good,bad,frame,report=join(a.shards,a.index,pd.read_parquet(a.snapshot,columns=["date","symbol"]))
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 good.to_csv(out/"FY2025_original_NSE_full_3FY_2023_2025_annual_financial_facts.csv",index=False)
 bad.to_csv(out/"FY2025_original_NSE_source_rejected.csv",index=False)
 frame.to_parquet(out/"FY2025_original_1314_companies_3FY_verified_research.parquet",index=False)
 (out/"FY2025_original_three_FY_annual_source_reconciliation.json").write_text(json.dumps(report,indent=2))
 print(json.dumps(report,indent=2),flush=True)
if __name__=="__main__":main()
