"""Strict consolidation of FY2022-24 original NSE annual numeric shards.

Checks disjoint 12 partitions, historical source SHA256, fiscal publication
and 2-year CAGR, then reattaches a NaN-preserving sidecar to all 1,345
original 2024 date stocks. No future label or predictor fit.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from v11_4_strict_annual_numeric_features import fold_close

BASE="2024-12-31"
YEARS=(2022,2023,2024)
def reconcile(original_index,shard_root,snapshot):
 idx=original_index.copy()
 companies=sorted(idx["symbol"].astype(str).str.upper().str.strip().unique())
 if len(idx)!=3*len(companies):raise ValueError("Original 2024 3-FY source catalog not exact triples")
 reports=list(Path(shard_root).rglob("FY2024_original_three_FY_numeric_source_report.json"))
 if len(reports)!=12:raise ValueError("Missing independent original fiscal annual shards")
 all_good=[];all_bad=[];seen=set()
 for p in reports:
  r=json.loads(p.read_text());n=int(r["shard"])
  if n in seen:raise ValueError("Repeated full-market financial partition")
  seen.add(n)
  expected=set(companies[n::12])
  if r["shards"]!=12 or r["all_2024_source_candidates"]!=len(companies) or r["attempted_companies"]!=len(expected):
   raise ValueError("Historical original NSE financial sharding inconsistent")
  op=p.parent/"FY2024_original_NSE_three_FY_strict_numeric_2022_to_2024.csv"
  rp=p.parent/"FY2024_original_NSE_three_FY_source_rejected.csv"
  if not op.is_file() or not rp.is_file():raise ValueError("Missing accepted/rejected source records")
  x=pd.read_csv(op) if r["three_fiscal_years_numeric_verified"] else pd.DataFrame()
  bad=pd.read_csv(rp) if r["original_source_rejected"] else pd.DataFrame()
  if len(x)!=r["three_fiscal_years_numeric_verified"] or len(bad)!=r["original_source_rejected"]:
   raise ValueError("Source shard summary disagrees with original fiscal files")
  accepted=set(x["symbol"]) if len(x) else set()
  rejected=set(bad["symbol"]) if len(bad) else set()
  if accepted&rejected or accepted|rejected!=expected:
   raise ValueError("Financial source acceptance/rejection mismatch")
  if len(x):all_good.append(x)
  if len(bad):all_bad.append(bad)
 if seen!=set(range(12)):raise ValueError("Missing 12-way historical partition")
 good=pd.concat(all_good,ignore_index=True)
 bad=pd.concat(all_bad,ignore_index=True) if all_bad else pd.DataFrame()
 if good["symbol"].duplicated().any() or len(good)+len(bad)!=len(companies):
  raise ValueError("Duplicated/unaccounted original three-fiscal-year company")
 cutoff=fold_close(BASE)
 for fy in YEARS:
  if not good[f"FY{fy}_source_sha256"].astype(str).str.fullmatch("[a-f0-9]{64}").all():
   raise ValueError("Missing NSE original fiscal SHA256")
  pub=pd.to_datetime(good[f"FY{fy}_published_utc"],utc=True,errors="coerce")
  if pub.isna().any() or (pub>cutoff).any():raise ValueError("Future financial filing")
  urls=good[f"FY{fy}_source_url"].astype(str)
  if not urls.str.match(r"^https://nsearchives\.nseindia\.com/[^?#]+\.xml$",case=False).all():
   raise ValueError("Unverified original NSE financial URL")
  revenue=pd.to_numeric(good[f"FY{fy}_revenue_INR"],errors="coerce")
  profit=pd.to_numeric(good[f"FY{fy}_PAT_INR"],errors="coerce")
  if not revenue.gt(0).all() or not profit.map(np.isfinite).all():raise ValueError("Corrupt original fiscal numeric values")
  if good[f"FY{fy}_annual_context_audit"].isna().any():raise ValueError("No original fiscal context proof")
 reconstructed=(good["FY2024_revenue_INR"]/good["FY2022_revenue_INR"])**.5-1
 if not np.allclose(reconstructed,good["FY2022_to_FY2024_sales_two_year_CAGR"],rtol=1e-10,atol=1e-10):
  raise ValueError("Three annual points incorrectly compounded")
 if not good["no_3year_CAGR_from_only_three_FY_observations"].astype(str).str.lower().eq("true").all():
  raise ValueError("Falsely claimed 3-year duration")
 if good["model_training_authorized"].astype(str).str.lower().eq("true").any():
  raise ValueError("Historical financial data used to fit model before separate validation")
 frame=snapshot.copy();frame["date"]=pd.to_datetime(frame["date"],errors="raise").dt.normalize()
 universe=sorted(set(frame.loc[frame["date"].eq(pd.Timestamp(BASE)),"symbol"].astype(str).str.upper().str.strip()))
 if len(universe)!=1345:raise ValueError("Unexpected original Dec2024 stock universe")
 keep=["symbol","original_reporting_mode","FY2024_vs_2023_revenue_yoy",
       "FY2023_vs_2022_revenue_yoy","FY2022_to_FY2024_sales_two_year_CAGR",
       "FY2022_to_FY2024_PAT_two_year_CAGR"] + [f"FY{fy}_source_sha256" for fy in YEARS]
 joined=pd.DataFrame({"symbol":universe,"date":BASE}).merge(good[keep],on="symbol",how="left",validate="1:1")
 joined["verified_original_three_FY_asof_2024"]=joined["FY2022_source_sha256"].notna()
 if len(joined)!=1345:raise ValueError("Source missing original 2024 historic equity identity")
 summary={
  "scope":"V11_4_2024_FOLD_FULL_ORIGINAL_3FY_NUMERIC_RECONCILIATION_NOT_TRAINING",
  "original_dec2024_equities":1345,
  "original_three_FY_source_link_sets":len(companies),
  "source_verified_FY2022_FY2023_FY2024_companies":len(good),
  "strict_source_rejections":len(bad),
  "three_FY_numeric_coverage_percent":round(100*len(good)/1345,2),
  "single_2024_fold_70pct_coverage_passed":bool(len(good)/1345>=.70),
  "two_year_sales_CAGR_verified":len(good),
  "two_year_profit_CAGR_verified":int(joined["FY2022_to_FY2024_PAT_two_year_CAGR"].notna().sum()),
  "one_year_end_source_fold_not_a_predictive_backtest":True,
  "exact_historic_stock_universe_preserved":True,
  "no_imputation":True,
  "frozen_V11_4_scores_changed":False
 }
 return good,bad,joined,summary
def main():
 p=argparse.ArgumentParser()
 p.add_argument("--index",required=True);p.add_argument("--shards",required=True)
 p.add_argument("--snapshot",required=True);p.add_argument("--out",required=True)
 a=p.parse_args()
 good,bad,joined,r=reconcile(pd.read_csv(a.index,dtype=str),a.shards,
    pd.read_parquet(a.snapshot,columns=["date","symbol"]))
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 good.to_csv(out/"strict_original_FY2022_2024_annual_facts.csv",index=False)
 bad.to_csv(out/"strict_rejected_original_2024_company_financial_sources.csv",index=False)
 joined.to_parquet(out/"original_dec2024_stock_universe_3FY_financial_sidecar.parquet",index=False)
 (out/"original_2024_full_3FY_reconciliation.json").write_text(json.dumps(r,indent=2))
 print(json.dumps(r,indent=2))
if __name__=="__main__":main()
