"""Standalone reproducible *historical* 2025 financial-only 3FY screening demo.

Not a trained V11.4 ML predictor. A transparent reduced-data companion for
user's request to try only 3 consecutive financial years, two growth
intervals. Never looks at six-month forward outcomes or future labels and
never replaces original frozen 2026-10-08 prospective 10 stocks.
"""
from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
BASE="2025-12-31"
REQ=("symbol","FY2023_revenue_INR","FY2024_revenue_INR","FY2025_revenue_INR",
     "FY2023_PAT_INR","FY2024_PAT_INR","FY2025_PAT_INR",
     "revenue_2023_to_2025_2yr_CAGR","revenue_FY2025_yoy",
     "profit_2023_to_2025_2yr_CAGR","FY2025_source_SHA256")
def score(financial,snapshot):
 if not set(REQ).issubset(financial):
  raise ValueError("Verified three FY source financial metrics missing")
 if financial["symbol"].duplicated().any():
  raise ValueError("Financial issuer duplicated")
 s=snapshot.copy()
 s["date"]=pd.to_datetime(s["date"],errors="raise").dt.normalize()
 s["symbol"]=s["symbol"].astype(str).str.upper().str.strip()
 s=s[s["date"].eq(pd.Timestamp(BASE))].copy()
 if len(s)!=1314 or s["symbol"].duplicated().any():
  raise ValueError("Missing original 2025 historical NSE securities")
 if not {"close","avg_turnover_63"}.issubset(s):
  raise ValueError("Missing original pre-decision-close trading characteristics")
 q=s[["date","symbol","close","avg_turnover_63"]].merge(financial[list(REQ)],on="symbol",how="left",validate="1:1")
 if len(q)!=1314:raise ValueError("Original stock universe was filtered before accounting")
 if q["FY2025_source_SHA256"].notna().sum()!=1013:
  raise ValueError("Three FY financial numeric coverage changed or backfilled")
 for year in (2023,2024,2025):
  if not pd.to_numeric(q[f"FY{year}_revenue_INR"],errors="coerce").dropna().gt(0).all():
   raise ValueError("Invalid official annual revenue")
 for col in ("close","avg_turnover_63",*REQ[1:-1]):
  q[col]=pd.to_numeric(q[col],errors="coerce")
 # ALL selection thresholds declared before examining any data or 6m returns.
 q["verified_three_FY"]=q["FY2025_source_SHA256"].notna()
 q["within_recorded_price_20_2000"]=q["close"].between(20,2000)
 q["minimum_mean_daily_turnover_20_lakh_INR"]=q["avg_turnover_63"].ge(2_000_000)
 q["revenue_two_year_CAGR_over_12pct"]=q["revenue_2023_to_2025_2yr_CAGR"].gt(.12)
 q["sales_both_recent_years_grew"]=q["FY2024_revenue_INR"].gt(q["FY2023_revenue_INR"]) & q["FY2025_revenue_INR"].gt(q["FY2024_revenue_INR"])
 q["sales_growth_accelerated"]=q["revenue_FY2025_yoy"].gt(q["FY2024_revenue_INR"]/q["FY2023_revenue_INR"]-1)
 q["PAT_positive_all_three_years"]=q[["FY2023_PAT_INR","FY2024_PAT_INR","FY2025_PAT_INR"]].gt(0).all(axis=1)
 q["PAT_two_year_CAGR_over_15pct"]=q["profit_2023_to_2025_2yr_CAGR"].gt(.15)
 q["PAT_FY2025_gt_FY2024"]=q["FY2025_PAT_INR"].gt(q["FY2024_PAT_INR"])
 # Avoid extreme percentage growth from tiny profits by reporting consecutive
 # positive-profit growth and two-year compounded growth as separate checks.
 conditions=["verified_three_FY","within_recorded_price_20_2000",
             "minimum_mean_daily_turnover_20_lakh_INR",
             "revenue_two_year_CAGR_over_12pct","sales_both_recent_years_grew",
             "sales_growth_accelerated","PAT_positive_all_three_years",
             "PAT_two_year_CAGR_over_15pct","PAT_FY2025_gt_FY2024"]
 q["historical_three_FY_screen_meets_all_rules"]=q[conditions].all(axis=1)
 # For explainability only; NO calibrated ML probability or forward outcome.
 qualified=q[q["historical_three_FY_screen_meets_all_rules"]].sort_values([
  "revenue_FY2025_yoy","revenue_2023_to_2025_2yr_CAGR","symbol"],ascending=[False,False,True]).copy()
 qualified.insert(0,"screen_rank",range(1,len(qualified)+1))
 report={"scope":"2025_12_31_3FY_ANNUAL_FINANCIAL_RULE_SCREEN_NOT_TRAINED_ML",
 "asof_IST":BASE,"original_market_stock_universe":1314,
 "verified_3FY_financial_company_count":int(q["verified_three_FY"].sum()),
 "stocks_passing_all_predeclared_short_history_rules":len(qualified),
 "stocks_passing_individual_rules":{c:int(q[c].sum()) for c in conditions},
 "rule_screen_is_not_probability_of_doubling":True,
 "no_six_month_forward_returns_or_labels_loaded":True,
 "no_retrospective_optimizing_cutoffs":True,
 "no_true_three_year_CAGR_claim":True,
 "stocks_without_exact_fiscal_data_left_in_full_universe":int((~q["verified_three_FY"]).sum()),
 "not_a_current_October_2026_stock_quotation":True,
 "original_frozen_live_model_unchanged":True,
 "production_approved":False}
 return q,qualified,report
def run(financial,snapshot,out):
 full,qualified,summary=score(financial,snapshot)
 out=Path(out);out.mkdir(parents=True,exist_ok=True)
 full.to_csv(out/"original_2025_complete_1314_company_short_history_screen_audit_PRIVATE.csv",index=False)
 qualified.to_csv(out/"original_2025_verified_three_FY_financial_rule_shortlist_PRIVATE.csv",index=False)
 (out/"original_2025_three_FY_reduced_history_screen_summary.json").write_text(json.dumps(summary,indent=2))
 print(json.dumps(summary,indent=2),flush=True)
 return summary
def main():
 p=argparse.ArgumentParser();p.add_argument("--financial",required=True)
 p.add_argument("--snapshot",required=True);p.add_argument("--out",required=True)
 a=p.parse_args()
 run(pd.read_csv(a.financial),pd.read_parquet(a.snapshot,columns=["date","symbol","close","avg_turnover_63"]),a.out)
if __name__=="__main__":main()
