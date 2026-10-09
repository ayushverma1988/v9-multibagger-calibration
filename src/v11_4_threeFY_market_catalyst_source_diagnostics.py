"""Independent source-only feature sparsity diagnostic for 8-fold NSE 3FY matrix.

No targets, stock rankings, calibrations or test outcome information loaded.
This is for identifying original-source gaps, NOT hyperparameter selection.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from v11_4_eightfold_3FY_market_catalyst_source_matrix import (
 FOLDS,EXPECT_UNIVERSE,EVENTS,FIN_FEATS,PRICE,FORBIDDEN)
def analyze(source):
 x=source.copy()
 if FORBIDDEN&set(x):raise ValueError("Source-only diagnostic cannot read outcomes")
 if not {"date","symbol","has_strict_3FY_original_fiscal_source","historical_asof_utc",*FIN_FEATS,*PRICE,*EVENTS}.issubset(x):
  raise ValueError("Incomplete original PIT financial+price+NSE event source")
 x["date"]=pd.to_datetime(x["date"],errors="raise").dt.strftime("%Y-%m-%d")
 if len(x)!=sum(EXPECT_UNIVERSE.values()) or set(x["date"])!=set(FOLDS):
  raise ValueError("Original eight fold source coverage identity drift")
 if x.duplicated(["date","symbol"]).any():raise ValueError("Original stock date duplicate")
 moments=pd.to_datetime(x["historical_asof_utc"],errors="coerce",utc=True,format="mixed")
 cutoff=(pd.to_datetime(x["date"]).dt.tz_localize("Asia/Kolkata")+
         pd.Timedelta(hours=15,minutes=30)).dt.tz_convert("UTC")
 if moments.isna().any() or moments.gt(cutoff).any():
  raise ValueError("Original NSE catalyst document unavailable at decision clock")
 output=[]
 for date in FOLDS:
  z=x[x["date"].eq(date)];verified=z["has_strict_3FY_original_fiscal_source"].eq(True)
  if int(verified.sum())!=FOLDS[date][1]:raise ValueError("Original source count changed")
  cats=z[list(EVENTS)].apply(pd.to_numeric,errors="coerce")
  pr=z[list(PRICE[:-1])].apply(pd.to_numeric,errors="coerce")
  finance=z[list(FIN_FEATS)].apply(pd.to_numeric,errors="coerce")
  if finance.loc[~verified].notna().any().any():
   raise ValueError("Unverified original FY source falsely imputed to numeric")
  output.append({
    "fold":date,"original_market_stock_rows":len(z),
    "original_verified_full_threeFY":int(verified.sum()),
    "full_threeFY_percent":round(100*verified.mean(),2),
    "threeFY_minimum_70percent_pass":bool(verified.mean()>=.7),
    "original_NSE_event_fields_present":len(EVENTS),
    "NSE_total_catalyst_positive_count":int((cats["nse_catalyst_total_180d"]>0).sum()),
    "NSE_total_catalyst_positive_share":round(float((cats["nse_catalyst_total_180d"]>0).mean()),4),
    "NSE_catalyst_any_numeric_missing_rows":int(cats.isna().any(axis=1).sum()),
    "NSE_capacity_signal_nonzero":int((cats["nse_capacity_expansion_180d_positive"]>0).sum()),
    "NSE_order_signal_nonzero":int((cats["nse_order_win_180d_positive"]>0).sum()),
    "NSE_promoter_signal_nonzero":int((cats["nse_promoter_activity_180d_positive"]>0).sum()),
    "original_price_feature_finite_all_rows":int(np.isfinite(pr).all(axis=1).sum()),
    "verified_fiscal_all9_features_finite":int(np.isfinite(finance.loc[verified]).all(axis=1).sum()),
    "source_only_no_forward_target_loaded":True})
 report={
  "scope":"EXACT_ORIGINAL_NSE_EIGHTFOLD_SOURCE_ONLY_FEATURE_COVERAGE_NOT_MODEL_FITTING",
  "original_stock_date_observations":len(x),
  "strict_threeFY_verified_stock_dates":int(x["has_strict_3FY_original_fiscal_source"].sum()),
  "folds":output,
  "all_stocks_original_even_if_missing_NSE_fiscal":True,
  "zero_NSE_catalyst_MAY_mean_no_disclosure_not_absence_of_corporate_progress":True,
  "RSI_14_not_in_original_source_no_imputation":True,
  "no_future_target_or_outcome_loaded":True,
  "prediction_or_model_tuning_performed":False,
  "unseen_scientific_holdout_assessed":False,
  "original_Oct2026_frozen_pick_order_unchanged":True}
 return report
def main():
 p=argparse.ArgumentParser();p.add_argument("--features",required=True);p.add_argument("--out",required=True);a=p.parse_args()
 r=analyze(pd.read_parquet(a.features))
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 (out/"original_8fold_source_only_NSE_catalyst_financial_coverage.json").write_text(json.dumps(r,indent=2))
 print(json.dumps(r,indent=2),flush=True)
if __name__=="__main__":main()
