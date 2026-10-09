"""Diagnose why 8 predeclared historic NSE promoter positive-event fields are ZERO.

Source-only: original 18fold canonical NSE metadata plus source-preserving
8fold PIT financial matrix. No labels. Distinguishes an empty promoter event
class from event_type present but unannotated positive direction.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd
FOLDS=("2022-06-30","2022-12-30","2023-06-30","2023-12-29",
       "2024-06-28","2024-12-31","2025-06-30","2025-12-31")
REQ=("nse_promoter_activity_180d","nse_promoter_activity_180d_positive",
     "nse_earnings_180d","nse_catalyst_total_180d",
     "nse_capacity_expansion_180d","nse_capacity_expansion_180d_positive",
     "nse_order_win_180d","nse_order_win_180d_positive")
def inspect(source):
 if not {"date","symbol",*REQ}.issubset(source):
  raise ValueError("Original raw NSE historical metadata category columns unavailable")
 f=source.copy()
 if {"y6","y12","y24","y6_mature_date","dd30_6m"}&set(f):
  raise ValueError("Unacceptable future labels in source-only promoter event audit")
 f["date"]=pd.to_datetime(f["date"],errors="raise").dt.strftime("%Y-%m-%d")
 f=f[f["date"].isin(FOLDS)].copy()
 if f.duplicated(["date","symbol"]).any() or len(f)!=9912:
  raise ValueError("Expected unchanged 9,912 original eightfold equities")
 per=[]
 for date in FOLDS:
  x=f[f["date"].eq(date)].copy()
  for key in REQ:x[key]=pd.to_numeric(x[key],errors="coerce")
  if x[list(REQ)].isna().any().any():
   raise ValueError("Missing canonical source event count")
  for kind in ("promoter_activity","capacity_expansion","order_win"):
   if (x[f"nse_{kind}_180d_positive"]>x[f"nse_{kind}_180d"]).any():
    raise ValueError("Positive event count larger than raw NSE record count")
  per.append({
   "fold":date,"source_issuer_dates":len(x),
   "promoter_any_raw_event_companies":int(x["nse_promoter_activity_180d"].gt(0).sum()),
   "promoter_any_positive_event_companies":int(x["nse_promoter_activity_180d_positive"].gt(0).sum()),
   "promoter_total_raw_event_records":int(x["nse_promoter_activity_180d"].sum()),
   "promoter_total_positive_direction_records":int(x["nse_promoter_activity_180d_positive"].sum()),
   "earnings_any_raw_event_companies":int(x["nse_earnings_180d"].gt(0).sum()),
   "broad_any_catalyst_companies":int(x["nse_catalyst_total_180d"].gt(0).sum()),
   "capacity_original_any_companies":int(x["nse_capacity_expansion_180d"].gt(0).sum()),
   "capacity_direction_positive_companies":int(x["nse_capacity_expansion_180d_positive"].gt(0).sum()),
   "order_original_any_companies":int(x["nse_order_win_180d"].gt(0).sum()),
   "order_direction_positive_companies":int(x["nse_order_win_180d_positive"].gt(0).sum())})
 all_raw=sum(x["promoter_total_raw_event_records"] for x in per)
 all_positive=sum(x["promoter_total_positive_direction_records"] for x in per)
 diagnostic=("ORIGINAL_PROMOTER_EVENT_CATEGORY_ABSENT" if all_raw==0
   else "PROMOTER_EVENTS_PRESENT_BUT_NOT_ANNOTATED_POSITIVE" if all_positive==0
   else "POSITIVE_PROMOTER_SOURCE_PRESENT")
 return {"scope":"CANONICAL_NSE_8FOLD_PROMOTER_EVENT_SOURCE_CLASS_DONT_TUNE_2025",
         "source_only_orig_stock_dates":len(f),
         "promoter_raw_180d_records_eightfold":all_raw,
         "promoter_positive_180d_records_eightfold":all_positive,
         "promoter_source_diagnostic_status":diagnostic,
         "per_original_fold":per,
         "no_2025_outcomes_or_private_rankings_read":True,
         "no_retraining_or_2025_threshold_tuning":True,
         "existing_2026_sealed_predictions_unchanged":True}
def main():
 p=argparse.ArgumentParser()
 p.add_argument("--original-18fold-source",required=True);p.add_argument("--out",required=True)
 a=p.parse_args()
 r=inspect(pd.read_parquet(a.original_18fold_source))
 f=Path(a.out);f.parent.mkdir(parents=True,exist_ok=True)
 f.write_text(json.dumps(r,indent=2))
 print(json.dumps(r,indent=2),flush=True)
if __name__=="__main__":main()
