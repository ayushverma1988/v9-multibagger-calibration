"""V11.4 source-only eight-fold audited RSI>70 + promoter-event truth sidecar.

Combine ALREADY PREDECLARED NSE financial+market+filing features with strictly
historic Wilder RSI14. Never consume returns, y6, fitted probabilities or
historical winners. A generic promoter filing is NOT verified promoter buying.
Neither false RSI on unknown prices nor false zero positive promoter ownership.

Third-party archived raw daily bars have frozen-close reconciliation, but
their exchange provenance across *all preceding sessions* has NOT been
independently cross-validated with official NSE tapes. Accordingly RSI remains
a separately flagged research sidecar; it is NOT a validated production feature
or a new predictor weight. Published 2025 outcome labels are never loaded.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd

ORIGINAL_DATES={
 "2022-06-30":1087,"2022-12-30":1123,"2023-06-30":1147,
 "2023-12-29":1276,"2024-06-28":1322,"2024-12-31":1345,
 "2025-06-30":1298,"2025-12-31":1314}
STRICT_STATUS="STRICT_ORIGINAL_120_SESSION_RSI70_AVAILABLE"
REQUIRED_MATERIAL=("nse_capacity_expansion_180d_positive",
 "nse_order_win_180d_positive","nse_regulatory_180d_positive")
LABEL_FORBIDDEN={"y6","y12","y24","y6_mature_date","dd30_6m",
 "days_to_2x","exploratory_p6_double_model_probability",
 "p6_double_calibrated","v11_4_p2x_calibrated",
 "exploratory_p6_double_combined_research_only","research_rank"}

def normalize(panel,title):
 x=panel.copy()
 if LABEL_FORBIDDEN.intersection(x):
  raise ValueError(f"{title}: future outcome, model rank or score inside feature-only source")
 if not {"date","symbol"}.issubset(x):
  raise ValueError(f"{title}: historical company and date key missing")
 x["date"]=pd.to_datetime(x["date"],errors="raise").dt.strftime("%Y-%m-%d")
 x["symbol"]=x["symbol"].astype(str).str.upper().str.strip()
 if x[["date","symbol"]].duplicated().any() or x["symbol"].eq("").any():
  raise ValueError(f"{title}: duplicate historical company-date")
 if set(x["date"])!=set(ORIGINAL_DATES) or len(x)!=sum(ORIGINAL_DATES.values()):
  raise ValueError(f"{title}: 8 original market-fold stock identities changed")
 if x.groupby("date").size().to_dict()!=ORIGINAL_DATES:
  raise ValueError(f"{title}: original company count drift")
 return x

def join_no_labels(original,archived_rsi,raw_nse_metadata,frozen_original_close):
 feature=normalize(original,"3FY NSE source-only matrix")
 rsi=normalize(archived_rsi,"Original eightfold historical RSI")
 events=normalize(raw_nse_metadata,"Original archived NSE event metadata")
 close=normalize(frozen_original_close,"Original immutable frozen stock closes")
 needs={"historical_asof_utc","has_strict_3FY_original_fiscal_source",
  "nse_promoter_activity_180d_positive","nse_catalyst_total_180d",*REQUIRED_MATERIAL}
 if not needs.issubset(feature):
  raise ValueError("Original PIT financial+NSE source evidence incomplete")
 if not {"rsi14_wilder_source","fourth_family_rsi_strictly_gt70",
         "source_verification","original_frozen_market_close_INR"}.issubset(rsi):
  raise ValueError("Historical Wilder RSI source identity and price audit absent")
 if not {"nse_promoter_activity_180d","nse_earnings_180d",
         "nse_promoter_activity_180d_positive"}.issubset(events):
  raise ValueError("Canonical original NSE promoter event count not available")
 if "close" not in close:raise ValueError("Frozen original reference close missing")
 for title,other in (("RSI",rsi),("events",events),("close",close)):
  if set(zip(feature["date"],feature["symbol"]))!=set(zip(other["date"],other["symbol"])):
   raise ValueError(f"{title} original stock-date identity differs from historical feature universe")
 cutoff=(pd.to_datetime(feature["date"]).dt.tz_localize("Asia/Kolkata")
       +pd.Timedelta(hours=15,minutes=30)).dt.tz_convert("UTC")
 available=pd.to_datetime(feature["historical_asof_utc"],utc=True,errors="coerce",format="mixed")
 if available.isna().any() or available.gt(cutoff).any():
  raise ValueError("Post-fold NSE feature metadata must never enter RSI sidecar")
 col=["date","symbol","rsi14_wilder_source","fourth_family_rsi_strictly_gt70",
      "source_verification","original_frozen_market_close_INR"]
 ecols=["date","symbol","nse_promoter_activity_180d",
        "nse_earnings_180d","nse_promoter_activity_180d_positive"]
 out=(feature.merge(rsi[col],on=["date","symbol"],how="left",validate="1:1")
       .merge(events[ecols],on=["date","symbol"],how="left",validate="1:1",
              suffixes=("","_canonical"))
       .merge(close[["date","symbol","close"]],on=["date","symbol"],
              how="left",validate="1:1"))
 if len(out)!=len(feature):raise ValueError("Historical universe was inner-filtered")
 if out["close"].isna().any():raise ValueError("Missing immutable original price")
 ref=pd.to_numeric(out["original_frozen_market_close_INR"],errors="coerce")
 close_ref=pd.to_numeric(out["close"],errors="coerce")
 if not np.isfinite(ref).all() or not np.allclose(ref,close_ref,rtol=0,atol=1e-6):
  raise ValueError("Historical RSI source does not reconcile to original immutable close")
 if "nse_promoter_activity_180d_positive_canonical" not in out:
  raise ValueError("Crosscheck source positive promoter event count missing")
 promold=pd.to_numeric(out["nse_promoter_activity_180d_positive"],errors="coerce")
 promoriginal=pd.to_numeric(out["nse_promoter_activity_180d_positive_canonical"],errors="coerce")
 raw=pd.to_numeric(out["nse_promoter_activity_180d"],errors="coerce")
 if promold.isna().any() or promoriginal.isna().any() or raw.isna().any():
  raise ValueError("Missing canonical NSE raw promoter filing evidence")
 if not np.allclose(promold,promoriginal,rtol=0,atol=0):
  raise ValueError("Promoter positive direction differs from canonical NSE raw event archive")
 if (promold!=0).any() or (raw<0).any():
  raise ValueError("Previously undocumented positive NSE promoter direction cannot be promoted without document audit")
 verified=out["source_verification"].eq(STRICT_STATUS)
 rsi_values=pd.to_numeric(out["rsi14_wilder_source"],errors="coerce")
 if (verified&(~rsi_values.between(0,100))).any():
  raise ValueError("Source-verified RSI outside standard Wilder 0-100 scale")
 if ((~verified)&rsi_values.notna()).any():
  raise ValueError("Invalid/unknown historical source leaked numerical RSI into research")
 passing=rsi_values.gt(70)
 recorded=out["fourth_family_rsi_strictly_gt70"].astype("boolean")
 if recorded.loc[verified].isna().any() or not recorded.loc[verified].eq(passing.loc[verified]).all():
  raise ValueError("Recorded historical RSI screening flag does not apply strict threshold >70")
 if recorded.loc[~verified].notna().any():
  raise ValueError("Unknown RSI historically substituted with FAIL or PASS")
 out["rsi14_Wilder_research_verified"]=verified
 out["rsi14_strict_gt70_source_verified"]=passing.where(verified,pd.NA).astype("boolean")
 out["condition_4_RSI14_GT70_source_status"]=np.where(
    verified,np.where(passing,"PASS","FAIL"),"UNKNOWN")
 # Do NOT invent purchases. All old direction-neutral source observations
 # are missing evidence of actual buying, not affirmative zero purchases.
 out["promoter_buying_verified_180d"]=pd.Series(pd.array([pd.NA]*len(out),dtype="Int64"))
 out["promoter_buying_direction_status"]="UNVERIFIED_NEUTRAL_OR_GENERIC_FILING"
 out["promoter_original_180d_filing_count"]=raw.astype("int64")
 out["nse_material_specific_positive_180d_count"]=0
 for col in REQUIRED_MATERIAL:
  values=pd.to_numeric(out[col],errors="coerce")
  if values.isna().any() or values.lt(0).any():
   raise ValueError("Specific audited positive NSE catalyst metadata invalid: "+col)
  out["nse_material_specific_positive_180d_count"]+=values.astype("int64")
 out["nse_material_specific_positive_present"]=out["nse_material_specific_positive_180d_count"].gt(0)
 # Previous 29-feature research model remains as archived historical control,
 # not a reranker. Do not overwrite the underlying nse_promoter_activity column.
 out=out.drop(columns=["nse_promoter_activity_180d_positive_canonical"])
 if LABEL_FORBIDDEN.intersection(out):
  raise ValueError("Unexpected future outcome or old predicted rank within source sidecar")
 folds=[]
 for d in ORIGINAL_DATES:
  f=out.loc[out["date"].eq(d)]
  ok=f["rsi14_Wilder_research_verified"]
  unknown=int((~ok).sum());passn=int(f["rsi14_strict_gt70_source_verified"].fillna(False).sum())
  folds.append({"date":d,"original_NSE_universe":len(f),
    "threeFY_original_numerical_sources_verified":int(f["has_strict_3FY_original_fiscal_source"].eq(True).sum()),
    "RSI14_source_verified":int(ok.sum()),"RSI14_unknown":unknown,
    "RSI14_strictly_above_70":passn,
    "RSI14_gt70_and_3FY_available":int((f["has_strict_3FY_original_fiscal_source"].eq(True)&
              f["rsi14_strict_gt70_source_verified"].fillna(False)).sum()),
    "promoter_generic_original_NSE_filing_records":int(f["promoter_original_180d_filing_count"].sum()),
    "actual_promoter_share_buys_document_verified":None,
    "material_capacity_order_regulatory_positive_presence":int(f["nse_material_specific_positive_present"].sum()),
    "broad_uncategorized_or_routine_NSE_catalyst_presence":int(pd.to_numeric(f["nse_catalyst_total_180d"]).gt(0).sum()),
    "source_only_no_historical_outcome_labels":True})
 summary={"scope":"V11_4_STRICT_SOURCE_ONLY_8FOLD_THREEFY_NSE_RSI14_GT70_PROMOTER_NOT_CONFIRMED",
  "historical_original_market_stock_date_rows":len(out),
  "original_three_fiscal_year_verified_stockdates":int(out["has_strict_3FY_original_fiscal_source"].eq(True).sum()),
  "historical_source_RSI14_verified":int(verified.sum()),
  "historical_source_RSI14_missing_UNKNOWN":int((~verified).sum()),
  "historical_RSI14_strictly_GT70":int(out["rsi14_strict_gt70_source_verified"].fillna(False).sum()),
  "all_RSI14_missing_represented_UNKNOWN_not_0":True,
  "promoter_raw_generic_NSE_filing_records_observed":int(raw.sum()),
  "promoter_buying_confirmed_stockdate_observations":None,
  "promoter_buying_replaced_by_false_zero":False,
  "positive_specific_NSE_event_features":list(REQUIRED_MATERIAL),
  "generic_broad_NSE_event_total_does_not_count_as_verified_earnings_acceleration":True,
  "original_NSE_filing_exchange_clock_1530_enforced":True,
  "historical_raw_daily_bar_archive_provenance_independently_verified_against_all_NSE_sessions":False,
  "not_new_predictive_backtest_no_labels_loaded":True,
  "not_train_or_production_model_not_a_probability_calibration":True,
  "frozen_Oct2026_stock_ranks_and_probabilities_unchanged":True,
  "original_stock_date_folds":folds}
 return out,summary

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--features",required=True)
 p.add_argument("--rsi",required=True)
 p.add_argument("--canonical-NSE",required=True)
 p.add_argument("--frozen-original-close",required=True)
 p.add_argument("--out",required=True)
 p.add_argument("--expected-verified-RSI",type=int,default=None)
 p.add_argument("--expected-GT70",type=int,default=None)
 p.add_argument("--expected-raw-promoter",type=int,default=None)
 args=p.parse_args()
 paths=(args.features,args.rsi,args.canonical_NSE,args.frozen_original_close)
 m=[pd.read_parquet(k) for k in paths]
 # Do not copy original 18fold y6 outcomes into training/source data.
 m[3]=m[3][["date","symbol","close"]].copy()
 frame,summary=join_no_labels(*m)
 for expected,actual,name in (
   (args.expected_verified_RSI,summary["historical_source_RSI14_verified"],"RSI verified"),
   (args.expected_GT70,summary["historical_RSI14_strictly_GT70"],"strict RSI >70"),
   (args.expected_raw_promoter,summary["promoter_raw_generic_NSE_filing_records_observed"],"raw promoter")):
  if expected is not None and actual!=expected:
   raise ValueError(f"Original {name} source inventory changed: expected {expected} found {actual}")
 out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
 frame.to_parquet(out/"eightfold_3FY_market_specificNSEevents_RSI70_source_only_PRIVATE.parquet",index=False,
                  compression="zstd")
 (out/"eightfold_3FY_RSI70_promoter_source_quality_summary.json").write_text(json.dumps(summary,indent=2))
 print(json.dumps(summary,indent=2),flush=True)
if __name__=="__main__":main()
