"""Predeclared RSI(14)>70 descriptive association in EARLY original source cohorts.

The user's strict >70 threshold is fixed; no RSI threshold optimization,
no 2025 June/Dec y6 outcomes, no new 2026 picks, and no re-fit of the
29-feature combined ML. Evaluate only 2022-Dec + 2023-Jun already-matured
training periods, and 2024-Jun already-matured calibration period.
Actual doubling frequency conditional on RSI is *observational*,
not a causal relation or a clean new out-of-sample prediction.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from v11_4_eightfold_3FY_market_catalyst_source_matrix import EXPECT_UNIVERSE

TRAIN_DATES=("2022-12-30","2023-06-30")
CAL_DATE="2024-06-28"
USED_DATES=(*TRAIN_DATES,CAL_DATE)
CAL_TRAIN_CUTOFF_UTC=pd.Timestamp("2024-06-28T10:00:00Z")
LATER_TEST_START_UTC=pd.Timestamp("2025-06-30T10:00:00Z")
NO_SOURCE_OUTCOMES={"y6","y12","y24","y6_mature_date",
 "dd30_6m","p6_double_calibrated",
 "exploratory_p6_double_combined_research_only","research_rank"}

def wilson(success,n,z=1.96):
 if not n:return [None,None]
 p=success/n;den=1+z*z/n
 a=(p+z*z/(2*n))/den
 b=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
 return [float(max(0,a-b)),float(min(1,a+b))]

def inspect(source,original_labels):
 if NO_SOURCE_OUTCOMES&set(source):
  raise ValueError("Future stock-doubling labels contaminated source-only RSI financial features")
 req={"date","symbol","close","avg_turnover_63","integrity_feature_clean",
  "has_strict_3FY_original_fiscal_source","rsi14_Wilder_research_verified",
  "RSI_120session_archive_available_PLUS_official_day_close",
  "RSI14_STRICT_GT70_when_official_NSE_DAY_close_confirmed",
  "promoter_buying_verified_180d"}
 if not req.issubset(source):raise ValueError("PIT original NSE 3FY+RSI70 day-close source absent")
 label_need={"date","symbol","y6","y6_mature_date","integrity_y6_clean","close","avg_turnover_63"}
 if not label_need.issubset(original_labels):raise ValueError("Original frozen 6-month outcomes incomplete")
 a=source.copy();b=original_labels.copy()
 for x in (a,b):
  x["date"]=pd.to_datetime(x["date"],errors="raise").dt.strftime("%Y-%m-%d")
  x["symbol"]=x["symbol"].astype(str).str.upper().str.strip()
  if x[["date","symbol"]].duplicated().any():raise ValueError("Repeated historical stock identity")
 if a.groupby("date").size().to_dict()!=EXPECT_UNIVERSE:
  raise ValueError("Original eight historical complete NSE market universes changed")
 if a["promoter_buying_verified_180d"].notna().any():
  raise ValueError("Unverified neutral promoter records cannot become documented buying")
 a=a[a["date"].isin(USED_DATES)].copy()
 b=b[b["date"].isin(USED_DATES)].copy()
 if set(zip(a["date"],a["symbol"]))!=set(zip(b["date"],b["symbol"])):
  raise ValueError("Original training/calibration stock universe and labels drifted")
 z=a.merge(b[["date","symbol","y6","y6_mature_date",
  "integrity_y6_clean","close","avg_turnover_63"]],
  on=["date","symbol"],how="left",validate="1:1",suffixes=("","_original"))
 if len(z)!=sum(EXPECT_UNIVERSE[x] for x in USED_DATES):
  raise ValueError("Original three early historical cohorts truncated by join")
 v=pd.to_numeric(z["close"],errors="coerce")
 old=pd.to_numeric(z["close_original"],errors="coerce")
 if v.isna().any() or not np.allclose(v,old,rtol=0,atol=1e-6):
  raise ValueError("Historical financial feature market close differs from frozen label record")
 volume=pd.to_numeric(z["avg_turnover_63"],errors="coerce")
 oldvol=pd.to_numeric(z["avg_turnover_63_original"],errors="coerce")
 if not np.allclose(volume,oldvol,rtol=1e-7,atol=1e-4,equal_nan=True):
  raise ValueError("Market turnover drift between historic sources")
 z["maturity"]=pd.to_datetime(z["y6_mature_date"],utc=True,errors="coerce",format="mixed")
 cutoff=z["date"].map(lambda d:CAL_TRAIN_CUTOFF_UTC if d in TRAIN_DATES else LATER_TEST_START_UTC)
 has_source=z["has_strict_3FY_original_fiscal_source"].eq(True)
 verified_rsi=z["RSI_120session_archive_available_PLUS_official_day_close"].eq(True)
 gate=z["RSI14_STRICT_GT70_when_official_NSE_DAY_close_confirmed"].astype("boolean")
 if gate.loc[verified_rsi].isna().any() or gate.loc[~verified_rsi].notna().any():
  raise ValueError("Original official NSE RSI source missing-data gate corrupt")
 eligible=(has_source&verified_rsi&
  z["integrity_feature_clean"].eq(True)&z["integrity_y6_clean"].eq(True)&
  z["y6"].isin([0,1])&z["close"].between(20,2000)&volume.gt(0)&
  z["maturity"].notna()&(z["maturity"]<cutoff))
 evalframe=z.loc[eligible].copy()
 if len(evalframe)<100:
  raise ValueError("Early historical known three-FY + RSI + mature target cohort too small")
 evalframe["strict_RSI_GT70"]=evalframe["RSI14_STRICT_GT70_when_official_NSE_DAY_close_confirmed"].astype(bool)
 evalframe["y6"]=evalframe["y6"].astype(int)
 reports=[]
 for name,subset in [("2022_Dec_train",evalframe[evalframe["date"].eq(TRAIN_DATES[0])]),
   ("2023_June_train",evalframe[evalframe["date"].eq(TRAIN_DATES[1])]),
   ("2024_June_calibration",evalframe[evalframe["date"].eq(CAL_DATE)]),
   ("all_three_early_existing_train_cal_dates_POOLED_not_independent",evalframe)]:
  n=len(subset);positive=int(subset["y6"].sum())
  exposed=subset["strict_RSI_GT70"]
  ne=int(exposed.sum());ny=int(subset.loc[exposed,"y6"].sum())
  nc=n-ne;cy=positive-ny
  odds,p=fisher_exact([[ny,ne-ny],[cy,nc-cy]]) if ne and nc else (None,None)
  exp_rate=ny/ne if ne else None
  other_rate=cy/nc if nc else None
  base=positive/n if n else None
  reports.append({"fold_or_group":name,
   "original_eligible_known_threeFY_clean_labels_and_official_dayclose_RSI":n,
   "actual_six_month_doublers":positive,
   "all_sample_2x_frequency":base,
   "RSI14_strict_gt70_count":ne,
   "RSI14_strict_gt70_actual_six_month_doublers":ny,
   "RSI14_strict_gt70_six_month_double_rate":exp_rate,
   "RSI14_gt70_double_rate_Wilson95":wilson(ny,ne),
   "RSI14_not_gt70_count":nc,
   "RSI14_not_gt70_actual_six_month_doublers":cy,
   "RSI14_not_gt70_six_month_double_rate":other_rate,
   "RSI70_over_nonRSI70_positive_rate_ratio":exp_rate/other_rate if exp_rate is not None and other_rate else None,
   "RSI70_sensitivity_fraction_of_all_historical_doublers_recovered":ny/positive if positive else None,
   "fisher_2sided_unadjusted_p_descriptive_only":float(p) if p is not None else None,
   "RSI_above70_group_not_cherrypicked_or_tuned":True,
   "not_a_new_independent_OOS_holdout":True})
 return {"scope":"V11_4_ORIGINAL_THREEFY_OFFICIAL_DAY_RECHECK_RSI_GT70_EARLY_TRAIN_CAL_ONLY_RESEARCH_ASSOCIATION",
  "original_8fold_features_seen_no_future_labels_as_input":True,
  "evaluated_only_stock_label_fold_dates":list(USED_DATES),
  "strict_calibration_and_train_label_maturity_dates_precede_subsequent_test_decisions":True,
  "no_2025_test_fold_six_month_doublers_used":True,
  "original_2026_forward_top10_and_fitted_probabilities_untouched":True,
  "this_is_observational_association_not_actual_backtested_model_gain":True,
  "no_new_independent_blinded_test_completed":True,
  "prospective_or_production_model_promoted":False,
  "RSI_120session_original_NSE_archive_independently_certified":False,
  "descriptive_by_early_training_and_calibration_period":reports}

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--source",required=True);p.add_argument("--original-mature-labels",required=True)
 p.add_argument("--out",required=True)
 a=p.parse_args()
 features=pd.read_parquet(a.source)
 labels=pd.read_parquet(a.original_mature_labels,columns=[
   "date","symbol","y6","y6_mature_date","integrity_y6_clean","close","avg_turnover_63"])
 report=inspect(features,labels)
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 (out/"RSI14_GT70_THREE_FY_early_training_calibration_ONLY_unadjusted_association.json").write_text(
  json.dumps(report,indent=2))
 print(json.dumps(report,indent=2),flush=True)
if __name__=="__main__":main()
