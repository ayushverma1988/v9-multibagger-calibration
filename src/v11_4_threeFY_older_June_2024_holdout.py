"""Independent OLDER June2024 three-FY historical test, same frozen 10 predictors.

Fit June2022 (FY2020-22), calibrate June2023 (FY2021-23), evaluate June2024
(FY2022-24), ALL source and six-month outcome data truly known in sequence.
Never read 2025 June/Dec backtests. Fixed L2 logistic C=.03 and original
calibration algorithm. A distinct historical holdout adds replication, not
production validation of 2x-stock claims.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
from v11_4_threeFY_June_clean_out_of_sample_ML import YEARS,make_features,EVAL_UTC
from v11_4_threeFY_chronological_research_ML import PREDICTORS,calibrated_for_test
from v11_4_strict_annual_numeric_features import fold_close
ORIGINAL_2022_JUNE=("2022-06-30",(2020,2021,2022))
def older_sealed_research_holdout(june22,june23,june24):
 if 2022 in YEARS and YEARS[2022]!=ORIGINAL_2022_JUNE:
  raise ValueError("Original fiscal vintage changed")
 YEARS[2022]=ORIGINAL_2022_JUNE
 train=make_features(june22,2022,fold_close("2023-06-30"))
 cal=make_features(june23,2023,fold_close("2024-06-28"))
 hold=make_features(june24,2024,EVAL_UTC)
 if max(train["date"])>=min(cal["date"]) or max(cal["date"])>=min(hold["date"]):
  raise ValueError("Historical clean chronology violated")
 if any(len(z)<600 for z in (train,cal,hold)):
  raise ValueError("Too few original 3FY verified 2022/23/24 historical securities")
 pos=[int(z["y6"].astype(int).sum()) for z in (train,cal,hold)]
 if min(pos)<12:raise ValueError("Too few original heldout or calibrating six-month actual 2X wins")
 model,p,method=calibrated_for_test(train,cal,hold)
 if not np.isfinite(p).all():raise ValueError("Invalid source-verified financial experimental ML values")
 scored=hold[["date","symbol","close","avg_turnover_63","y6",*PREDICTORS]].copy()
 scored["exploratory_original_2024_June_p6_2x_NONPRODUCTION"]=p
 scored=scored.sort_values(["exploratory_original_2024_June_p6_2x_NONPRODUCTION","symbol"],
  ascending=[False,True]).reset_index(drop=True)
 scored.insert(0,"research_rank",np.arange(1,len(scored)+1))
 y=hold["y6"].astype(int).to_numpy();wins=int(scored.head(10)["y6"].sum())
 stat={
  "scope":"ORIGINAL_3FY_OLDER_JUNE2022_TRAIN_2023_CAL_2024_SINGLE_HELDOUT",
  "original_train_decision_IST":"2022-06-30",
  "original_calibration_decision_IST":"2023-06-30",
  "original_heldout_test_decision_IST":"2024-06-28",
  "source_and_outcome_available_asof_original_next_stage":True,
  "original_matured_training_stock_rows":len(train),
  "original_matured_calibration_stock_rows":len(cal),
  "original_matured_heldout_test_stock_rows":len(hold),
  "training_sixmonth_actual_2x_event_count":pos[0],
  "calibration_sixmonth_actual_2x_event_count":pos[1],
  "heldout_sixmonth_actual_2x_event_count":pos[2],
  "heldout_market_sixmonth_2x_base_rate":float(y.mean()),
  "heldout_top10_actual_2x_doublers":wins,
  "heldout_top10_precision":wins/10,
  "heldout_lift_over_market_2x_base":float((wins/10)/y.mean()) if y.mean()>0 else None,
  "heldout_ROC_AUC":float(roc_auc_score(y,p)),
  "heldout_average_precision":float(average_precision_score(y,p)),
  "heldout_Brier_score":float(brier_score_loss(y,p)),
  "calibration_method":method,
  "same_10_financial_predictors_and_L2_logistic_as_prior":True,
  "NO_2025_June_or_Dec_financial_test_loaded":True,
  "historically_earlier_than_2025_prospective_test":True,
  "historically_one_unique_older_holdout_not_12":True,
  "June2024_labels_seen_as_calibration_in_earlier_candidate":True,
  "therefore_not_a_new_blinded_independent_test":True,
  "not_independently_sufficient_for_production":True,
  "original_Oct2026_prospective_V11_4_unchanged":True,
  "scientific_status":"OLDER_HISTORICAL_HOLDOUT_ONLY_DO_NOT_PROMOTE_TO_TRADING"
 }
 return scored,stat
def main():
 p=argparse.ArgumentParser()
 for y in (2022,2023,2024):p.add_argument(f"--june-{y}",required=True)
 p.add_argument("--out",required=True);a=p.parse_args()
 frames=[pd.read_parquet(getattr(a,f"june_{y}")) for y in (2022,2023,2024)]
 score,report=older_sealed_research_holdout(*frames)
 root=Path(a.out);root.mkdir(parents=True,exist_ok=True)
 score.to_csv(root/"older_2024_June_threeFY_ranks_and_actual_mature_2x_labels_PRIVATE.csv",index=False)
 (root/"2024_June_older_threeFY_historical_holdout_aggregate_science.json").write_text(json.dumps(report,indent=2))
 print(json.dumps(report,indent=2),flush=True)
if __name__=="__main__":main()
