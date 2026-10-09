"""Fail-closed scientific rejection gate for the actual 29-feature combined 3FY model.

Matured 2025 June and December retrospective outcomes had been consulted in
earlier experiments; re-evaluating a new model on them does not create truly
new blinded test evidence. Never flip this status to PASS because a workflow
ran successfully or December retrospectively found one winner.
"""
from __future__ import annotations
import argparse,json,math,os
from pathlib import Path

REQUIRED_FOLDS=12
MIN_FOLD_DOUBLERS=12
MIN_ROC_AUC=.60
MIN_TOP10_WINS=2
MIN_LIFT=2.

def assess(original):
  needed=("scope","original_October_2026_frozen_V11_4_selections_unchanged",
          "not_a_new_blind_independent_test_2025_labels_seen_in_earlier_finance_only_tests",
          "no_real_money_trading_or_production_approval",
          "independent_clean_12fold_scientific_promotion_gate_satisfied",
          "per_historic_fold_research_replication","train_rows","train_actual_2x_6m",
          "calibration_rows","calibration_actual_2x_6m")
  if any(k not in original for k in needed):raise ValueError("Actual historic-science source report incomplete")
  if original["scope"]!="V11_4_RESEARCH_FROZEN_RECIPE_3FY_MOMENTUM_NSE_CATALYST_COMBINED_2X_HISTORIC_REPLICATION":
    raise ValueError("Wrong source data or spoofed combined model performance")
  if original["original_October_2026_frozen_V11_4_selections_unchanged"] is not True:
    raise ValueError("User's original frozen stock selection is no longer immutable")
  if original["no_real_money_trading_or_production_approval"] is not True:
    raise ValueError("Unvalidated model falsely claims production")
  folds=original["per_historic_fold_research_replication"]
  if len(folds)!=2 or [f.get("fold") for f in folds]!=["2025-06-30","2025-12-31"]:
    raise ValueError("Folds misidentified, missing, or falsely presented as new")
  failures=[]
  if original["not_a_new_blind_independent_test_2025_labels_seen_in_earlier_finance_only_tests"] is not True:
    raise ValueError("Known-outcome retrospective tests cannot be presented as blinded")
  if original["independent_clean_12fold_scientific_promotion_gate_satisfied"] is not False:
    raise ValueError("Only two previously examined fold outcomes cannot satisfy 12-fold gate")
  failures.extend(["NO_TRULY_NEW_UNSEEN_BLINDED_HOLDOUTS",
                   "REQUIRES_12_INDEPENDENT_MATURED_FOLDS"])
  if original["train_actual_2x_6m"]<12 or original["calibration_actual_2x_6m"]<12:
    failures.append("INSUFFICIENT_TRAIN_OR_CALIBRATION_REAL_2X_EVENTS")
  for fold in folds:
    date=fold["fold"]
    n=int(fold["evaluated_label_matured_source_clean_stocks"])
    positive=int(fold["six_month_2x_positive_stocks"])
    win=int(fold["top10_actual_six_month_doublers"])
    base=float(fold["market_2x_baseline_frequency_in_evaluated_cohort"])
    prec=float(fold["top10_precision"])
    auc=fold["roc_auc"]
    lift=fold["top10_lift_over_evaluated_cohort"]
    if n<10 or positive<=0 or positive>n or not 0<=win<=10:
      raise ValueError("Impossible original historic positive/market count")
    if not math.isclose(base,positive/n,rel_tol=1e-8):
      raise ValueError("Historic source outcome rate inconsistent")
    if not math.isclose(prec,win/10,rel_tol=1e-8):
      raise ValueError("Historic top10 precision inconsistent")
    if lift is not None and not math.isclose(float(lift),prec/base,rel_tol=1e-8):
      raise ValueError("Historic lift calculation inconsistent")
    if positive<MIN_FOLD_DOUBLERS:
      failures.append(date+"_TOO_FEW_REAL_DOUBLERS_IN_HOLDOUT")
    if win<MIN_TOP10_WINS:
      failures.append(date+"_TOP10_PRECISION_BELOW_MINIMUM")
    if auc is None or not math.isfinite(float(auc)) or float(auc)<MIN_ROC_AUC:
      failures.append(date+"_ROC_AUC_BELOW_0_60")
    if lift is None or not math.isfinite(float(lift)) or float(lift)<MIN_LIFT:
      failures.append(date+"_TOP10_LIFT_BELOW_2X")
  return {
   "scope":"COMBINED_29_FEATURE_THREEFY_NSE_MODEL_EXECUTION_SUCCESS_BUT_SCIENCE_REJECTED",
   "actual_source_integrity_tests_succeeded":True,
   "historical_scores_generated_privately":True,
   "research_model_quality":"REJECTED_FOR_SCIENTIFIC_PROMOTION",
   "production_or_real_money_investing_approved":False,
   "previously_examined_holdout_dates":2,
   "genuinely_new_blinded_independent_holdout_dates":0,
   "minimum_required_independent_folds":REQUIRED_FOLDS,
   "training_real_six_month_doublers":original["train_actual_2x_6m"],
   "calibration_real_six_month_doublers":original["calibration_actual_2x_6m"],
   "folds":[{"date":f["fold"],"top10_doublers":f["top10_actual_six_month_doublers"],
             "ROC_AUC":f["roc_auc"],"holdout_event_count":f["six_month_2x_positive_stocks"]}
             for f in folds],
   "rejection_reasons":failures,
   "forward_2026_10_08_top10_unchanged":True,
   "no_retrospective_hyperparameter_tuning_performed_in_this_gate":True}

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--input",required=True);p.add_argument("--out",required=True)
 a=p.parse_args()
 original=json.loads(Path(a.input).read_text())
 result=assess(original)
 target=Path(a.out);target.parent.mkdir(parents=True,exist_ok=True)
 target.write_text(json.dumps(result,indent=2))
 print(json.dumps(result,indent=2),flush=True)
 if os.getenv("GITHUB_STEP_SUMMARY"):
  with open(os.environ["GITHUB_STEP_SUMMARY"],"a") as f:
   f.write("## Combined V11.4 fiscal + NSE momentum + catalyst model\n\n"
           "**Scientific status: REJECTED** despite passing computational tests.\n\n"
           "Train 2022-Dec, 2023-Jun; calibrate 2024-Jun; research replication June/Dec 2025.\n\n"
           +", ".join(result["rejection_reasons"])+"\n")
 if os.getenv("GITHUB_ACTIONS")=="true":
  print("::warning title=Combined-model research NOT ready::"+", ".join(result["rejection_reasons"]),flush=True)
if __name__=="__main__":main()
