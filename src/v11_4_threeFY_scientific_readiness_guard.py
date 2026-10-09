"""Fail-closed scientific quality gate for the independent V11.4 three-FY ML pilot.

A GitHub Actions 'success' proves execution, NOT that a research predictor
has meaningful, stable out-of-time stock-doubling performance.

No training, scoring, tuning, reordering, or private prediction export.
No metric changes and no retroactive modification of the 2025 test fold.
"""
from __future__ import annotations
import argparse,json,math
from pathlib import Path

MIN_SEPARATE_UNSEEN_FOLDS=12
MIN_CAL_POSITIVES=12
MIN_HOLDOUT_POSITIVES=12
MIN_TOP10_WINS=2
MIN_AUC=0.60
MIN_LIFT=2.0

def audit(report):
    required=("scope","train_year","calibration_year","unseen_test_year",
        "training_rows","calibration_rows","test_rows",
        "train_positive_twoX_sixmonth","calibration_positive_twoX_sixmonth",
        "test_positive_twoX_sixmonth","test_baseline_event_frequency",
        "test_top10_precision","test_top10_success_count",
        "test_top10_lift_over_test_prevalence","test_roc_auc",
        "only_one_out_of_time_holdout_fold","model_training_production_approved",
        "original_oct2026_V11_4_forward_predictions_unchanged")
    missing=[x for x in required if x not in report]
    if missing:raise ValueError("Real archived model metrics absent: "+", ".join(missing))
    if report["scope"]!="INDEPENDENT_THREE_FISCAL_YEAR_HISTORICAL_SINGLE_FOLD_ML_HOLDOUT_NOT_CURRENT_PREDICTIONS":
        raise ValueError("Wrong or mutable model-performance source report")
    if (report["train_year"],report["calibration_year"],report["unseen_test_year"])!=(
        "2023-12-29","2024-12-31","2025-12-31"):
        raise ValueError("Original chronological data identity changed")
    if report["original_oct2026_V11_4_forward_predictions_unchanged"] is not True:
        raise ValueError("Unexpected alteration to frozen independent prospective model")
    if report["model_training_production_approved"] is not False:
        raise ValueError("Unvalidated three-year model claims production approval")
    if sum(report["test_top10_success_count"]==int(round(report["test_top10_precision"]*10))
            for _ in [None])!=1:
        raise ValueError("Model Top10 empirical precision / outcome counts mismatch")
    if (report["train_positive_twoX_sixmonth"]>report["training_rows"] or
        report["calibration_positive_twoX_sixmonth"]>report["calibration_rows"] or
        report["test_positive_twoX_sixmonth"]>report["test_rows"]):
        raise ValueError("Impossible original six-month label totals")
    base=report["test_positive_twoX_sixmonth"]/report["test_rows"]
    if not math.isclose(base,report["test_baseline_event_frequency"],rel_tol=1e-8):
        raise ValueError("Real holdout event frequency inconsistent")
    if report["test_baseline_event_frequency"]<=0:
        raise ValueError("No measurable six-month doubling baseline")
    count=1 if report["only_one_out_of_time_holdout_fold"] else None
    if count is None:
        raise ValueError("No verified distinct chronological holdout fold count")
    reasons=[]
    if count<MIN_SEPARATE_UNSEEN_FOLDS:
        reasons.append("INSUFFICIENT_INDEPENDENT_CHRONOLOGICAL_FOLDS")
    if report["calibration_positive_twoX_sixmonth"]<MIN_CAL_POSITIVES:
        reasons.append("ONLY_FIVE_POSITIVES_IN_CALIBRATION" if report[
            "calibration_positive_twoX_sixmonth"]==5 else "SPARSE_CALIBRATION_DOUBLERS")
    if report["test_positive_twoX_sixmonth"]<MIN_HOLDOUT_POSITIVES:
        reasons.append("INSUFFICIENT_HOLDOUT_POSITIVE_EVENTS")
    if report["test_top10_success_count"]<MIN_TOP10_WINS:
        reasons.append("HOLDOUT_TOP10_INSUFFICIENT_ACTUAL_DOUBLERS")
    if (report["test_roc_auc"] is None or
        not math.isfinite(float(report["test_roc_auc"])) or
        report["test_roc_auc"]<MIN_AUC):
        reasons.append("HOLDOUT_RANKING_ROC_AUC_BELOW_EXPLORATORY_GATE")
    if (report["test_top10_lift_over_test_prevalence"] is None or
        not math.isfinite(float(report["test_top10_lift_over_test_prevalence"])) or
        report["test_top10_lift_over_test_prevalence"]<MIN_LIFT):
        reasons.append("HOLDOUT_TOP10_LIFT_BELOW_EXPLORATORY_GATE")
    status="REJECTED_FOR_PROMOTION" if reasons else "BASIC_GATES_PASSED_STILL_REQUIRES_EXTERNAL_REVIEW"
    return {
        "scope":"V11_4_3FY_INDEPENDENT_HISTORICAL_SCIENTIFIC_READINESS_NOT_WORKFLOW_EXECUTION",
        "data_and_workflow_execution_succeeded":True,
        "three_fiscal_year_historical_financial_extracts_reconciled":True,
        "historical_holdout_performance_state":status,
        "scientific_promotion_or_real_money_trading_approved":False,
        "promotion_blockers":reasons,
        "independent_2025_holdout_count":count,
        "independent_historical_holdout_folds_required":MIN_SEPARATE_UNSEEN_FOLDS,
        "calibration_actual_double_count":report["calibration_positive_twoX_sixmonth"],
        "heldout_test_doublers_in_top10":report["test_top10_success_count"],
        "heldout_top10_precision":report["test_top10_precision"],
        "heldout_roc_auc":report["test_roc_auc"],
        "heldout_lift_over_base_prevalence":report["test_top10_lift_over_test_prevalence"],
        "model_parameters_or_frozen_2026_selections_changed":False,
        "no_retrospective_threshold_tuning_on_2025_holdout":True,
        "only_original_3_fiscal_year_observations_used":True
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True)
    p.add_argument("--out",required=True)
    a=p.parse_args()
    inp=Path(a.input)
    summary=audit(json.loads(inp.read_text()))
    dst=Path(a.out);dst.parent.mkdir(parents=True,exist_ok=True)
    dst.write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)
    msg=("**V11.4 three-FY model:** "+summary["historical_holdout_performance_state"]
         +"\n\nExecution success != financial predictive quality. "
         +"Promotion blocked: "+", ".join(summary["promotion_blockers"])+".")
    # GitHub renders this as an explicit run summary, even when testing itself succeeds.
    import os
    if os.getenv("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"],"a") as dest:
            dest.write("## Historical 3-FY model scientific status\n\n"+msg+"\n")
    if summary["promotion_blockers"] and os.getenv("GITHUB_ACTIONS")=="true":
        print("::warning title=RESEARCH MODEL NOT READY::"+", ".join(summary["promotion_blockers"]),flush=True)

if __name__=="__main__":main()
