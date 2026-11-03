import unittest
from v11_4_threeFY_scientific_readiness_guard import audit
def recorded_result():
 return {
  "scope":"INDEPENDENT_THREE_FISCAL_YEAR_HISTORICAL_SINGLE_FOLD_ML_HOLDOUT_NOT_CURRENT_PREDICTIONS",
  "train_year":"2023-12-29","calibration_year":"2024-12-31","unseen_test_year":"2025-12-31",
  "training_rows":960,"calibration_rows":1060,"test_rows":1010,
  "train_positive_twoX_sixmonth":61,"calibration_positive_twoX_sixmonth":5,
  "test_positive_twoX_sixmonth":33,"test_baseline_event_frequency":33/1010,
  "test_top10_precision":0.0,"test_top10_success_count":0,"test_top10_lift_over_test_prevalence":0.0,
  "test_roc_auc":0.5487112682609101,"only_one_out_of_time_holdout_fold":True,
  "model_training_production_approved":False,
  "original_oct2026_V11_4_forward_predictions_unchanged":True}
class Guard(unittest.TestCase):
 def test_actual_exploratory_holdout_rejected_on_no_wins(self):
  s=audit(recorded_result())
  self.assertEqual(s["historical_holdout_performance_state"],"REJECTED_FOR_PROMOTION")
  self.assertIn("HOLDOUT_TOP10_INSUFFICIENT_ACTUAL_DOUBLERS",s["promotion_blockers"])
  self.assertIn("INSUFFICIENT_INDEPENDENT_CHRONOLOGICAL_FOLDS",s["promotion_blockers"])
  self.assertIn("ONLY_FIVE_POSITIVES_IN_CALIBRATION",s["promotion_blockers"])
  self.assertFalse(s["scientific_promotion_or_real_money_trading_approved"])
 def test_fake_green_workflow_cannot_override_bad_science(self):
  a=recorded_result();a["test_roc_auc"]=0.99;a["test_top10_precision"]=0.3
  a["test_top10_success_count"]=3;a["test_top10_lift_over_test_prevalence"]=0.3/(33/1010)
  s=audit(a)
  self.assertEqual(s["historical_holdout_performance_state"],"REJECTED_FOR_PROMOTION")
  self.assertIn("INSUFFICIENT_INDEPENDENT_CHRONOLOGICAL_FOLDS",s["promotion_blockers"])
 def test_future_model_change_aborts_source_verification(self):
  a=recorded_result();a["original_oct2026_V11_4_forward_predictions_unchanged"]=False
  with self.assertRaisesRegex(ValueError,"alteration"):
   audit(a)
 def test_chronological_fold_dates_fail_closed(self):
  a=recorded_result();a["train_year"]="2025-12-31"
  with self.assertRaisesRegex(ValueError,"chronological"):
   audit(a)
 def test_inconsistent_holdout_doubling_count_rejected(self):
  a=recorded_result();a["test_top10_precision"]=0.3
  with self.assertRaisesRegex(ValueError,"precision"):
   audit(a)
 def test_unverifiable_source_report_rejected(self):
  a=recorded_result();del a["test_roc_auc"]
  with self.assertRaisesRegex(ValueError,"absent"):
   audit(a)
if __name__=="__main__":unittest.main()
