"""Reject historically known outcomes no matter whether CI says success."""
import unittest
from copy import deepcopy
from v11_4_threeFY_combined_scientific_rejection_gate import assess

def source():
 return {
  "scope":"V11_4_RESEARCH_FROZEN_RECIPE_3FY_MOMENTUM_NSE_CATALYST_COMBINED_2X_HISTORIC_REPLICATION",
  "original_October_2026_frozen_V11_4_selections_unchanged":True,
  "not_a_new_blind_independent_test_2025_labels_seen_in_earlier_finance_only_tests":True,
  "no_real_money_trading_or_production_approval":True,
  "independent_clean_12fold_scientific_promotion_gate_satisfied":False,
  "train_rows":1654,"train_actual_2x_6m":113,
  "calibration_rows":976,"calibration_actual_2x_6m":45,
  "per_historic_fold_research_replication":[
    {"fold":"2025-06-30","evaluated_label_matured_source_clean_stocks":918,
     "six_month_2x_positive_stocks":9,
     "market_2x_baseline_frequency_in_evaluated_cohort":9/918,
     "top10_actual_six_month_doublers":0,
     "top10_precision":0.,"top10_lift_over_evaluated_cohort":0.,"roc_auc":.3992176995},
    {"fold":"2025-12-31","evaluated_label_matured_source_clean_stocks":1010,
     "six_month_2x_positive_stocks":33,
     "market_2x_baseline_frequency_in_evaluated_cohort":33/1010,
     "top10_actual_six_month_doublers":1,
     "top10_precision":.1,"top10_lift_over_evaluated_cohort":.1/(33/1010),
     "roc_auc":.5443069383}]
 }
class ScientificGate(unittest.TestCase):
 def test_actual_combined_historic_results_rejected_even_if_ci_success(self):
  x=assess(source())
  self.assertEqual(x["research_model_quality"],"REJECTED_FOR_SCIENTIFIC_PROMOTION")
  self.assertEqual(x["genuinely_new_blinded_independent_holdout_dates"],0)
  self.assertFalse(x["production_or_real_money_investing_approved"])
  self.assertTrue(any("2025-06-30_ROC" in r for r in x["rejection_reasons"]))
 def test_previous_2025_retrospective_attempt_marked_blinded_must_fail(self):
  x=source();x["not_a_new_blind_independent_test_2025_labels_seen_in_earlier_finance_only_tests"]=False
  with self.assertRaisesRegex(ValueError,"cannot be presented"):
   assess(x)
 def test_fake_12_independent_success_rejected(self):
  x=source();x["independent_clean_12fold_scientific_promotion_gate_satisfied"]=True
  with self.assertRaisesRegex(ValueError,"12-fold"):
   assess(x)
 def test_fake_win_claim_rejected(self):
  x=source();x["per_historic_fold_research_replication"][0]["top10_actual_six_month_doublers"]=4
  with self.assertRaisesRegex(ValueError,"precision inconsistent"):
   assess(x)
 def test_2026_frozen_model_immutable(self):
  x=source();x["original_October_2026_frozen_V11_4_selections_unchanged"]=False
  with self.assertRaisesRegex(ValueError,"no longer immutable"):
   assess(x)
 def test_source_changed_labels_cannot_pass(self):
  x=source();x["per_historic_fold_research_replication"][1]["six_month_2x_positive_stocks"]=200
  with self.assertRaisesRegex(ValueError,"inconsistent"):
   assess(x)
if __name__=="__main__":unittest.main()
