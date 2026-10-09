"""Original RSI>70 user screen early-only descriptive outcomes, never tune 2025 folds."""
import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np
import v11_4_RSI70_threeFY_earlier_training_only_signal_association as m

def synthetic():
 feature=[];labels=[]
 days=(("2022-12-30","2023-07-20T00:00:00Z"),
       ("2023-06-30","2024-01-20T00:00:00Z"),
       ("2024-06-28","2025-01-20T00:00:00Z"))
 for day,mature in days:
  for i in range(110):
   sym=f"A{i:04d}"
   rsi_gt=i%5==0
   feature.append({"date":day,"symbol":sym,"close":100+i*.1,
    "avg_turnover_63":1500000.+i*120,
    "integrity_feature_clean":True,
    "has_strict_3FY_original_fiscal_source":True,
    "rsi14_Wilder_research_verified":True,
    "RSI_120session_archive_available_PLUS_official_day_close":True,
    "RSI14_STRICT_GT70_when_official_NSE_DAY_close_confirmed":rsi_gt,
    "promoter_buying_verified_180d":pd.NA})
   labels.append({"date":day,"symbol":sym,"close":100+i*.1,
    "avg_turnover_63":1500000.+i*120,"y6":int(i%11==0),
    "y6_mature_date":mature,"integrity_y6_clean":True})
 return pd.DataFrame(feature),pd.DataFrame(labels)

class Association(unittest.TestCase):
 def setUp(self):
  self.f,self.l=synthetic()
  self.patch=patch.dict(m.EXPECT_UNIVERSE,{d:110 for d in m.USED_DATES},clear=True)
  self.patch.start();self.addCleanup(self.patch.stop)
 def test_train_cal_only_descriptive_RSI70_not_model_promotion(self):
  x=m.inspect(self.f,self.l)
  self.assertEqual(len(x["descriptive_by_early_training_and_calibration_period"]),4)
  self.assertTrue(x["no_2025_test_fold_six_month_doublers_used"])
  self.assertEqual(x["evaluated_only_stock_label_fold_dates"],list(m.USED_DATES))
  self.assertTrue(x["this_is_observational_association_not_actual_backtested_model_gain"])
  self.assertFalse(x["prospective_or_production_model_promoted"])
 def test_wrong_matured_date_not_counted_in_2023_train(self):
  self.l.loc[self.l["date"].eq("2023-06-30")&self.l["symbol"].eq("A0000"),
    "y6_mature_date"]="2025-01-01T00:00:00Z"
  r=m.inspect(self.f,self.l)
  second=r["descriptive_by_early_training_and_calibration_period"][1]
  self.assertEqual(second["original_eligible_known_threeFY_clean_labels_and_official_dayclose_RSI"],109)
 def test_future_2025_label_only_is_ignored_as_holdout(self):
  baseline=m.inspect(self.f,self.l)
  added=pd.DataFrame([{"date":"2025-12-31","symbol":"A0000","close":30.,
    "avg_turnover_63":2e6,"y6":1,"y6_mature_date":"2026-07-20T00:00:00Z",
    "integrity_y6_clean":True}])
  result=m.inspect(self.f,pd.concat([self.l,added],ignore_index=True))
  self.assertEqual(baseline,result)
 def test_unknown_RSI_as_false_rejected(self):
  self.f.loc[0,"RSI_120session_archive_available_PLUS_official_day_close"]=False
  with self.assertRaisesRegex(ValueError,"missing-data gate"):
   m.inspect(self.f,self.l)
 def test_fitted_score_in_PIT_source_not_allowed(self):
  self.f["p6_double_calibrated"]=.99
  with self.assertRaisesRegex(ValueError,"contaminated"):
   m.inspect(self.f,self.l)
 def test_change_maturity_cannot_use_future_training_2024_cal(self):
  self.l.loc[self.l["date"].eq("2024-06-28")&self.l["symbol"].eq("A0000"),
    "y6_mature_date"]="2026-01-01T00:00:00Z"
  x=m.inspect(self.f,self.l)
  cal=x["descriptive_by_early_training_and_calibration_period"][2]
  self.assertEqual(cal["original_eligible_known_threeFY_clean_labels_and_official_dayclose_RSI"],109)
if __name__=="__main__":unittest.main()
