import unittest
import pandas as pd
from v11_4_threeFY_June_clean_out_of_sample_ML import YEARS,make_features,june_threeFY_single_holdout
class StrictJuneCalibratedML(unittest.TestCase):
 def src(self,y):
  date,years=YEARS[y]
  frames=[]
  for i in range(1030):
   r={"date":date,"symbol":f"S{i:04d}","close":float(45+(i%200)),
    "avg_turnover_63":float(3_000_000+(i%87)*33_000),
    "y6":int(i%22==0) if y!=2025 else int(i%113==0),
    "y6_mature_date":{2023:"2024-01-31T00:00:00Z",2024:"2025-01-31T00:00:00Z",2025:"2026-01-31T00:00:00Z"}[y],
    "integrity_y6_clean":True,"verified_3FY_before_actual_June_fold_close":True,
    "source_eligible_mature_six_month_research_label":True}
   for j,fy in enumerate(years):
    r[f"FY{fy}_revenue_INR"]=float((10+j)*(1000+i%15))
    r[f"FY{fy}_PAT_INR"]=float((j+2)*(100+i%45))
   frames.append(r)
  return pd.DataFrame(frames)
 def test_heldout_three_independent_Junes_never_2025_Dec(self):
  x,y,z=[self.src(a) for a in YEARS]
  result,summary=june_threeFY_single_holdout(x,y,z)
  self.assertEqual((summary["train_original_date"],summary["unseen_original_test_date"]),("2023-06-30","2025-06-30"))
  self.assertEqual(summary["train_actual_2x_6m_event_count"],47)
  self.assertTrue(summary["test_holdout_minimum_12_events_guard_passed"] is False)
  self.assertTrue(summary["original_2025_Dec_test_labels_scores_NEVER_READ"])
  self.assertFalse(summary["training_or_trading_production_approved"])
  self.assertEqual(len(result),1030)
 def test_calibration_labels_later_than_test_decision_must_not_train(self):
  x=self.src(2024)
  x.loc[:100,"y6_mature_date"]="2025-07-02T10:00:00Z"
  out=make_features(x,2024,pd.Timestamp("2025-06-30T10:00:00Z"))
  self.assertEqual(len(out),929)
 def test_few_calibration_positive_events_rejected(self):
  x,y,z=[self.src(a) for a in YEARS]
  y["y6"]=0;y.loc[:4,"y6"]=1
  with self.assertRaisesRegex(ValueError,"too few"):
   june_threeFY_single_holdout(x,y,z)
 def test_source_numeric_gaps_not_imputed_to_true_growth(self):
  x=self.src(2023);x.loc[0,"FY2022_revenue_INR"]=None
  with self.assertRaisesRegex(ValueError,"null numeric"):
   make_features(x,2023,pd.Timestamp("2024-06-28T10:00:00Z"))
if __name__=="__main__":unittest.main()
