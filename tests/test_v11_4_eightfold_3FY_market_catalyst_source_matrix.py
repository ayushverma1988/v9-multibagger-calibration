import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import pandas as pd
from v11_4_eightfold_3FY_market_catalyst_source_matrix import (
 combine,FOLDS,EXPECT_UNIVERSE,PRICE,EVENTS,FIN_FEATS)
class EightfoldNoLeakage(unittest.TestCase):
 def tiny(self,year="2025-06-30"):
  yrs=(2023,2024,2025)
  prices=[]
  for i in range(4):
   r={"date":year,"symbol":f"Z{i:03d}","historical_asof_utc":"2025-06-30T10:00:00Z"}
   r.update({c:1. for c in PRICE})
   r.update({c:0 for c in EVENTS})
   r["integrity_feature_clean"]=True
   prices.append(r)
  f=[]
  for i in range(3):
   row={"date":year,"symbol":f"Z{i:03d}","verified_3FY_before_actual_June_fold_close":True}
   for k,fy in enumerate(yrs):
    row[f"FY{fy}_revenue_INR"]=100+k*20
    row[f"FY{fy}_PAT_INR"]=10+k*4
   f.append(row)
  return pd.DataFrame(prices),pd.DataFrame(f)
 def test_source_preserves_original_market_universe_and_unknowns(self):
  base,fin=self.tiny()
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/"f.parquet";fin.to_parquet(p,index=False)
   with patch.dict(FOLDS,{"2025-06-30":((2023,2024,2025),3,"june")},clear=True):
    with patch.dict(EXPECT_UNIVERSE,{"2025-06-30":4},clear=True):
     joint,report=combine(base,{"2025-06-30":p})
   self.assertEqual(len(joint),4)
   self.assertEqual(report["original_NSE_strict_3FY_company_date_rows"],3)
   self.assertTrue(joint.loc[joint["symbol"].eq("Z003"),list(FIN_FEATS)].isna().all().all())
   self.assertFalse(report["2022_June_70pct_coverage_threshold_passed"] is False)
 def test_source_rejects_future_NSE_catalyst(self):
  base,fin=self.tiny();base.loc[0,"historical_asof_utc"]="2025-07-02T10:00:00Z"
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/"f.parquet";fin.to_parquet(p,index=False)
   with patch.dict(FOLDS,{"2025-06-30":((2023,2024,2025),3,"june")},clear=True):
    with patch.dict(EXPECT_UNIVERSE,{"2025-06-30":4},clear=True):
     with self.assertRaisesRegex(ValueError,"violates historical"):
      combine(base,{"2025-06-30":p})
 def test_cannot_load_any_outcome_as_original_training_input(self):
  base,fin=self.tiny();base["y6"]=1
  with patch.dict(FOLDS,{"2025-06-30":((2023,2024,2025),3,"june")},clear=True):
   with patch.dict(EXPECT_UNIVERSE,{"2025-06-30":4},clear=True):
    with self.assertRaisesRegex(ValueError,"Future outcome"):
     combine(base,{"2025-06-30":"unused"})
 def test_future_fiscal_document_excluded_from_Dec2025_original_matrix(self):
  base,fin=self.tiny("2025-12-31")
  for fy in (2023,2024,2025):
   fin[f"FY{fy}_published_utc"]=f"{fy}-05-01T06:00:00Z"
   fin[f"FY{fy}_source_SHA256"]="a"*64
  fin.loc[0,"FY2025_published_utc"]="2026-01-01T00:00:00Z"
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/"f.csv";fin.drop(columns=["date","verified_3FY_before_actual_June_fold_close"]).to_csv(p,index=False)
   with patch.dict(FOLDS,{"2025-12-31":((2023,2024,2025),3,"december")},clear=True):
    with patch.dict(EXPECT_UNIVERSE,{"2025-12-31":4},clear=True):
     with self.assertRaisesRegex(ValueError,"published after decision"):
      combine(base,{"2025-12-31":p})
if __name__=="__main__":unittest.main()
