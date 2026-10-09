import unittest
import pandas as pd
from v11_4_threeFY_June_PIT_supplement import FOLDS,reopen_june_source
class JunePITSourceReopen(unittest.TestCase):
 def sample(self,year):
  day,yrs=FOLDS[year]
  m=pd.DataFrame([{"date":day,"symbol":f"J{i:04d}","close":120.,
   "avg_turnover_63":2000000.,"y6":int(i%13==0),
   "y6_mature_date":"2026-07-10T00:00:00Z","integrity_y6_clean":True} for i in range(1050)])
  fin=[]
  for i in range(1000):
   r={"symbol":f"J{i:04d}"}
   for fy in yrs:
    r[f"FY{fy}_revenue_INR"]=10_000_000*(fy-yrs[0]+1)
    r[f"FY{fy}_PAT_INR"]=100000
    r[f"FY{fy}_published_utc"]=f"{fy}-05-20T06:00:00Z"
    r[f"FY{fy}_source_sha256"]="f"*64
   fin.append(r)
  return pd.DataFrame(fin),m
 def test_all_three_june_folds_without_backdate(self):
  for year in FOLDS:
   with self.subTest(year=year):
    f,m=self.sample(year)
    joined,report=reopen_june_source(f,m,year)
    self.assertEqual(len(joined),1050)
    self.assertEqual(report["all_3FY_sources_with_original_June_stock_identity"],1000)
    self.assertEqual(report["eligible_matured_6month_twoX_labels"],1000)
    self.assertFalse(report["new_financial_ML_training_or_calibration_done"])
 def test_fiscal_report_published_after_June_must_fail_closed(self):
  f,m=self.sample(2025)
  f.loc[0,"FY2025_published_utc"]="2025-07-01T05:00:00Z"
  _,r=reopen_june_source(f,m,2025)
  self.assertEqual(r["all_3FY_sources_with_original_June_stock_identity"],999)
 def test_missing_3FY_original_hash_cannot_be_guessed(self):
  f,m=self.sample(2024)
  f.loc[0,"FY2023_source_sha256"]="INVALID"
  _,r=reopen_june_source(f,m,2024)
  self.assertEqual(r["all_3FY_sources_with_original_June_stock_identity"],999)
 def test_no_december_only_symbol_backfill(self):
  f,m=self.sample(2023)
  f.loc[0,"symbol"]="COMPANY_FROM_DEC_ONLY"
  _,r=reopen_june_source(f,m,2023)
  self.assertEqual(r["all_3FY_sources_with_original_June_stock_identity"],999)
if __name__=="__main__":unittest.main()
