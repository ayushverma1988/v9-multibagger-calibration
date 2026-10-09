"""No-lookahead and immutable source checks for 3FY chronological pilot."""
import unittest
import numpy as np
import pandas as pd
from v11_4_threeFY_chronological_research_ML import (
  original_financial_features,eligibility,historical_holdout,fold_close,BASE)

def synthetic_year(y,syms):
 cfg=BASE[y];a,b,c=cfg["years"]
 rows=[]
 for i,s in enumerate(syms):
  scale=1.+(i%21)*.02
  rows.append({
   "symbol":s,
   f"FY{a}_revenue_INR":100.*scale,
   f"FY{b}_revenue_INR":115.*scale,
   f"FY{c}_revenue_INR":145.*scale,
   f"FY{a}_PAT_INR":10.*scale,
   f"FY{b}_PAT_INR":12.*scale,
   f"FY{c}_PAT_INR":17.*scale})
 return pd.DataFrame(rows)

class Chronological3FYTests(unittest.TestCase):
 def make(self):
  frames=[];fin={}
  for y in (2023,2024,2025):
   date=BASE[y]["date"];rows=[];syms=[f"S{i:04d}" for i in range(1050)]
   fin[y]=synthetic_year(y,syms)
   for i,sym in enumerate(syms):
    mature={2023:"2024-07-15T10:00:00Z",2024:"2025-07-15T10:00:00Z",2025:"2026-07-15T10:00:00Z"}[y]
    rows.append({"date":date,"symbol":sym,"close":100+i*.001,
      "avg_turnover_63":3_000_000.+i*50,"y6":int((i%31)==0),
      "y6_mature_date":mature,"integrity_y6_clean":True})
   frames.append(pd.DataFrame(rows))
  return fin,pd.concat(frames,ignore_index=True)
 def test_complete_threefiscal_sources_can_yield_matured_chronological_test(self):
  fi,x=self.make()
  picks,meta=historical_holdout(fi[2023],fi[2024],fi[2025],x)
  self.assertEqual((meta["training_rows"],meta["calibration_rows"],meta["test_rows"]),(1050,1050,1050))
  self.assertFalse(meta["model_training_production_approved"])
  self.assertTrue(meta["only_one_out_of_time_holdout_fold"])
  self.assertEqual(len(picks),1050)
  self.assertTrue(picks["exploratory_p6_double_model_probability"].between(0,1).all())
 def test_future_maturity_excludes_training_stock(self):
  fi,x=self.make()
  x.loc[(x["date"]=="2023-12-29")&(x["symbol"]=="S0000"),"y6_mature_date"]="2026-01-01T00:00:00Z"
  z=original_financial_features(fi[2023],x.assign(date=pd.to_datetime(x["date"])),2023)
  matured=eligibility(z,fold_close("2024-12-31"))
  self.assertNotIn("S0000",set(matured["symbol"]))
 def test_incomplete_source_cannot_reuse_company_twice(self):
  fi,x=self.make();fi[2024].loc[1,"symbol"]="S0000"
  with self.assertRaisesRegex(ValueError,"Duplicate"):
   historical_holdout(fi[2023],fi[2024],fi[2025],x)
 def test_no_profit_source_imputation(self):
  fi,x=self.make();fi[2025].loc[0,"FY2025_PAT_INR"]=None
  with self.assertRaisesRegex(ValueError,"incomplete"):
   historical_holdout(fi[2023],fi[2024],fi[2025],x)
if __name__=="__main__":unittest.main()
