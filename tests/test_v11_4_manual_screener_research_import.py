"""Strict optional manually obtained Screener source boundary."""
import unittest
from datetime import date
import pandas as pd
from v11_4_manual_screener_research_import import transform_csv
class ManualResearchImport(unittest.TestCase):
 def sample(self):
  return pd.DataFrame({"NSE Code":["INFY","TCS"],
    "Sales growth 5Years":["15.2","12.0"],
    "Profit growth 7Years":["13.4","14.2"],
    "Promoter holding":["15.0","71.8"],
    "P/E":["25.1","33.2"],
    "Market Cap Rs.Cr.":["100000","200000"],
    "Debt to equity":["0.02","0.08"]})
 def test_import_fraction_ratio_and_unverified_state(self):
  z,m=transform_csv(self.sample(),str(date.today()))
  self.assertEqual(z.iloc[0]["sales_growth_5y"],.152)
  self.assertEqual(z.iloc[0]["profit_growth_7y"],.134)
  self.assertEqual(z.iloc[1]["promoter_holding"],.718)
  self.assertFalse(z["historical_2018_to_2025_training_eligible"].any())
  self.assertFalse(z["model_retraining_or_re_ranking_authorized"].any())
  self.assertEqual(len(m),7)
 def test_external_names_not_stock_identity(self):
  x=self.sample().drop(columns=["NSE Code"]).assign(Name=["Infosys","Tata"])
  with self.assertRaisesRegex(ValueError,"NSE Code"):
   transform_csv(x,str(date.today()))
 def test_duplicate_stock_rejected(self):
  x=self.sample();x.loc[1,"NSE Code"]="INFY"
  with self.assertRaisesRegex(ValueError,"Duplicate NSE"):
   transform_csv(x,str(date.today()))
 def test_blank_symbol_rejected(self):
  x=self.sample();x.loc[1,"NSE Code"]=""
  with self.assertRaisesRegex(ValueError,"Invalid or missing"):
   transform_csv(x,str(date.today()))
 def test_mixed_percent_not_allowed(self):
  x=self.sample();x.loc[0,"Sales growth 5Years"]="15%"
  with self.assertRaisesRegex(ValueError,"Mixed percent"):
   transform_csv(x,str(date.today()))
 def test_no_false_2022_pit_claim(self):
  x,_=transform_csv(self.sample(),"2026-10-08")
  self.assertTrue(x["third_party_values_as_reported_now_NOT_PIT_verified"].all())
 def test_reject_ambiguous_aliases(self):
  x=self.sample();x["Sales growth 5Year"]=x["Sales growth 5Years"]
  with self.assertRaisesRegex(ValueError,"Ambiguous duplicate"):
   transform_csv(x,str(date.today()))
 def test_reject_future_snapshot(self):
  with self.assertRaisesRegex(ValueError,"future"):
   transform_csv(self.sample(),"2099-01-01")
if __name__=="__main__":unittest.main()
