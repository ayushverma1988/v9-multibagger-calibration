import unittest
import pandas as pd
from v11_4_three_fy_financial_rule_demo import score
class ThreeYearFinancialDemo(unittest.TestCase):
 def fixtures(self):
  rows=[]
  stock=[]
  for i in range(1314):
   sy=f"C{i:04d}"
   stock.append({"symbol":sy,"date":"2025-12-31","close":150,"avg_turnover_63":5_000_000})
   if i<1013:
    rows.append({"symbol":sy,"FY2023_revenue_INR":100.,"FY2024_revenue_INR":118.,
      "FY2025_revenue_INR":150.,"FY2023_PAT_INR":10.,
      "FY2024_PAT_INR":12.,"FY2025_PAT_INR":20.,
      "revenue_2023_to_2025_2yr_CAGR":(1.5**.5)-1,
      "revenue_FY2025_yoy":150/118-1,
      "profit_2023_to_2025_2yr_CAGR":(2.**.5)-1,
      "FY2025_source_SHA256":"a"*64})
  return pd.DataFrame(rows),pd.DataFrame(stock)
 def test_three_year_only_financial_no_model_probability(self):
  f,s=self.fixtures();full,picks,summary=score(f,s)
  self.assertEqual(len(full),1314)
  self.assertEqual(len(picks),1013)
  self.assertEqual(summary["stocks_without_exact_fiscal_data_left_in_full_universe"],301)
  self.assertTrue(summary["no_six_month_forward_returns_or_labels_loaded"])
  self.assertFalse(summary["production_approved"])
  self.assertFalse(any("probab" in c.lower() or "y6"==c for c in full))
 def test_liquidity_price_and_growth_separately_exclude(self):
  f,s=self.fixtures()
  s.loc[0,"avg_turnover_63"]=0
  s.loc[1,"close"]=3000
  f.loc[2,"FY2025_PAT_INR"]=-1
  f.loc[3,"revenue_FY2025_yoy"]=0
  _,picks,report=score(f,s)
  self.assertEqual(len(picks),1009)
  self.assertEqual(report["stocks_passing_all_predeclared_short_history_rules"],1009)
 def test_duplicate_financial_identity_rejected(self):
  f,s=self.fixtures();f.loc[1,"symbol"]=f.loc[0,"symbol"]
  with self.assertRaisesRegex(ValueError,"duplicated"):
   score(f,s)
 def test_source_count_sentinel_prevents_imputation(self):
  f,s=self.fixtures();f=f.iloc[:-1]
  with self.assertRaisesRegex(ValueError,"coverage"):
   score(f,s)
if __name__=="__main__":unittest.main()
