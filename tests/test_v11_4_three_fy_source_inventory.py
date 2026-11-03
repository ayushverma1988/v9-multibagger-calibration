import unittest
import pandas as pd
from v11_4_three_fy_source_inventory import candidates
class ThreeFiscalYearSourceContracts(unittest.TestCase):
 def base(self):
  stocks=pd.DataFrame({"date":["2023-12-29"]*510,"symbol":[f"C{i:03d}" for i in range(510)]})
  z=[]
  for i in range(510):
   s=f"C{i:03d}"
   for yr in (2021,2022,2023):
    z.append({"symbol":s,"fy_end":f"{yr}-03-31","available_at_utc":f"{yr}-06-30T08:00:00Z",
     "consolidated":"Consolidated","xbrl_url":f"https://nsearchives.nseindia.com/CORP/{s}_{yr}.xml"})
  return pd.DataFrame(z),stocks
 def test_3_fiscal_points_give_2_year_not_3yr_cagr(self):
  z,stocks=self.base();selected,report=candidates(z,stocks)
  self.assertEqual(report["original_three_consecutive_FY_source_URL_sets"],510)
  self.assertFalse(report["allows_true_3y_sales_cagr"])
  self.assertEqual(len(selected),1530)
 def test_future_filing_not_in_original_2023_fold(self):
  z,stocks=self.base();z.loc[(z["symbol"]=="C001")&(z["fy_end"]=="2023-03-31"),"available_at_utc"]="2024-01-01T00:00:00Z"
  selected,report=candidates(z,stocks)
  self.assertEqual(report["original_three_consecutive_FY_source_URL_sets"],509)
  self.assertNotIn("C001",set(selected["symbol"]))
 def test_fy2021_missing_fails_consecutive(self):
  z,stocks=self.base();z=z.loc[~((z["symbol"]=="C001")&(z["fy_end"]=="2021-03-31"))]
  _,report=candidates(z,stocks)
  self.assertEqual(report["original_FY2022_FY2023_source_pairs"],510)
  self.assertEqual(report["original_three_consecutive_FY_source_URL_sets"],509)
 def test_mixed_reporting_mode_not_paired(self):
  z,stocks=self.base();z.loc[(z["symbol"]=="C001")&(z["fy_end"]=="2022-03-31"),"consolidated"]="Non-Consolidated"
  _,report=candidates(z,stocks)
  self.assertEqual(report["original_three_consecutive_FY_source_URL_sets"],509)
 def test_foreign_unapproved_document_host_rejected(self):
  z,stocks=self.base();z.loc[(z["symbol"]=="C001")&(z["fy_end"]=="2021-03-31"),"xbrl_url"]="https://example.com/file.xml"
  _,report=candidates(z,stocks)
  self.assertEqual(report["original_three_consecutive_FY_source_URL_sets"],509)
if __name__=="__main__":unittest.main()
