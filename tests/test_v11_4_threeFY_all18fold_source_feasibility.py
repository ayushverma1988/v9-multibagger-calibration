import unittest
import pandas as pd
from v11_4_threeFY_all18fold_source_feasibility import eligible_three_fy
DATES=["2017-06-30","2017-12-29","2018-06-29","2018-12-31","2019-06-28","2019-12-31",
 "2020-06-30","2020-12-31","2021-06-30","2021-12-31","2022-06-30","2022-12-30",
 "2023-06-30","2023-12-29","2024-06-28","2024-12-31","2025-06-30","2025-12-31"]
class SourceFeasibility(unittest.TestCase):
 def make(self):
  stocks=[{"date":day,"symbol":f"A{i:04d}"} for day in DATES for i in range(320)]
  docs=[{"symbol":f"A{i:04d}","fy_end":f"{yr}-03-31",
    "available_at_utc":f"{yr}-06-01T07:00:00Z",
    "consolidated":"Consolidated","xbrl_url":f"https://nsearchives.nseindia.com/file{i}_{yr}.xml"}
    for i in range(320) for yr in range(2014,2026)]
  return pd.DataFrame(docs),pd.DataFrame(stocks),pd.DataFrame({"date":DATES})
 def test_historical_filing_before_1530_ist_only(self):
  docs,market,folds=self.make()
  inventory,links=eligible_three_fy(docs,market,folds)
  self.assertEqual(len(inventory),18)
  self.assertTrue(inventory["point_in_time_source_url_coverage_pct"].eq(100).all())
  self.assertEqual(len(links),18*320*3)
 def test_late_june_fy_source_excluded_not_backdated(self):
  docs,market,folds=self.make()
  docs.loc[(docs["symbol"]=="A0001")&(docs["fy_end"]=="2023-03-31"),"available_at_utc"]="2023-07-01T09:00:00Z"
  report,_=eligible_three_fy(docs,market,folds)
  self.assertEqual(int(report.loc[report["historical_fold"]=="2023-06-30","source_filing_link_sets_before_fold_close"].iloc[0]),319)
  self.assertEqual(int(report.loc[report["historical_fold"]=="2023-12-29","source_filing_link_sets_before_fold_close"].iloc[0]),320)
 def test_same_issuer_requires_exact_mode(self):
  docs,market,folds=self.make()
  docs.loc[(docs["symbol"]=="A0001")&(docs["fy_end"]=="2023-03-31"),"consolidated"]="Standalone"
  report,_=eligible_three_fy(docs,market,folds)
  self.assertEqual(int(report.loc[report["historical_fold"]=="2023-12-29","source_filing_link_sets_before_fold_close"].iloc[0]),319)
 def test_never_guess_missing_fiscal_year(self):
  docs,market,folds=self.make()
  docs=docs[~((docs["symbol"]=="A0001")&(docs["fy_end"]=="2022-03-31"))]
  report,_=eligible_three_fy(docs,market,folds)
  self.assertEqual(int(report.loc[report["historical_fold"]=="2023-12-29","source_filing_link_sets_before_fold_close"].iloc[0]),319)
if __name__=="__main__":unittest.main()
