import unittest
import pandas as pd
from v11_4_covid_regime_sensitivity import intersects, summarize

class COVIDRegimeTests(unittest.TestCase):
 def test_june2019_does_not_overlap_covid_crash(self):
  self.assertFalse(intersects("2019-06-28","2019-12-31","2020-02-20","2020-06-30"))
 def test_december2019_future_six_month_window_overlaps(self):
  self.assertTrue(intersects("2019-12-31","2020-06-30","2020-02-20","2020-06-30"))
 def test_june2020_selection_covid_affected(self):
  self.assertTrue(intersects("2020-06-30","2020-12-31","2020-02-20","2020-06-30"))
 def test_june2023_not_affected(self):
  self.assertFalse(intersects("2023-06-30","2023-12-31","2020-02-20","2021-12-31"))
 def test_no_threshold_relaxation(self):
  f=pd.DataFrame({"date":pd.to_datetime(["2019-06-28","2023-06-30"]),
                  "status":["tested","tested"],"matured_selections":[10,10],
                  "six_month_double_hits":[0,3]})
  s=pd.DataFrame({"date":pd.to_datetime(["2019-06-28","2023-06-30"]),
                  "top10_jaccard":[1/9,.8]})
  out=summarize(f,s)
  self.assertFalse(out["twelve_fold_gate"])
  self.assertFalse(out["worst_p05_gate"])
  self.assertFalse(out["production_approved"])
if __name__=="__main__":unittest.main(verbosity=2)
