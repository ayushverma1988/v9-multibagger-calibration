"""Same-mode FY link + historically available publication proof, 2020-2021."""
import unittest
from unittest.mock import patch
import pandas as pd
import v11_4_recover_2020_2021_threeFY_PIT_source_triage as m
def sample():
 doc=[];stock=[]
 for day,meta in m.TARGETS.items():
  for i in range(10):
   sym=f"{day[:4]}{day[5:7]}SYM{i}"
   stock.append({"date":day,"symbol":sym})
   for yr in meta["fys"]:
    statement="Consolidated"
    if i==7 and yr==meta["fys"][-1]:statement="Standalone"
    if i==9 and yr==meta["fys"][-1]:continue
    pub=f"{yr}-06-01T09:00:00Z"
    if i==8 and yr==meta["fys"][-1]:pub=f"{yr}-07-01T09:00:00Z"
    doc.append({"symbol":sym,"fy_end":f"{yr}-03-31",
      "available_at_utc":pub,"consolidated":statement,
      "xbrl_url":f"https://nsearchives.nseindia.com/{sym}_{yr}_{statement}.xml"})
 return pd.DataFrame(doc),pd.DataFrame(stock)
class Triage(unittest.TestCase):
 def setUp(self):
  self.meta={
   "2020-06-30":{"fys":(2018,2019,2020),"universe":10,"original_link_sets":7},
   "2020-12-31":{"fys":(2018,2019,2020),"universe":10,"original_link_sets":8},
   "2021-06-30":{"fys":(2019,2020,2021),"universe":10,"original_link_sets":7},
   "2021-12-31":{"fys":(2019,2020,2021),"universe":10,"original_link_sets":8}}
  self.p=patch.object(m,"TARGETS",self.meta);self.p.start()
  self.addCleanup(self.p.stop)
  self.docs,self.stocks=sample()
 def test_older_PIT_no_late_year_and_no_mixed_mode(self):
  result,rows=m.triage(self.docs,self.stocks)
  by={x["original_fold"]:x for x in result["per_date"]}
  self.assertEqual(by["2021-06-30"]["original_eligible_same_mode_threeFY_NSE_XML_link_sets"],7)
  self.assertEqual(by["2021-12-31"]["original_eligible_same_mode_threeFY_NSE_XML_link_sets"],8)
  self.assertTrue(rows["later_docs_not_usable_for_historical_selection"].all())
  self.assertEqual(result["four_frozen_historical_pre2022_folds"],4)
  self.assertTrue(result["source_link_percent_not_equivalent_to_numeric_coverage"])
 def test_malformed_late_source_not_backfilled_asof(self):
  self.docs.loc[(self.docs["symbol"].str.endswith("SYM0"))&(self.docs["fy_end"]=="2021-03-31"),"available_at_utc"]="2022-06-01T07:00:00Z"
  with self.assertRaisesRegex(ValueError,"index candidate count changed"):
   m.triage(self.docs,self.stocks)
 def test_never_mix_standalone_and_consolidated(self):
  self.docs.loc[(self.docs["symbol"].str.endswith("SYM6"))&(self.docs["fy_end"]=="2021-03-31"),"consolidated"]="Standalone"
  with self.assertRaisesRegex(ValueError,"index candidate count changed"):
   m.triage(self.docs,self.stocks)
 def test_missing_stock_denominator_rejected(self):
  x=self.stocks.drop(self.stocks.index[0])
  with self.assertRaisesRegex(ValueError,"denominator drift"):
   m.triage(self.docs,x)
 def test_future_market_targets_are_banned(self):
  x=self.stocks.copy();x["y6"]=1
  with self.assertRaisesRegex(ValueError,"No future stock"):
   m.triage(self.docs,x)
 def test_wrong_host_cannot_recover_link(self):
  self.docs.loc[(self.docs["symbol"].str.endswith("SYM1"))&(self.docs["fy_end"]=="2021-03-31"),"xbrl_url"]="https://fake-nsearchives.example/invalid.xml"
  with self.assertRaisesRegex(ValueError,"index candidate count changed"):
   m.triage(self.docs,self.stocks)
if __name__=="__main__":unittest.main()
