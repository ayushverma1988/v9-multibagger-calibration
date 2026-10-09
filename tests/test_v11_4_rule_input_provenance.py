"""Strict missing vs valid source auditing for all four V11.4 user rules."""
import json, unittest
from pathlib import Path
import pandas as pd
import numpy as np
from v11_4_rule_input_provenance import enrich_and_audit,source_gate,parse_value
from v11_4_four_family_live_screener import analyze
config=json.loads(Path("config/v11_4_four_screener_families.json").read_text())
date="2026-10-08"
def market():
 base={"date":[date]*3,"symbol":["AAA","BBB","CCC"],
       "historical_asof_utc":["2026-10-08T10:00:00Z"]*3,
       "rsi14_wilder":[85.,70.,30.],
       "price_gt_dma50_prev":[True,False,True],
       "price_lt_dma200_prev":[True,False,True],
       "is_sme":[False]*3,
       "up_from_52w_low":[.55,.25,.75],
       "down_from_52w_high":[.1,.3,.02]}
 return pd.DataFrame(base)
def rec(sym,metric,value,ts="2026-10-08T09:30:00Z",url="https://nsearchives.nseindia.com/reports/report.xml",
        verified=True,period="2026-03-31",units="fraction"):
 return {"symbol":sym,"metric":metric,"value":value,"source_available_utc":ts,
         "source_url":url,"source_verified":verified,"source_period_end_utc":period,
         "reporting_mode":"consolidated","metric_units":units}
class Provenance(unittest.TestCase):
 def test_market_immediate_source_cover_six_fields_but_not_XBRL(self):
  x,coverage,info=enrich_and_audit(market(),config,date)
  self.assertEqual(info["screen_condition_count"],4)
  self.assertEqual(info["fields_with_any_verified_observations"],6)
  self.assertIn("profit_growth_7y",info["fields_still_fully_missing"])
  y=analyze(x,config)
  self.assertEqual(y["condition_4_RSI14_high_momentum_status"].tolist(),["PASS","FAIL","FAIL"])
  self.assertEqual(y["condition_3_earnings_acceleration_status"].tolist(),["UNKNOWN"]*3)
 def test_future_filings_untrusted_hosts_and_bad_units_all_rejected(self):
  recs=[
   rec("AAA","debt_to_equity",".4","2026-10-09T07:00:00Z"),
   rec("BBB","debt_to_equity",".4",url="https://fakesource-nseindia.com/x.xml"),
   rec("CCC","debt_to_equity","5",units="percent"),
   rec("AAA","promoter_holding",".51",verified=False),
  ]
  valid,summary=source_gate(pd.DataFrame(recs),date,{"debt_to_equity","promoter_holding"})
  self.assertEqual(len(valid),0)
  self.assertGreater(summary["rejected_future_or_missing_publication"],0)
  self.assertGreater(summary["rejected_provenance"],0)
 def test_eligible_historical_record_allowed_only_if_older_than_close(self):
  facts=pd.DataFrame([rec("AAA","debt_to_equity","0.40")])
  x,coverage,summary=enrich_and_audit(market(),config,date,facts)
  self.assertEqual(float(x.loc[x.symbol=="AAA","debt_to_equity"].iloc[0]),.4)
  self.assertTrue(pd.isna(x.loc[x.symbol=="BBB","debt_to_equity"].iloc[0]))
  self.assertEqual(summary["supplementary_source_metrics_audit"]["valid_original_point_in_time_metrics"],1)
 def test_overwriting_technical_or_RSI_is_never_allowed(self):
  facts=pd.DataFrame([rec("AAA","rsi14_wilder","99",units="wilder_0_100")])
  _,summary=source_gate(facts,date,{"debt_to_equity"})
  self.assertEqual(summary["valid_original_point_in_time_metrics"],0)
 def test_conflicting_same_time_facts_quarantined(self):
  facts=pd.DataFrame([rec("AAA","debt_to_equity","0.4"),
                      rec("AAA","debt_to_equity","0.8")])
  with self.assertRaisesRegex(ValueError,"Ambiguous"):
   enrich_and_audit(market(),config,date,facts)
 def test_bad_live_cutoff_fails_audit(self):
  x=market()
  x.loc[1,"historical_asof_utc"]="2026-10-08T10:01:00Z"
  with self.assertRaisesRegex(ValueError,"cutoff mismatch"):
   enrich_and_audit(x,config,date)
if __name__=="__main__":unittest.main(verbosity=2)
