"""No leaked targets or fabricated positive promoter buys in eightfold RSI>70 research."""
import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np
import v11_4_threeFY_RSI70_promoter_source_ready as m

def cases():
 days=("2023-12-29","2025-12-31")
 f=[];r=[];e=[];close=[]
 for d in days:
  for j in range(2):
   sym=f"TEST{j}"
   q={"date":d,"symbol":sym,"historical_asof_utc":d+"T09:00:00Z",
     "has_strict_3FY_original_fiscal_source":True,
     "nse_promoter_activity_180d_positive":0,
     "nse_catalyst_total_180d":7,
     "nse_capacity_expansion_180d_positive":2,
     "nse_order_win_180d_positive":1,
     "nse_regulatory_180d_positive":0}
   f.append(q)
   r.append({"date":d,"symbol":sym,"rsi14_wilder_source":71 if j==0 else np.nan,
     "fourth_family_rsi_strictly_gt70":True if j==0 else pd.NA,
     "source_verification":m.STRICT_STATUS if j==0 else "PRICE_SOURCE_UNKNOWN",
     "original_frozen_market_close_INR":400+j})
   e.append({"date":d,"symbol":sym,"nse_promoter_activity_180d":3+j,
             "nse_earnings_180d":4,
             "nse_promoter_activity_180d_positive":0})
   close.append({"date":d,"symbol":sym,"close":400+j})
 return list(map(pd.DataFrame,(f,r,e,close)))

class HistoricalRSI70Source(unittest.TestCase):
 def setUp(self):
  self.source,self.rsi,self.events,self.close=cases()
  self.dates={"2023-12-29":2,"2025-12-31":2}
  self.patch=patch.object(m,"ORIGINAL_DATES",self.dates)
  self.patch.start()
  self.addCleanup(self.patch.stop)
 def run_source(self):
  return m.join_no_labels(self.source,self.rsi,self.events,self.close)
 def test_verifiable_rsi_gt70_and_unknown_not_false(self):
  frame,summary=self.run_source()
  self.assertEqual(len(frame),4)
  self.assertEqual(summary["historical_source_RSI14_verified"],2)
  self.assertEqual(summary["historical_source_RSI14_missing_UNKNOWN"],2)
  self.assertEqual(summary["historical_RSI14_strictly_GT70"],2)
  self.assertEqual(frame["condition_4_RSI14_GT70_source_status"].tolist(),
                   ["PASS","UNKNOWN","PASS","UNKNOWN"])
  self.assertEqual(frame["nse_material_specific_positive_180d_count"].tolist(),[3]*4)
  self.assertTrue(frame["promoter_buying_verified_180d"].isna().all())
  self.assertTrue(summary["frozen_Oct2026_stock_ranks_and_probabilities_unchanged"])
  self.assertFalse(summary["historical_raw_daily_bar_archive_provenance_independently_verified_against_all_NSE_sessions"])
 def test_true_rsi70_does_not_pass_and_exactly_70_missing_not_passed(self):
  self.rsi.loc[0,"rsi14_wilder_source"]=70.0
  self.rsi.loc[0,"fourth_family_rsi_strictly_gt70"]=False
  a,_=self.run_source()
  self.assertEqual(a.iloc[0]["condition_4_RSI14_GT70_source_status"],"FAIL")
  self.assertEqual(a.iloc[1]["condition_4_RSI14_GT70_source_status"],"UNKNOWN")
 def test_unknown_rsi_substituted_as_false_rejected(self):
  self.rsi.loc[1,"fourth_family_rsi_strictly_gt70"]=False
  with self.assertRaisesRegex(ValueError,"Unknown RSI"):
   self.run_source()
 def test_verified_rsi_missing_recorded_verdict_fails(self):
  self.rsi.loc[0,"fourth_family_rsi_strictly_gt70"]=pd.NA
  with self.assertRaisesRegex(ValueError,"threshold"):
   self.run_source()
 def test_false_claim_generic_promoter_equals_verified_buying_fails(self):
  self.source.loc[0,"nse_promoter_activity_180d_positive"]=1
  with self.assertRaisesRegex(ValueError,"Promoter positive direction"):
   self.run_source()
 def test_even_canonical_positive_promoter_requires_individual_document_review(self):
  self.source.loc[0,"nse_promoter_activity_180d_positive"]=1
  self.events.loc[0,"nse_promoter_activity_180d_positive"]=1
  with self.assertRaisesRegex(ValueError,"undocumented positive"):
   self.run_source()
 def test_future_NSE_filing_clock_fails(self):
  self.source.loc[0,"historical_asof_utc"]="2026-01-01T10:00:00Z"
  with self.assertRaisesRegex(ValueError,"Post-fold"):
   self.run_source()
 def test_frozen_market_close_mismatch_fails(self):
  self.close.loc[0,"close"]=499.
  with self.assertRaisesRegex(ValueError,"immutable close"):
   self.run_source()
 def test_changing_original_market_row_count_fails(self):
  self.source=self.source.iloc[:-1]
  with self.assertRaisesRegex(ValueError,"identities changed"):
   self.run_source()
 def test_outcome_in_research_source_fails(self):
  self.source["y6"]=1
  with self.assertRaisesRegex(ValueError,"future outcome"):
   self.run_source()
 def test_private_rank_or_probability_in_source_fails(self):
  self.source["p6_double_calibrated"]=.9
  with self.assertRaisesRegex(ValueError,"rank or score"):
   self.run_source()
if __name__=="__main__":unittest.main()
