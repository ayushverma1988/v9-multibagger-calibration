"""NSE original day-close provenance audit adversarial tests, no future labels."""
import unittest
from unittest.mock import patch
import io
import pandas as pd
import v11_4_original_NSE_historical_daily_close_crosscheck as m

def quote_tables():
 return pd.DataFrame({"symbol":[f"SY{i:04d}" for i in range(11)],
                      "official_close_INR":[100.+i for i in range(11)]})
def original(date="2024-12-31"):
 return pd.DataFrame({"date":[date]*10,
  "symbol":[f"SY{i:04d}" for i in range(10)],
  "close":[100.+i for i in range(10)]})
class CheckHistoricClose(unittest.TestCase):
 def setUp(self):
  self.p=patch.dict(m.COUNTS,{"2024-12-31":10},clear=False)
  self.p.start();self.addCleanup(self.p.stop)
  self.a=original();self.x=quote_tables()
 def test_two_official_file_families_reconcile_all(self):
  z=m.validate("2024-12-31",self.a,self.x,self.x)
  self.assertEqual(z["strict_two_exchange_file_family_match_count"],10)
  self.assertEqual(z["original_market_price_official_dual_archive_validation_pct"],100.)
  self.assertFalse(z["raw_RSI_120_day_history_source_independently_certified"])
 def test_conflicting_secondary_quarantines_company(self):
  b=self.x.copy();b.loc[0,"official_close_INR"]+=2
  z=m.validate("2024-12-31",self.a,self.x,b)
  self.assertEqual(z["strict_two_exchange_file_family_match_count"],9)
  self.assertEqual(z["source_unavailable_unknown_stocks"],1)
 def test_missing_all_archive_sources_is_unknown_not_falsely_proven(self):
  z=m.validate("2024-12-31",self.a,None,None)
  self.assertEqual(z["source_unavailable_unknown_stocks"],10)
  self.assertEqual(z["strict_two_exchange_file_family_match_count"],0)
 def test_single_official_source_is_not_two_source_proof(self):
  z=m.validate("2024-12-31",self.a,self.x,None)
  self.assertEqual(z["strict_two_exchange_file_family_match_count"],0)
  self.assertEqual(z["single_official_quote_reference_matched_count"],10)
 def test_missing_original_price_prohibited(self):
  self.a.loc[0,"close"]=float("nan")
  with self.assertRaisesRegex(ValueError,"market closes invalid"):
   m.validate("2024-12-31",self.a,self.x,self.x)
 def test_cannot_load_2025_six_month_actual_doublers(self):
  self.a["y6"]=1
  with self.assertRaisesRegex(ValueError,"No future"):
   m.validate("2024-12-31",self.a,self.x,self.x)
 def test_duplicate_company_market_quote_forbidden(self):
  z=self.x.copy();z.loc[1,"symbol"]=z.loc[0,"symbol"]
  with self.assertRaisesRegex(ValueError,"Ambiguous"):
   m.validate("2024-12-31",self.a,z,self.x)
 def test_parser_udiff_and_full_independent_columns(self):
  raw="TckrSymb,SctySrs,ClsPric\n"+ "\n".join(
     f"SY{i:04d},EQ,{100+i}" for i in range(600))+"\n"
  frame=m.quote_frame(raw.encode(),"udiff")
  self.assertEqual(len(frame),600)
  raw2="SYMBOL,SERIES,CLOSE_PRICE\n"+ "\n".join(
     f"SY{i:04d},EQ,{100+i}" for i in range(600))+"\n"
  frame2=m.quote_frame(raw2.encode(),"full")
  self.assertEqual(len(frame2),600)
  self.assertEqual(frame.iloc[0]["official_close_INR"],100.)
  self.assertEqual(frame2.iloc[0]["official_close_INR"],100.)
if __name__=="__main__":unittest.main()
