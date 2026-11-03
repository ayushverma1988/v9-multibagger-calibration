"""Conservative original NSE dual-day-close + RSI70 shadow; no leakage or zero-imputed unknown."""
import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np
import v11_4_exchange_NSE_day_close_RSI70_source_gate as m

def fixtures():
 days=("2023-06-30","2024-12-31")
 f=[];q=[]
 for ix,day in enumerate(days):
  for j in range(2):
   sy=f"EX{j}"
   price=100+10*j
   good_rsi=(j==0 or ix==1)
   rsi=74. if j==0 else (66. if good_rsi else np.nan)
   row={"date":day,"symbol":sy,"close":price,
    "rsi14_wilder_source":rsi,
    "rsi14_Wilder_research_verified":good_rsi,
    "rsi14_strict_gt70_source_verified":bool(rsi>70) if good_rsi else pd.NA,
    "promoter_buying_verified_180d":pd.NA,
    "has_strict_3FY_original_fiscal_source":True,
    "promoter_original_180d_filing_count":3,
    "nse_material_specific_positive_present":True}
   for col in m.FIN_FEATS:row[col]=.1
   f.append(row)
   q.append({"date":day,"symbol":sy,
    "official_dual_day_close_verified":not(ix==1 and j==0),
    "official_close_validation_status":"OFFICIAL_NSE_TWO_ARCHIVE_DAY_CLOSE_CONFIRMED"
         if not(ix==1 and j==0) else "ORIGINAL_HISTORICAL_ARCHIVE_CLOSE_DISAGREES_WITH_NSE",
    "original_frozen_close_INR":price,
    "official_primary_close_INR":price if not(ix==1 and j==0) else price+5,
    "official_full_close_INR":price if not(ix==1 and j==0) else price+5})
 return pd.DataFrame(f),pd.DataFrame(q)

class DayCloseSourceGate(unittest.TestCase):
 def setUp(self):
  self.f,self.q=fixtures()
  self.patch=patch.dict(m.EXPECT_UNIVERSE,{"2023-06-30":2,"2024-12-31":2},clear=True)
  self.patch.start();self.addCleanup(self.patch.stop)
 def test_original_equity_denominators_and_strict_gt70_no_zero_missing(self):
  rows,s=m.annotate(self.f,self.q)
  self.assertEqual(len(rows),4)
  self.assertEqual(s["NSE_original_day_close_confirmed_from_dual_exchange_archive_file_families"],3)
  self.assertEqual(s["RSI14_GT70_AND_official_original_date_close_reconfirmed"],1)
  self.assertEqual(s["RSI14_GT70_archived_original_source_before_official_day_recheck"],2)
  self.assertEqual(rows["fourth_screener_RSI70_official_day_close_shadow_status"].tolist(),
      ["PASS","UNKNOWN","UNKNOWN","FAIL"])
  self.assertTrue(rows["complete_120day_Wilder_RSI_source_official_NSE_certified"].eq(False).all())
 def test_any_prior_unknown_rsi_cannot_become_false_instead_of_unknown(self):
  rows,s=m.annotate(self.f,self.q)
  x=rows.loc[rows["date"].eq("2023-06-30")&rows["symbol"].eq("EX1")].iloc[0]
  self.assertTrue(pd.isna(x["RSI14_STRICT_GT70_when_official_NSE_DAY_close_confirmed"]))
 def test_mismatch_rsi70_signal_fail_closed(self):
  self.f.loc[0,"rsi14_strict_gt70_source_verified"]=False
  with self.assertRaisesRegex(ValueError,"threshold changed"):
   m.annotate(self.f,self.q)
 def test_no_pretend_confirmed_promoter_buys(self):
  self.f.loc[0,"promoter_buying_verified_180d"]=2
  with self.assertRaisesRegex(ValueError,"promoter activity"):
   m.annotate(self.f,self.q)
 def test_wrong_original_close_out_of_sync(self):
  self.q.loc[0,"original_frozen_close_INR"]=999.
  with self.assertRaisesRegex(ValueError,"frozen stock prices"):
   m.annotate(self.f,self.q)
 def test_missing_historical_company_evidence_forbidden(self):
  self.q=self.q.iloc[:-1]
  with self.assertRaisesRegex(ValueError,"original eightfold"):
   m.annotate(self.f,self.q)
 def test_future_y6_label_in_source_forbidden(self):
  self.f["y6"]=1
  with self.assertRaisesRegex(ValueError,"future target"):
   m.annotate(self.f,self.q)
 def test_fitted_model_score_in_official_quote_forbidden(self):
  self.q["p6_double_calibrated"]=.2
  with self.assertRaisesRegex(ValueError,"future target"):
   m.annotate(self.f,self.q)
if __name__=="__main__":unittest.main()
