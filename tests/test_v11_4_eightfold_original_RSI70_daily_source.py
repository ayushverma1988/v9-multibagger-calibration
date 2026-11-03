"""Adversarial provenance regression tests for eightfold RSI14>70 source-only backfill."""
import unittest
from unittest.mock import patch
import pandas as pd
from v11_4_eightfold_original_RSI70_daily_source import extract_original_rsi,DAYS

def synthetic():
    days=pd.bdate_range("2021-01-01","2025-12-31")
    market=pd.DataFrame({
      "date":days,"symbol":"SYN","series":"EQ","isin":"INE123456789",
      "close": [100+i*.2 for i in range(len(days))]})
    frozen=market.loc[market["date"].dt.strftime("%Y-%m-%d").isin(DAYS),
        ["date","symbol","close"]].copy()
    actions=pd.DataFrame(columns=["symbol","type","ex_date"])
    return market,actions,frozen
class StrictEightfoldRSI70(unittest.TestCase):
    def setUp(self):
        self.a,self.b,self.c=synthetic()
        self.p=patch("v11_4_eightfold_original_RSI70_daily_source.UNIVERSES",
          {s:1 for s in DAYS})
        self.p.start();self.addCleanup(self.p.stop)
    def test_actual_wilder_above70_and_exact_all8_keys(self):
        rows,meta=extract_original_rsi(self.a,self.b,self.c)
        self.assertEqual(len(rows),8)
        self.assertEqual(meta["user_new_strict_threshold"],70)
        self.assertEqual(rows["fourth_family_rsi_strictly_gt70"].tolist(),[True]*8)
        self.assertEqual(rows["source_verification"].nunique(),1)
        self.assertTrue(meta["labels_news_or_future_prices_read"] is False)
    def test_future_2026_corporate_action_does_not_affect_2025_signal(self):
        self.b=pd.DataFrame([{"symbol":"SYN","type":"split","ex_date":"2026-01-01"}])
        rows,_=extract_original_rsi(self.a,self.b,self.c)
        self.assertTrue(rows["fourth_family_rsi_strictly_gt70"].fillna(False).all())
    def test_prior_split_without_publication_clock_must_be_unknown(self):
        self.b=pd.DataFrame([{"symbol":"SYN","type":"bonus","ex_date":"2025-06-01"}])
        rows,_=extract_original_rsi(self.a,self.b,self.c)
        q=rows.loc[rows["date"].eq("2025-06-30")].iloc[0]
        self.assertIn("SPLIT_OR_BONUS",q["source_verification"])
        self.assertTrue(pd.isna(q["fourth_family_rsi_strictly_gt70"]))
    def test_split_missing_ex_date_fails_closed(self):
        self.b=pd.DataFrame([{"symbol":"SYN","type":"split","ex_date":None}])
        rows,_=extract_original_rsi(self.a,self.b,self.c)
        self.assertTrue(rows["fourth_family_rsi_strictly_gt70"].isna().all())
    def test_price_mismatch_returns_unknown(self):
        self.c.loc[self.c["date"].eq(pd.Timestamp("2025-12-31")),"close"]*=1.1
        rows,_=extract_original_rsi(self.a,self.b,self.c)
        item=rows[rows["date"].eq("2025-12-31")].iloc[0]
        self.assertIn("NOT_EQUAL_FROZEN",item["source_verification"])
        self.assertTrue(pd.isna(item["fourth_family_rsi_strictly_gt70"]))
    def test_nontrading_or_price_stale_unknown(self):
        self.a=self.a[~self.a["date"].eq(pd.Timestamp("2025-12-31"))]
        rows,_=extract_original_rsi(self.a,self.b,self.c)
        self.assertIn("MISSING_ORIGINAL",rows.iloc[-1]["source_verification"])
    def test_no_future_label_columns_in_frozen_reference(self):
        self.c["y6"]=1
        with self.assertRaisesRegex(ValueError,"future stock outcomes"):
            extract_original_rsi(self.a,self.b,self.c)
if __name__=="__main__":unittest.main()
