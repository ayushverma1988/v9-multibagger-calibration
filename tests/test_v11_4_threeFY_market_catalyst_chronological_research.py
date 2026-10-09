"""Regression protections for the predeclared combined 3FY + price + NSE event chronology."""
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import v11_4_threeFY_market_catalyst_chronological_research as m
from v11_4_eightfold_3FY_market_catalyst_source_matrix import FOLDS,EXPECT_UNIVERSE,FIN_FEATS,PRICE,EVENTS

DAYS=list(FOLDS)

def dummy():
    feat=[];lab=[];n=40
    mature={"2022-06-30":"2023-01-15T00:00:00Z",
            "2022-12-30":"2023-07-15T00:00:00Z",
            "2023-06-30":"2024-01-15T00:00:00Z",
            "2023-12-29":"2024-07-15T00:00:00Z",
            "2024-06-28":"2025-01-15T00:00:00Z",
            "2024-12-31":"2025-07-15T00:00:00Z",
            "2025-06-30":"2026-01-15T00:00:00Z",
            "2025-12-31":"2026-07-15T00:00:00Z"}
    for date in DAYS:
        for i in range(n):
            sym=f"F{i:04d}";source=(i!=0 or date!="2022-06-30")
            r={"date":date,"symbol":sym,
               "historical_asof_utc":date+"T09:20:00Z",
               "has_strict_3FY_original_fiscal_source":source,
               "f3_provenance_original_fiscal_FY":"2023|2024|2025" if source else np.nan}
            for k in PRICE:r[k]=True if k=="integrity_feature_clean" else .1+i/400.
            r["avg_turnover_63"]=3_000_000+i*1111
            for k in FIN_FEATS:r[k]=.1+i/500 if source else np.nan
            for k in EVENTS:r[k]=int(i%9==0)
            feat.append(r)
            lab.append({"date":date,"symbol":sym,"close":100+i,
             "avg_turnover_63":3_000_000+i*1111,
             "integrity_y6_clean":True,"y6":int(i%9==0),
             "y6_mature_date":mature[date]})
    return pd.DataFrame(feat),pd.DataFrame(lab)

class ChronologicalCombined(unittest.TestCase):
    def setUp(self):
        self.a,self.b=dummy()
        self.p1=patch.dict(EXPECT_UNIVERSE,{s:40 for s in DAYS},clear=True)
        self.p2=patch.dict(FOLDS,{s:((2023,2024,2025),39 if s=="2022-06-30" else 40,"june") for s in DAYS},clear=True)
        self.p3=patch.object(m,"MIN_TRAIN_ROWS",20)
        self.p4=patch.object(m,"MIN_CAL_ROWS",20)
        self.p5=patch.object(m,"MIN_TEST_ROWS",20)
        self.p6=patch.object(m,"MIN_TRAIN_POS",2)
        self.p7=patch.object(m,"MIN_CAL_POS",2)
        for x in (self.p1,self.p2,self.p3,self.p4,self.p5,self.p6,self.p7):x.start()
        self.addCleanup(lambda:[x.stop() for x in (self.p7,self.p6,self.p5,self.p4,self.p3,self.p2,self.p1)])
    def test_joint_result_with_no_fake_scientific_promotion(self):
        frame,meta=m.research_train_score(self.a,self.b)
        self.assertEqual(meta["train_rows"],80)
        self.assertEqual(meta["calibration_rows"],40)
        self.assertEqual(len(frame),80)
        self.assertEqual(len(meta["per_historic_fold_research_replication"]),2)
        self.assertFalse(meta["independent_clean_12fold_scientific_promotion_gate_satisfied"])
        self.assertTrue(meta["not_a_new_blind_independent_test_2025_labels_seen_in_earlier_finance_only_tests"])
        self.assertTrue(frame["exploratory_p6_double_combined_research_only"].between(0,1).all())
    def test_future_market_source_fails_closed(self):
        self.a.loc[0,"historical_asof_utc"]="2029-01-01T00:00:00Z"
        with self.assertRaisesRegex(ValueError,"Future NSE catalyst"):
            m.check_and_join(self.a,self.b)
    def test_train_future_label_not_used_before_calibration_date(self):
        self.b.loc[(self.b["date"]=="2023-06-30")&(self.b["symbol"]=="F0000"),
                   "y6_mature_date"]="2025-01-15T00:00:00Z"
        x=m.check_and_join(self.a,self.b)
        fit=m.eligible_at(x,"2023-06-30",m.safe_cutoff(m.CAL_DATE))
        self.assertNotIn("F0000",fit["symbol"].tolist())
    def test_2024_june_calibration_label_not_matured_until_test(self):
        self.b.loc[(self.b["date"]=="2024-06-28")&(self.b["symbol"]=="F0000"),
                   "y6_mature_date"]="2025-12-01T00:00:00Z"
        x=m.check_and_join(self.a,self.b)
        cal=m.eligible_at(x,m.CAL_DATE,m.safe_cutoff("2025-06-30"))
        self.assertNotIn("F0000",cal["symbol"].tolist())
    def test_financial_coverage_below_gate_cannot_enter_fit(self):
        x=m.check_and_join(self.a,self.b)
        with self.assertRaisesRegex(ValueError,"2022 June"):
            m.eligible_at(x,"2022-06-30",m.safe_cutoff(m.CAL_DATE))
    def test_original_market_turnover_disagreement_rejected(self):
        self.b.loc[0,"avg_turnover_63"]=99999999
        with self.assertRaisesRegex(ValueError,"turnover"):
            m.check_and_join(self.a,self.b)
    def test_training_features_cannot_include_target(self):
        self.a["y6"]=1
        with self.assertRaisesRegex(ValueError,"future stock outcomes"):
            m.check_and_join(self.a,self.b)
    def test_half_year_overlap_not_called_independent_holdouts(self):
        _,meta=m.research_train_score(self.a,self.b)
        self.assertTrue(meta["source_mutually_correlated_sixmonth_overlapping_dates_not_independent"])
        self.assertTrue(meta["RSI14_over80_UNVERIFIED_OMITTED"])
if __name__=="__main__":unittest.main()
