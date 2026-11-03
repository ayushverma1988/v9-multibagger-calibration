"""V11.4 strict research risk overlay regressions; never change frozen Top 10."""
import unittest
import pandas as pd
from v11_4_tradability_risk_overlay import audit_tradability
class RiskOverlay(unittest.TestCase):
 def fixtures(self):
  symbols=["MBECL"]+[f"STOCK{i}" for i in range(2,11)]
  picks=pd.DataFrame({"symbol":symbols,"date":[pd.Timestamp("2026-10-08")]*10,
     "rank":list(range(1,11)),"close":[474.30]+[500.]*9,
     "p6_double_calibrated":[.2892]+[.039]*9})
  features=pd.DataFrame({"symbol":symbols,"date":[pd.Timestamp("2026-10-08")]*10,
     "last_session_turnover_INR":[474.3*520]+[4_000_000.]*9,
     "median_20_session_turnover_INR":[400_000.]+[5_000_000.]*9,
     "observed_trading_sessions_last_35d":[20]*10,"avg_turnover_63":[5_000_000.]*10,
     "integrity_feature_clean":[True]*10})
  screening=pd.DataFrame({"symbol":symbols,
     "condition_1_status":["UNKNOWN"]*10,
     "condition_2_status":["FAIL"]*10,
     "condition_3_status":["UNKNOWN"]*10,
     "condition_4_status":["PASS"]*3+["FAIL"]*7})
  return features,picks,screening
 def test_thin_volume_high_top_prob_flagged_but_frozen_stocks_unmodified(self):
  x,y,z=self.fixtures();saved=y.copy(deep=True)
  overlay,audit=audit_tradability(x,y,z)
  self.assertEqual(overlay.iloc[0]["rank"],1)
  self.assertEqual(overlay.iloc[0]["tradability_assessment"],"HIGH_CAUTION")
  self.assertIn("LAST_SESSION_TURNOVER_BELOW_5_LAKH",overlay.iloc[0]["tradability_issues"])
  self.assertIn("TOP_PROBABILITY_EXCEEDS_4X",overlay.iloc[0]["risk_flags"])
  self.assertTrue(audit["probability_outlier_alert"])
  self.assertEqual(audit["stocks_with_low_recent_tradability_evidence"],1)
  pd.testing.assert_frame_equal(saved,y,check_exact=True)
 def test_unknown_turnover_does_not_imply_buyable(self):
  x,y,z=self.fixtures();x=x.drop(columns=["last_session_turnover_INR","median_20_session_turnover_INR"])
  overlay,audit=audit_tradability(x,y,z)
  self.assertEqual(audit["stocks_cleared_for_execution"],0)
  self.assertEqual(audit["stocks_with_missing_tradability_evidence"],10)
  self.assertTrue(overlay["tradability_assessment"].eq("UNKNOWN").all())
 def test_reject_missing_predicted_symbol(self):
  x,y,z=self.fixtures();x=x.iloc[:-1]
  with self.assertRaisesRegex(ValueError,"not in contemporaneous"):
   audit_tradability(x,y,z)
 def test_reject_duplicate_symbols(self):
  x,y,z=self.fixtures();x.loc[1,"symbol"]="MBECL"
  with self.assertRaisesRegex(ValueError,"Invalid complete"):
   audit_tradability(x,y,z)
 def test_invalid_probabilities_fail(self):
  x,y,z=self.fixtures();y.loc[0,"p6_double_calibrated"]=2.0
  with self.assertRaisesRegex(ValueError,"probabilities invalid"):
   audit_tradability(x,y,z)
 def test_missing_rule_family_fails_closed(self):
  x,y,z=self.fixtures();z=z.drop(columns=["condition_4_status"])
  with self.assertRaisesRegex(ValueError,"Four original rule"):
   audit_tradability(x,y,z)
 def test_no_false_full_market_liquidity_approval(self):
  x,y,z=self.fixtures();x["last_session_turnover_INR"]=50_000_000.
  risk,s=audit_tradability(x,y,z)
  self.assertEqual(s["stocks_cleared_for_execution"],0)
  self.assertTrue(risk["frozen_model_ranking_unchanged"].all())
if __name__=="__main__":unittest.main()
