"""Four original user screener families plus RSI(14) Wilder over 80."""
import json,unittest
from pathlib import Path
import pandas as pd
from v11_4_four_family_live_screener import analyze,eval_rule,evaluate_family
from v11_4_live_nse_market_catalyst import rsi_wilder_last
CONFIG="config/v11_4_four_screener_families.json"
class FourFamilyTests(unittest.TestCase):
 def setUp(self):
  self.config=json.loads(Path(CONFIG).read_text())
 def test_all_three_original_rules_and_fourth_present(self):
  counts=[len(v["hard_rules"]) for v in self.config["conditions"].values()]
  self.assertEqual(counts,[20,4,8,1])
  self.assertEqual(self.config["conditions"]["condition_4_RSI14_high_momentum"]["hard_rules"],
                   [["rsi14_wilder",">",80]])
 def test_rsi_wilder_numerical_extremes(self):
  self.assertEqual(rsi_wilder_last([100.]*30),50.)
  self.assertAlmostEqual(rsi_wilder_last(list(range(1,45))),100.)
  self.assertAlmostEqual(rsi_wilder_last(list(range(45,1,-1))),0.)
  self.assertTrue(pd.isna(rsi_wilder_last([1.]*14)))
 def test_strict_above_80_not_greater_equal(self):
  self.assertTrue(eval_rule(80.1,">",80))
  self.assertFalse(eval_rule(80.,">",80))
  self.assertIsNone(eval_rule(float("nan"),">",80))
 def test_missing_fundamentals_is_unknown_not_fail(self):
  x=pd.DataFrame({"date":pd.to_datetime(["2026-10-08"]*3),
    "symbol":["STRONG","BORDER","WEAK"],
    "rsi14_wilder":[90.0,80.0,20.0]})
  res=analyze(x,self.config)
  self.assertEqual(res["rsi14_gt80"].fillna(False).tolist(),[True,False,False])
  self.assertEqual(res["condition_1_quality_compounder_status"].tolist(),["UNKNOWN"]*3)
  self.assertEqual(res["condition_2_recovery_value_setup_status"].tolist(),["UNKNOWN"]*3)
  self.assertEqual(res["condition_3_earnings_acceleration_status"].tolist(),["UNKNOWN"]*3)
  self.assertEqual(res["condition_4_RSI14_high_momentum_status"].tolist(),["PASS","FAIL","FAIL"])
  self.assertTrue((res["immutable_frozen_model_ranking_changed"]==False).all())
 def test_partial_family_with_known_fail_is_fail(self):
  family=self.config["conditions"]["condition_2_recovery_value_setup"]
  self.assertEqual(evaluate_family({"price_gt_dma50_prev":True},family)["status"],"UNKNOWN")
  self.assertEqual(evaluate_family({"price_gt_dma50_prev":False},family)["status"],"FAIL")
if __name__=="__main__":unittest.main(verbosity=2)
