import unittest,pandas as pd
from v11_4_backtest_readiness_pit import validate_feature_names,fold_close
class BacktestReadinessTests(unittest.TestCase):
 def test_six_month_label_leak_block(self):
  with self.assertRaises(ValueError):
   validate_feature_names(pd.DataFrame({"y6":[1],"historical_asof_utc":["2024-12-31T10:00:00Z"]}))
 def test_maturity_label_leak_block(self):
  with self.assertRaises(ValueError):
   validate_feature_names(pd.DataFrame({"y6_mature_date":["2025-06-30"],"historical_asof_utc":["2024-12-31T10:00:00Z"]}))
 def test_valid_feature_schema(self):
  self.assertTrue(validate_feature_names(pd.DataFrame({"symbol":["ABC"],"historical_asof_utc":["2024-12-31T10:00:00Z"]})))
 def test_close(self):
  self.assertEqual(fold_close("2024-12-31").isoformat(),"2024-12-31T10:00:00+00:00")
if __name__=="__main__":unittest.main(verbosity=2)
