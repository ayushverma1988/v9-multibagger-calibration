import unittest
import pandas as pd
from v11_4_untrained_catalyst_only_probe import score_without_labels,FIXED_WEIGHTS
class StaticCatalystDiagnosticTests(unittest.TestCase):
 def test_labels_cannot_enter_feature_score(self):
  x=pd.DataFrame({**{k:[1] for k in FIXED_WEIGHTS},"y6":[1]})
  with self.assertRaises(ValueError):score_without_labels(x)
 def test_score_independent_of_future_label(self):
  x=pd.DataFrame({k:[0,1] for k in FIXED_WEIGHTS})
  self.assertEqual(len(score_without_labels(x)),2)
 def test_negative_source_counts_blocked(self):
  x=pd.DataFrame({k:[0] for k in FIXED_WEIGHTS})
  x["nse_order_win_180d_positive"]=-1
  with self.assertRaises(ValueError):score_without_labels(x)
 def test_missing_source_field_blocked(self):
  x=pd.DataFrame({"nse_order_win_180d_positive":[1]})
  with self.assertRaises(ValueError):score_without_labels(x)
if __name__=="__main__":unittest.main(verbosity=2)
