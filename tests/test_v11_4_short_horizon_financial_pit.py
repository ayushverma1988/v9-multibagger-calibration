import unittest
from v11_4_short_horizon_financial_pit import positive_growth,close_utc
class TestAnnualShortHorizon(unittest.TestCase):
 def test_growth(self):self.assertAlmostEqual(positive_growth(120,100),20)
 def test_zero_base_rejected(self):self.assertIsNone(positive_growth(120,0))
 def test_negative_base_rejected(self):self.assertIsNone(positive_growth(120,-100))
 def test_negative_current_rejected(self):self.assertIsNone(positive_growth(-100,120))
 def test_exact_close(self):self.assertEqual(close_utc("2024-12-31").isoformat(),"2024-12-31T10:00:00+00:00")
if __name__=="__main__":unittest.main(verbosity=2)
