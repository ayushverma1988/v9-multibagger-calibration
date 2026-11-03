import unittest
from v11_4_short_horizon_2fold_matrix import fold_close
class HistoricalMatrixGateTests(unittest.TestCase):
    def test_frozen_2024_close(self):
        self.assertEqual(fold_close("2024-12-31").isoformat(),"2024-12-31T10:00:00+00:00")
    def test_frozen_2025_close(self):
        self.assertEqual(fold_close("2025-12-31").isoformat(),"2025-12-31T10:00:00+00:00")
if __name__=="__main__":unittest.main()
