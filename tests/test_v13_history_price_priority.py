import unittest

import numpy as np
import pandas as pd

from v13_collection import public_history_refresh_priority


class HistoryRefreshPricePolicy(unittest.TestCase):
    def test_same_turnover_priority_at_all_valid_prices(self):
        for price in (.5, 19.99, 100., 2000.01, 100000.):
            with self.subTest(price=price):
                history = pd.DataFrame({'close': [price] * 127,
                    'volume': [4_000_000 / price] * 127})
                self.assertTrue(public_history_refresh_priority(history))

    def test_bad_prices_missing_history_and_low_turnover_do_not_jump_queue(self):
        good = pd.DataFrame({'close': [100.] * 127, 'volume': [30000.] * 127})
        for modified in (good.iloc[:126], good.assign(close=0.),
                         good.assign(close=np.inf), good.assign(volume=1.)):
            with self.subTest(length=len(modified)):
                self.assertFalse(public_history_refresh_priority(modified))


if __name__ == '__main__':
    unittest.main()
