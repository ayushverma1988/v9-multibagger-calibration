import unittest

import numpy as np
import pandas as pd

from v13_financial_evidence import REQUIRED, financial_quality
from v13_model import gates
from v13_screening_policy import annual_cfo_pat_min


class DatedCashFlowGate(unittest.TestCase):
    def test_both_gates_use_same_dated_floor_and_include_exact_boundary(self):
        good = dict(zip(REQUIRED, [1, .2, .2, .1, .35, .5, 2.]))
        rows = [
            dict(good, date='2026-11-27', annual_CFO_PAT=.3648),
            dict(good, date='2026-11-30', annual_CFO_PAT=.3648),
            dict(good, date='2026-11-30', annual_CFO_PAT=.35),
            dict(good, date='2026-11-30', annual_CFO_PAT=.349999),
            dict(good, date='2026-11-30', annual_CFO_PAT=np.nan),
            dict(good, date='2026-11-30', annual_CFO_PAT=.3648, dated_debt_equity=2.),
        ]
        frame = pd.DataFrame(rows).assign(close=100., distance_ma50=.02, log_ret63=.10)
        verified, discovery, rejected = gates(frame)
        self.assertEqual(verified.tolist(), [False, True, True, False, False, False])
        self.assertEqual(discovery.tolist(), [False, False, False, False, True, False])
        self.assertEqual(rejected.tolist(), [True, False, False, True, False, True])
        self.assertEqual([financial_quality(row)['status'] for row in rows],
                         ['KNOWN_FINANCIAL_FAILURE','PASS_AVAILABLE_SEVEN_CHECKS',
                          'PASS_AVAILABLE_SEVEN_CHECKS','KNOWN_FINANCIAL_FAILURE',
                          'UNKNOWN_FINANCIAL_INPUTS','KNOWN_FINANCIAL_FAILURE'])
        self.assertEqual(financial_quality(rows[-1])['failed_checks'], ['dated_debt_equity'])

    def test_pre_effective_and_undated_views_keep_the_original_floor(self):
        self.assertEqual(annual_cfo_pat_min(None), .40)
        self.assertEqual(annual_cfo_pat_min(pd.NaT), .40)
        self.assertEqual(annual_cfo_pat_min('2026-11-27'), .40)
        self.assertEqual(annual_cfo_pat_min('2026-11-28'), .35)
        dates = pd.Series([None, '2026-11-27', '2026-11-30'], index=[8, 3, 6])
        pd.testing.assert_series_equal(annual_cfo_pat_min(dates),
                                       pd.Series([.4, .4, .35], index=dates.index))


if __name__ == '__main__':
    unittest.main()
