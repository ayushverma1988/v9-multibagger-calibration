import unittest

import numpy as np
import pandas as pd

import v13_bottom_recovery as bottom
import v13_redevelopment as rd
from v13_financial_evidence import REQUIRED


class BottomRecovery(unittest.TestCase):
    def bars(self):
        # An older peak followed by a 30-session stable base.
        p = np.r_[np.linspace(80, 160, 180), np.linspace(160, 72, 65),
                  np.linspace(72, 78, 55)]
        return pd.DataFrame({'date': pd.bdate_range('2025-01-01', periods=len(p)),
            'isin': 'INE000000001', 'close': p, 'adj_close': p,
            'high': p * 1.01, 'low': p * .99, 'volume': 10000.})

    def row(self, features):
        r = {k: np.nan for k in REQUIRED}
        r.update(date='2026-10-15', isin='INE000000001', symbol='TEST', close=78.,
            feature_eligible=True, distance_ma50=0., log_ret63=0., p6=.1, p12=.2,
            loss12=.3, joint12=.1, rsi14=43.)
        r.update(features)
        return pd.DataFrame([r])

    def test_past_bottom_features_are_unchanged_by_future_prices(self):
        b = self.bars(); a = rd.daily(b.iloc[:280])
        b.loc[280:, ['close', 'adj_close', 'high', 'low']] *= 3
        b.loc[280:, 'volume'] *= 20
        z = rd.daily(b)
        pd.testing.assert_frame_equal(a[rd.MARKET_FEATURES + rd.BOTTOM_FEATURES],
                                      z.iloc[:280][rd.MARKET_FEATURES + rd.BOTTOM_FEATURES])

    def test_split_adjustment_does_not_create_a_false_bottom(self):
        a = self.bars(); b = a.copy(); b[['close', 'high', 'low']] *= 2
        pd.testing.assert_frame_equal(rd.daily(a)[rd.BOTTOM_FEATURES],
                                      rd.daily(b)[rd.BOTTOM_FEATURES])

    def test_base_below_rsi70_and_no_capex_is_visible_as_shadow(self):
        b = self.bars(); b.loc[297, 'volume'] = 25000.
        x = rd.daily(b).iloc[-1]
        features = {k: x[k] for k in rd.MARKET_FEATURES + rd.BOTTOM_FEATURES}
        original = self.row(features)
        d, audit = bottom.select(original)
        self.assertTrue(d.iloc[0].bottom_recovery_eligible)
        self.assertEqual(d.iloc[0].bottom_recovery_state, 'EARLY_RECOVERY')
        self.assertEqual(d.iloc[0].bottom_recovery_rank, 1)
        self.assertEqual(d.iloc[0].bottom_recovery_catalyst_status,
                         'NO_OBSERVED_RECENT_POSITIVE_DISCLOSURE')
        pd.testing.assert_frame_equal(original[['p6', 'p12', 'loss12', 'joint12']],
                                      d[['p6', 'p12', 'loss12', 'joint12']])
        self.assertFalse(audit['promotion_approved'])

    def test_unknown_history_financial_failure_and_adverse_news_exclude(self):
        features = {'rd_peak_drawdown252': -.45, 'rd_base_range30': .14,
            'rd_base_return30': .04, 'rd_low_distance30': .12,
            'rd_recent_volume_burst5': 2., 'rd_recovery_turn': 1.}
        for change in ({'rd_peak_drawdown252': np.nan},
                       {'hd_dilution_90d': 1.}, {'owner_PAT_yoy': -.2},
                       {'close': 0.}):
            with self.subTest(change=change):
                d, _ = bottom.select(self.row({**features, **change}))
                self.assertFalse(d.iloc[0].bottom_recovery_eligible)


if __name__ == '__main__':
    unittest.main()
