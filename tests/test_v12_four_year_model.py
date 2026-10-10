import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import v12_four_year_model as model
import v12_coherent_hazard_model as coherent
from v12_public_history import bounds, parse_chart


def daily(start='2022-01-03', periods=1250):
    dates = pd.bdate_range(start, periods=periods)
    price = 100*np.exp(np.arange(periods)*.001)
    return pd.DataFrame({'date': dates, 'open': price, 'high': price*1.02,
        'low': price*.98, 'close': price, 'adj_close': price,
        'volume': 1000000., 'symbol': 'PUBLIC', 'isin': 'INE000A01000'})


class FourYearContract(unittest.TestCase):
    def test_hazard_union_is_coherent_without_projection(self):
        early = np.linspace(0, 1, 21)
        late = np.linspace(1, 0, 21)
        total = coherent.union_probability(early, late)
        self.assertTrue((total >= early).all())
        self.assertTrue((total <= 1).all())
        self.assertAlmostEqual(coherent.union_probability(.2, .3), .44)

    def test_invalid_conditional_probability_rejected(self):
        with self.assertRaises(ValueError):
            coherent.union_probability(.2, 1.1)

    def test_late_training_target_condition_is_not_a_predictor(self):
        p = pd.DataFrame({'hit2_6': [1, 0, 0, np.nan], 'hit2_12': [1, 1, 0, np.nan],
            'loss30_12': [1, 1, 0, np.nan], 'hit2_before_loss30_12': [1, 0, 0, np.nan],
            'rsi14': [65]*4, 'distance_ma50': [.1]*4, 'distance_ma200': [-.1]*4,
            'log_ret63': [.2]*4})
        q = coherent.prepare(p)
        self.assertTrue(pd.isna(q.late_hit2_12.iloc[0]))
        self.assertEqual(q.late_hit2_12.iloc[1], 1)
        self.assertEqual(q.loss30_before_hit2_12.iloc[0], 0)
        self.assertEqual(q.loss30_before_hit2_12.iloc[1], 1)
        self.assertNotIn('late_hit2_12', model.FEATURES)

    def test_four_year_calendar_boundary(self):
        low, high = bounds('2026-10-09')
        self.assertEqual(str(low.date()), '2022-10-09')
        self.assertEqual(str(high.date()), '2026-10-09')

    def test_trims_before_any_rolling_or_RSI(self):
        x = daily()
        y = x.copy()
        y.loc[y.date.lt('2022-10-09'), ['close', 'adj_close']] *= 1000
        a = model.make_security_features(x, '2026-10-09')
        b = model.make_security_features(y, '2026-10-09')
        pd.testing.assert_frame_equal(a, b)
        self.assertGreaterEqual(a.date.min(), pd.Timestamp('2022-10-09'))

    def test_no_infinite_market_features(self):
        x = model.make_security_features(daily(), '2026-10-09')
        required = [k for k in model.FEATURES if k in x]
        self.assertTrue(np.isfinite(x.loc[x.feature_eligible, required].to_numpy(float)).all())

    def test_low_liquidity_never_passes(self):
        d = daily()
        d['volume'] = 1
        self.assertFalse(model.make_security_features(d, '2026-10-09').feature_eligible.any())

    def test_zero_volume_is_not_an_observed_trading_session(self):
        d = daily()
        d['volume'] = 0
        self.assertEqual(len(model.make_security_features(d, '2026-10-09')), 0)

    def test_cutler_RSI_flat_and_rising(self):
        self.assertEqual(model.rsi(pd.Series([10.]*20)).iloc[-1], 50)
        self.assertEqual(model.rsi(pd.Series(np.arange(20, dtype=float))).iloc[-1], 100)

    def test_full_horizon_required_even_early_doubler(self):
        d = daily(periods=60)
        d.loc[1:, ['open', 'close', 'adj_close', 'high', 'low']] *= 3
        self.assertIsNone(model.forward_outcome(d, 0, 6, '2026-10-09'))

    def test_calendar_target_not_126_session_assumption(self):
        d = daily(start='2023-01-02', periods=500)
        o = model.forward_outcome(d, 0, 6, '2026-10-09')
        self.assertEqual(o['mature_date'], '2023-07-03')

    def test_intraday_high_only_not_2x(self):
        d = daily(start='2023-01-02', periods=500)
        d['high'] = d.close*4
        o = model.forward_outcome(d, 0, 6, '2026-10-09')
        self.assertEqual(o['hit2'], 0)

    def test_costs_can_prevent_nominal_double(self):
        d = daily(start='2023-01-02', periods=500)
        d.loc[0, ['open', 'close', 'adj_close']] = 100
        d.loc[1, ['open', 'close', 'adj_close']] = 100
        d.loc[2:, ['open', 'close', 'adj_close']] = 200
        # Avoid an unverified one-day action by reaching 2x gradually.
        d.loc[2:11, ['open', 'close', 'adj_close']] = np.repeat(np.linspace(105, 195, 10)[:, None], 3, axis=1)
        d['high'] = d.close*1.01
        d['low'] = d.close*.99
        o = model.forward_outcome(d, 0, 6, '2026-10-09')
        self.assertEqual(o['hit2'], 0)

    def test_invalid_long_suspension_is_unknown_outcome(self):
        d = daily(start='2023-01-02', periods=500)
        d = d.drop(index=range(10, 30)).reset_index(drop=True)
        self.assertIsNone(model.forward_outcome(d, 0, 6, '2026-10-09'))

    def test_missing_future_cannot_be_zero(self):
        d = daily(start='2023-01-02', periods=200)
        self.assertIsNone(model.forward_outcome(d, 170, 12, '2026-10-09'))

    def test_chronological_purge_prevents_label_leakage(self):
        records = []
        for date in pd.date_range('2023-01-31', '2025-12-31', freq='ME'):
            for i in range(300):
                records.append({'date': date, 'symbol': 'P'+str(i), 'hit2_6': i%5 == 0,
                                'mature_date_6': date+pd.DateOffset(months=6)})
        p = pd.DataFrame(records)
        train, cal = model.chronological_parts(p, '2025-12-31', 'hit2_6')
        self.assertTrue(train.mature_date_6.lt(cal.date.min()).all())
        self.assertTrue(cal.mature_date_6.lt(pd.Timestamp('2025-12-31')).all())
        self.assertTrue(train.date.lt(cal.date.min()).all())

    def test_unknown_training_labels_excluded(self):
        p = pd.DataFrame({'date': pd.to_datetime(['2023-01-31']*600),
            'hit2_6': np.nan, 'mature_date_6': pd.Timestamp('2023-07-31')})
        self.assertIsNone(model.chronological_parts(p, '2025-12-31', 'hit2_6'))

    def test_rank_is_deterministic_under_ties(self):
        d = pd.DataFrame({'symbol': ['Z','A'], 'probability': [.3,.3]})
        self.assertEqual(model.rank(d, 'probability').symbol.tolist(), ['A','Z'])

    def test_calibration_cannot_reverse_ranking(self):
        raw = np.linspace(.01, .9, 200)
        y = (raw < .2).astype(int)
        params = model.calibrate(raw, y, y)
        self.assertGreater(params['slope'], 0)

    def test_issuer_omission_keeps_or_drops_whole_firm(self):
        d = pd.DataFrame({'symbol': np.repeat(['S'+str(i) for i in range(100)], 5)})
        keep = model.issuer_omission(d, 17)
        self.assertTrue(keep.symbol.value_counts().eq(5).all())
        self.assertEqual(keep.symbol.nunique(), 95)

    def test_recipe_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as p:
            model.write_recipe(p, p, '2026-10-09')
            with self.assertRaises(FileExistsError):
                model.write_recipe(p, p, '2026-10-09')

    def finance_inputs(self):
        current = pd.DataFrame([{'symbol': 'PUBLIC', 'isin': 'INE000A01000',
            'close': 200., 'date': pd.Timestamp('2026-10-09'), 'technical_pass': True}])
        audit = pd.DataFrame([{'symbol': 'PUBLIC', 'isin': 'INE000A01000',
            'status': 'CURRENT_THREE_WAY_CORROBORATED', 'market_cap_verified_crore': 1000.,
            'latest_annual_revenue_period': '2026-03-31', 'latest_quarter_revenue_period': '2026-06-30',
            'quarterly_sales_yoy_growth': .25, 'quarterly_profit_yoy_growth': .4,
            'opm_annual': .15, 'debt_equity': .3, 'latest_annual_cfo_pat_ratio': 1.2,
            'negative_or_missing_equity': False, 'PBT_components_reconcile_all_four_quarters': True,
            'four_quarter_PBT_excluding_reported_exceptional_items': 1e8,
            'original_fact_disagreements': 0, 'issuer_name_matches': True}])
        reference = pd.DataFrame([{'isin': 'INE000A01000', 'amfi_size_category': 'Small Cap'}])
        official = pd.DataFrame([{'symbol': 'PUBLIC', 'isin': 'INE000A01000',
            'date': pd.Timestamp('2026-10-09'), 'close': 200., 'series': 'EQ'}])
        return current, audit, reference, official

    def test_missing_CFO_remains_unknown(self):
        c, a, r, o = self.finance_inputs()
        a.loc[0, 'latest_annual_cfo_pat_ratio'] = np.nan
        q = model.current_financial_gate(c, a, r, o, '2026-10-09')
        self.assertEqual(q.current_quality_status.iloc[0], 'UNKNOWN')

    def test_exceptional_profit_cannot_create_financial_pass(self):
        c, a, r, o = self.finance_inputs()
        a.loc[0, 'four_quarter_PBT_excluding_reported_exceptional_items'] = -1
        q = model.current_financial_gate(c, a, r, o, '2026-10-09')
        self.assertEqual(q.current_quality_status.iloc[0], 'REJECT')

    def test_verified_complete_source_checks_can_pass(self):
        q = model.current_financial_gate(*self.finance_inputs(), '2026-10-09')
        self.assertEqual(q.current_quality_status.iloc[0], 'CURRENT_SOURCE_CHECKS_PASS')
        self.assertFalse(q.financial_gate_historically_calibrated.iloc[0])

    def test_wrong_venue_price_cannot_pass(self):
        c, a, r, o = self.finance_inputs()
        o.loc[0, 'close'] = 300
        q = model.current_financial_gate(c, a, r, o, '2026-10-09')
        self.assertEqual(q.current_quality_status.iloc[0], 'REJECT')

    def test_original_filing_conflict_is_not_ignored(self):
        c, a, r, o = self.finance_inputs()
        a.loc[0, 'original_fact_disagreements'] = 1
        self.assertEqual(model.current_financial_gate(c, a, r, o, '2026-10-09').current_quality_status.iloc[0], 'REJECT')


if __name__ == '__main__':
    unittest.main()
