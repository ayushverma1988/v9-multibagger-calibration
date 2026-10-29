import json
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import v13_chart_interpretation as chart


def bars(n=320):
    p = 100 + np.arange(n) * .1 + np.sin(np.arange(n) * .2)
    return pd.DataFrame({'date':pd.bdate_range('2024-01-01', periods=n),
        'isin':'INE000000001', 'close':p, 'adj_close':p,
        'high':p + 1., 'low':p - 1., 'volume':100_000.})


def attached(**changes):
    r = dict(date=pd.Timestamp('2026-10-23'), symbol='ALPHA', isin='INE000000001',
        close=100., adj_close=100., rsi14=50., feature_eligible=True,
        chart_history_status='EXACT_DATE_MATCH_PROVIDER_ACTIONS_UNVERIFIED',
        chart_bb_percentile126=.1, chart_prior_squeeze5=1.,
        chart_gap_atr=-.5, chart_volume_dry10_50=.8, chart_base_range63=.2,
        chart_atr14=2., chart_adjustment_factor=1., chart_extension50_atr=1.,
        chart_daily_trend=1., chart_completed_week_trend=1.,
        chart_adjustment_jump63=0., chart_first_breakout=0.,
        chart_retest_hold=0., chart_failed_breakout=0.,
        chart_rounded_handle_proxy=0., financial_quality_status='PASS_AVAILABLE_SEVEN_CHECKS',
        p6=.04, p12=.10, loss12=.4, research_selected=True, research_rank=1)
    r.update(changes)
    return pd.DataFrame([r])


class ChartEvidence(unittest.TestCase):
    def test_wilder_arithmetic_seed_and_recursion(self):
        x = pd.Series([1., 2., 3., 6.])
        y = chart.wilder(x, 3)
        self.assertTrue(y.iloc[:2].isna().all())
        self.assertAlmostEqual(y.iloc[2], 2.)
        self.assertAlmostEqual(y.iloc[3], 10/3)

    def test_atr_accounts_for_overnight_gap(self):
        b = bars(30)
        b.loc[20:, ['close','adj_close','high','low']] += 30
        x = chart.daily(b)
        self.assertGreater(x.loc[20, 'chart_atr14'], 4.)
        self.assertAlmostEqual(x.loc[13, 'chart_atr14'], 2.)

    def test_bollinger_definition_and_long_percentile_warmup(self):
        b = bars()
        d = chart.daily(b)
        a = b.adj_close.iloc[-20:].to_numpy()
        self.assertAlmostEqual(d.iloc[-1].chart_bb_width20, 4*np.std(a, ddof=0)/a.mean())
        self.assertTrue(d.chart_bb_percentile126.iloc[:144].isna().all())
        self.assertTrue(d.chart_bb_percentile126.iloc[144:].between(0,1).all())

    def test_feature_prefix_does_not_change_with_future_bars(self):
        b = bars(350)
        prefix = chart.daily(b.iloc[:271])
        b.loc[271:, ['close','adj_close','high','low']] *= 4
        b.loc[271:, 'volume'] *= 10
        full = chart.daily(b)
        pd.testing.assert_frame_equal(prefix[chart.FEATURES], full.iloc[:271][chart.FEATURES])

    def test_partial_week_does_not_use_its_future_friday(self):
        b = bars(350)
        wed = b.index[b.date.dt.weekday.eq(2)][-2]
        a = chart.daily(b.iloc[:wed+1])
        b.loc[wed+1:, ['close','adj_close','high','low']] *= 100
        c = chart.daily(b)
        self.assertEqual(a.iloc[-1].chart_completed_week_trend,
                         c.iloc[wed].chart_completed_week_trend)
        friday = b.index[(b.index < wed) & b.date.dt.weekday.eq(4)][-1]
        self.assertEqual(b.loc[friday, 'date'].weekday(), 4)
        self.assertEqual(a.iloc[-1].chart_completed_week_trend,
                         a.loc[friday, 'chart_completed_week_trend'])

    def test_resistance_and_volume_denominators_exclude_signal_bar(self):
        b = bars()
        old_high = b.high.iloc[-64:-1].max()
        b.loc[len(b)-1, ['close','adj_close','high','low']] += 10
        b.loc[len(b)-1, 'volume'] = 300_000
        d = chart.daily(b)
        self.assertAlmostEqual(d.iloc[-1].chart_resistance63, old_high)
        self.assertEqual(d.iloc[-1].chart_volume_ratio20, 3.)

    def test_split_adjusted_price_geometry_is_invariant(self):
        b = bars()
        other = b.copy()
        other.loc[:150, ['close','high','low']] *= 2
        a, c = chart.daily(b), chart.daily(other)
        cols = [x for x in chart.FEATURES if x not in
            ('chart_adjustment_factor','chart_history_close','chart_adjustment_jump63')]
        pd.testing.assert_frame_equal(a[cols], c[cols])
        self.assertEqual(c.loc[151, 'chart_adjustment_jump63'], 1.)

    def test_constant_prices_do_not_manufacture_squeeze_breakout(self):
        b = bars()
        b[['close','adj_close','high','low']] = 100.
        d = chart.daily(b)
        self.assertFalse(d.chart_first_breakout.eq(1).any())
        self.assertTrue(d.chart_gap_atr.isna().all())

    def test_invalid_dates_identity_and_prices_rejected(self):
        for col, value in [('date', pd.Timestamp('2024-01-01')),
                            ('isin', 'OTHER'), ('volume', -1), ('adj_close', np.nan),
                            ('low', 200.)]:
            b = bars(); b.loc[2,col] = value
            with self.subTest(col=col), self.assertRaises(ValueError):
                chart.daily(b)

    def test_zero_volume_bar_is_not_an_observed_trade(self):
        b = bars(); b.loc[30,'volume'] = 0
        d = chart.daily(b)
        self.assertEqual(len(d), len(b)-1)
        self.assertNotIn(b.loc[30,'date'], d.date.tolist())

    def test_rsi50_inclusive_and_optional_dma_band(self):
        for value, eligible in [(49.999,False),(50.,True),(50.001,True),(np.nan,False)]:
            with self.subTest(rsi=value):
                d = chart.select(attached(rsi14=value, distance_ma200=.4))
                self.assertEqual(bool(d.iloc[0].chart_candidate), eligible)

    def test_squeeze_is_a_watch_until_direction_confirms(self):
        d = chart.select(attached())
        self.assertEqual(d.iloc[0].chart_state, 'PRE_BREAKOUT_WATCH')
        self.assertFalse(d.iloc[0].chart_signal_is_2x_probability)
        d = chart.select(attached(chart_gap_atr=-4.))
        self.assertFalse(d.iloc[0].chart_candidate)

    def test_weekly_unknown_cannot_be_confirmed_trend(self):
        d = chart.select(attached(chart_first_breakout=1., chart_completed_week_trend=np.nan))
        self.assertEqual(d.iloc[0].chart_state, 'BREAKOUT_TREND_REVIEW')
        d = chart.select(attached(chart_first_breakout=1.))
        self.assertEqual(d.iloc[0].chart_state, 'CONFIRMED_BREAKOUT')

    def test_failed_breakout_extension_action_and_financial_gates(self):
        for change, state in [
            ({'chart_failed_breakout':1.}, 'FAILED_BREAKOUT'),
            ({'chart_extension50_atr':6.}, 'EXTENDED_REVIEW'),
            ({'chart_adjustment_jump63':1.}, 'CORPORATE_ACTION_REVIEW'),
            ({'chart_bb_percentile126':np.nan}, 'INSUFFICIENT_HISTORY'),
            ({'chart_history_status':'MISSING_EXACT_HISTORY'}, 'UNKNOWN_CURRENT_MARKET')]:
            with self.subTest(state=state):
                d = chart.select(attached(**change))
                self.assertEqual(d.iloc[0].chart_state, state)
                self.assertFalse(d.iloc[0].chart_candidate)
        for change in [{'financial_quality_status':'KNOWN_FINANCIAL_FAILURE'},
                       {'owner_PAT_yoy':-.1}, {'hd_dilution_90d':1}]:
            self.assertFalse(chart.select(attached(**change)).iloc[0].chart_candidate)

    def test_existing_probabilities_selection_and_input_are_identical(self):
        original = attached()
        before = original.copy(deep=True)
        d = chart.select(original)
        protected = ['p6','p12','loss12','research_selected','research_rank']
        pd.testing.assert_frame_equal(before[protected], d[protected])
        pd.testing.assert_frame_equal(before, original)
        self.assertEqual(d.iloc[0].chart_3atr_below_close_INR, 94.)

    def test_rounded_handle_proxy_accepts_dwell_but_rejects_v(self):
        lead = np.linspace(70, 100, 90)
        cup = np.r_[np.linspace(100,75,20),np.full(20,75.),np.linspace(75,100,20)]
        handle = np.r_[100,99,98,99,100]
        prices = np.r_[lead,cup,handle,101.]
        b = bars(len(prices));b['close']=prices;b['adj_close']=prices
        b['high']=prices+1;b['low']=prices-1
        r = chart.rounded_handle_proxy(b)
        self.assertEqual(r['chart_rounded_handle_proxy'],1.)
        v = np.r_[np.linspace(100,75,30),np.linspace(75,100,30)]
        b.loc[90:149,['close','adj_close']] = np.c_[v,v]
        b['high']=b.close+1;b['low']=b.close-1
        self.assertEqual(chart.rounded_handle_proxy(b)['chart_rounded_handle_proxy'],0.)

    def test_attach_clips_future_and_never_substitutes_date(self):
        b = bars()
        last = b.iloc[-2]
        f = attached(date=last.date,close=last.close)
        with tempfile.TemporaryDirectory() as temp:
            (Path(temp)/'INE000000001.parquet').touch()
            with patch.object(pd,'read_parquet',return_value=b):
                d = chart.attach(f,temp)
                self.assertTrue(d.iloc[0].chart_history_status.startswith('EXACT_DATE_MATCH'))
                b.loc[len(b)-2,'volume']=0
                d = chart.attach(f,temp)
                self.assertEqual(d.iloc[0].chart_history_status,'NO_TRADED_BAR_ON_DECISION_DATE')

    def test_collector_symbol_hash_is_resolved_but_identity_conflicts_fail(self):
        b=bars();last=b.iloc[-1]
        f=attached(date=last.date,close=last.close)
        with tempfile.TemporaryDirectory() as temp:
            (Path(temp)/(hashlib.sha256(b'ALPHA').hexdigest()+'.parquet')).touch()
            with patch.object(pd,'read_parquet',return_value=b):
                self.assertTrue(chart.attach(f,temp).iloc[0].chart_history_status.startswith('EXACT_DATE_MATCH'))
                b['isin']='INE000000002'
                self.assertEqual(chart.attach(f,temp).iloc[0].chart_history_status,'NO_TRADED_BAR_ON_DECISION_DATE')

    def test_export_is_reproducible_with_immutable_cohort(self):
        b=bars();last=b.iloc[-1]
        f=attached(date=last.date,close=last.close,adj_close=last.adj_close)
        f=f.drop(columns=[c for c in f if c.startswith('chart_')])
        with tempfile.TemporaryDirectory() as temp:
            h=Path(temp)/'history';h.mkdir();(h/'INE000000001.parquet').touch()
            with patch.object(pd,'read_parquet',return_value=b):
                summary=chart.export(f,temp,{'sha256':'fixed'},history=h,rsi_min=50)
                chart.export(f,temp,{'sha256':'fixed'},history=h,rsi_min=50)
            p=Path(temp)/'chart_interpretation'
            self.assertEqual(len(list((p/'cohorts').glob('*.json'))),1)
            self.assertFalse(summary['probabilities_recalibrated'])
            saved=json.loads((p/'latest.json').read_text())
            self.assertEqual(saved['records'][0]['p6'],.04)


if __name__ == '__main__':
    unittest.main()
