import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import v13_early_potential as early
import v13_redevelopment as rd


class EarlyPotential(unittest.TestCase):
    def row(self, **changes):
        r = dict(date='2026-10-16', isin='INE000000001', symbol='ALPHA',
            close=100., rsi14=60., latest_issued_shares=50_000_000, feature_eligible=True,
            rd_peak_drawdown756=-.45, rd_base_range30=.20,
            rd_base_return30=.01, rd_low_distance30=.10,
            rd_recent_volume_burst5=2., quarter_sales_yoy=.12,
            owner_PAT_yoy=np.nan, owner_PAT_margin=np.nan,
            financial_quality_status='PASS_AVAILABLE_SEVEN_CHECKS',
            current_owner_profit_status='UNKNOWN', p6=.04, p12=.09,
            loss12=.80)
        r.update(changes)
        return pd.DataFrame([r])

    def test_volume_and_sales_surface_pre_breakout_candidate(self):
        d, s = early.select(self.row(rsi14=50., distance_ma200=-.20))
        self.assertEqual(d.iloc[0].early_pattern, 'DEEP_RESET_BASE')
        self.assertEqual(d.iloc[0].early_state, 'EARLY_POTENTIAL')
        self.assertEqual(d.iloc[0].early_rank, 1)
        self.assertFalse(s['weights_are_probabilities'])
        self.assertFalse(s['frozen_model_changed'])

    def test_rsi50_is_inclusive_and_lower_or_missing_values_are_excluded(self):
        for value,passes in [(49.999,False),(50.,True),(50.001,True),(np.nan,False)]:
            with self.subTest(rsi=value):
                frame=self.row(rsi14=value);d,_=early.select(frame)
                self.assertEqual(bool(d.iloc[0].early_eligible),passes)
                pd.testing.assert_frame_equal(frame[['p6','p12','loss12']],d[['p6','p12','loss12']])

    def test_earnings_supported_reset_without_spike_is_visible(self):
        d, _ = early.select(self.row(rd_base_range30=.39,
            rd_recent_volume_burst5=.6, owner_PAT_yoy=.29, owner_PAT_margin=.11))
        self.assertEqual(d.iloc[0].early_state, 'EARLY_POTENTIAL')
        self.assertFalse(d.iloc[0].early_volume_cue)

    def test_short_history_uses_126_observed_prior_sessions(self):
        d, _ = early.select(self.row(rd_peak_drawdown756=np.nan,
            rd_observed_sessions=208, rd_peak_drawdown126=-.11,
            rd_base_range30=.18, rd_base_return30=-.02,
            quarter_sales_yoy=np.nan))
        self.assertEqual(d.iloc[0].early_pattern, 'SHORT_HISTORY_BASE')
        self.assertEqual(d.iloc[0].early_state, 'CHART_WATCH')
        d, _ = early.select(self.row(rd_peak_drawdown756=np.nan,
            rd_observed_sessions=85, rd_peak_drawdown126=np.nan))
        self.assertEqual(d.iloc[0].early_state, 'NO_OBSERVED_SETUP')

    def test_bad_financials_and_adverse_filing_are_review_not_potential(self):
        for change, expected in [
            ({'financial_quality_status':'KNOWN_FINANCIAL_FAILURE'},'FINANCIAL_CONFLICT_REVIEW'),
            ({'owner_PAT_yoy':-.2},'FINANCIAL_CONFLICT_REVIEW'),
            ({'hd_dilution_90d':1},'ADVERSE_NEWS_REVIEW'),
            ({'feature_eligible':False},'NO_VALID_CURRENT_MARKET'),
            ({'latest_issued_shares':np.nan},'SIZE_UNKNOWN_REVIEW'),
            ({'latest_issued_shares':10_000_000_000},'OUTSIDE_SIZE_SCOPE')]:
            with self.subTest(change=change):
                d, _ = early.select(self.row(**change))
                self.assertEqual(d.iloc[0].early_state, expected)
                self.assertFalse(d.iloc[0].early_eligible)

    def test_symbol_does_not_change_rank_or_forecast_and_exports_all(self):
        original = self.row()
        renamed = self.row(symbol='USER_EXAMPLE')
        a, _ = early.select(original);b, _ = early.select(renamed)
        self.assertEqual(a.iloc[0].early_score, b.iloc[0].early_score)
        pd.testing.assert_frame_equal(original[['p6','p12','loss12']],a[['p6','p12','loss12']])
        with tempfile.TemporaryDirectory() as root:
            pd.DataFrame([{'isin':'INE000000001','nse_current_symbol':'ALPHA'},
                          {'isin':'INE000000002','nse_current_symbol':'BETA'}]).to_parquet(
                Path(root)/'universe.parquet',index=False)
            audit = early.export(original, root, {'sha256':'frozen'})
            self.assertEqual(audit['early_potentials'],1)
            self.assertEqual(audit['unscored_coverage_queue'],1)
            coverage=json.loads((Path(root)/'early_potential'/'coverage_queue.json').read_text())
            self.assertEqual(coverage['records'][0]['symbol'],'BETA')
            self.assertEqual(coverage['records'][0]['history_status'],'NO_STORED_HISTORY')
            x=json.loads((Path(root)/'early_potential'/'latest.json').read_text())
            self.assertEqual(len(x['candidates']),1)
            self.assertIn('ALPHA',(Path(root)/'early_potential'/'latest.md').read_text())
            cohort=Path(root)/audit['prospective_cohort']
            self.assertTrue(cohort.exists())
            self.assertEqual(len(list(cohort.parent.glob('*.json'))),1)
            early.export(original,root,{'sha256':'frozen'})
            self.assertEqual(len(list(cohort.parent.glob('*.json'))),1)
            changed=self.row(close=101.)
            early.export(changed,root,{'sha256':'frozen'})
            self.assertEqual(len(list(cohort.parent.glob('*.json'))),2)
            self.assertEqual(json.loads(cohort.read_text())['rows'][0]['close'],100.)

    def test_history_features_are_prefix_causal(self):
        p = np.r_[np.linspace(100, 180, 150), np.linspace(180, 95, 80)]
        b = pd.DataFrame({'date':pd.bdate_range('2025-01-01',periods=len(p)),
            'isin':'INE000000001','close':p,'adj_close':p,'high':p*1.01,
            'low':p*.99,'volume':10000.})
        first=rd.daily(b.iloc[:200])
        b.loc[200:,['close','adj_close','high','low']]*=3
        last=rd.daily(b)
        pd.testing.assert_frame_equal(first[rd.BOTTOM_FEATURES],
                                      last.iloc[:200][rd.BOTTOM_FEATURES])

    def test_pattern_queue_surfaces_retest_outside_global_top20(self):
        leaders=[self.row(isin=f'INE{i:09d}',symbol=f'LEADER{i}',
            rd_base_range30=.08,rd_low_distance30=.02,
            rd_recent_volume_burst5=4.,owner_PAT_yoy=.25,
            owner_PAT_margin=.12).iloc[0] for i in range(25)]
        retest=self.row(isin='INE999999999',symbol='RETEST',
            rd_surge_retest_age=12,rd_surge_peak_retrace=-.15,
            rd_surge_base_lift=.25,rd_recent_volume_burst5=.2,
            owner_PAT_yoy=.1,owner_PAT_margin=.1).iloc[0]
        frame=pd.DataFrame(leaders+[retest]);d,_=early.select(frame)
        self.assertGreater(d[d.symbol.eq('RETEST')].iloc[0].early_rank,20)
        with tempfile.TemporaryDirectory() as root:
            early.export(frame,root,{})
            md=(Path(root)/'early_potential'/'latest.md').read_text()
            self.assertIn('## Surge pullbacks',md)
            self.assertIn('| 1 | RETEST |',md)


if __name__=='__main__':unittest.main()
