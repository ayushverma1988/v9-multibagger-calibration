import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from v13_available_financials import (assess, enrich, export, screen, DISCOVERY,
                                     TURNAROUND, QUALIFIED)
from v13_model import transform


class AvailableFinancials(unittest.TestCase):
    def test_bank_uses_bank_safety_fields_not_industrial_cashflow_leverage(self):
        w=self.wide();w.loc[w.period_kind.eq('annual'),'cfo']=-25.
        w.loc[w.period_kind.eq('instant'),'borrowings_current']=500.
        for k,v in {'bank_roa':1.2,'bank_capital_buffer_pp':3.,'bank_net_npa_change_pp':-.2}.items():
            w.loc[w.period_kind.eq('quarter'),k]=v
        r=assess(w,pd.Timestamp('2026-10-30'),'BANK')
        self.assertEqual(r['tier'],'AVAILABLE_CURRENT_FINANCIALS_RESEARCH')
        self.assertFalse(r['current_safety_failures'])
        self.assertEqual(r['check_states']['annual_CFO_PAT'],'NOT_APPLICABLE_INDUSTRIAL_RATIO_FOR_FINANCIAL_SECTOR')
        self.assertEqual(r['sector_evidence_score'],1.)

    def test_missing_bank_safety_is_discovery_and_observed_weakness_still_excluded(self):
        w=self.wide();r=assess(w,pd.Timestamp('2026-10-30'),'BANK')
        self.assertEqual(r['tier'],DISCOVERY)
        self.assertIn('bank_capital_buffer_pp',r['core_missing_fields'])
        w.loc[w.period_kind.eq('quarter'),'bank_capital_buffer_pp']=-1.
        r=assess(w,pd.Timestamp('2026-10-30'),'BANK')
        self.assertEqual(r['tier'],'OBSERVED_CURRENT_FINANCIAL_WEAKNESS')

    def wide(self, **changes):
        base = {'security_isin': 'INE000000001', 'symbol': 'A',
            'reporting_mode': 'standalone', 'available': pd.Timestamp('2026-08-01T00:00Z'),
            'available_at_utc': '2026-08-01T00:00:00Z', 'source_sha256': 'a'*64,
            'source_url': 'https://nsearchives.nseindia.com/a.xml'}
        q = {**base, 'period_kind': 'quarter', 'period': pd.Timestamp('2026-06-30'),
             'period_end': '2026-06-30', 'revenue': 100., 'pat': 10., **changes}
        old = {**q, 'period': pd.Timestamp('2025-06-30'), 'period_end': '2025-06-30',
               'revenue': 80., 'pat': 8., 'source_sha256': 'b'*64}
        annual = {**base, 'period_kind': 'annual', 'period': pd.Timestamp('2026-03-31'),
                  'period_end': '2026-03-31', 'revenue': 300., 'pat': 30., 'cfo': 25.,
                  'pbt': 40., 'finance_cost': 2.}
        balance = {**annual, 'period_kind': 'instant', 'equity': 100., 'cash': 20.,
                   'borrowings_current': 5., 'borrowings_noncurrent': 5.}
        return pd.DataFrame([q, old, annual, balance])

    def frame(self):
        return pd.DataFrame([{'date': pd.Timestamp('2026-10-30'), 'symbol': 'A',
            'symbol_official': 'A', 'isin': 'INE000000001', 'close': 100.,
            'source_status': 'VALID_POSITIVE_VOLUME_SESSION', 'official_source_sha256': 'f'*64,
            'information_cutoff_utc': '2026-10-30T16:30:00Z',
            'distance_ma50': .1, 'distance_ma200': np.nan, 'log_ret63': .1,
            'rsi14': 71., 'p6': .1, 'p12': .2, 'joint12': .15, 'loss12': .4,
            'four_profitable_quarters': np.nan, 'quarter_sales_yoy': .25,
            'quarter_pat_yoy': .25, 'quarter_pat_margin': .1, 'annual_CFO_PAT': 25/30,
            'dated_debt_equity': .1, 'annual_EBIT_proxy_interest_coverage': 21.}])

    def zero_cost(self, w):
        w.loc[w.period_kind.eq('annual'), 'finance_cost'] = 0.
        w.loc[w.period_kind.eq('instant'), ['borrowings_current', 'borrowings_noncurrent']] = 0.
        return w

    def test_single_source_and_missing_supplementary_checks_are_usable(self):
        d, rows, audit = screen(self.frame(), self.wide())
        self.assertTrue(d.available_research_selected.iloc[0])
        self.assertIn(rows[0]['tier'], QUALIFIED)
        self.assertFalse(rows[0]['strict_financial_complete'])
        self.assertEqual(audit['classified_check_positions'], 7)
        self.assertTrue(audit['single_source_allowed'])

    def test_zero_interest_is_na_only_with_matched_no_borrowing_proof(self):
        w = self.zero_cost(self.wide())
        r = assess(w, pd.Timestamp('2026-10-30'))
        self.assertTrue(r['zero_interest_no_borrowings_NA'])
        self.assertTrue(r['check_states']['annual_EBIT_proxy_interest_coverage'].startswith('NOT_APPLICABLE_'))
        self.assertFalse(r['strict_financial_complete'])
        for key in ('borrowings_current', 'borrowings_noncurrent'):
            for val in (np.nan, 1.):
                x = w.copy();x.loc[x.period_kind.eq('instant'), key] = val
                a = assess(x, pd.Timestamp('2026-10-30'))
                self.assertFalse(a['zero_interest_no_borrowings_NA'])
                self.assertEqual(a['tier'], DISCOVERY)

    def test_negative_equity_and_negative_cashflow_are_observed_weakness(self):
        for key, kind in [('equity', 'instant'), ('cash', 'instant'), ('cfo', 'annual')]:
            w = self.wide();w.loc[w.period_kind.eq(kind), key] = -1.
            d, rows, _ = screen(self.frame(), w)
            self.assertFalse(d.available_research_selected.iloc[0])
            self.assertEqual(rows[0]['tier'], 'OBSERVED_CURRENT_FINANCIAL_WEAKNESS')

    def test_a_loss_base_is_turnaround_not_infinite_growth_or_quality_pass(self):
        w = self.wide();w.loc[w.period_end.eq('2025-06-30'), 'pat'] = -8.
        d, rows, _ = screen(self.frame(), w)
        r = rows[0]
        self.assertEqual(r['tier'], TURNAROUND)
        self.assertTrue(r['observed_turnaround'])
        self.assertEqual(r['check_states']['quarter_pat_yoy'], 'TURNAROUND_COMPARISON_NOT_PERCENT_GROWTH')
        self.assertEqual(r['alternatives']['PAT_change_INR_year_on_year'], 18.)
        self.assertTrue(d.available_research_selected.iloc[0])
        self.assertFalse(r['strict_financial_quality_pass'])

    def test_missing_core_is_discovery_not_verified_or_zero_filled(self):
        w = self.wide();w.loc[w.period_kind.eq('instant'), 'cash'] = np.nan
        d, rows, _ = screen(self.frame(), w)
        self.assertEqual(rows[0]['tier'], DISCOVERY)
        self.assertIn('balance_cash', rows[0]['core_missing_fields'])
        self.assertTrue(np.isnan(rows[0]['raw_observations']['balance_cash']))
        self.assertTrue(d.available_research_selected.iloc[0])

    def test_current_loss_cannot_be_accepted_as_turnaround(self):
        w = self.wide(pat=-1.)
        d, rows, _ = screen(self.frame(), w)
        self.assertFalse(d.available_research_selected.iloc[0])
        self.assertFalse(rows[0]['observed_turnaround'])

    def test_owner_conflict_and_ambiguous_latest_original_are_quarantined(self):
        w = self.wide(pat_owners=0., pat_noncontrolling=0.)
        for x in (w, pd.concat([self.wide(), self.wide().iloc[:1]], ignore_index=True)):
            d, rows, _ = screen(self.frame(), x)
            self.assertFalse(d.available_research_eligible.iloc[0])
            self.assertEqual(rows[0]['tier'], 'QUARANTINED_SOURCE_CONFLICT')

    def test_reconciled_negative_owner_profit_is_not_ignored(self):
        w = self.wide(pat_owners=-2., pat_noncontrolling=12.)
        d, rows, _ = screen(self.frame(), w)
        self.assertFalse(d.available_research_eligible.iloc[0])
        self.assertIn('OBSERVED_NONPOSITIVE_RECONCILED_OWNER_PAT', rows[0]['current_safety_failures'])

    def test_future_review_stale_period_and_wrong_isin_cannot_fill_a_gap(self):
        for variant in ('future', 'stale', 'wrong_identity', 'missing_source'):
            w = self.wide()
            if variant == 'future':
                w['available'] = pd.Timestamp('2026-11-02T00:00Z')
                w['available_at_utc'] = '2026-11-02T00:00:00Z'
            elif variant == 'stale':
                w['period'] -= pd.DateOffset(years=2)
                w['period_end'] = w.period.dt.strftime('%Y-%m-%d')
            elif variant == 'wrong_identity':
                w['security_isin'] = 'INE000000002'
            else:
                w['source_sha256'] = None
            _, rows, _ = screen(self.frame(), w)
            self.assertEqual(rows[0]['tier'], DISCOVERY, variant)
            self.assertIn('current_PAT', rows[0]['core_missing_fields'], variant)

    def test_no_standalone_substitute_for_missing_current_group_quarter(self):
        w = self.wide()
        consolidated = w[w.period_kind.isin(['annual', 'instant'])].copy()
        consolidated['reporting_mode'] = 'consolidated'
        _, rows, _ = screen(self.frame(), pd.concat([w, consolidated], ignore_index=True))
        self.assertEqual(rows[0]['preferred_reporting_basis'], 'consolidated')
        self.assertEqual(rows[0]['tier'], DISCOVERY)

    def test_strict_rsi70_boundary_identity_and_probabilities_stay_required(self):
        f = self.frame()
        for key, val in [('rsi14', 70.), ('rsi14', np.nan), ('official_source_sha256', None),
                         ('symbol_official', 'OTHER'), ('p6', .3), ('loss12', .99)]:
            x = f.copy();x.at[0, key] = val
            self.assertFalse(screen(x, self.wide())[0].available_research_eligible.iloc[0], (key, val))

    def test_no_price_band_or_mandatory_dma_catalyst_or_valuation(self):
        f = self.frame();f['close'] = 5.;f['distance_ma200'] = .8
        d, _, _ = screen(f, self.wide())
        self.assertTrue(d.available_research_selected.iloc[0])

    def test_source_facts_features_and_probabilities_are_not_overwritten(self):
        f = self.frame();w = self.wide();original_f = f.copy(deep=True);original_w = w.copy(deep=True)
        d, _, _ = screen(f, w)
        pd.testing.assert_frame_equal(f, original_f)
        pd.testing.assert_frame_equal(w, original_w)
        pd.testing.assert_frame_equal(d[f.columns], f)
        self.assertTrue(np.isnan(d.four_profitable_quarters.iloc[0]))

    def test_missing_model_input_is_distinct_from_observed_zero(self):
        p = {'features': ['x'], 'median': np.array([5.]), 'lo': np.array([-10.]), 'hi': np.array([10.])}
        d = pd.DataFrame({'x': [np.nan, 0., np.inf]})
        np.testing.assert_array_equal(transform(p, d), [[5., 1.], [0., 0.], [5., 1.]])
        self.assertTrue(pd.isna(d.x.iloc[0]))

    def test_replay_link_validates_names_probabilities_and_preserves_unselected(self):
        f = self.frame();f['rsi14'] = 70.
        with tempfile.TemporaryDirectory() as t:
            buf = io.BytesIO();f.drop(columns=['p6', 'p12', 'loss12', 'joint12']).to_parquet(buf,index=False)
            inputs = buf.getvalue();input_sha = hashlib.sha256(inputs).hexdigest()
            root = Path(t);folder = root/'prediction_replay'/input_sha;folder.mkdir(parents=True)
            (folder/'inputs.parquet').write_bytes(inputs)
            payload = {'model_sha256': 'c'*64, 'isins': f['isin'].tolist(),
                       'probabilities': {k: f[k].tolist() for k in ('p6', 'p12', 'loss12', 'joint12')}}
            raw = json.dumps(payload).encode();key = hashlib.sha256(raw).hexdigest()
            (folder/('champion_'+key+'.json')).write_bytes(raw)
            receipt = {'rows': 1, 'replay_verified': True, 'input_parquet_sha256': input_sha,
                       'model_sha256': 'c'*64, 'receipt_sha256': key}
            result = export(f, self.wide(), root, receipt)
            saved = json.loads((root/'available_financial_research'/(result['snapshot_sha256']+'.json')).read_text())
            self.assertTrue(saved['unselected_included'])
            self.assertEqual(len(saved['all_names']), 1)
            self.assertFalse(saved['all_names'][0]['selected'])
            self.assertEqual(saved['selected'], 0)
            self.assertIsNone(saved['all_names'][0]['alternatives']['sales_qoq'])
            self.assertFalse(saved['formal_forecasts_changed'])
            bad = f.copy();bad['p12'] = .21
            with self.assertRaisesRegex(ValueError, 'probabilities'):
                export(bad, self.wide(), root, receipt)
            changed = f.copy();changed['isin'] = 'INE000000002'
            with self.assertRaisesRegex(ValueError, 'inputs|cohort'):
                export(changed, self.wide(), root, receipt)
            changed = f.copy();changed['annual_CFO_PAT'] = .1
            with self.assertRaisesRegex(ValueError, 'inputs'):
                export(changed, self.wide(), root, receipt)

    def test_unknown_original_information_clock_is_never_retrofitted(self):
        f = self.frame();f['information_cutoff_utc'] = None
        with self.assertRaisesRegex(ValueError, 'cutoff'):
            screen(f, self.wide())


if __name__ == '__main__':
    unittest.main()
