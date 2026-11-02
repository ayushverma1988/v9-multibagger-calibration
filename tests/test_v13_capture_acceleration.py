import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from v13_capture_acceleration import financial_profile, fixed_cohort_comparison, terminal_history_error, planning_financials
from v13_financial_evidence import completion_queue
from v13_parallel_collection import bounded_map
from v13_collection import current_history


class CaptureAcceleration(unittest.TestCase):
    def wide(self, symbol='A', isin='INE000000001', prior_pat=10., missing_cfo=False):
        common = {'symbol': symbol, 'security_isin': isin,
            'reporting_mode': 'consolidated',
            'available': pd.Timestamp('2026-08-01T12:00Z'),
            'available_at_utc': '2026-08-01T12:00:00+00:00',
            'source_sha256': 'a' * 64, 'source_url': 'https://nsearchives.nseindia.com/a.xml'}
        rows = []
        for date in ('2026-06-30', '2026-03-31', '2025-12-31', '2025-09-30', '2025-06-30'):
            rows.append({**common, 'period': pd.Timestamp(date), 'period_end': date,
                'period_kind': 'quarter', 'pat': prior_pat if date == '2025-06-30' else 12.,
                'revenue': 100. if date == '2025-06-30' else 120.})
        rows.append({**common, 'period': pd.Timestamp('2026-03-31'), 'period_end': '2026-03-31',
            'period_kind': 'annual', 'pat': 40., 'revenue': 400., 'pbt': 44.,
            'finance_cost': 4., **({} if missing_cfo else {'cfo': 40.})})
        rows.append({**common, 'period': pd.Timestamp('2026-03-31'), 'period_end': '2026-03-31',
            'period_kind': 'instant', 'equity': 100., 'borrowings_current': 10.,
            'borrowings_noncurrent': 10.})
        return pd.DataFrame(rows)

    def master(self):
        return pd.DataFrame({'nse_current_symbol': ['A'], 'isin': ['INE000000001']})

    def test_nonpositive_base_is_known_raw_data_not_a_missing_download(self):
        source = self.wide(prior_pat=-1.)
        original = source.copy(deep=True)
        profile = financial_profile(source, self.master(), {'A'}, '2026-10-30')
        self.assertEqual(profile['status_counts'], {'OBSERVED_NONPOSITIVE_BASE_ONLY': 1})
        self.assertEqual(profile['raw_fact_complete_rows'], 1)
        self.assertEqual(profile['recoverable_missing_fact_rows'], 0)
        self.assertFalse(profile['rows'][0]['financial_complete'])
        self.assertIn('quarter_pat_yoy', profile['rows'][0]['observed_nonpositive_fields'])
        pd.testing.assert_frame_equal(source, original)

    def test_fixed_cohort_records_actual_cash_recovery_without_new_denominator(self):
        before = financial_profile(self.wide(missing_cfo=True), self.master(), {'A'}, '2026-10-30')
        after = financial_profile(self.wide(), self.master(), {'A'}, '2026-10-30')
        result = fixed_cohort_comparison(before, after)
        self.assertEqual(result['complete_before'], 0)
        self.assertEqual(result['complete_after'], 1)
        self.assertEqual(result['fixed_cohort_rows'], 1)
        self.assertEqual(result['newly_complete'], [{'symbol': 'A', 'isin': 'INE000000001'}])
        self.assertEqual(result['missing_field_positions_before'], 1)
        self.assertEqual(result['missing_field_positions_after'], 0)

    def test_current_master_identity_and_public_cohort_are_required(self):
        profile = financial_profile(self.wide(isin='INE000000002'), self.master(), {'A', 'PRIVATE'}, '2026-10-30')
        self.assertEqual(profile['cohort_rows'], 1)
        self.assertEqual(profile['raw_fact_complete_rows'], 0)
        self.assertEqual(profile['rows'][0]['isin'], 'INE000000001')
        self.assertEqual(profile['rows'][0]['priority'], 0)

    def test_future_or_outside_four_year_facts_cannot_fill_the_profile(self):
        for source in (self.wide().assign(available=pd.Timestamp('2026-11-01T00:00Z')),
                       self.wide().assign(period=pd.Timestamp('2021-06-30'))):
            with self.subTest(rows=len(source)):
                result = financial_profile(source, self.master(), {'A'}, '2026-10-30')
                self.assertEqual(result['raw_fact_complete_rows'], 0)
                self.assertEqual(result['complete_rows'], 0)

    def test_comparison_rejects_changed_identity_or_source_date(self):
        before = financial_profile(self.wide(), self.master(), {'A'}, '2026-10-30')
        for after in ({**before, 'source_asof': '2026-10-29'},
                      {**before, 'rows': [{**before['rows'][0], 'isin': 'INE000000002'}]}):
            with self.assertRaises(ValueError):
                fixed_cohort_comparison(before, after)

    def test_newly_incomplete_row_is_reported_not_forced_complete(self):
        before = financial_profile(self.wide(), self.master(), {'A'}, '2026-10-30')
        after = financial_profile(self.wide(missing_cfo=True), self.master(), {'A'}, '2026-10-30')
        result = fixed_cohort_comparison(before, after)
        self.assertEqual(result['newly_incomplete'], [{'symbol': 'A', 'isin': 'INE000000001'}])
        self.assertEqual(result['complete_after'], 0)

    def test_explicit_public_issuer_recovery_priority_precedes_hash_order(self):
        rows = []
        for symbol in ('A', 'B'):
            for date in ('2026-06-30', '2026-03-31', '2025-12-31', '2025-09-30', '2025-06-30'):
                rows.append({'symbol': symbol, 'period_end': date, 'reporting_mode': 'consolidated',
                    'available_at_utc': '2026-08-01T00:00Z', 'last_attempt': '0000', 'filed_period': 'Quarterly'})
        result = completion_queue(pd.DataFrame(rows), pd.DataFrame(), '2026-10-30', 1, {'A': 1, 'B': 0})
        self.assertEqual(result.symbol.tolist(), ['B'])

    def test_six_inflight_jobs_can_overlap_without_sequential_worker_deadlock(self):
        gate = threading.Barrier(6, timeout=10)
        def work(item):
            gate.wait()
            return {'item': item}
        results = list(bounded_map(range(6), work, workers=6, seconds=20))
        self.assertEqual(sorted(r['item'] for _, r in results), list(range(6)))

    def test_access_stop_with_more_workers_does_not_submit_the_remainder(self):
        results = list(bounded_map(range(30), lambda item: {'error': '403 Forbidden'}, workers=6))
        self.assertLessEqual(len(results), 6)

    def test_definite_wrong_instrument_does_not_retry_two_full_history_variants(self):
        class Store:
            calls = 0
            def get(self, url):
                self.calls += 1
                return b'{}', {}
        store = Store()
        with patch('v13_collection.parse_chart', side_effect=ValueError('Wrong symbol, venue, currency or instrument')):
            result = current_history(store, 'A', 'INE000000001', '2026-10-30', pd.DataFrame(), pd.DataFrame())
        self.assertEqual(store.calls, 1)
        self.assertEqual(result['redundant_variants_skipped'], 2)
        self.assertTrue(result['terminal_source_error'])

    def test_omitted_sessions_stale_data_and_timeouts_still_allow_variants(self):
        for error in ('Provider refresh omitted 3 accepted sessions',
                      'Fewer than 127 valid daily observations', 'Read timed out'):
            self.assertFalse(terminal_history_error(error))
        self.assertTrue(terminal_history_error('404 Client Error: Not Found'))

    def test_old_isin_facts_cannot_hide_a_current_identity_download_gap(self):
        old = self.wide(isin='INE000000002')
        original = old.copy(deep=True)
        view = planning_financials(old, self.master(), '2026-10-30')
        self.assertTrue(view.empty)
        queue = pd.DataFrame([{'symbol': 'A', 'period_end': '2026-06-30',
            'reporting_mode': 'consolidated', 'available_at_utc': '2026-08-01T00:00Z',
            'last_attempt': '0000', 'filed_period': ''}])
        self.assertEqual(len(completion_queue(queue, view, '2026-10-30', 1)), 1)
        pd.testing.assert_frame_equal(old, original)

    def test_same_isin_rename_is_normalised_only_in_the_planning_copy(self):
        old = self.wide(symbol='OLD')
        result = planning_financials(old, self.master(), '2026-10-30')
        self.assertEqual(set(result.symbol), {'A'})
        self.assertEqual(set(old.symbol), {'OLD'})
        self.assertEqual(set(result.source_sha256), {'a' * 64})

    def test_planning_rejects_ambiguous_master_and_filters_unavailable_facts(self):
        with self.assertRaises(ValueError):
            planning_financials(self.wide(), pd.concat([self.master(), self.master()]), '2026-10-30')
        source = self.wide().assign(available=pd.Timestamp('2026-11-01T00:00Z'))
        self.assertTrue(planning_financials(source, self.master(), '2026-10-30').empty)

    def test_unlabelled_integrated_filing_can_fill_observed_annual_cash_gap(self):
        source = self.wide(missing_cfo=True)
        queue = pd.DataFrame([{'symbol': 'A', 'period_end': '2026-03-31',
            'reporting_mode': 'consolidated', 'available_at_utc': '2026-08-01T00:00Z',
            'last_attempt': '0000', 'filed_period': ''}])
        self.assertEqual(len(completion_queue(queue, source, '2026-10-30', 1)), 1)


if __name__ == '__main__':
    unittest.main()
