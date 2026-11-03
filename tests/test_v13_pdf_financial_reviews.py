import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import requests

from v13_pdf_financial_reviews import amount, reviewed_facts, collect, original_get


class ReviewedFinancialPDF(unittest.TestCase):
    def record(self):
        path = Path(__file__).resolve().parents[1] / 'v13_evidence_seed/financial_pdf_reviews_20261102.json'
        return json.loads(path.read_text())['documents'][0]

    def text(self, record):
        return '\n'.join([record[k] for k in ('issuer_excerpt', 'basis_excerpt', 'units_excerpt', 'header_excerpt')]
                         + [r['row_excerpt'] for r in record['fields']])

    def parse(self, record):
        return reviewed_facts(record, self.text(record), '2026-11-02T10:00Z',
                              '2026-11-02', clock='2026-11-02T11:00Z')

    def test_original_comparatives_have_actual_later_availability_and_owner_pat(self):
        rows = self.parse(self.record()); fields = {r['metric']: r['value'] for r in rows}
        self.assertEqual(fields['revenue'], 1422650000.)
        self.assertEqual(fields['pat'], 243800000.)
        self.assertEqual(fields['pat_owners'], 232060000.)
        self.assertEqual(fields['pat_noncontrolling'], 11740000.)
        self.assertEqual({r['period_end'] for r in rows}, {'2025-06-30'})
        self.assertEqual({r['available_at_utc'] for r in rows}, {'2026-11-02T10:00:00+00:00'})
        self.assertTrue(all(pd.Timestamp(r['original_publication_utc']) < pd.Timestamp(r['available_at_utc']) for r in rows))

    def test_blank_and_dash_are_not_zero_but_printed_zero_is_numeric(self):
        for token in ('-', '', 'nil', 'N/A', '1.23.45'):
            with self.assertRaises(ValueError): amount(token)
        self.assertEqual(amount('0.00'), 0.)
        self.assertEqual(amount('(236)'), -236.)
        self.assertEqual(amount('1.422.65'), 1422.65)

    def test_missing_source_excerpt_is_rejected(self):
        record = self.record()
        with self.assertRaises(ValueError):
            reviewed_facts(record, 'Unrelated company report', '2026-11-02T10:00Z',
                           '2026-11-02', clock='2026-11-02T11:00Z')

    def test_basis_units_unreviewed_input_and_wrong_column_are_rejected(self):
        for changes in ({'reporting_mode': 'standalone'}, {'scale_INR': 100000},
                        {'review_status': 'AUTOMATIC_TEXT_CANDIDATE'},
                        {'column_index': 99}, {'column_index': 3}, {'source_sha256': 'bad'}):
            record = {**self.record(), **changes}
            with self.subTest(changes=changes), self.assertRaises(ValueError): self.parse(record)

    def test_future_review_or_retrieval_cannot_be_backdated(self):
        record = {**self.record(), 'first_reviewed_utc': '2026-11-03T10:00Z'}
        with self.assertRaises(ValueError): self.parse(record)
        record = {**self.record(), 'first_reviewed_utc': '2026-08-01T10:00Z'}
        with self.assertRaises(ValueError): self.parse(record)

    def test_annual_and_out_of_window_columns_cannot_become_quarters(self):
        record = self.record(); record['period_start'] = '2025-04-01'
        record['column_dates'][2] = '2026-03-31'
        with self.assertRaises(ValueError): self.parse(record)
        record = self.record(); record['period_start'] = '2021-04-01'; record['column_dates'][2] = '2021-06-30'
        with self.assertRaises(ValueError): self.parse(record)

    def test_reported_cell_order_cannot_be_swapped(self):
        record = self.record(); record['fields'][0]['reported_cells'].reverse()
        with self.assertRaises(ValueError): self.parse(record)

    def test_eps_and_derived_profit_are_not_accepted_as_money_facts(self):
        for metric in ('eps_diluted_continuing', 'forward_PAT', 'pat_owners_by_subtraction'):
            record = self.record(); record['fields'][0]['metric'] = metric
            with self.assertRaises(ValueError): self.parse(record)

    def test_rounding_does_not_allow_a_material_owner_total_contradiction(self):
        record = self.record()
        row = next(r for r in record['fields'] if r['metric'] == 'pat_owners')
        row['row_excerpt'] = row['row_excerpt'].replace('232.06', '132.06')
        row['reported_cells'][2] = '132.06'
        with self.assertRaisesRegex(ValueError, 'reconcile'): self.parse(record)

    def test_source_review_record_is_not_modified_by_validation(self):
        record = self.record(); original = copy.deepcopy(record)
        self.parse(record)
        self.assertEqual(record, original)

    def setup_collect(self, folder, wrong_identity=False, existing=False):
        root = Path(folder); record = self.record()
        raw = b'unit-test original PDF bytes'
        record['source_sha256'] = hashlib.sha256(raw).hexdigest()
        record['first_reviewed_utc'] = '2026-09-01T00:00Z'
        seed = root / 'seed.json'; seed.write_text(json.dumps({'documents': [record]}))
        pd.DataFrame({'nse_current_symbol': [record['symbol']],
            'isin': ['INE000000001' if wrong_identity else record['isin']]}).to_parquet(root/'universe.parquet', index=False)
        (root/'frozen_forecasts.json').write_bytes(b'frozen originals')
        old = pd.DataFrame([{'security_isin': record['isin'], 'reporting_mode': 'consolidated',
            'period_kind': 'quarter', 'period_end': '2025-06-30', 'revenue': 999., 'pat': 111.}])
        if existing: old.to_parquet(root/'financial_wide.parquet', index=False)
        catalog = [{'symbol': record['symbol'], 'sm_isin': record['isin'],
            'attchmntFile': record['source_url'], 'exchdisstime': '12-Aug-2026 14:43:27'}]
        proof = {'sha256': record['source_sha256'], 'first_retrieved_utc': '2026-09-02T00:00Z'}
        source = MagicMock()
        source.get.side_effect = lambda url: (json.dumps(catalog).encode(), {'sha256':'b'*64}) if url == record['catalogue_url'] else (raw, proof)
        page = MagicMock(); page.extract_text.return_value = self.text(record)
        pdf = MagicMock(); pdf.__enter__.return_value.pages = [page]*8
        return seed, source, pdf, old

    def test_existing_original_quarter_is_not_overwritten_and_marker_is_idempotent(self):
        with tempfile.TemporaryDirectory() as t:
            seed, source, pdf, old = self.setup_collect(t, existing=True)
            with patch('v13_pdf_financial_reviews.Store', return_value=source), patch('v13_pdf_financial_reviews.pdfplumber.open', return_value=pdf):
                result = collect(t, seed, '2026-10-30')
                again = collect(t, seed, '2026-10-30')
            self.assertTrue(result['complete']); self.assertEqual(result['missing_quarter_fact_observations_added'], 0)
            pd.testing.assert_frame_equal(pd.read_parquet(Path(t)/'financial_wide.parquet'), old)
            self.assertEqual(result, again); self.assertEqual(source.get.call_count, 2)
            self.assertEqual((Path(t)/'frozen_forecasts.json').read_bytes(), b'frozen originals')

    def test_wrong_master_identity_stops_before_source_fetch_or_financial_write(self):
        with tempfile.TemporaryDirectory() as t:
            seed, source, pdf, _ = self.setup_collect(t, wrong_identity=True)
            with patch('v13_pdf_financial_reviews.Store', return_value=source):
                result = collect(t, seed, '2026-10-30')
            self.assertFalse(result['complete']); source.get.assert_not_called()
            self.assertFalse((Path(t)/'financial_wide.parquet').exists())

    def test_retrieved_pdf_bytes_must_match_both_receipt_and_reviewed_hash(self):
        with tempfile.TemporaryDirectory() as t:
            seed, source, pdf, _ = self.setup_collect(t)
            original = source.get.side_effect
            source.get.side_effect = lambda url: original(url) if url.endswith('12-08-2026') else (b'changed', original(url)[1])
            with patch('v13_pdf_financial_reviews.Store', return_value=source):
                result = collect(t, seed, '2026-10-30')
            self.assertFalse(result['complete']); self.assertFalse((Path(t)/'financial_wide.parquet').exists())

    def test_transient_source_timeout_retries_but_permission_failure_does_not(self):
        first = MagicMock(); first.folder = Path('/unused-test-cache')
        first.get.side_effect = requests.exceptions.ReadTimeout('Read timed out')
        fresh = MagicMock(); fresh.get.return_value = (b'original', {'sha256': 'a'*64})
        with patch('v13_pdf_financial_reviews.Store', return_value=fresh), patch('v13_pdf_financial_reviews.time.sleep'):
            raw, proof = original_get(first, 'https://www.nseindia.com/public')
        self.assertEqual(raw, b'original'); self.assertEqual(proof['bounded_fetch_attempts'], 2)
        first.get.side_effect = requests.exceptions.HTTPError('403 Forbidden')
        with patch('v13_pdf_financial_reviews.Store') as factory, self.assertRaises(requests.exceptions.HTTPError):
            original_get(first, 'https://www.nseindia.com/public')
        factory.assert_not_called()

    def test_transient_source_failure_is_bounded_to_three_attempts(self):
        source = MagicMock(); source.folder = Path('/unused-test-cache')
        source.get.side_effect = requests.exceptions.ReadTimeout('Read timed out')
        with patch('v13_pdf_financial_reviews.Store', return_value=source), patch('v13_pdf_financial_reviews.time.sleep'), self.assertRaises(requests.exceptions.ReadTimeout):
            original_get(source, 'https://www.nseindia.com/public')
        self.assertEqual(source.get.call_count, 3)


if __name__ == '__main__': unittest.main()
