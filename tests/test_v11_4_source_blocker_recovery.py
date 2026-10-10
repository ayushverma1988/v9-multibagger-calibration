import copy
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from v11_4_verified_current_financials import parse_financial_document, derive_metrics, numeric
from v11_4_exchange_reference_universe import build_universe, isin_valid
from v11_4_source_blocker_recovery import current_catalog, select_current_filings
from v11_4_primary_document_catalysts import classify_document
from v11_4_independent_evaluation_ledger import (
    freeze_protocol, record_observation, load_protocol, encoded, fingerprint, assess_accessible_event)


def document(extra="", quarter=False):
    start = "2026-01-01" if quarter else "2025-04-01"
    return f'''<xbrli:xbrl xmlns:xbrli="http://www.xbrl.org/2003/instance" xmlns:x="http://example">
    <xbrli:context id="D"><xbrli:period><xbrli:startDate>{start}</xbrli:startDate><xbrli:endDate>2026-03-31</xbrli:endDate></xbrli:period></xbrli:context>
    <xbrli:unit id="INR"><xbrli:measure>iso4217:INR</xbrli:measure></xbrli:unit>
    <x:Symbol contextRef="D">TEST</x:Symbol><x:ISIN contextRef="D">INE002A01018</x:ISIN>
    <x:NatureOfReportStandaloneConsolidated contextRef="D">Consolidated</x:NatureOfReportStandaloneConsolidated>
    <x:LevelOfRounding contextRef="D">Lakhs</x:LevelOfRounding>
    <x:RevenueFromOperations contextRef="D" unitRef="INR">20000000</x:RevenueFromOperations>
    <x:ProfitLossForPeriod contextRef="D" unitRef="INR">3000000</x:ProfitLossForPeriod>{extra}</xbrli:xbrl>'''.encode()


HTML = b'<table><tr><td>Revenue from operations</td><td>200.00</td></tr><tr><td>Total profit (loss) for period</td><td>30.00</td></tr></table>'


class FinancialEvidenceTests(unittest.TestCase):
    def test_xbrl_already_rupees_render_lakhs_not_double_scaled(self):
        rows, _ = parse_financial_document(document(), "TEST", "2026-03-31", "Consolidated", HTML)
        self.assertEqual(rows[0]["value_INR"], 20000000)
        self.assertTrue(all(r["rendered_statement_agrees"] for r in rows))

    def test_render_mismatch_cannot_derive_ratio(self):
        rows, _ = parse_financial_document(document(), "TEST", "2026-03-31", "Consolidated", HTML.replace(b'200.00', b'2.00'))
        self.assertFalse(rows[0]["rendered_statement_agrees"])

    def test_issuer_or_basis_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError, "symbol"):
            parse_financial_document(document(), "OTHER", "2026-03-31", "Consolidated", HTML)
        with self.assertRaisesRegex(ValueError, "basis"):
            parse_financial_document(document(), "TEST", "2026-03-31", "Standalone", HTML)

    def test_quarter_cannot_be_annual(self):
        rows, _ = parse_financial_document(document(quarter=True), "TEST", "2026-03-31", "Consolidated", HTML)
        self.assertEqual({r["period_kind"] for r in rows}, {"quarter"})
        self.assertEqual(derive_metrics(rows), {})

    def test_duplicate_conflicting_same_period_fact_is_rejected(self):
        other = '<x:RevenueFromOperations contextRef="D" unitRef="INR">50000000</x:RevenueFromOperations>'
        rows, audit = parse_financial_document(document(other), "TEST", "2026-03-31", "Consolidated", HTML)
        self.assertEqual([r["metric"] for r in rows], ["pat"])
        self.assertEqual(len(audit["ambiguous_facts_rejected"]), 1)

    def test_monetary_fact_with_per_share_unit_rejected(self):
        xml = document().replace(b'<xbrli:measure>iso4217:INR</xbrli:measure>', b'<xbrli:measure>iso4217:INR</xbrli:measure><xbrli:measure>xbrli:shares</xbrli:measure>')
        rows, _ = parse_financial_document(xml, "TEST", "2026-03-31", "Consolidated", HTML)
        self.assertEqual(rows, [])

    def test_zero_loss_parentheses_and_infinite_numbers(self):
        self.assertEqual(numeric(0), 0)
        self.assertEqual(numeric('(1,234.5)'), -1234.5)
        self.assertIsNone(numeric('NaN'))
        self.assertIsNone(numeric('25 crore'))

    def test_missing_debt_component_not_synthetic_zero(self):
        def f(k, v, kind):
            return {"metric": k, "value_INR": v, "period_end": "2026-03-31", "period_kind": kind,
                    "reporting_mode": "consolidated", "rendered_statement_agrees": True}
        rows = [f("pat", 20, "annual"), f("equity", 100, "instant"), f("borrowings_current", 5, "instant"), f("reserves", 50, "instant")]
        metrics = derive_metrics(rows)
        self.assertNotIn("debt_to_equity", metrics)
        self.assertNotIn("reserves_gt_borrowings", metrics)

    def test_three_observations_do_not_fabricate_three_year_cagr(self):
        rows = [{"metric": "revenue", "value_INR": v, "period_end": f"{y}-03-31", "period_kind": "annual",
                 "reporting_mode": "consolidated", "rendered_statement_agrees": True}
                for y, v in [(2024, 100), (2025, 120), (2026, 160)]]
        self.assertNotIn("sales_growth_3y", derive_metrics(rows))

    def test_future_exchange_receipt_time_is_excluded(self):
        class Store:
            def json(self, *args):
                return {"data": [{"symbol": "TEST", "qe_Date": "30-JUN-2026", "broadcast_Date": "09-Oct-2026 15:29:00",
                                   "creation_Date": "09-Oct-2026 15:31:00", "consolidated": "Standalone"}], "totalCount": 1}, {"sha256": "a"*64}
        self.assertEqual(current_catalog(Store(), "TEST", pd.Timestamp("2026-10-09T10:00:00Z")), [])

    def test_filing_choice_never_mixes_reporting_basis(self):
        rows = [{"symbol": "TEST", "period_end": end, "available_at_utc": pub, "reporting_mode": basis}
                for end, pub, basis in [('2026-06-30', '2026-08-01T10:00:00Z', 'consolidated'),
                                        ('2026-03-31', '2026-05-01T10:00:00Z', 'consolidated'),
                                        ('2026-03-31', '2026-05-02T10:00:00Z', 'standalone')]]
        self.assertEqual({r['reporting_mode'] for r in select_current_filings(rows)}, {'consolidated'})


class ExchangeAndCatalystTests(unittest.TestCase):
    def test_isin_checksum(self):
        self.assertTrue(isin_valid('INE002A01018'))
        self.assertFalse(isin_valid('INE002A01019'))

    def test_same_symbol_changed_isin_does_not_match_size_or_exchange(self):
        amfi = pd.DataFrame([{'isin': 'INE002A01018', 'nse_reference_symbol': 'SAME', 'bse_reference_symbol': 'SAME', 'amfi_size_category': 'SMALL'}])
        nse = pd.DataFrame([{'isin': 'INE040A01034', 'nse_current_symbol': 'SAME'}])
        x, report = build_universe(amfi, nse)
        self.assertEqual(len(x), 2)
        self.assertEqual(report['current_NSE_ISINs_with_official_size_category'], 0)
        self.assertEqual(report['NSE_BSE_reference_overlap_ISINs'], 0)
        self.assertEqual(report['NSE_same_symbol_changed_ISIN_not_auto_joined'], 1)

    def test_reference_bse_presence_cannot_be_current_active_coverage(self):
        amfi = pd.DataFrame([{'isin': 'INE002A01018', 'nse_reference_symbol': 'TEST', 'bse_reference_symbol': 'TEST', 'amfi_size_category': 'SMALL'}])
        nse = pd.DataFrame([{'isin': 'INE002A01018', 'nse_current_symbol': 'TEST'}])
        _, report = build_universe(amfi, nse)
        self.assertFalse(report['current_active_BSE_master_complete'])
        self.assertFalse(report['all_listed_and_tradeable_equities_claim_supported'])

    def test_order_value_is_disclosure_candidate_not_verified_earnings(self):
        r = classify_document('Test Industries Limited has received an order. Order value INR 200 crores for supply of transformers.', 'TEST', 'Test Industries Limited')
        self.assertEqual(r['quantified_evidence'][0]['quantity'], 200)
        self.assertFalse(r['complete_causal_chain_verified'])
        self.assertFalse(r['quantified_evidence'][0]['earnings_impact_verified'])

    def test_proposed_capacity_is_conditional(self):
        r = classify_document('Test Industries Limited plans a new plant. Production capacity 500 MW subject to approval.', 'TEST', 'Test Industries Limited')
        self.assertEqual(r['quantified_evidence'][0]['status'], 'CONDITIONAL_CANDIDATE')

    def test_unrelated_issuer_attachment_cannot_activate_event(self):
        r = classify_document('Other issuer received order value INR 200 crores', 'TEST', 'Test Industries Limited')
        self.assertFalse(r['issuer_identity_read'])
        self.assertEqual(r['quantified_evidence'], [])


class ProspectiveLedgerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = self.root / 'model.joblib'; self.model.write_bytes(b'test-frozen-weights')
        self.protocol, self.digest = freeze_protocol(self.root / 'ledger', self.model,
            {'feature_names': ['a'], 'source_code_sha256': {'module': 'a'*64}}, '2026-10-10T01:00:00Z')
        self.candidates = pd.DataFrame({'isin': [f'ISIN{i}' for i in range(11)], 'symbol': [f'TEST{i}' for i in range(11)],
                                        'rank': range(1,12), 'probability': [.20-.005*i for i in range(11)]})

    def record(self, **changes):
        args = dict(folder=self.root/'ledger', protocol=self.protocol, protocol_sha256=self.digest, model_path=self.model,
                    ranked_candidates=self.candidates, source_sha256={'prices': 'b'*64},
                    source_first_seen_utc={'prices': '2026-10-12T15:40:00Z'}, decision_close_utc='2026-10-12T10:00:00Z',
                    next_session_open_utc='2026-10-13T03:45:00Z', first_recorded_utc='2026-10-12T16:00:00Z')
        args.update(changes)
        return record_observation(**args)

    def test_backfilled_oct9_recovery_cannot_be_blind_observation(self):
        with self.assertRaisesRegex(ValueError, 'Backfilled'):
            self.record(decision_close_utc='2026-10-09T10:00:00Z')

    def test_immutable_observation_preserves_exact_ten(self):
        record = self.record()
        self.assertEqual(len(record['selected']), 10)
        self.assertEqual(record['selected'][-1]['symbol'], 'TEST9')
        with self.assertRaises(FileExistsError): self.record()

    def test_weights_cannot_be_retrained_after_freeze(self):
        self.model.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'Weights'): self.record()

    def test_source_after_recording_or_decision_after_next_open_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Source acquired'):
            self.record(source_first_seen_utc={'prices': '2026-10-12T16:01:00Z'})
        with self.assertRaisesRegex(ValueError, 'Backfilled'):
            self.record(first_recorded_utc='2026-10-13T04:00:00Z')

    def test_missing_or_future_labels_cannot_filter_cohort(self):
        with self.assertRaisesRegex(ValueError, 'Future outcome'):
            self.record(ranked_candidates=self.candidates.assign(integrity_y6_clean=True))

    def test_protocol_tampering_detected_against_external_receipt(self):
        path = self.root/'ledger'/'protocol.json'
        path.write_bytes(encoded({**self.protocol, 'acceptance_gates': {'mean_top10_jaccard': 0}}))
        with self.assertRaisesRegex(ValueError, 'pinned receipt'): load_protocol(path, self.digest)

    def test_not_mature_cannot_report_success_or_failure(self):
        sessions = pd.date_range('2026-10-13', periods=126, freq='B', tz='UTC') + pd.Timedelta(hours=10)
        r = assess_accessible_event(100, pd.DataFrame(), sessions, '2026-10-20T20:00:00Z')
        self.assertEqual(r['status'], 'NOT_MATURE'); self.assertIsNone(r['observed_2x'])

    def test_missing_session_cannot_be_negative_or_replace_selection(self):
        sessions = pd.date_range('2026-10-13', periods=126, freq='B', tz='UTC') + pd.Timedelta(hours=10)
        prices = pd.DataFrame({'session_close_utc': sessions[:-1], 'adjusted_close': 100, 'turnover_INR': 6000000, 'source_verified': True})
        r = assess_accessible_event(100, prices, sessions, sessions[-1])
        self.assertEqual(r['status'], 'UNKNOWN_MISSING_OR_UNVERIFIED_SESSIONS'); self.assertIsNone(r['observed_2x'])

    def test_mature_liquid_three_session_event(self):
        sessions = pd.date_range('2026-10-13', periods=126, freq='B', tz='UTC') + pd.Timedelta(hours=10)
        prices = pd.DataFrame({'session_close_utc': sessions, 'adjusted_close': 100., 'turnover_INR': 6000000, 'source_verified': True})
        prices.loc[60:62, 'adjusted_close'] = 210
        self.assertEqual(assess_accessible_event(100, prices, sessions, sessions[-1])['observed_2x'], 1)
        prices.loc[60:62, 'turnover_INR'] = 1000000
        self.assertEqual(assess_accessible_event(100, prices, sessions, sessions[-1])['observed_2x'], 0)


if __name__ == '__main__':
    unittest.main()
