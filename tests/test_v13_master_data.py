import unittest
from unittest.mock import patch
from pathlib import Path
import tempfile
import pandas as pd
from v13_master_data import reconcile, parse_screener_inputs,cached_primary_index

class MasterEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.key=('INE123456789','standalone','annual','2026-03-31','revenue')
        self.primary={self.key:{'value_INR':1000.,'source_url':'https://issuer.test/a.pdf','source_sha256':'a'*64}}
    def obs(self,family,value=1000,**kw):
        r=dict(zip(('isin','reporting_mode','period_kind','period_end','metric'),self.key))
        r.update(symbol='TEST',provider_family=family,value_INR=value,rounding_half_width_INR=.5,
            source_url='https://'+family.lower()+'.test/a',first_observed_utc='2026-10-20T10:00:00+00:00',identity_proof='EXACT_ISIN')
        r.update(kw);return r
    def check(self,rows,primary=None):
        return reconcile(self.primary if primary is None else primary,rows,'2026-10-19','2026-10-20T12:00:00+00:00',required_additional=2)[0]
    def test_two_additional_families_match_primary(self):
        r=self.check([self.obs('A'),self.obs('B')])[0]
        self.assertTrue(r['validated_for_new_master_input'])
        self.assertFalse(r['canonical_original_overwritten'])
    def test_user_requirement_needs_only_one_additional_source(self):
        r=reconcile(self.primary,[self.obs('A')],'2026-10-19','2026-10-20T12:00:00Z')[0][0]
        self.assertTrue(r['validated_for_new_master_input'])
        self.assertFalse(r['primary_plus_two_additional_sources_agree'])
    def test_missing_original_can_be_recovered_by_two_distinct_sources(self):
        r=reconcile({},[self.obs('A'),self.obs('B')],'2026-10-19','2026-10-20T12:00:00Z')[0][0]
        self.assertTrue(r['validated_for_new_master_input']);self.assertEqual(r['recovered_value_INR'],1000)
        self.assertIsNone(r['primary_value_INR']);self.assertFalse(r['canonical_original_overwritten'])
    def test_wrapper_and_yahoo_do_not_validate_each_other(self):
        r=reconcile({},[self.obs('YAHOO_FINANCE'),self.obs('INDIAN_STOCK_MARKET_API')],'2026-10-19','2026-10-20T12:00:00Z')[0][0]
        self.assertFalse(r['validated_for_new_master_input'])
    def test_two_source_recovery_rejects_conflicting_third_source(self):
        r=reconcile({},[self.obs('A'),self.obs('B'),self.obs('C',1100)],'2026-10-19','2026-10-20T12:00:00Z')[0][0]
        self.assertFalse(r['validated_for_new_master_input'])
    def test_original_family_cannot_corroborate_itself(self):
        r=reconcile(self.primary,[self.obs('NSE')],'2026-10-19','2026-10-20T12:00:00Z')[0][0]
        self.assertFalse(r['validated_for_new_master_input'])
    def test_mirror_of_one_provider_counts_once(self):
        self.assertFalse(self.check([self.obs('A'),self.obs('A')])[0]['validated_for_new_master_input'])
    def test_secondary_agreement_does_not_replace_missing_primary(self):
        self.assertFalse(self.check([self.obs('A'),self.obs('B')],{})[0]['validated_for_new_master_input'])
    def test_consolidated_and_standalone_are_separate(self):
        r=self.check([self.obs('A'),self.obs('B',reporting_mode='consolidated')])
        self.assertFalse(any(x['validated_for_new_master_input'] for x in r))
    def test_periods_and_kinds_are_separate(self):
        r=self.check([self.obs('A'),self.obs('B',period_kind='quarter')])
        self.assertFalse(any(x['validated_for_new_master_input'] for x in r))
    def test_future_review_observation_excluded(self):
        r=self.check([self.obs('A'),self.obs('B',first_observed_utc='2026-10-21T00:00:00+00:00')])
        self.assertFalse(r[0]['validated_for_new_master_input'])
    def test_conflict_not_resolved_by_majority(self):
        r=self.check([self.obs('A',1010),self.obs('B',1010),self.obs('C',1010)])[0]
        self.assertEqual(r['status'],'NUMERIC_CONFLICT_REVIEW')
    def test_conflicting_values_from_one_family_excluded(self):
        r=self.check([self.obs('A'),self.obs('A',1002),self.obs('B')])[0]
        self.assertFalse(r['validated_for_new_master_input']);self.assertEqual(r['within_family_conflicts'],['A'])
    def test_ambiguous_provider_profit_not_admitted(self):
        r=self.check([self.obs('A',definition_status='UNRESOLVED'),self.obs('B')])[0]
        self.assertFalse(r['validated_for_new_master_input'])
    def test_rounding_does_not_allow_percentage_error(self):
        r=self.check([self.obs('A',1001),self.obs('B',1001)])[0]
        self.assertFalse(r['validated_for_new_master_input'])
    def test_original_index_cache_reopens_when_information_clock_admits_filing(self):
        wide=pd.DataFrame({'period':pd.to_datetime(['2026-03-31']*2),
            'available':pd.to_datetime(['2026-10-20T10:00:00Z','2026-10-20T13:00:00Z'])})
        with tempfile.TemporaryDirectory() as temp:
            (Path(temp)/'financial_wide.parquet').write_bytes(b'fixture-source-matrix')
            with patch('v13_master_data.primary_index',return_value=self.primary) as index:
                first=cached_primary_index(temp,wide,'2026-10-19','2026-10-20T12:00:00Z')
                self.assertEqual(first,self.primary)
                self.assertEqual(cached_primary_index(temp,wide,'2026-10-19','2026-10-20T12:30:00Z'),first)
                self.assertEqual(index.call_count,1)
                cached_primary_index(temp,wide,'2026-10-19','2026-10-20T14:00:00Z')
                self.assertEqual(index.call_count,2)
    def test_parser_keeps_units_reporting_basis_and_profit_definition(self):
        raw=b'''<div id="company-info" data-consolidated="false"></div>
        <a href="https://www.nseindia.com/get-quotes/equity?symbol=TEST">NSE</a>
        <section id="quarters">Standalone Figures in Rs. Crores<table>
        <tr><th></th><th>Jun 2026</th></tr><tr><td>Sales +</td><td>12.34</td></tr>
        <tr><td>Net Profit +</td><td>1.00</td></tr></table></section>'''
        rows=parse_screener_inputs(raw,'TEST','INE123456789',{
            'requested_url':'https://www.screener.in/company/TEST/',
            'sha256':'a'*64,'first_retrieved_utc':'2026-10-20T10:00:00+00:00'})
        self.assertEqual(rows[0]['value_INR'],123400000)
        self.assertEqual(rows[0]['reporting_mode'],'standalone')
        self.assertEqual(rows[1]['definition_status'],'UNRESOLVED')

if __name__=='__main__':unittest.main()
