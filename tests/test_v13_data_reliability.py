import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from v12_5_features import features,reporting_basis_status
from v13_financial_evidence import financial_quality,financial_readiness,completion_queue
from v13_collection import current_history

class BasisContinuity(unittest.TestCase):
    def fixture(self):
        rows=[]
        for basis,period,rev,pat in [('consolidated','2026-03-31',100,10),
          ('standalone','2026-06-30',30,2),('standalone','2025-06-30',40,4)]:
            rows.append(dict(reporting_mode=basis,period_kind='quarter',period=pd.Timestamp(period),
              period_end=period,available=pd.Timestamp('2026-08-01T00:00:00Z'),
              available_at_utc='2026-08-01T00:00:00Z',source_sha256='a'*64,revenue=rev,pat=pat))
        return pd.DataFrame(rows)

    def test_missing_group_quarter_cannot_create_standalone_decline(self):
        w=self.fixture();date=pd.Timestamp('2026-10-19')
        f,_=features(w,date)
        self.assertTrue(np.isnan(f['quarter_pat_yoy']))
        self.assertEqual(financial_quality(f)['status'],'UNKNOWN_FINANCIAL_INPUTS')
        status=financial_readiness(w,date,f)
        self.assertEqual(status['preferred_reporting_basis'],'consolidated')
        self.assertIn('NO_STANDALONE_SUBSTITUTION',status['reporting_basis_status'])
        self.assertFalse(status['preferred_quarter_fresh'])

    def test_pending_group_quarter_prioritized_despite_newer_standalone(self):
        w=self.fixture()
        q=pd.DataFrame([dict(symbol='A',reporting_mode='consolidated',period_end='2026-06-30',
          available_at_utc='2026-07-21T12:00:00Z',filed_period='Quarterly',last_attempt='0000',
          source_url='https://nsearchives.nseindia.com/a.xml',version_key='a')])
        w['symbol']='A'
        result=completion_queue(q,w,'2026-10-19',2)
        self.assertEqual(len(result),1)
        self.assertEqual(result.reporting_mode.iloc[0],'consolidated')

    def test_new_group_quarter_uses_group_comparison_only(self):
        w=self.fixture();extra=[]
        for period,rev,pat in [('2026-06-30',120,12),('2025-06-30',100,10)]:
            r=w.iloc[0].to_dict();r.update(period=pd.Timestamp(period),period_end=period,revenue=rev,pat=pat)
            extra.append(r)
        f,_=features(pd.concat([w,pd.DataFrame(extra)]),pd.Timestamp('2026-10-19'))
        self.assertAlmostEqual(f['quarter_pat_yoy'],.2)

class CurrentHistory(unittest.TestCase):
    def bars(self,date):
        return pd.DataFrame([dict(date=pd.Timestamp(date),symbol='A',isin='INE000000001',open=100.,high=101.,low=99.,close=100.,volume=10.,adj_close=100.)])
    def test_stale_success_retries_and_only_source_aligned_current_passes(self):
        old=self.bars('2026-10-16');fresh=self.bars('2026-10-19')
        official=fresh.drop(columns='adj_close').assign(source_status='VALID_POSITIVE_VOLUME_SESSION')
        class Store:
            calls=[]
            def get(self,url):
                self.calls.append(url);return b'{}',{}
        store=Store()
        replies=[(old,{'latest_bar':'2026-10-16','source_sha256':'a'*64}),
                 (fresh,{'latest_bar':'2026-10-19','source_sha256':'b'*64})]
        with patch('v13_collection.parse_chart',side_effect=replies),patch('v13_history_integrity.validate_refresh'):
            r=current_history(store,'A','INE000000001','2026-10-19',old,official)
        self.assertTrue(r['official_aligned']);self.assertEqual(len(store.calls),2)
        self.assertEqual(r['info']['request_variant'],'extended')
    def test_all_stale_is_retained_as_partial_and_never_counted_current(self):
        old=self.bars('2026-10-16')
        class Store:
            def get(self,url):return b'{}',{}
        with patch('v13_collection.parse_chart',return_value=(old,{'latest_bar':'2026-10-16','source_sha256':'a'*64})):
            r=current_history(Store(),'A','INE000000001','2026-10-19',pd.DataFrame(),pd.DataFrame())
        self.assertFalse(r['fresh']);self.assertFalse(r['official_aligned'])
        self.assertEqual(len(r['attempts']),3)
    def test_rate_limit_stops_variant_retries(self):
        class Store:
            calls=0
            def get(self,url):
                self.calls+=1;raise RuntimeError('429 Too Many Requests')
        store=Store();r=current_history(store,'A','INE000000001','2026-10-19',pd.DataFrame(),pd.DataFrame())
        self.assertEqual(store.calls,1);self.assertIn('429',r['error'])

if __name__=='__main__':unittest.main()

class CrossSourceComparison(unittest.TestCase):
    def test_same_provider_mirrors_cannot_supply_two_confirmations(self):
        from v13_financial_crosscheck import compare
        wide=pd.DataFrame([dict(security_isin='INE000000001',reporting_mode='consolidated',period_kind='annual',
          period=pd.Timestamp('2026-03-31'),available=pd.Timestamp('2026-05-01T00:00:00Z'),revenue=100000000,
          source_sha256='a'*64,source_url='https://nsearchives.nseindia.com/a.xml')])
        obs=dict(symbol='A',isin='INE000000001',reporting_mode='consolidated',period_kind='annual',
          period_end='2026-03-31',metric='revenue',value_INR=100000000,rounding_half_width_INR=500000,
          provider_family='SP_GLOBAL',source_url='https://example.org/a',first_observed_utc='2026-10-20T00:00:00Z')
        rs,triples=compare(wide,[obs,dict(obs,source_url='https://mirror.org/a')],'2026-10-19')
        self.assertTrue(all(r['comparison_status']=='NUMERIC_AGREEMENT_WITHIN_DISPLAY_ROUNDING' for r in rs))
        self.assertEqual(triples,[])
        _,triples=compare(wide,[obs,dict(obs,provider_family='SCREENER')],'2026-10-19')
        self.assertEqual(len(triples),1);self.assertFalse(triples[0]['truth_certified'])
    def test_different_period_basis_and_profit_definition_cannot_match(self):
        from v13_financial_crosscheck import compare
        w=pd.DataFrame([dict(security_isin='INE000000001',reporting_mode='standalone',period_kind='annual',
          period=pd.Timestamp('2026-03-31'),available=pd.Timestamp('2026-05-01T00:00:00Z'),revenue=100.,pat=10.)])
        obs=dict(symbol='A',isin='INE000000001',reporting_mode='consolidated',period_kind='annual',
          period_end='2026-03-31',metric='revenue',value_INR=100.,rounding_half_width_INR=.5,
          provider_family='SCREENER',source_url='https://example.org',first_observed_utc='2026-10-20T00:00:00Z')
        rs,_=compare(w,[obs],'2026-10-19');self.assertEqual(rs[0]['comparison_status'],'PRIMARY_FACT_MISSING_OR_AMBIGUOUS')
        rs,_=compare(w,[dict(obs,metric='provider_net_profit',value_INR=10.,reporting_mode='standalone')],'2026-10-19')
        self.assertEqual(rs[0]['comparison_status'],'PROFIT_DEFINITION_REVIEW_REQUIRED')
