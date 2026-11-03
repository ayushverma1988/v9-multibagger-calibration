import unittest,tempfile,json,hashlib
from pathlib import Path
import pandas as pd
from v13_source_quality import business_order_status,bound_number,capacity_disclosure
from v13_collection import announcement_counts
from v13_catalyst_execution import reviewed
from v13_financial_evidence import apply_scenarios
from v13_action_audit import validate_action_notice
from v13_lifecycle import promotion_gate,PAIRED_COLUMNS

class SourceQuality(unittest.TestCase):
    def test_commissioner_or_brokerage_commission_is_not_capacity(self):
        self.assertFalse(capacity_disclosure('Order passed by Income Tax Commissioner'))
        self.assertFalse(capacity_disclosure('Brokerage commission increased by 20%'))
        self.assertTrue(capacity_disclosure('The company plans capacity expansion'))
        self.assertTrue(capacity_disclosure('The plant has been commissioned'))
    def test_legal_and_adverse_orders_are_not_sales_catalysts(self):
        for text in ['Action(s) initiated or orders passed. ED Order of Seizure under FEMA.',
                     'Received a High Court order in arbitration.', 'Tax adjudication penalty order']:
            self.assertEqual(business_order_status(text),'LEGAL_REGULATORY_NOT_BUSINESS_ORDER')
        for text in ['Received a work order of INR 50 crore','Bagging/Receiving of orders/contracts','Won a contract to build bridges']:
            self.assertEqual(business_order_status(text),'BUSINESS_ORDER_DISCLOSURE_NOT_REVENUE')
        self.assertNotEqual(business_order_status('Plans to bid for construction contracts'),'BUSINESS_ORDER_DISCLOSURE_NOT_REVENUE')
        self.assertNotEqual(business_order_status('Terminated the purchase order'),'BUSINESS_ORDER_DISCLOSURE_NOT_REVENUE')

    def test_cached_old_legal_order_flag_cannot_survive_feature_join(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);folder=root/'sources/x';folder.mkdir(parents=True)
            pd.DataFrame([{'event_id':'1','available_at_utc':'2026-10-13T04:00:00Z','symbol':'A','isin':'INE000000001',
              'catalog_sha256':'a','headline':'Action(s) initiated or orders passed','text':'ED order of seizure under FEMA',
              'order_disclosure':True,'capacity_disclosure':False}]).to_parquet(folder/'events.parquet')
            out=announcement_counts(pd.DataFrame([{'isin':'INE000000001','symbol':'A'}]),root,'2026-10-13')
            self.assertEqual(out.order_disclosure_count90.iloc[0],0)

    def test_repeated_catalogues_are_not_repeated_orders_or_backdated_features(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);folder=root/'sources/x';folder.mkdir(parents=True)
            row={'event_id':'1','available_at_utc':'2026-10-13T04:00:00Z','symbol':'A','isin':'INE000000001',
              'headline':'Received a work order','text':'Received a work order INR 50 crore','order_disclosure':True,'capacity_disclosure':False}
            pd.DataFrame([{**row,'catalog_sha256':'a'},{**row,'catalog_sha256':'b'}]).to_parquet(folder/'events.parquet')
            f=pd.DataFrame([{'isin':'INE000000001','symbol':'A','date':pd.Timestamp('2026-06-30')},
                            {'isin':'INE000000001','symbol':'A','date':pd.Timestamp('2026-10-13')}])
            out=announcement_counts(f,root,'2026-10-13')
            self.assertTrue(pd.isna(out.order_disclosure_count90.iloc[0]))
            self.assertEqual(out.order_disclosure_count90.iloc[1],1)

    def test_numeric_binding_requires_amount_scale_and_explicit_currency(self):
        self.assertTrue(bound_number(10000000,'Expected forward PAT INR 1 crore',currency=True))
        self.assertFalse(bound_number(10000000,'Expected forward PAT 1 crore',currency=True))
        self.assertFalse(bound_number(1,'Expected forward PAT INR 1 crore',currency=True))
        self.assertFalse(bound_number(True,'1 share'))
        self.assertTrue(bound_number(480000000,'Rs.48,00,00,000/-',currency=True))
        self.assertTrue(bound_number(67500000,'Rs.6.75 crores',currency=True))
        self.assertFalse(bound_number(2027,'Expected PAT INR 100 in 2027',currency=True))

    def test_exact_legal_excerpt_cannot_be_a_firm_business_order(self):
        text='Received order of seizure for Rs. 10 crore under FEMA.'
        e={'isin':'INE000000001','symbol':'A','attachment_url':'https://nsearchives.nseindia.com/a.pdf','source_sha256':'a'*64,
           'text_sha256':hashlib.sha256(text.encode()).hexdigest(),'issuer_identity_read':True,'first_retrieved_utc':'2026-10-14T02:00:00Z'}
        rec={k:e[k] for k in ('isin','symbol','attachment_url','source_sha256')}
        rec.update(firm_order=True,field_excerpts={'firm_order':text})
        with self.assertRaises(ValueError):reviewed(rec,e,text)

    def test_scenario_quote_cannot_bind_an_arbitrary_larger_forecast(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);(root/'evidence').mkdir();raw=b'Expected forward PAT attributable to shareholders INR 100 for 2026-04-01 to 2027-03-31. Forward diluted shares 10. Sector PE 20.'
            (root/'source.bin').write_bytes(raw);digest=hashlib.sha256(raw).hexdigest()
            rec={'isin':'INE000000001','available_at_utc':'2026-10-13T00:00:00Z','first_reviewed_utc':'2026-10-14T02:00:00Z',
              'input_source_sha256':{k:digest for k in ('forward_PAT','forward_diluted_shares','sector_PE')},
              'input_source_excerpts':{'forward_PAT':'Expected forward PAT attributable to shareholders INR 100 for 2026-04-01 to 2027-03-31.','forward_diluted_shares':'Forward diluted shares 10.','sector_PE':'Sector PE 20.'},
              'forward_PAT_INR':1000,'forward_diluted_shares':10,'sector_PE':20,'forward_PAT_basis':'OWNERS_ATTRIBUTABLE',
              'horizon_months':12,'forecast_period_start':'2026-04-01','forecast_period_end':'2027-03-31',
              'valuation_target_date':'2027-10-14','forward_share_basis':'FULLY_DILUTED_INCLUDING_REVIEWED_ACTIONS'}
            (root/'evidence/reviewed_earnings_scenarios.json').write_text(json.dumps([rec]))
            f=pd.DataFrame([{'isin':'INE000000001','date':pd.Timestamp('2026-10-14'),'close':100,'ttm_reported_PAT_INR':100,'latest_diluted_shares':10,'earnings_scenario_status':'UNKNOWN'}])
            self.assertEqual(apply_scenarios(f,root).earnings_scenario_status.iloc[0],'UNKNOWN')
            rec['forward_PAT_INR']=100;(root/'evidence/reviewed_earnings_scenarios.json').write_text(json.dumps([rec]))
            self.assertEqual(apply_scenarios(f,root).earnings_scenario_status.iloc[0],'SCENARIO_NOT_PROBABILITY')

    def test_action_date_and_factor_cannot_be_free_numbers(self):
        raw=b'Trading Symbol: A. Ex date 13 October 2026. Shares split from 1 to 2.'
        a={'isin':'INE000000001','symbol':'A','kind':'split','ex_date':'2026-10-13','share_multiplier':2,
           'source_share_ratio':{'kind':'share_counts','old':1,'new':2},'source_sha256':hashlib.sha256(raw).hexdigest(),
           'field_excerpts':{'ex_date':'Ex date 13 October 2026','share_multiplier':'Shares split from 1 to 2'}}
        self.assertTrue(validate_action_notice(a,raw))
        for wrong in [{'ex_date':'2026-10-14'},{'share_multiplier':3},{'source_share_ratio':{}},
                      {'source_share_ratio':{'kind':'share_counts','old':2,'new':1},'share_multiplier':.5}]:
            with self.assertRaises(ValueError):validate_action_notice({**a,**wrong},raw)

    def test_record_date_does_not_certify_exchange_ex_date(self):
        raw=b'Trading Symbol: A. Record date 13 October 2026. Dividend Rs. 2 per share.'
        a={'isin':'INE000000001','symbol':'A','kind':'dividend','ex_date':'2026-10-13','cash_per_old_share':2,
           'source_sha256':hashlib.sha256(raw).hexdigest(),'field_excerpts':{'ex_date':'Record date 13 October 2026','cash_per_old_share':'Dividend Rs. 2 per share'}}
        with self.assertRaises(ValueError):validate_action_notice(a,raw)

    def test_new_policy_cannot_promote_without_future_frozen_baseline(self):
        out=promotion_gate(pd.DataFrame(columns=PAIRED_COLUMNS),'2026-10-11T00:00:00Z',source_complete=True,require_baseline=True)
        self.assertFalse(out['promote'])
        self.assertIn('missing_verified_future_frozen_baseline_comparison',out['reasons'])

if __name__=='__main__':unittest.main()
