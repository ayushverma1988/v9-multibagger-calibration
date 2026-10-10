import json,unittest
import numpy as np
import pandas as pd
from v12_dated_training_inputs import exact_facts,normalize_financial_catalog,exchange_clock
from v12_prospective_registry import adjudicate_one,digest
from v12_bse_transfer import parse_bse_chart

def xml_fixture():
    return b'''<xbrl xmlns="http://www.xbrl.org/2003/instance" xmlns:f="urn:financial">
    <context id="OneD"><entity><identifier scheme="issuer">TEST</identifier></entity><period><startDate>2023-04-01</startDate><endDate>2023-06-30</endDate></period></context>
    <context id="FourD"><entity><identifier scheme="issuer">TEST</identifier></entity><period><startDate>2023-04-01</startDate><endDate>2023-06-30</endDate></period></context>
    <unit id="INR"><measure>iso4217:INR</measure></unit>
    <f:Symbol contextRef="OneD">TEST</f:Symbol><f:NatureOfReportStandaloneConsolidated contextRef="OneD">Standalone</f:NatureOfReportStandaloneConsolidated>
    <f:DateOfStartOfReportingPeriod contextRef="OneD">2023-04-01</f:DateOfStartOfReportingPeriod><f:DateOfEndOfReportingPeriod contextRef="OneD">2023-06-30</f:DateOfEndOfReportingPeriod>
    <f:DateOfStartOfReportingPeriod contextRef="FourD">2022-04-01</f:DateOfStartOfReportingPeriod><f:DateOfEndOfReportingPeriod contextRef="FourD">2023-06-30</f:DateOfEndOfReportingPeriod>
    <f:RevenueFromOperations contextRef="OneD" unitRef="INR">100</f:RevenueFromOperations><f:ProfitLossForPeriod contextRef="OneD" unitRef="INR">10</f:ProfitLossForPeriod>
    <f:RevenueFromOperations contextRef="FourD" unitRef="INR">999</f:RevenueFromOperations>
    </xbrl>'''

class DatedSourceTests(unittest.TestCase):
    def record(self):return {'symbol':'TEST','period_end':'2023-06-30','catalog_isin':'INE001F01019','reporting_mode':'standalone'}
    def test_primary_catalog_binding_when_legacy_xml_omits_isin(self):
        facts,identity=exact_facts(xml_fixture(),self.record(),'2026-10-09')
        self.assertEqual(identity['identity_source'],'ORIGINAL_EXCHANGE_CATALOG_ISIN')
        self.assertEqual(identity['document_isin'],'')
        self.assertEqual({r['metric']:r['value_INR'] for r in facts},{'revenue':100.,'pat':10.})
        self.assertEqual(identity['contexts_with_conflicting_economic_dates_rejected'],['FourD'])
    def test_unrelated_bad_ytd_does_not_destroy_valid_quarter(self):
        facts,_=exact_facts(xml_fixture(),self.record(),'2026-10-09')
        self.assertTrue(all(r['context']=='OneD' and r['period_kind']=='quarter' for r in facts))
    def test_period_before_four_year_boundary_is_rejected(self):
        raw=xml_fixture().replace(b'2023-04-01',b'2022-04-01').replace(b'2023-06-30',b'2022-06-30');r=self.record();r['period_end']='2022-06-30'
        with self.assertRaises(ValueError):exact_facts(raw,r,'2026-10-09')
    def test_wrong_basis_cannot_be_combined(self):
        r=self.record();r['reporting_mode']='consolidated'
        with self.assertRaises(ValueError):exact_facts(xml_fixture(),r,'2026-10-09')
    def test_infinite_primary_fact_is_missing(self):
        raw=xml_fixture().replace(b'>100<',b'>Infinity<');facts,_=exact_facts(raw,self.record(),'2026-10-09')
        self.assertFalse(any(f['metric']=='revenue' for f in facts))
    def test_wrong_symbol_cannot_use_matching_catalog_isin(self):
        r=self.record();r['symbol']='OTHER'
        with self.assertRaises(ValueError):exact_facts(xml_fixture(),r,'2026-10-09')
    def test_disclosed_period_never_becomes_publication_date(self):
        items=[{'schema':'legacy','venue_index':'equities','catalog_sha256':'a'*64,'record':{'toDate':'30-Jun-2023','symbol':'TEST','isin':'INE001F01019','xbrl':'https://nsearchives.nseindia.com/corporate/xbrl/test.xml'}}]
        with self.assertRaises(ValueError):normalize_financial_catalog(items,'2026-10-09')
    def test_later_revision_sets_available_clock(self):
        r={'symbol':'TEST','qe_Date':'30-Jun-2023','broadcast_Date':'01-Jul-2023 12:00:00','creation_Date':'01-Jul-2023 12:01:00','revised_Date':'10-Aug-2023 13:00:00','consolidated':'Standalone','xbrl':'https://nsearchives.nseindia.com/corporate/xbrl/test.xml','seq_Id':'1'}
        x=normalize_financial_catalog([{'schema':'integrated','venue_index':'equities','catalog_sha256':'a'*64,'record':r}],'2026-10-09')
        self.assertEqual(x.available_at_utc.iloc[0],'2023-08-10T07:30:00+00:00')
    def test_ist_clock_not_utc_clock(self):self.assertEqual(exchange_clock('01-Jul-2023 12:00:00').isoformat(),'2023-07-01T06:30:00+00:00')

class ProspectiveTests(unittest.TestCase):
    def record(self):return {'isin':'INE001F01019','symbol':'TEST','decision_at_utc':'2026-10-10T08:30:00Z'}
    def bars(self):
        dates=pd.bdate_range('2026-10-12','2027-04-14');n=len(dates)
        return pd.DataFrame({'date':dates,'open':100.,'high':210.,'low':95.,'close':205.,'adj_close':205.,'isin':'INE001F01019','venue':'NSE','currency':'INR','entry_proof_sha256':'b'*64,'source_sha256':'a'*64,'first_retrieved_utc':'2027-04-14T11:00:00Z','first_trade_after_decision_verified':[True]+[False]*(n-1)})
    def test_no_future_outcome_when_no_entry(self):
        self.assertIsNone(adjudicate_one(self.record(),pd.DataFrame(),6,'2026-10-10T08:31:00Z')['outcome'])
    def test_early_double_is_pending_until_complete_horizon(self):
        b=self.bars().head(3);b['first_retrieved_utc']='2026-10-14T11:00:00Z'
        self.assertEqual(adjudicate_one(self.record(),b,6,'2026-10-14T12:00:00Z')['status'],'PENDING_HORIZON')
    def test_unverified_first_trade_stays_unknown(self):
        b=self.bars();b['first_trade_after_decision_verified']=False
        self.assertEqual(adjudicate_one(self.record(),b,6,'2027-04-14T12:00:00Z')['status'],'UNKNOWN_FIRST_ACTUAL_TRADE_NOT_VERIFIED')
    def test_price_infinity_does_not_become_success(self):
        b=self.bars();b.loc[50,'high']=np.inf
        self.assertEqual(adjudicate_one(self.record(),b,6,'2027-04-14T12:00:00Z')['status'],'UNKNOWN_INVALID_PRICES')
    def test_complete_proxy_uses_net_closing_target(self):
        r=adjudicate_one(self.record(),self.bars(),6,'2027-04-14T12:00:00Z')
        self.assertEqual(r['outcome'],1);self.assertEqual(r['deadline'],'2027-04-12')
    def test_future_retrieval_is_rejected(self):
        with self.assertRaises(ValueError):adjudicate_one(self.record(),self.bars(),6,'2026-10-10T12:00:00Z')
    def test_loss_after_successful_target_exit_is_not_pre_target_risk(self):
        b=self.bars();b.loc[10,'low']=50.
        self.assertEqual(adjudicate_one(self.record(),b,6,'2027-04-14T12:00:00Z')['loss30_before_hit2'],0)
    def test_loss_before_first_double_counts_in_risk(self):
        b=self.bars();b.loc[0,['close','adj_close']]=150.;b.loc[0,'low']=60.
        self.assertEqual(adjudicate_one(self.record(),b,6,'2027-04-14T12:00:00Z')['loss30_before_hit2'],1)
    def test_registration_hash_changes_if_selection_is_changed(self):
        self.assertNotEqual(digest({'selected':False}),digest({'selected':True}))

if __name__=='__main__':unittest.main()
