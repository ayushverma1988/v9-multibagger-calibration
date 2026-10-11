import unittest
from xml.etree import ElementTree as ET
import numpy as np
import pandas as pd
from v12_5_expanded_facts import unit_types,parse_extra
from v12_5_features import features,ratio,source_wide

class UnitsTest(unittest.TestCase):
    def test_instant_assets_with_null_start_are_retained(self):
        raw=b'''<xbrl xmlns="http://www.xbrl.org/2003/instance" xmlns:f="urn:test">
        <context id="q"><period><startDate>2026-04-01</startDate><endDate>2026-06-30</endDate></period></context>
        <context id="i"><period><instant>2026-06-30</instant></period></context>
        <unit id="INR"><measure>iso4217:INR</measure></unit>
        <f:Symbol>TEST</f:Symbol><f:ISIN>INE123A01012</f:ISIN><f:NatureOfReportStandaloneConsolidated>Standalone</f:NatureOfReportStandaloneConsolidated>
        <f:RevenueFromOperations contextRef="q" unitRef="INR">100</f:RevenueFromOperations>
        <f:Assets contextRef="i" unitRef="INR">500</f:Assets></xbrl>'''
        record={'symbol':'TEST','catalog_isin':'INE123A01012','reporting_mode':'standalone','period_end':'2026-06-30'}
        f=parse_extra(raw,record,'2026-10-09')
        self.assertEqual(next(x['value'] for x in f if x['metric']=='assets'),500)
    def test_per_share_division_direction(self):
        good=ET.fromstring('<x><unit id="u"><divide><unitNumerator><measure>iso:INR</measure></unitNumerator><unitDenominator><measure>x:shares</measure></unitDenominator></divide></unit></x>')
        self.assertEqual(unit_types(good),{'u':'INR/SHARES'})
        bad=ET.fromstring('<x><unit id="u"><divide><unitNumerator><measure>x:shares</measure></unitNumerator><unitDenominator><measure>iso:INR</measure></unitDenominator></divide></unit></x>')
        self.assertNotIn('u',unit_types(bad))
    def test_nonpositive_denominators_unknown(self):
        self.assertTrue(np.isnan(ratio(1,0)));self.assertTrue(np.isnan(ratio(1,-1)))

def rows():
    r=[]
    for period,rev,pat,eps in [('2026-06-30',150,20,2),('2025-06-30',100,10,1),('2026-03-31',120,15,1.5),('2025-03-31',100,10,1)]:
        r.append(dict(period_kind='quarter',period=pd.Timestamp(period),period_end=period,
                      available=pd.Timestamp('2026-08-01',tz='UTC'),available_at_utc='2026-08-01T00:00:00Z',
                      reporting_mode='consolidated',source_sha256='same',revenue=rev,pat=pat,
                      eps_diluted_continuing=eps,paidup_equity_capital=100,face_value=10))
    return pd.DataFrame(r)

class FeatureTest(unittest.TestCase):
    def test_sales_acceleration_and_same_document_eps(self):
        x,_=features(rows(),pd.Timestamp('2026-10-09'))
        self.assertAlmostEqual(x['sales_growth_acceleration'],.3)
        self.assertAlmostEqual(x['reported_diluted_EPS_yoy_same_document'],1)
        self.assertTrue(np.isnan(x['verified_recurring_diluted_EPS_growth']))
    def test_eps_cannot_splice_versions(self):
        d=rows();d.loc[d.period_end.eq('2025-06-30'),'source_sha256']='older'
        x,_=features(d,pd.Timestamp('2026-10-09'))
        self.assertTrue(np.isnan(x['reported_diluted_EPS_yoy_same_document']))
    def test_face_value_change_is_not_dilution(self):
        d=rows();d.loc[d.period_end.eq('2025-06-30'),'face_value']=5
        x,_=features(d,pd.Timestamp('2026-10-09'))
        self.assertTrue(np.isnan(x['reported_share_count_yoy']))
    def test_stale_quarter_does_not_populate(self):
        x,_=features(rows(),pd.Timestamp('2027-06-30'))
        self.assertTrue(all(np.isnan(v) for v in x.values()))
    def test_same_clock_conflicting_filing_rejected(self):
        d=rows();r=d.iloc[[0]].copy();r['source_sha256']='conflict'
        x,_=features(pd.concat([d,r]),pd.Timestamp('2026-10-09'))
        self.assertTrue(np.isnan(x['sales_growth_acceleration']))
    def test_basis_not_mixed(self):
        d=rows();d.loc[d.period_end.eq('2025-06-30'),'reporting_mode']='standalone'
        x,_=features(d,pd.Timestamp('2026-10-09'))
        self.assertTrue(np.isnan(x['sales_growth_acceleration']))

class CatalystClockTest(unittest.TestCase):
    def test_mutually_exclusive_joint_loss_consistency(self):
        from v12_5_expanded_model import joint_consistent
        self.assertFalse(joint_consistent(.4,.3,.2))
        self.assertFalse(joint_consistent(.3,.4,.8))
        self.assertTrue(joint_consistent(.2,.3,.7))
    def test_after_close_and_wrong_identity_excluded(self):
        from v12_5_catalyst_join import observed
        r=dict(isin='INE123A01012',issuer_identity_read=True,available_at_utc='2026-10-09T10:01:00Z',commissioned=True,source_sha256='x')
        f,_=observed([r],'INE123A01012','2026-10-09')
        self.assertTrue(np.isnan(f['commissioned_document_count90']))
        f,_=observed([r],'INE123A01013','2026-10-10')
        self.assertTrue(np.isnan(f['commissioned_document_count90']))
    def test_contract_scale_not_profit_or_revenue(self):
        from v12_5_catalyst_join import observed
        r=dict(isin='INE123A01012',issuer_identity_read=True,available_at_utc='2026-10-09T09:00:00Z',firm_order=True,order_due='2026-12-31',order_id='one',order_value_INR_as_disclosed=20,source_sha256='x')
        f,_=observed([r,r],'INE123A01012','2026-10-09',100)
        self.assertEqual(f['disclosed_order_value_due12_to_sales'],.2)
        self.assertTrue(np.isnan(f['verified_incremental_profit12_to_PAT']))
        r['order_due']='2028-12-31'
        f,_=observed([r],'INE123A01012','2026-10-09',100)
        self.assertTrue(np.isnan(f['disclosed_order_value_due12_to_sales']))
    def test_absence_is_unknown_not_zero(self):
        from v12_5_catalyst_join import observed
        f,_=observed([],'INE123A01012','2026-10-09')
        self.assertTrue(all(np.isnan(x) for x in f.values()))

if __name__=='__main__':unittest.main()
