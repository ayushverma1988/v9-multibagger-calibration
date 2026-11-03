import unittest
import numpy as np
import pandas as pd
from v12_3_point_in_time import financial_row, coherent_period, FIELDS
from v12_3_selective_evaluation import choose, extended_outcome, block_interval
from v12_3_catalyst_evidence import enrich
from v12_3_official_monitor import reconcile_observations


def fact_rows(period,available,values,kind='quarter',source='a',basis='consolidated'):
    return [dict(metric=k,value_INR=v,period=pd.Timestamp(period),available=pd.Timestamp(available,tz='UTC'),
                 security_isin='INE000000001',reporting_mode=basis,period_kind=kind,source_sha256=source,
                 source_url='https://example.test/'+source,catalog_sha256='c',available_at_utc=available+'Z',
                 period_end=period,period_start=period) for k,v in values.items()]


class PointInTimeTests(unittest.TestCase):
    def test_prior_year_comparative_survives_450_day_boundary(self):
        f=pd.DataFrame(fact_rows('2024-03-31','2024-05-10',{'revenue':100,'pat':10},source='old')+
                       fact_rows('2025-03-31','2025-05-10',{'revenue':120,'pat':15}))
        r,_=financial_row(f,'2025-07-31','INE000000001')
        self.assertAlmostEqual(r['quarter_sales_yoy'],.2)
        self.assertAlmostEqual(r['quarter_pat_yoy'],.5)

    def test_later_revision_does_not_leak(self):
        f=pd.DataFrame(fact_rows('2024-03-31','2024-05-10',{'revenue':100,'pat':10},source='old')+
                       fact_rows('2025-03-31','2025-05-10',{'revenue':120,'pat':15})+
                       fact_rows('2025-03-31','2025-08-01',{'revenue':900,'pat':90},source='future'))
        r,links=financial_row(f,'2025-07-31','INE000000001')
        self.assertAlmostEqual(r['quarter_sales_yoy'],.2)
        self.assertNotIn('future',[x['source_sha256'] for x in links])

    def test_incomplete_revision_does_not_splice_old_metrics(self):
        f=pd.DataFrame(fact_rows('2025-03-31','2025-05-10',{'revenue':120,'pat':15})+
                       fact_rows('2025-03-31','2025-06-01',{'revenue':125},source='rev'))
        r,_=financial_row(f,'2025-07-31','INE000000001')
        self.assertTrue(np.isnan(r['quarter_pat_margin']))

    def test_stale_and_wrong_identity_are_unknown(self):
        f=pd.DataFrame(fact_rows('2025-03-31','2025-05-10',{'revenue':120,'pat':15}))
        for date,isin in [('2026-02-01','INE000000001'),('2025-07-31','INE999999999')]:
            r,_=financial_row(f,date,isin)
            self.assertEqual(r['financial_status'],'UNKNOWN')

    def test_same_clock_conflicting_documents_abstain(self):
        f=pd.DataFrame(fact_rows('2025-03-31','2025-05-10',{'revenue':120},source='a')+
                       fact_rows('2025-03-31','2025-05-10',{'revenue':125},source='b'))
        self.assertTrue(coherent_period(f,'quarter').empty)


class SelectionTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame([dict(symbol='A',technical_pass=True,close=100,relative_strength63=.9,rsi14=75,
             quarter_sales_yoy=.2,quarter_pat_yoy=.3,quarter_pat_margin=.12,annual_CFO_PAT=1.,dated_debt_equity=.2,
             p_hit2_6=.2,p_hit2_12=.4,p_loss30_12=.3)])

    def test_strict_allows_zero_and_missing_never_passes(self):
        f=self.frame();self.assertEqual(len(choose(f,'enriched_selective_max5',6)),1)
        for col in FIELDS:
            g=f.copy();g[col]=np.nan
            self.assertEqual(len(choose(g,'enriched_selective_max5',6)),0)
        f['p_loss30_12']=.8
        self.assertEqual(len(choose(f,'enriched_selective_max5',6)),0)

    def test_no_loss_head_means_no_selective_pick(self):
        self.assertTrue(choose(self.frame().drop(columns='p_loss30_12'),'enriched_selective_max5',12).empty)

    def test_not_enough_independent_time_blocks(self):
        f=pd.DataFrame({'date':pd.date_range('2025-01-01',periods=4),'hits':[1]*4,'selected':[10]*4})
        self.assertIsNone(block_interval(f,12))

    def history(self):
        dates=pd.bdate_range('2025-01-01','2025-08-01')
        d=pd.DataFrame({'date':dates,'open':100.,'high':102.,'low':98.,'close':100.,'adj_close':100.,'volume':1000.})
        return d

    def test_early_hit_does_not_mature_future_horizon(self):
        d=self.history();d.loc[1:,'adj_close']=210.
        r=extended_outcome(d,'2024-12-31',6,'2025-02-01')
        self.assertEqual(r['status'],'PENDING_HORIZON')

    def test_zero_volume_not_entry(self):
        d=self.history();d.loc[0,'volume']=0
        r=extended_outcome(d,'2024-12-31',6,'2025-08-01')
        self.assertEqual(r['entry_date'],'2025-01-02')

    def test_finite_ohlc_and_price_jump_required(self):
        d=self.history();d.loc[1,'adj_close']=1000
        self.assertEqual(extended_outcome(d,'2024-12-31',6,'2025-08-01')['status'],'UNKNOWN_CORPORATE_ACTION_OR_JUMP')

    def test_catalyst_extraction_never_invents_profit(self):
        info={'quantified_evidence':[{'category':'order','unit_as_disclosed':'crore','quantity':50}], 'issuer_identity_read':True}
        r=enrich('Delivery within 12 months, subject to approvals. Funded from internal accruals.',info)
        self.assertEqual(r['quantified_evidence'][0]['order_value_INR'],5e8)
        self.assertFalse(r['incremental_profit_verified']);self.assertFalse(r['complete_causal_chain_verified'])

    def test_primary_observation_reconciliation_rejects_zero_volume_and_conflict(self):
        d=pd.DataFrame([dict(date='2026-10-09',isin='INE000000001',open=100.,high=102.,low=98.,close=101.,volume=1000.)])
        o=d.copy();o['source_status']='VALID_POSITIVE_VOLUME_SESSION'
        self.assertEqual(reconcile_observations(d,o).primary_observation_check.iloc[0],'MATCHED_RAW_PRICE_AND_VOLUME')
        d['volume']=0
        self.assertEqual(reconcile_observations(d,o).primary_observation_check.iloc[0],'UNKNOWN_OR_CONFLICT')
        d['volume']=1000;d['close']=105
        self.assertEqual(reconcile_observations(d,o).primary_observation_check.iloc[0],'UNKNOWN_OR_CONFLICT')


if __name__=='__main__':unittest.main()
