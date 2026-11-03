import json,unittest
import numpy as np,pandas as pd
import v13_discovery_diagnostics as dd
import v13_historical_discovery as hd
import v13_sector_quality as sq
from v13_project_evidence import scenario_bridge
from v13_discovery_current import watchlists
class Discovery(unittest.TestCase):
    def bars(self):
        p=100+np.sin(np.arange(260)/3)*2
        return pd.DataFrame(dict(date=pd.bdate_range('2025-01-01',periods=260),isin='INE000000001',close=p,adj_close=p,high=p+1,low=p-1,volume=10000.))
    def event(self,**changes):
        r=dict(symbol='TEST',isin='INE000000001',source='NSE',event_id='event1',published_ts=pd.Timestamp('2025-01-31T09:00Z'),exchange_received_ts=None,document_url='https://nsearchives.nseindia.com/corporate/test.pdf',headline='Updates',details='The company received a purchase order.')
        r.update(changes);r['source_json']=json.dumps(dict(symbol=r['symbol'],sm_isin=r['isin'],an_dt='31-Jan-2025 14:30:00',attchmntFile=r['document_url'],desc=r['headline'],attchmntText=r['details']));return r
    def test_future_prices_cannot_rewrite_breakout(self):
        b=self.bars();a=dd.daily(b.iloc[:230]);b.loc[230:,['close','adj_close','high','low']]*=4
        pd.testing.assert_frame_equal(a,dd.daily(b).iloc[:230])
    def test_cross_without_volume_has_reason(self):
        b=self.bars();b.loc[259,['close','adj_close','high']]=[110,110,111];d=dd.daily(b).iloc[-1]
        self.assertIn('VOLUME_NOT_CONFIRMED',d.dd_breakout_reason);self.assertEqual(d.dd_cross_age,0);self.assertEqual(d.dd_first_breakout,0)
    def test_no_first_observation_rsi_cross(self):self.assertTrue(dd.daily(self.bars().iloc[:15]).dd_rsi_cross_age.isna().all())
    def test_bank_not_failed_by_industrial_debt(self):
        q=sq.assess(dict(industry='Banks',dated_debt_equity=10,annual_CFO_PAT=-1));self.assertEqual(q['status'],'UNKNOWN_SECTOR_FINANCIAL_INPUTS');self.assertEqual(q['failed_checks'],[])
    def test_bank_capital_failure(self):
        q=sq.assess(dict(industry='Banks',four_profitable_quarters=1,quarter_pat_yoy=.2,bank_roa=.01,bank_capital_buffer_pp=-.1,bank_net_npa_change_pp=-.2));self.assertEqual(q['failed_checks'],['bank_capital_buffer_pp'])
    def test_unknown_sector_not_pass(self):self.assertFalse(sq.assess({})['quality_pass'])
    def test_future_membership_excluded(self):
        f=pd.DataFrame([dict(date='2025-01-31',isin='A',symbol='A')]);m=pd.DataFrame([dict(isin='A',symbol='A',available_at_utc='2026-01-01T00:00Z',source_sha256='a'*64,industry='Banks',size_category='Large Cap')]);self.assertIsNone(sq.dated_members(f,m).iloc[0].industry)
    def test_unknown_size_not_small_mid(self):self.assertEqual(sq.annotate(pd.DataFrame([dict(industry='Banks')])).iloc[0].size_status,'UNKNOWN_DATED_SIZE')
    def test_source_identity_bound(self):
        r=self.event();r['isin']='WRONG';d,a=hd.normalize(pd.DataFrame([r]));self.assertTrue(d.empty);self.assertEqual(a['rejected']['SOURCE_IDENTITY_MISMATCH'],1)
    def test_after_close_received_excluded(self):
        e,_=hd.normalize(pd.DataFrame([self.event(exchange_received_ts='2025-01-31T11:00Z')]));d=hd.attach(pd.DataFrame([dict(date='2025-01-31',isin='INE000000001')]),e,'2025-02-28T23:59Z');self.assertTrue(pd.isna(d.iloc[0].hd_order_90d))
    def test_archive_end_not_zero(self):
        e,_=hd.normalize(pd.DataFrame([self.event()]));f=pd.DataFrame([dict(date=x,isin='INE000000001') for x in ['2025-01-31','2025-03-31']]);d=hd.attach(f,e,'2025-02-28T23:59Z');self.assertEqual(d.iloc[0].hd_order_90d,1);self.assertTrue(pd.isna(d.iloc[1].hd_order_90d))
    def test_legal_media_not_orders(self):
        self.assertEqual(hd.signals('News Verification','Order received according to media; response awaited')['order'],0);self.assertEqual(hd.signals('Updates','Court order received imposing penalty')['order'],0)
    def test_delay_not_commissioned(self):
        s=hd.signals('Updates','Commissioning of the plant has been delayed');self.assertEqual(s['adverse_project'],1);self.assertEqual(s['commissioning_language'],0)
    def test_duplicate_source_not_double_counted(self):
        r=self.event();e,_=hd.normalize(pd.DataFrame([r,r]));self.assertEqual(len(e),1)
    def test_demat_cancellation_not_project_failure(self):
        self.assertEqual(hd.signals('Certificate under Regulation 74','Securities for dematerialisation have been cancelled; certificate which you will find in order.')['adverse_project'],0)
    def test_ownership_approval_not_operating(self):self.assertEqual(hd.signals('Updates','RBI approval granted for acquiring aggregate holding of 9.5% of voting rights')['approval_language'],0)
    def test_operating_approval_signal(self):self.assertEqual(hd.signals('Updates','The company received environmental approval for its plant expansion')['approval_language'],1)
    def test_missing_margin_not_profit(self):self.assertEqual(scenario_bridge(100,200,None,10,10,5)['status'],'UNKNOWN_INPUTS')
    def test_dilution_reduces_upside(self):
        a=scenario_bridge(100,200,.1,10,10,5);b=scenario_bridge(100,200,.1,20,10,5);self.assertAlmostEqual(a['net_multiple']/2,b['net_multiple']);self.assertIsNone(a['probability'])
    def watch_fixture(self,**changes):
        row=dict(isin='INE000000001',symbol='TEST',sector_status='UNKNOWN_SECTOR_FINANCIAL_INPUTS',sector_sector_kind='BANK',close=100.,feature_eligible=True,p6=.1,p12=.2,loss12=.3,joint12=.15,size_status='SMALL_MID_VERIFIED',rsi14=50.,distance_ma50=.1,log_ret63=.1,redevelopment_price_discipline=.8,research_within_200dma_band=False)
        row.update(changes)
        doc=dict(isin=row['isin'],source_sha256='a'*64,issuer_identity_read=True,available_at_utc='2025-01-01T10:00Z',first_retrieved_utc='2025-01-01T11:00Z',headline='Updates',text='The company received a purchase order.')
        return pd.DataFrame([row]),[doc]
    def test_inclusive_rsi50_and_optional_dma(self):
        f,e=self.watch_fixture(rsi14=49.999);self.assertFalse(watchlists(f,e,'2025-01-02T12:00Z').iloc[0].rsi70_research_eligible)
        f.loc[0,'rsi14']=50.;self.assertTrue(watchlists(f,e,'2025-01-02T12:00Z').iloc[0].rsi70_research_eligible)
    def test_early_watch_does_not_require_rsi70(self):
        f,e=self.watch_fixture(rsi14=60.);self.assertTrue(watchlists(f,e,'2025-01-02T12:00Z').iloc[0].early_watch_eligible)
    def test_unknown_sector_preserves_known_financial_failure(self):
        f,e=self.watch_fixture(sector_sector_kind='UNKNOWN',research_financial_tier='OBSERVED_FINANCIAL_FAILURE');self.assertFalse(watchlists(f,e,'2025-01-02T12:00Z').iloc[0].early_watch_eligible)
    def test_future_review_cannot_trigger_early_watch(self):
        f,e=self.watch_fixture();e[0]['first_retrieved_utc']='2025-01-03T10:00Z';self.assertFalse(watchlists(f,e,'2025-01-02T12:00Z').iloc[0].early_watch_eligible)
    def test_watchlist_preserves_frozen_probabilities(self):
        f,e=self.watch_fixture();d=watchlists(f,e,'2025-01-02T12:00Z');pd.testing.assert_frame_equal(f[['p6','p12','joint12','loss12']],d[['p6','p12','joint12','loss12']])
    def test_price_band_removed_both_ends(self):
        for price in [.5,19.99,2000.01,100000.]:
            with self.subTest(price=price):
                f,e=self.watch_fixture(close=price,rsi14=75.)
                d=watchlists(f,e,'2025-01-02T12:00Z').iloc[0]
                self.assertTrue(d.rsi70_research_eligible)
                self.assertTrue(d.early_watch_eligible)
    def test_invalid_prices_still_excluded(self):
        for price in [0.,-1.,np.nan,np.inf,-np.inf]:
            with self.subTest(price=price):
                f,e=self.watch_fixture(close=price,rsi14=75.)
                d=watchlists(f,e,'2025-01-02T12:00Z').iloc[0]
                self.assertFalse(d.rsi70_research_eligible)
                self.assertFalse(d.early_watch_eligible)
    def test_model_gate_has_no_price_band(self):
        import v13_model as model
        from v13_financial_evidence import REQUIRED
        rows=[]
        for price in [1.,5000.,0.,np.inf]:
            r={k:np.nan for k in REQUIRED};r.update(close=price,distance_ma50=.1,log_ret63=.1);rows.append(r)
        d=pd.DataFrame(rows)
        for name in model.old.BASE_FIELDS:
            if name not in d:d[name]=np.nan
        verified,discovery,bad=model.gates(d)
        self.assertEqual((verified|discovery).tolist(),[True,True,False,False])
if __name__=='__main__':unittest.main()
