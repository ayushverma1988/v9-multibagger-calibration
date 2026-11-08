import unittest
import numpy as np
import pandas as pd
import v13_signal_timing_study as s

def bars(n=300):
    dates=pd.bdate_range('2024-01-02',periods=n)
    close=np.full(n,100.)
    return pd.DataFrame({'date':dates,'open':100.,'high':101.,'low':99.,'close':close,'adj_close':close,'volume':100000.,'isin':'INE000000001','symbol':'TEST'})

class TimingTests(unittest.TestCase):
    def test_rsi_boundaries_and_nesting(self):
        d=pd.DataFrame({'feature_eligible':True,'close':100.,'distance_ma50':.1,'log_ret63':.1,'rsi14':[50.,50.01,70.,70.01]})
        self.assertEqual(s.eligible(d,50).tolist(),[False,True,True,True])
        self.assertEqual(s.eligible(d,70).tolist(),[False,False,False,True])

    def test_entry_uses_next_open_with_costs(self):
        d=bars();d.loc[1,'open']=80.
        out=s.outcome(d,0,1,d.date.max())
        expected=100/80*(1-s.COST)/(1+s.COST)-1
        self.assertEqual(out['status'],'MATURE_VALID_PROVIDER_PROXY')
        self.assertEqual(out['entry_date'],d.loc[1,'date'])
        self.assertAlmostEqual(out['endpoint_return'],expected)

    def test_immature_window_has_no_invented_endpoint(self):
        d=bars(30);out=s.outcome(d,15,6,d.date.max())
        self.assertEqual(out['status'],'CENSORED_NOT_MATURE')
        self.assertNotIn('endpoint_return',out)

    def test_drawdown_and_low_excursion_are_different(self):
        d=bars();d.loc[5:10,'adj_close']=120.;d.loc[11:15,'adj_close']=90.
        d['close']=d.adj_close;d['high']=d.adj_close+1;d['low']=d.adj_close-1
        out=s.outcome(d,0,1,d.date.max())
        self.assertAlmostEqual(out['max_close_drawdown'],-.25)
        self.assertLess(out['worst_adjusted_low_return'],-.1)

    def test_future_and_after_close_filings_are_excluded(self):
        date=pd.Timestamp('2025-02-03');cut=s.signal_cutoff(date)
        w=pd.DataFrame({'available':[cut-pd.Timedelta(minutes=1),cut+pd.Timedelta(minutes=1)],
             'period':[pd.Timestamp('2024-12-31')]*2,'reporting_mode':['standalone']*2,'period_kind':['quarter']*2,'value':[1.,999.]})
        q=s.known_financials(w,date)
        self.assertEqual(q.value.tolist(),[1.])
        self.assertEqual(len(s.known_financials(w,date+pd.Timedelta(days=1))),1)
        self.assertEqual(s.known_financials(w,date+pd.Timedelta(days=1)).value.tolist(),[999.])

    def test_tied_filing_versions_are_not_averaged(self):
        cut=s.signal_cutoff('2025-02-03')
        w=pd.DataFrame({'available':[cut,cut],'period':[pd.Timestamp('2024-12-31')]*2,
            'reporting_mode':['standalone']*2,'period_kind':['quarter']*2,'value':[1.,999.]})
        self.assertEqual(len(s.known_financials(w,'2025-02-03')),2)

    def test_event_cooldown_keeps_first_and_not_future_winners(self):
        mask=pd.Series([False]*150);mask.iloc[[2,5,40,66,67,130]]=True
        self.assertEqual(s.event_indices(mask),[2,66,130])

    def test_failed_breakout_remains_recorded_even_if_later_recovers(self):
        d=bars();d['prior_resistance63']=100.;d.loc[3,'adj_close']=95.;d.loc[10:,'adj_close']=110.
        r=s.failure20(d,0)
        self.assertTrue(r['failed_breakout20'])
        self.assertEqual(r['breakout_failure_date'],d.loc[3,'date'])

    def test_catalyst_after_signal_not_used(self):
        cut=s.signal_cutoff('2025-02-03')
        e=pd.DataFrame({'available':[cut+pd.Timedelta(seconds=1)],'study_order':[True],'study_capacity':[False]})
        r=s.catalyst_snapshot(e,'2025-02-03')
        self.assertIsNone(r['dated_order_count90'])
        self.assertEqual(r['catalyst_evidence'],'UNKNOWN_CATALOGUE_COVERAGE')

    def test_causal_prices_do_not_change_when_future_appended(self):
        try:
            from v13_features import daily_features
            from v13_redevelopment import daily
        except ModuleNotFoundError:
            if __import__('os').environ.get('CI'):raise
            self.skipTest('Repository feature dependencies exercised on study runner')
        d=bars(450);rng=np.random.default_rng(221)
        d['close']=100*np.exp(np.cumsum(rng.normal(.0005,.01,len(d))));d['adj_close']=d.close
        d['open']=d.close*.997;d['high']=d.close*1.01;d['low']=d.close*.99
        d['volume']=rng.integers(100000,500000,len(d))
        def compute(v):
            f=daily_features(v,str(d.date.max().date()));r=daily(f)
            for c in r:
                if c.startswith('rd_'):f[c]=r[c]
            return s.stage_frame(f)
        a=compute(d.iloc[:350]);b=compute(d)
        cols=['rsi14','rd_first_breakout','rd_breakout_age','prior_resistance63','prior_base_range20','prior_rally63_ending20','entry_stage']
        pd.testing.assert_frame_equal(a[cols],b.loc[:349,cols],check_exact=True)

if __name__=='__main__': unittest.main(verbosity=2)
