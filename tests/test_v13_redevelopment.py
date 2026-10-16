import unittest
import numpy as np
import pandas as pd
from scipy.special import expit,logit
import v13_redevelopment as rd

class Redevelopment(unittest.TestCase):
    def bars(self):
        p=100+np.sin(np.arange(260)/3)*2
        return pd.DataFrame({'date':pd.bdate_range('2025-01-01',periods=260),
          'isin':'INE000000001','close':p,'adj_close':p,'high':p+1,'low':p-1,'volume':10000.})

    def test_future_price_and_volume_cannot_change_past_features(self):
        bars=self.bars();before=rd.daily(bars.iloc[:230])
        bars.loc[230:,'close']*=4;bars.loc[230:,'adj_close']*=4;bars.loc[230:,'volume']*=100
        after=rd.daily(bars)
        pd.testing.assert_frame_equal(before[rd.MARKET_FEATURES],after.loc[:229,rd.MARKET_FEATURES])

    def test_resistance_and_volume_exclude_breakout_session(self):
        bars=self.bars();bars.loc[259,['close','adj_close']]=110;bars.loc[259,'high']=111;bars.loc[259,'volume']=30000
        d=rd.daily(bars)
        self.assertAlmostEqual(d.iloc[-1].rd_volume_ratio,3)
        self.assertAlmostEqual(d.iloc[-1].rd_breakout_gap,110/bars.high.iloc[196:259].max()-1)

    def test_missing_history_never_means_observed_base_or_breakout(self):
        d=rd.daily(self.bars().iloc[:20])
        self.assertTrue(d.rd_first_breakout.isna().all())
        self.assertTrue(d.rd_breakout_age.isna().all())

    def test_first_breakout_requires_twenty_observed_prior_resistances(self):
        bars=self.bars().iloc[:83].copy()
        bars.loc[82,['close','adj_close','high']]=[110,110,111]
        bars.loc[82,'volume']=30000
        self.assertFalse(rd.daily(bars).rd_first_breakout.fillna(0).any())

    def test_calibration_monotonic_finite_for_rare_and_zero_events(self):
        raw=np.linspace(.01,.9,80)
        for y in [np.zeros(80),np.ones(80),np.arange(80)%3==0]:
            c=rd.shrink_calibration(raw,y,np.arange(400)%5==0)
            p=expit(c['slope']*logit(raw)+c['intercept'])
            self.assertTrue(np.isfinite(p).all());self.assertTrue((np.diff(p)>0).all())
            self.assertGreater(c['slope'],0)

    def test_split_adjustment_keeps_price_features_consistent(self):
        a=self.bars();b=a.copy();b[['close','high','low']]*=2
        pd.testing.assert_frame_equal(rd.daily(a)[rd.MARKET_FEATURES],rd.daily(b)[rd.MARKET_FEATURES])

if __name__=='__main__':unittest.main()
