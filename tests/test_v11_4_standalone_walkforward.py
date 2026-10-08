import unittest
import pandas as pd
import numpy as np
from v11_4_standalone_train_walkforward import (
    fold_close,keep_train,eligible_asof,top10_similarity,
    safe_featureize,CATALYST_PREFIXES,PRICE_MIN,PRICE_MAX,
    MODEL_FEATURES,train_once,calibrate_logit,calibration_mode,MIN_CALIBRATION_POSITIVES
)

class StandaloneV114Tests(unittest.TestCase):
    def test_india_cutoff(self):
        self.assertEqual(fold_close("2020-06-30").isoformat(),"2020-06-30T10:00:00+00:00")

    def test_training_drops_future_or_same_fold_outcomes(self):
        d=pd.DataFrame({
            "date":pd.to_datetime(["2019-12-31","2019-12-31","2020-06-30","2019-12-31"]),
            "y6_mature_date":["2020-06-29","2020-07-01","2020-01-01",None],
            "y6":[1.,1.,1.,1.],
            "integrity_y6_clean":[True]*4,
            "integrity_feature_clean":[True]*4,
            "close":[500.]*4,"avg_turnover_63":[1000.]*4
        })
        result=keep_train(d,pd.Timestamp("2020-06-30"))
        self.assertEqual(result.tolist(),[True,False,False,False])

    def test_ineligible_prices_and_liquidity(self):
        d=pd.DataFrame({
            "integrity_feature_clean":[True]*5,
            "close":[20.,2000.,19.99,2000.01,200.],
            "avg_turnover_63":[1.,1.,1.,1.,0.]
        })
        self.assertEqual(eligible_asof(d).tolist(),[True,True,False,False,False])

    def test_top10_jaccard(self):
        self.assertAlmostEqual(top10_similarity(["A","B","C"],["A","B","D"]),.5)
        self.assertEqual(top10_similarity(["A","B"],["A","B"]),1)

    def test_ranker_is_deterministic_and_bounded(self):
        n=100
        train=pd.DataFrame({"date":pd.to_datetime(["2018-06-29"]*50+["2018-12-31"]*50),
                            "y6":[0,1]*50})
        for i,c in enumerate(MODEL_FEATURES):
            train[c]=np.linspace(0,3,n)*(1+(i%4)/10) + np.sin(np.arange(n)+i)
        cal=train.iloc[:30].copy()
        today=train.iloc[30:43].copy()
        p1=train_once(train,cal,today)
        p2=train_once(train,cal,today)
        self.assertEqual(len(p1),len(today))
        self.assertTrue(np.all(np.isfinite(p1)))
        self.assertTrue(np.all((p1>0)&(p1<1)))
        np.testing.assert_array_equal(p1,p2)
        self.assertEqual(MIN_CALIBRATION_POSITIVES,12)

    def test_zero_winner_calibration_is_shrunk_not_zeroed(self):
        prior=np.array([0]*1900+[1]*100)
        rare=np.zeros(900,dtype=int)
        calp=np.full(900,.04)
        testp=np.array([.01,.04,.15])
        self.assertEqual(calibration_mode(rare),"sparse_empirical_bayes_intercept")
        calibrated=calibrate_logit(testp,calp,rare,prior)
        self.assertTrue(np.all((calibrated>0)&(calibrated<1)))
        self.assertTrue(np.all(np.diff(calibrated)>0))
        target=(200*(100/2000))/(900+200)
        calibrated_cal=calibrate_logit(calp,calp,rare,prior)
        self.assertAlmostEqual(float(calibrated_cal.mean()),target,places=6)

    def test_sparse_winner_fold_never_fits_unstable_platt(self):
        y=np.array([1]*7+[0]*846,dtype=int)
        self.assertEqual(calibration_mode(y),"sparse_empirical_bayes_intercept")
        self.assertEqual(calibration_mode(np.array([1]*12+[0]*30)),"platt")

    def test_filing_count_cannot_be_negative(self):
        df=pd.DataFrame({"nse_order_win_90d":[-2,0,3]})
        for name in CATALYST_PREFIXES:
            if name not in df:df[name]=[0,0,0]
        for name in ("ret_20","ret_60","ret_120","ret_252","mom_accel",
                     "vol_accel","turnover_accel","off_high_252","above_low_252",
                     "trend_consistency_60","volatility_60"):
            df[name]=1.
        transformed=safe_featureize(df)
        np.testing.assert_allclose(transformed["nse_order_win_90d"].values,np.log1p([0,0,3]))

if __name__=="__main__":unittest.main(verbosity=2)
