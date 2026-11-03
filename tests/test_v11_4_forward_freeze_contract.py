import unittest
import numpy as np
import pandas as pd
from v11_4_forward_research_release import (
 fit_calibration_params,apply_frozen_calibration,strip_pandemic_matured_outcomes,
 validate_prospective_candidates,score_prospective,VERSION,FLOAT_FEATURES
)
from v11_4_standalone_train_walkforward import calibrate_logit

class FreezeContractTests(unittest.TestCase):
 def test_calibration_parameterizes_existing_platt_exactly(self):
  y=np.array([0]*80+[1]*20)
  raw=np.array([.02]*80+[.45]*20)
  prior=np.array([0]*1900+[1]*100)
  params=fit_calibration_params(raw,y,prior)
  expect=calibrate_logit(np.array([.01,.10,.4]),raw,y,prior)
  got=apply_frozen_calibration(np.array([.01,.10,.4]),params)
  self.assertTrue(np.allclose(expect,got,atol=1e-7))
  self.assertGreater(params["slope"],0)
 def test_calibration_parameterizes_sparse_intercept(self):
  y=np.array([0]*90+[1]*5)
  raw=np.array([.03]*90+[.15]*5)
  prior=np.array([0]*1900+[1]*100)
  params=fit_calibration_params(raw,y,prior)
  expect=calibrate_logit(np.array([.01,.10,.4]),raw,y,prior)
  self.assertTrue(np.allclose(apply_frozen_calibration(np.array([.01,.10,.4]),params),expect,atol=1e-7))
  self.assertEqual(params["slope"],1.)
 def test_covid_only_crossing_shock_not_june_2019(self):
  df=pd.DataFrame({"date":pd.to_datetime(["2019-06-28","2019-12-31","2020-06-30","2023-06-30"]),
                   "y6_mature_date":["2019-12-31","2020-06-30","2020-12-31","2023-12-31"]})
  self.assertEqual(strip_pandemic_matured_outcomes(df).tolist(),[True,False,False,True])
  self.assertTrue(strip_pandemic_matured_outcomes(df,enabled=False).all())
 def test_no_legacy_model_predictions_as_prospective_features(self):
  cols={"date":["2026-10-08"]*10,"symbol":[f"STOCK{i}" for i in range(10)],
        "close":[150.]*10,"avg_turnover_63":[1e6]*10,
        "integrity_feature_clean":[True]*10,
        "historical_asof_utc":["2026-10-08T10:00:00+00:00"]*10}
  for c in FLOAT_FEATURES:cols[c]=[1.]*10
  self.df=pd.DataFrame(cols)
  self.meta={"original_NSE_event_catalog_verified":True,"market_close_source_verified":True,
             "NSE_event_coverage_through_utc":"2026-10-08T10:00:00Z",
             "market_data_coverage_through_utc":"2026-10-08T10:00:00Z"}
  self.assertEqual(str(validate_prospective_candidates(self.df,"2026-10-08",self.meta)),"2026-10-08 10:00:00+00:00")
  with self.assertRaisesRegex(ValueError,"Future outcome"):
   validate_prospective_candidates(self.df.assign(y6=0),"2026-10-08",self.meta)
 def test_bad_exchange_coverage_cannot_rank(self):
  cols={"date":["2026-10-08"]*10,"symbol":[f"STOCK{i}" for i in range(10)],
        "close":[150.]*10,"avg_turnover_63":[1e6]*10,
        "integrity_feature_clean":[True]*10,
        "historical_asof_utc":["2026-10-08T10:00:00+00:00"]*10}
  for c in FLOAT_FEATURES:cols[c]=[1.]*10
  x=pd.DataFrame(cols)
  meta={"original_NSE_event_catalog_verified":True,"market_close_source_verified":True,
        "NSE_event_coverage_through_utc":"2026-10-08T09:59:00Z",
        "market_data_coverage_through_utc":"2026-10-08T10:00:00Z"}
  with self.assertRaisesRegex(ValueError,"coverage"):
   validate_prospective_candidates(x,"2026-10-08",meta)
  x["date"]="2026-09-30"
  with self.assertRaisesRegex(ValueError,"Stale"):
   validate_prospective_candidates(x,"2026-10-08",{**meta,"NSE_event_coverage_through_utc":"2026-10-08T10:00:00Z"})
 def test_no_backtest_date_passes_forward_contract(self):
  cols={"date":["2025-12-31"]*10,"symbol":[f"STOCK{i}" for i in range(10)],
        "close":[150.]*10,"avg_turnover_63":[1e6]*10,
        "integrity_feature_clean":[True]*10,
        "historical_asof_utc":["2025-12-31T10:00:00+00:00"]*10}
  for c in FLOAT_FEATURES:cols[c]=[1.]*10
  meta={"original_NSE_event_catalog_verified":True,"market_close_source_verified":True,
        "NSE_event_coverage_through_utc":"2026-10-08T10:00:00Z",
        "market_data_coverage_through_utc":"2026-10-08T10:00:00Z"}
  with self.assertRaises(ValueError):
   validate_prospective_candidates(pd.DataFrame(cols),"2026-10-08",meta)

if __name__=="__main__":unittest.main(verbosity=2)
