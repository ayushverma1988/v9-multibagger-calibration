"""Offline deterministic tests for V11.4 live exchange features and PIT gates."""
import io
import unittest
import numpy as np
import pandas as pd
from v11_4_live_nse_market_catalyst import (
 market_udiff,market_full,validate_market,classify_event,
 with_catalyst_features,parse_nse_ts,market_features,adjustment_factors
)
from v11_4_standalone_train_walkforward import CATALYST_PREFIXES,PRICE_COLUMNS
from v11_4_forward_research_release import validate_prospective_candidates

class NSELiveSourceTests(unittest.TestCase):
 def test_two_official_bhavcopy_sources_agree(self):
  target="2026-10-08"
  a=pd.DataFrame({
   "TradDt":["08-10-2026"]*510,
   "TckrSymb":[f"X{i}" for i in range(510)],
   "SctySrs":["EQ"]*510,"ISIN":[f"INE{i:09d}" for i in range(510)],
   "OpnPric":[99.]*510,"HghPric":[101.]*510,"LwPric":[98.]*510,
   "ClsPric":[100.]*510,"TtlTradgVol":[10000]*510,"TtlTrfVal":[1000000]*510})
  b=pd.DataFrame({"SYMBOL":a["TckrSymb"],"SERIES":["EQ"]*510,
                   "CLOSE_PRICE":[100.]*510})
  pri=market_udiff(a.to_csv(index=False).encode(),target)
  sec=market_full(b.to_csv(index=False).encode())
  s=validate_market(pri,sec)
  self.assertTrue(pri["date"].eq(pd.Timestamp(target)).all())
  self.assertEqual(s["official_overlap_rows"],510)
  b.loc[0,"CLOSE_PRICE"]=300
  with self.assertRaisesRegex(ValueError,"disagree"):
   validate_market(pri,market_full(b.to_csv(index=False).encode()))
 def test_all_catalyst_predictions_timestamp_gate(self):
  self.assertIn(("order_win",1),classify_event("Receipt of order","new supply contract"))
  self.assertIn(("capacity_expansion",1),classify_event("Commencement of operations","plant"))
  self.assertIn(("regulatory",-1),classify_event("Warning letter","regulator"))
  self.assertEqual(classify_event("Financial results","Quarterly audited results"),[])
  self.assertNotIn(("promoter_activity",1),classify_event("Promoter pledge release","stake"))
  self.assertEqual(parse_nse_ts("08-Oct-2026 15:00:00").isoformat(),
                   "2026-10-08T09:30:00+00:00")
  day=pd.Timestamp("2026-10-08")
  market=pd.DataFrame({"date":[day,day],"symbol":["A","B"],"close":[150.,200.]})
  e=pd.DataFrame([
    {"symbol":"A","type":"order_win","direction":1,
     "available_utc":pd.Timestamp("2026-10-08T09:50:00Z")},
    {"symbol":"A","type":"order_win","direction":1,
     "available_utc":pd.Timestamp("2026-09-20T09:00:00Z")},
    {"symbol":"B","type":"regulatory","direction":-1,
     "available_utc":pd.Timestamp("2026-10-08T09:00:00Z")},])
  x=with_catalyst_features(market,e,day)
  self.assertEqual(x.loc[x.symbol.eq("A"),"nse_order_win_90d"].iloc[0],2)
  self.assertEqual(x.loc[x.symbol.eq("A"),"nse_order_win_180d"].iloc[0],2)
  self.assertEqual(x.loc[x.symbol.eq("B"),"nse_regulatory_90d_positive"].iloc[0],0)
  for k in CATALYST_PREFIXES:self.assertIn(k,x)
  e.loc[0,"available_utc"]=pd.Timestamp("2026-10-08T10:01:00Z")
  with self.assertRaisesRegex(ValueError,"post-close"):
   with_catalyst_features(market,e,day)
 def test_split_adjustment_uses_only_actions_past_current_close(self):
  history=pd.DataFrame({
    "symbol":["A","A","A"],"date":pd.to_datetime([
         "2026-01-01","2026-04-01","2026-07-01"]),
    "close":[200.,100.,100.]})
  action=pd.DataFrame({"symbol":["A","A"],
      "ex_date":pd.to_datetime(["2026-04-01","2026-12-01"]),
      "factor":[.5,.5]})
  out=adjustment_factors(history,action,pd.Timestamp("2026-07-01"))
  self.assertEqual(out["adj_close"].tolist(),[100.,100.,100.])
 def test_live_momentum_exact_shape_with_253_sessions(self):
  dates=pd.bdate_range("2025-10-20",periods=260)
  day=dates[-1]
  g=pd.DataFrame({"date":dates,"symbol":["DEMO"]*len(dates),
    "close":np.linspace(100,150,len(dates)),
    "adj_close":np.linspace(100,150,len(dates)),
    "volume":[10000.]*len(dates),"turnover":[1e6]*len(dates)})
  features=market_features(g,day,min_company_rows=1)
  self.assertEqual(len(features),1)
  self.assertTrue(features["integrity_feature_clean"].iloc[0])
  self.assertTrue(np.isfinite(features[list(PRICE_COLUMNS)].to_numpy(float)).all())
 def test_future_feed_missing_catalog_blocks_scorer(self):
  frame={"date":["2026-10-08"]*10,
   "symbol":[f"C{i}" for i in range(10)],
   "close":[125.]*10,"avg_turnover_63":[20000.]*10,
   "integrity_feature_clean":[True]*10,
   "historical_asof_utc":["2026-10-08T10:00:00Z"]*10}
  for c in PRICE_COLUMNS+CATALYST_PREFIXES:frame[c]=[0.1]*10
  meta={"original_NSE_event_catalog_verified":False,
    "market_close_source_verified":True,
    "NSE_event_coverage_through_utc":"2026-10-08T10:00:00Z",
    "market_data_coverage_through_utc":"2026-10-08T10:00:00Z"}
  with self.assertRaisesRegex(ValueError,"incomplete"):
   validate_prospective_candidates(pd.DataFrame(frame),"2026-10-08",meta)

if __name__=="__main__":unittest.main(verbosity=2)
