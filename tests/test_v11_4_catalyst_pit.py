import unittest
import pandas as pd
from v11_4_historical_catalyst_pit_features import fold_close,source_availability,fold_event_counts
class CatalystPITTests(unittest.TestCase):
 def test_market_close(self):
  self.assertEqual(fold_close("2024-12-31").isoformat(),"2024-12-31T10:00:00+00:00")
 def test_later_exchange_received_timestamp_overrides_early_publication(self):
  a=source_availability(pd.Series(["2024-12-31T08:00:00Z"]),pd.Series(["2024-12-31T11:00:00Z"]))
  self.assertEqual(a.iloc[0].isoformat(),"2024-12-31T11:00:00+00:00")
 def test_missing_received_uses_publication(self):
  a=source_availability(pd.Series(["2024-12-31T08:00:00Z"]),pd.Series([None]))
  self.assertEqual(a.iloc[0].isoformat(),"2024-12-31T08:00:00+00:00")
 def test_after_close_never_counted(self):
  df=pd.DataFrame({
    "symbol":["ABC","ABC","ABC","XYZ"],
    "event_type":["order_win"]*4,
    "published_ts":["2024-12-31T09:00:00Z","2024-12-31T11:00:00Z",
      "2024-12-31T09:00:00Z","2024-12-31T09:00:00Z"],
    "exchange_received_ts":[None,None,"2024-12-31T13:00:00Z",None]
  })
  e=fold_event_counts(df,{"ABC"},"2024-12-31",lookback=180)
  self.assertEqual(e.loc[("ABC","order_win")],1)
 def test_lookback_blocks_older_event(self):
  df=pd.DataFrame({"symbol":["ABC"],"event_type":["order_win"],
                   "published_ts":["2022-12-31T09:00:00Z"],
                   "exchange_received_ts":[None]})
  self.assertEqual(len(fold_event_counts(df,{"ABC"},"2024-12-31",180)),0)
if __name__=="__main__":unittest.main(verbosity=2)
