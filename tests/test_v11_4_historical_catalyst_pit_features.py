import unittest
import pandas as pd
from v11_4_historical_catalyst_pit_features import fold_close,source_availability,fold_event_counts
class HistoricalCatalystPITTests(unittest.TestCase):
 def test_market_close_exact(self):
  self.assertEqual(fold_close("2024-12-31").isoformat(),"2024-12-31T10:00:00+00:00")
 def test_later_received_wins(self):
  p=pd.Series(["2024-12-31T09:00:00Z"])
  r=pd.Series(["2024-12-31T12:00:00Z"])
  self.assertEqual(source_availability(p,r).iloc[0].isoformat(),"2024-12-31T12:00:00+00:00")
 def test_post_market_close_excluded(self):
  x=pd.DataFrame({
   "symbol":["ABC","ABC"],"event_type":["order_win","order_win"],
   "published_ts":["2024-12-31T09:00:00Z","2024-12-31T11:00:00Z"],
   "exchange_received_ts":[pd.NaT,pd.NaT],
  })
  counts=fold_event_counts(x,{"ABC"},"2024-12-31",180)
  self.assertEqual(counts[("ABC","order_win")],1)
 def test_later_exchange_receipt_excluded(self):
  x=pd.DataFrame({"symbol":["ABC"],"event_type":["capacity_expansion"],
   "published_ts":["2024-12-31T08:00:00Z"],
   "exchange_received_ts":["2025-01-01T11:00:00Z"]})
  self.assertTrue(fold_event_counts(x,{"ABC"},"2024-12-31",180).empty)
 def test_other_symbol_never_leaks(self):
  x=pd.DataFrame({"symbol":["XYZ"],"event_type":["order_win"],
   "published_ts":["2024-12-31T08:00:00Z"],"exchange_received_ts":[pd.NaT]})
  self.assertTrue(fold_event_counts(x,{"ABC"},"2024-12-31",180).empty)
 def test_stale_events_excluded(self):
  x=pd.DataFrame({"symbol":["ABC"],"event_type":["order_win"],
   "published_ts":["2023-01-01T08:00:00Z"],"exchange_received_ts":[pd.NaT]})
  self.assertTrue(fold_event_counts(x,{"ABC"},"2024-12-31",180).empty)
if __name__=="__main__":unittest.main(verbosity=2)
