import unittest
import pandas as pd
from v11_4_fy2023_historical_annual_pilot import choose_pairs

def data():
    return pd.DataFrame([
        {"symbol":"AAA","fy_end":"2022-03-31","available_at_utc":"2022-05-20T08:00:00Z",
         "consolidated":"Consolidated","xbrl_url":"https://nsearchives.nseindia.com/corp/aaa22.xml"},
        {"symbol":"AAA","fy_end":"2023-03-31","available_at_utc":"2023-05-20T08:00:00Z",
         "consolidated":"Consolidated","xbrl_url":"https://nsearchives.nseindia.com/corp/aaa23.xml"},
        {"symbol":"BBB","fy_end":"2022-03-31","available_at_utc":"2022-05-20T08:00:00Z",
         "consolidated":"Consolidated","xbrl_url":"https://nsearchives.nseindia.com/corp/bbb22.xml"},
        {"symbol":"BBB","fy_end":"2023-03-31","available_at_utc":"2024-01-05T08:00:00Z",
         "consolidated":"Consolidated","xbrl_url":"https://nsearchives.nseindia.com/corp/bbb23.xml"},
        {"symbol":"CCC","fy_end":"2022-03-31","available_at_utc":"2022-05-20T08:00:00Z",
         "consolidated":"Consolidated","xbrl_url":"https://nsearchives.nseindia.com/corp/ccc22.xml"},
        {"symbol":"CCC","fy_end":"2023-03-31","available_at_utc":"2023-05-20T08:00:00Z",
         "consolidated":"Standalone","xbrl_url":"https://nsearchives.nseindia.com/corp/ccc23.xml"},
        {"symbol":"DDD","fy_end":"2022-03-31","available_at_utc":"2022-05-20T08:00:00Z",
         "consolidated":"Consolidated","xbrl_url":"https://untrusted.example.com/fake22.xml"},
        {"symbol":"DDD","fy_end":"2023-03-31","available_at_utc":"2023-05-20T08:00:00Z",
         "consolidated":"Consolidated","xbrl_url":"https://nsearchives.nseindia.com/corp/ddd23.xml"},
        {"symbol":"AAA","fy_end":"2023-03-31","available_at_utc":"2023-06-20T08:00:00Z",
         "consolidated":"Consolidated","xbrl_url":"https://nsearchives.nseindia.com/corp/aaa23_revised.xml"},
    ])
class OriginalFY2023PairTests(unittest.TestCase):
    def test_only_original_preclose_same_reporting_mode_paired(self):
        result=choose_pairs(data(),{"AAA","BBB","CCC","DDD"},"2023-12-29")
        self.assertEqual(len(result),1)
        self.assertEqual(result[0][0],"AAA")
        self.assertEqual(result[0][1],"consolidated")
        self.assertTrue(any(u.endswith("/aaa23.xml") for u in result[0][2]["xbrl_url"]))
        self.assertFalse(any("revised" in x for x in result[0][2]["xbrl_url"]))
    def test_future_filing_cannot_be_backfilled(self):
        self.assertEqual(len(choose_pairs(data(),{"BBB"},"2023-12-29")),0)
    def test_nonofficial_and_mixed_mode_rejected(self):
        self.assertEqual(len(choose_pairs(data(),{"CCC","DDD"},"2023-12-29")),0)
    def test_wrong_year_request_rejected(self):
        with self.assertRaisesRegex(ValueError,"consecutive"):
            choose_pairs(data(),{"AAA"},"2023-12-29",years=(2019,2023))
if __name__=="__main__":unittest.main()
