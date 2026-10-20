import json
import unittest
from v13_alternate_sources import parse_quote, normal_name, sector_kind
from v13_financial_evidence import completion_queue
import pandas as pd

class AlternateSourcesTests(unittest.TestCase):
    def test_sector_kinds_keep_financials_separate(self):
        self.assertEqual(sector_kind(['Financial Services','Banks','Private Sector Bank']),'BANK')
        self.assertEqual(sector_kind(['Financial Services','Insurance']),'INSURER')
        self.assertEqual(sector_kind(['Financial Services','Capital Markets']),'OTHER_FINANCIAL')
        self.assertEqual(sector_kind(['Information Technology','IT - Software']),'NONFINANCIAL')
        self.assertEqual(sector_kind(['']),'UNKNOWN')
    def test_name_match_keeps_country_and_business_names(self):
        self.assertEqual(normal_name('Singer India share price'),normal_name('Singer India Limited'))
        self.assertNotEqual(normal_name('Singer India'),normal_name('Singer Sri Lanka'))
    def test_exchange_metadata_requires_exact_isin_symbol_series(self):
        issuer={'nse_current_symbol':'TEST','isin':'INE000000001','nse_current_series':'BE'}
        body={'equityResponse':[{'metaData':{'symbol':'TEST','isinCode':'INE000000001','series':'BE'},
            'secInfo':{'macro':'Industrials','basicIndustry':'Machinery'},'tradeInfo':{}}]}
        receipt={'requested_url':'https://www.nseindia.com/api/NextApi/apiClient/GetQuoteApi','sha256':'a'*64,'first_retrieved_utc':'2026-10-20T12:00:00+00:00'}
        self.assertEqual(parse_quote(json.dumps(body).encode(),issuer,receipt)['sector_kind'],'NONFINANCIAL')
        body['equityResponse'][0]['metaData']['series']='EQ'
        with self.assertRaisesRegex(ValueError,'identity/series'):parse_quote(json.dumps(body).encode(),issuer,receipt)
    def test_annual_group_basis_without_current_group_endpoint_does_not_crash(self):
        queue=pd.DataFrame([{'symbol':'TEST','reporting_mode':'standalone','period_end':'2026-06-30',
            'available_at_utc':'2026-08-01T12:00:00Z','last_attempt':'0000','filed_period':'Quarterly'}])
        wide=pd.DataFrame([{'symbol':'TEST','reporting_mode':'consolidated','period_kind':'annual',
            'period_end':'2026-03-31','period':pd.Timestamp('2026-03-31'),'available':pd.Timestamp('2026-05-01T12:00:00Z')},
            {'symbol':'TEST','reporting_mode':'standalone','period_kind':'quarter',
            'period_end':'2026-06-30','period':pd.Timestamp('2026-06-30'),'available':pd.Timestamp('2026-08-01T12:00:00Z')}])
        self.assertTrue(completion_queue(queue,wide,'2026-10-19',100).empty)

if __name__=='__main__':unittest.main()
