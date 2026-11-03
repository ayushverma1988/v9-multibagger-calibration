import json
import unittest
from v13_yahoo_financials import parse

class YahooFinancialsTests(unittest.TestCase):
    def setUp(self):
        self.issuer={'isin':'INE123456789','nse_current_symbol':'TEST','nse_current_name':'Test Limited'}
        self.proof={'requested_url':'https://query1.finance.yahoo.com/a','sha256':'a'*64,'first_retrieved_utc':'2026-10-21T00:00:00Z'}
        self.chart={'chart':{'result':[{'meta':{'symbol':'TEST.NS','longName':'Test Limited','exchangeName':'NSI','currency':'INR','instrumentType':'EQUITY'}}]}}
        blocks=[]
        for name,value in [('TotalRevenue',1000),('TotalAssets',2000),('CashAndCashEquivalents',300)]:
            key='annual'+name
            blocks.append({'meta':{'symbol':['TEST.NS'],'type':[key]},key:[{'asOfDate':'2026-03-31','periodType':'12M','currencyCode':'INR','reportedValue':{'raw':value}}]})
        self.raw={'timeseries':{'result':blocks,'error':None}}
        self.primary={('INE123456789','consolidated',kind,'2026-03-31',metric):{'value_INR':value,'source_url':'https://nse.test/a','source_sha256':'b'*64}
            for kind,metric,value in [('annual','revenue',1000),('instant','assets',2000)]}
    def run_parse(self,primary=None):
        return parse(json.dumps(self.raw).encode(),json.dumps(self.chart).encode(),self.issuer,self.proof,self.proof,self.primary if primary is None else primary)
    def test_independent_basis_anchors_allow_dated_missing_cash_observation(self):
        rows=self.run_parse();self.assertEqual(len(rows),3)
        cash=next(r for r in rows if r['metric']=='cash')
        self.assertEqual(cash['reporting_mode'],'consolidated');self.assertEqual(cash['value_INR'],300)
        self.assertEqual(cash['provider_family'],'YAHOO_FINANCE')
    def test_no_guess_when_original_basis_unknown(self):
        rows=self.run_parse({});self.assertEqual(len(rows),3)
        self.assertEqual(rows[0]['reporting_mode'],'consolidated')
        self.assertFalse(rows[0]['reporting_basis_independently_confirmed'])
    def test_identical_standalone_and_group_anchors_are_ambiguous(self):
        primary=dict(self.primary)
        for key,value in self.primary.items():primary[(key[0],'standalone',*key[2:])]=value
        rows=self.run_parse(primary)
        self.assertFalse(rows[0]['reporting_basis_independently_confirmed'])
    def test_currency_cannot_be_assumed_from_listing(self):
        self.raw['timeseries']['result'][0]['annualTotalRevenue'][0]['currencyCode']='USD'
        rows=self.run_parse();self.assertNotIn('revenue',{r['metric'] for r in rows})
    def test_wrong_exchange_or_issuer_is_rejected(self):
        self.chart['chart']['result'][0]['meta']['longName']='Other Limited'
        with self.assertRaisesRegex(ValueError,'mismatch'):self.run_parse()
    def test_ttm_is_not_treated_as_annual(self):
        self.raw['timeseries']['result'][0]['annualTotalRevenue'][0]['periodType']='TTM'
        rows=self.run_parse();self.assertNotIn('revenue',{r['metric'] for r in rows})
    def test_noncontrolling_profit_is_labeled_as_same_api_calculation(self):
        for name,value in [('NetIncomeIncludingNoncontrollingInterests',100),('NetIncome',80)]:
            key='annual'+name
            self.raw['timeseries']['result'].append({'meta':{'symbol':['TEST.NS'],'type':[key]},key:[{
                'asOfDate':'2026-03-31','periodType':'12M','currencyCode':'INR','reportedValue':{'raw':value}}]})
        rows=self.run_parse({});minority=next(r for r in rows if r['metric']=='pat_noncontrolling')
        self.assertEqual(minority['value_INR'],20)
        self.assertEqual(minority['definition_status'],'DERIVED_FROM_SAME_API_FINANCIAL_LINES')

if __name__=='__main__':unittest.main()
