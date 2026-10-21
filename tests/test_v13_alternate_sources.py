import json,tempfile
from pathlib import Path
import unittest
from v13_alternate_sources import parse_quote, parse_finology, normal_name, sector_kind,apply_sector_master
from v13_financial_evidence import completion_queue
import pandas as pd
from unittest.mock import patch,Mock

class AlternateSourcesTests(unittest.TestCase):
    def test_sector_kinds_keep_financials_separate(self):
        self.assertEqual(sector_kind(['Financial Services','Banks','Private Sector Bank']),'BANK')
        self.assertEqual(sector_kind(['Financial Services','Insurance']),'INSURER')
        self.assertEqual(sector_kind(['Financial Services','Capital Markets']),'OTHER_FINANCIAL')
        self.assertEqual(sector_kind(['Information Technology','IT - Software']),'NONFINANCIAL')
        self.assertEqual(sector_kind(['BPO/ITeS']),'NONFINANCIAL')
        self.assertEqual(sector_kind(['Textile - Weaving']),'NONFINANCIAL')
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
    def test_abbreviated_vendor_name_needs_exact_exchange_mapping(self):
        raw=b'''<h1>Abbreviated name share price</h1><span>BSE: <span>123456</span></span>
        <input id="mainContent_hfMode" value="C"><p>Data Feed provided by Accord Fintech</p>
        <div id="profit"><h4>Profit &amp; Loss (All Figures in Cr. Adjusted EPS in Rs.)</h4>
        <table><tr><th>PARTICULARS</th><th>Mar 2026</th></tr><tr><td>Net Sales</td><td>12.34</td></tr></table></div>'''
        issuer={'nse_current_symbol':'TEST','nse_current_name':'Different full name Limited','isin':'INE000000001'}
        receipt={'requested_url':'https://ticker.finology.in/company/TEST?mode=C','sha256':'a'*64,'first_retrieved_utc':'2026-10-20T12:00:00Z'}
        with self.assertRaisesRegex(ValueError,'issuer-name'):parse_finology(raw,issuer,receipt)
        rows,meta=parse_finology(raw,{**issuer,'bse_code':'123456'},receipt)
        self.assertEqual(meta['reporting_mode'],'consolidated')
        self.assertEqual(rows[0]['value_INR'],123400000)
        with self.assertRaisesRegex(ValueError,'issuer-name'):parse_finology(raw,{**issuer,'bse_code':'999999'},receipt)
        ticker_raw=raw.replace(b'<h1>',b'<input id="mainContent_hfSymbol" value="TEST"><h1>')
        rows,meta=parse_finology(ticker_raw,issuer,receipt)
        self.assertIn('EXACT_CURRENT_NSE_SYMBOL',meta['identity_proof'])
        self.assertEqual(rows[0]['reporting_mode'],'consolidated')
    def test_ownership_only_group_page_falls_back_to_real_standalone_statements(self):
        from v13_alternate_sources import collect
        shared='''<a href="https://www.nseindia.com/get-quotes/equity?symbol=TEST">NSE</a>
        <section id="peers"><a href="/market/IN/1">Industrials</a><a href="/market/IN/2">Industrials</a>
        <a href="/market/IN/3">Capital Goods</a><a href="/market/IN/4">Machinery</a></section>
        <section id="shareholding"><table><tr><th></th><th>Jun 2026</th></tr>
        <tr><td>Promoters +</td><td>50.0%</td></tr></table></section>'''
        group=('<div id="company-info" data-consolidated="true"></div>'+shared).encode()
        standalone=('''<div id="company-info" data-consolidated="false"></div>'''+shared+'''
        <section id="quarters">Figures in Rs. Crores<table><tr><th></th><th>Jun 2026</th></tr>
        <tr><td>Sales +</td><td>12.34</td></tr></table></section>''').encode()
        requested=[]
        def response(session,url,**kwargs):
            requested.append(url);r=Mock();r.content=group if url.endswith('/consolidated/') else standalone
            r.url=url;r.status_code=200;r.raise_for_status.return_value=None;return r
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'evidence').mkdir()
            pd.DataFrame([{'nse_current_symbol':'TEST','nse_current_name':'Test Limited','nse_current_series':'EQ','isin':'INE000000001'}]).to_parquet(root/'universe.parquet')
            (root/'evidence/universe_financial_coverage.json').write_text(json.dumps({'rows':[{'isin':'INE000000001','preferred_reporting_basis':'consolidated'}]}))
            with patch('v13_alternate_sources.requests.Session.get',response):
                result=collect(root,budget=1,seconds=3,sources=('SCREENER',))
            self.assertEqual(result['SCREENER']['new_accepted'],1)
            self.assertEqual(requested,['https://www.screener.in/company/TEST/consolidated/','https://www.screener.in/company/TEST/'])
            saved=json.loads((root/'master_data/provider_inputs/INE000000001_screener.json').read_text())
            statement=next(r for r in saved if r['metric']=='revenue')
            self.assertEqual(statement['reporting_mode'],'standalone');self.assertEqual(statement['value_INR'],123400000)
    def test_sector_recovery_needs_three_sources_before_review_clock(self):
        frame=pd.DataFrame([{'isin':'INE000000001','symbol':'TEST','industry':None,'sector_sector_kind':'UNKNOWN','p6':.2}])
        sources=[{'isin':'INE000000001','symbol':'TEST','provider_family':p,'source_sha256':'a'*64,
            'source_url':'https://example.test/'+p,'identity_proof':'EXACT_ISIN',
            'first_observed_utc':'2026-10-20T12:00:00Z'} for p in ('NSE','SCREENER','ACCORD_FINTECH')]
        evidence={'isin':'INE000000001','symbol':'TEST','industry':'Household Appliances','sector_kind':'NONFINANCIAL',
            'primary_plus_two_additional_sources_agree':True,'available_at_utc':'2026-10-20T12:00:00Z','sources':sources}
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'master_data';path.mkdir();(path/'sector_master.json').write_text(json.dumps([evidence]))
            early=apply_sector_master(frame,temp,'2026-10-20T11:00:00Z')
            self.assertFalse(early.sector_recovered_from_three_sources.iloc[0])
            late=apply_sector_master(frame,temp,'2026-10-20T13:00:00Z')
            self.assertTrue(late.sector_recovered_from_three_sources.iloc[0]);self.assertEqual(late.p6.iloc[0],.2)
            self.assertEqual(late.sector_sector_kind.iloc[0],'NONFINANCIAL')
            stale=apply_sector_master(frame,temp,'2026-11-21T13:00:00Z')
            self.assertFalse(stale.sector_recovered_from_three_sources.iloc[0])

if __name__=='__main__':unittest.main()
