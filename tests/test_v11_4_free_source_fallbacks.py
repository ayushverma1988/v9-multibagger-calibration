import copy
import gzip
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from pypdf import PdfWriter

from v11_4_free_source_fallbacks import (PublicEvidenceStore, finite_number,
    parse_interoperability_master, parse_yahoo_bse, parse_finology,
    parse_table, reconcile_provider_facts, parse_stockanalysis_financials, sha)
from v11_4_free_financial_current_audit import verify_current_market_cap, current_financial_metrics, verify_receipt_date
from v11_4_discovery_signal_research import source_features, freeze_ranking, freeze_recipe, FEATURES
from v11_4_catalyst_ocr_recovery import recover_pdf
from v11_4_free_source_recovery_run import financial_items
from v11_4_BSE_secondary_financial_recovery import parse_directory, unique_name_map


SEEN = '2026-10-10T01:00:00+00:00'
ISIN = 'INE467B01029'


def yahoo():
    timestamp = int(pd.Timestamp('2026-10-09T03:45:00Z').timestamp())
    return {'chart': {'error': None, 'result': [{'meta': {
        'symbol': 'TCS.BO', 'exchangeName': 'BSE', 'currency': 'INR', 'instrumentType': 'EQUITY',
        'exchangeTimezoneName': 'Asia/Kolkata', 'regularMarketPrice': 102,
        'regularMarketTime': int(pd.Timestamp('2026-10-09T10:00:00Z').timestamp()),
        'longName': 'Tata Consultancy Services Limited'}, 'timestamp': [timestamp],
        'indicators': {'quote': [{'open': [100], 'high': [103], 'low': [99],
                                 'close': [102], 'volume': [1000]}],
                       'adjclose': [{'adjclose': [51]}]}, 'events': {'splits': {'x': {}}}}]}}


def parse_chart(j, seen=SEEN):
    return parse_yahoo_bse(json.dumps(j).encode(), 'TCS.BO', seen,
                           '2026-10-09', 'Tata Consultancy Services Ltd.')


def html(checked='checked', units='Figures are in Crores', date='Mar 2026'):
    return f'''<div id="mainContent_pnlCompanyDetails"><h1 id="mainContent_ltrlCompName">Tata Consultancy Services Ltd.</h1></div>
    <input id="mainContent_togBtn" {checked}><span class="on">Consolidated</span><span class="off">Standalone</span>
    <section id="profit"><p>{units}</p><table><tr><th></th><th>{date}</th></tr>
    <tr><td>Net Sales</td><td>200.00</td></tr><tr><td>Profit After Tax</td><td>10.00</td></tr>
    <tr><td>Consolidated Net Profit</td><td>9.00</td></tr><tr><td>Other Income</td><td>-</td></tr></table></section>
    <div id="mainContent_divROE">3 Year 21.30% 5 Year 22.50%</div>
    <div id="mainContent_clsprice">100 NSE: 9 Oct 04:00 PM</div><input id="mainContent_hfExchangeType" value="BSE">
    <div id="companyessentials"><div class="compess"><small>Market Cap</small><p>100.00 Cr.</p></div>
    <div class="compess"><small>No. of Shares</small><p>1.00 Cr.</p></div>
    <div class="compess"><small>P/E</small><p>10.00</p></div><div class="compess"><small>EPS (TTM)</small><p>10.00</p></div></div>'''.encode()


def parse_page(raw):
    return parse_finology(raw,'https://ticker.finology.in/company/TCS?mode=C',
                         'TCS','Tata Consultancy Services Limited',SEEN)


class FreeSourceEvidenceTests(unittest.TestCase):
    def test_bse_directory_code_link_must_agree_and_names_cannot_join_ambiguously(self):
        raw=b'<h1>Bombay Stock Exchange Stocks</h1><table><tr><td>1</td><td><a href="/quote/bom/532540/">532540</a></td><td>Tata Consultancy Services Limited</td></tr></table>'
        rows=parse_directory(raw);self.assertEqual(rows[0]['bse_provider_code'],'532540')
        with self.assertRaises(ValueError):parse_directory(raw.replace(b'href="/quote/bom/532540/',b'href="/quote/bom/500325/'))
        self.assertEqual(unique_name_map(rows+[{**rows[0],'bse_provider_code':'500325'}]),{})

    def test_outbound_financial_universe_uses_all_public_listings_only(self):
        public=pd.DataFrame({'nse_current_symbol':['B','A'],
            'nse_current_name':['Beta Ltd','Alpha Ltd'],'isin':[ISIN,'INE002A01018']})
        items=financial_items(public)
        self.assertEqual([x['symbol'] for x in items],['A','B'])
        self.assertTrue(all(x['preferred_mode']=='consolidated' for x in items))
        self.assertEqual(len(items),len(public))
        # There is deliberately no private-cohort argument or selection field.
        with self.assertRaises(ValueError):financial_items(pd.concat([public,public]))

    def test_numbers_preserve_zero_and_reject_missing_nonfinite_units(self):
        self.assertEqual(finite_number('(1,200.5)'),-1200.5)
        self.assertEqual(finite_number(0),0)
        for value in [True,np.bool_(False),'-',None,'NaN','inf','12%','25 crore']:
            self.assertIsNone(finite_number(value))

    def test_interoperability_reference_is_not_ordinary_nse_trading(self):
        frame=pd.DataFrame([{'TckrSymb':'TCS$', 'SctySrs':'EQ', 'ISIN':ISIN,
            'PrtdToTrad':2,'DelFlg':'N','FinInstrmNm':'Tata Consultancy','IssdCptl':1000,'ParVal':1}])
        raw=gzip.compress(frame.to_csv(index=False).encode())
        result,_=parse_interoperability_master(raw,'2026-10-09')
        self.assertEqual(len(result),1)
        self.assertFalse(result.ordinary_NSE_trading_permitted.iloc[0])
        self.assertFalse(result.BSE_active_trading_and_price_coverage_proven.iloc[0])
        frame.loc[0,'TckrSymb']='TCS'
        with self.assertRaisesRegex(ValueError,'suffix'):
            parse_interoperability_master(gzip.compress(frame.to_csv(index=False).encode()),'2026-10-09')

    def test_yahoo_venue_identity_and_timezone_are_required(self):
        for field,value in [('exchangeName','NSI'),('currency','USD'),('symbol','TCS.NS'),
                            ('instrumentType','ETF'),('exchangeTimezoneName','UTC')]:
            j=yahoo();j['chart']['result'][0]['meta'][field]=value
            with self.assertRaises(ValueError):parse_chart(j)

    def test_zero_or_future_market_timestamp_is_not_latest_quote(self):
        for value in [0,None,int(pd.Timestamp('2026-10-11T01:00:00Z').timestamp())]:
            j=yahoo();j['chart']['result'][0]['meta']['regularMarketTime']=value
            with self.assertRaises(ValueError):parse_chart(j)

    def test_adjusted_close_never_replaces_price_and_split_not_applied_twice(self):
        info,bars,_=parse_chart(yahoo())
        self.assertEqual(bars[0]['close'],102)
        self.assertTrue(info['official_reference_name_matches'])
        self.assertFalse(info['ISIN_in_quote_response'])
        self.assertFalse(info['guaranteed_realtime_latency_proven'])
        self.assertFalse(info['corporate_action_adjustment_independently_verified'])

    def test_incomplete_or_missing_candle_is_unknown_not_zero(self):
        j=yahoo();j['chart']['result'][0]['indicators']['quote'][0]['close']=[None]
        _,bars,errors=parse_chart(j)
        self.assertEqual(bars,[]);self.assertEqual(len(errors),1)
        j=yahoo();j['chart']['result'][0]['meta']['regularMarketTime']=int(pd.Timestamp('2026-10-09T07:00:00Z').timestamp())
        _,bars,errors=parse_chart(j,seen='2026-10-09T08:00:00Z')
        self.assertEqual(bars,[]);self.assertIn('not_closed',errors[0]['reason'])

    def test_duplicate_or_misaligned_daily_bars_are_rejected(self):
        j=yahoo();j['chart']['result'][0]['timestamp']*=2
        with self.assertRaisesRegex(ValueError,'Misaligned'):parse_chart(j)
        for values in j['chart']['result'][0]['indicators']['quote'][0].values():values*=2
        with self.assertRaisesRegex(ValueError,'Duplicate'):parse_chart(j)

    def test_finology_real_basis_toggle_beats_requested_mode_and_css_labels(self):
        self.assertEqual(parse_page(html())['identity']['reporting_mode'],'consolidated')
        self.assertEqual(parse_page(html(checked=''))['identity']['reporting_mode'],'standalone')

    def test_financial_units_pat_scope_and_missing_facts(self):
        data=parse_page(html());facts={f['metric']:f for f in data['facts']}
        self.assertEqual(facts['revenue']['value_INR'],200e7)
        self.assertEqual(facts['pat']['value_INR'],10e7)
        self.assertEqual(facts['pat_parent']['value_INR'],9e7)
        self.assertNotIn('other_income_provider',facts)
        self.assertFalse(facts['revenue']['source_independent_original_filing_verified'])
        self.assertFalse(data['identity']['historical_PIT_training_eligible'])
        with self.assertRaisesRegex(ValueError,'unit'):parse_page(html(units='INR millions'))

    def test_visible_exchange_beats_hidden_field_and_future_period_rejected(self):
        data=parse_page(html())
        self.assertEqual(data['quote']['venue'],'NSE')
        self.assertTrue(data['quote']['cap_shares_quote_arithmetic_agrees'])
        with self.assertRaisesRegex(ValueError,'future period'):parse_page(html(date='Mar 2027'))

    def test_other_company_name_rejected(self):
        with self.assertRaisesRegex(ValueError,'issuer'):
            parse_page(html().replace(b'Tata Consultancy Services Ltd.',b'Another Company Ltd.'))

    def test_reconciliation_requires_exact_basis_period_and_pat_scope(self):
        f=parse_page(html())['facts'][1]
        original={**f,'rendered_statement_agrees':True,'source_sha256':'original'}
        self.assertEqual(reconcile_provider_facts([f],[original])[0]['status'],'AGREES_WITH_ORIGINAL')
        for key,value in [('metric','pat_parent'),('reporting_mode','standalone'),
                          ('period_kind','quarter'),('period_end','2025-03-31')]:
            mismatch={**original,key:value}
            self.assertEqual(reconcile_provider_facts([f],[mismatch])[0]['status'],'NO_UNIQUE_ORIGINAL_MATCH')
        mismatch={**original,'value_INR':f['value_INR']+1e6}
        self.assertEqual(reconcile_provider_facts([f],[mismatch])[0]['status'],'DISAGREES_WITH_ORIGINAL')

    def test_corrupt_cache_http_error_and_nonpublic_url_cannot_become_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=PublicEvidenceStore(tmp)
            with self.assertRaisesRegex(ValueError,'Unapproved'):store.get('http://ticker.finology.in/company/TCS')
            with self.assertRaisesRegex(ValueError,'Unapproved'):store.get('https://example.com/private')
            url='https://ticker.finology.in/company/TCS';key=sha(url.encode())
            Path(tmp,key+'.bin').write_bytes(b'x')
            Path(tmp,key+'.json').write_text(json.dumps({'requested_url':url,'sha256':sha(b'x'),'status':500}))
            with self.assertRaisesRegex(ValueError,'HTTP 500'):store.get(url)
            Path(tmp,key+'.bin').write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError,'changed'):store.get(url)

    def test_standardized_provider_units_and_definition_names_are_preserved(self):
        raw=b'''<h1>Tata Consultancy Services Income Statement</h1><p>BOM:532540</p>
        <p>Financials in millions INR.</p><table><tr><th>Fiscal Year</th><th>FY 2026</th></tr>
        <tr><td>Period Ending</td><td>Mar '26 Mar 31, 2026</td></tr>
        <tr><td>Net Income</td><td>49,210</td></tr></table>'''
        x=parse_stockanalysis_financials(raw,'532540','Tata Consultancy Services Ltd',SEEN)
        self.assertEqual(x['facts'][0]['value_INR'],49210e6)
        self.assertEqual(x['facts'][0]['metric'],'provider_standardized_net_income')
        self.assertFalse(x['facts'][0]['original_PAT_and_OPM_definition_equivalence_verified'])
        self.assertFalse(x['historical_PIT_training_eligible'])
        for changed in [raw.replace(b'532540',b'500325'),raw.replace(b'millions',b'thousands'),
                        raw.replace(b'Mar 31, 2026',b'Mar 31, 2027')]:
            with self.assertRaises(ValueError):parse_stockanalysis_financials(changed,'532540','Tata Consultancy Services Ltd',SEEN)


def valuation_fixture():
    company=parse_page(html());company['isin']=ISIN
    market=pd.DataFrame([{'symbol':'TCS','isin':ISIN,'date':'2026-10-09','close':100}])
    master=pd.DataFrame([{'TckrSymb':'TCS','ISIN':ISIN,'SctySrs':'EQ','DelFlg':'N',
                          'PrtdToTrad':0,'IssdCptl':10000000}])
    return company,market,master


class CurrentFinancialAuditTests(unittest.TestCase):
    def test_bse_cap_can_corroborate_share_units_without_replacing_nse_price(self):
        company,market,master=valuation_fixture()
        company['quote'].update({'venue':'BSE','latest_price_INR':102,'reported_market_cap_crore':102})
        bse=pd.DataFrame([{'isin':ISIN,'same_asof_trade_day':True,
            'bse_quote_verified_current_identity_and_name':True,'latest_price_INR':102}])
        x=verify_current_market_cap(company,market,master,'2026-10-09',bse)
        self.assertEqual(x['market_cap_verified_crore'],100)
        self.assertEqual(x['provider_same_venue_cap_corroborated_crore'],102)
        self.assertEqual(x['market_cap_verified_venue'],'NSE')
        bse['same_asof_trade_day']=False
        self.assertEqual(verify_current_market_cap(company,market,master,'2026-10-09',bse)['status'],'UNKNOWN')

    def test_receipt_original_bytes_and_trading_date_are_required(self):
        receipt={'status':200,'sha256':sha(b'x'),
                 'url':'https://nsearchives.nseindia.com/content/cm/NSE_CM_security_09102026.csv.gz'}
        verify_receipt_date(b'x',receipt,'2026-10-09','%d%m%Y')
        for raw,asof in [(b'tampered','2026-10-09'),(b'x','2026-10-08')]:
            with self.assertRaises(ValueError):verify_receipt_date(raw,receipt,asof,'%d%m%Y')

    def test_share_unit_is_per_security_corroborated_before_market_cap(self):
        company,market,master=valuation_fixture()
        result=verify_current_market_cap(company,market,master,'2026-10-09')
        self.assertEqual(result['market_cap_verified_crore'],100)
        master.loc[0,'IssdCptl']=1000000  # nominal capital cannot masquerade as shares
        self.assertEqual(verify_current_market_cap(company,market,master,'2026-10-09')['status'],'UNKNOWN')

    def test_cross_venue_stale_quote_wrong_isin_or_conflicting_count_rejected(self):
        for key,value in [('venue','BSE'),('displayed_quote_day_IST','2026-10-08'),('latest_price_INR',105)]:
            company,market,master=valuation_fixture();company['quote'][key]=value
            self.assertEqual(verify_current_market_cap(company,market,master,'2026-10-09')['status'],'UNKNOWN')
        company,market,master=valuation_fixture();master.loc[0,'ISIN']='INE002A01018'
        self.assertEqual(verify_current_market_cap(company,market,master,'2026-10-09')['status'],'UNKNOWN')
        company,market,master=valuation_fixture();master=pd.concat([master,master.assign(IssdCptl=12000000)])
        self.assertEqual(verify_current_market_cap(company,market,master,'2026-10-09')['status'],'UNKNOWN')

    def test_five_annual_rows_are_four_elapsed_years_not_five_year_growth(self):
        company=parse_page(html());company['facts']=[]
        for year in range(2022,2027):
            company['facts'].append({'metric':'revenue','period_end':f'{year}-03-31',
                'period_kind':'annual','value_INR':100+year})
        result=current_financial_metrics(company,'2026-10-09')
        self.assertIsNone(result['sales_growth_5y_reproduced'])
        self.assertIsNone(result['sales_growth_7y_reproduced'])
        self.assertIn('provider_reported_roe_5y_avg',result)
        self.assertFalse(result['full_original_four_screener_inputs_completed'])

    def test_negative_equity_missing_cashflow_and_negative_pat_do_not_pass(self):
        company=parse_page(html())
        company['facts'] += [{'metric':k,'period_end':'2026-03-31','period_kind':kind,'value_INR':v}
            for k,kind,v in [('share_capital','instant',10),('reserves','instant',-20),
                            ('borrowings_provider','instant',10)]]
        result=current_financial_metrics(company,'2026-10-09')
        self.assertIsNone(result['debt_equity']);self.assertIsNone(result['latest_annual_cfo_pat_ratio'])
        self.assertTrue(result['negative_or_missing_equity'])

    def test_exceptional_profit_diagnostic_requires_same_basis_pbt_reconciliation(self):
        company=parse_page(html());company['facts']=[]
        for date in ['2025-09-30','2025-12-31','2026-03-31','2026-06-30']:
            for metric,value in [('revenue',20),('pbt',100),('operating_profit_provider',-10),
                ('depreciation',1),('finance_cost',2),('other_income_provider',3),('exceptional_items_provider',110)]:
                company['facts'].append({'metric':metric,'period_end':date,'period_kind':'quarter',
                    'value_INR':value,'rendered_display_rounding_INR':.005})
        x=current_financial_metrics(company,'2026-10-09')
        self.assertTrue(x['PBT_components_reconcile_all_four_quarters'])
        self.assertEqual(x['four_quarter_PBT_excluding_reported_exceptional_items'],-40)
        company['facts'].pop()  # missing is never a synthetic zero
        self.assertIsNone(current_financial_metrics(company,'2026-10-09')['four_quarter_PBT_excluding_reported_exceptional_items'])


class FixedResearchAndOcrTests(unittest.TestCase):
    def test_ranking_does_not_depend_on_outcomes_and_resolves_ties(self):
        frame=pd.DataFrame({'symbol':['B','A','C'],'probability':[.7,.7,.2],'y6':[1,0,1]})
        self.assertEqual(freeze_ranking(frame).symbol.tolist(),['A','B','C'])
        self.assertEqual(freeze_ranking(frame.assign(y6=1-frame.y6)).symbol.tolist(),['A','B','C'])
        with self.assertRaises(ValueError):freeze_ranking(frame.assign(probability=np.nan))

    def test_already_transformed_events_are_not_log_transformed_twice(self):
        import v11_4_standalone_train_walkforward as core
        frame=pd.DataFrame({k:[1.] for k in core.MODEL_FEATURES})
        frame['ret_252']=2.;frame['volatility_60']=.2
        first=source_features(frame);again=source_features(core.safe_featureize(frame),already_safe=True)
        pd.testing.assert_frame_equal(first,again)
        frame['nse_order_win_90d']=np.nan
        self.assertTrue(pd.isna(source_features(frame)['specific_catalyst_momentum_alignment'].iloc[0]))
        frame['mom_accel']=np.inf
        with self.assertRaises(ValueError):source_features(frame)

    def test_recipe_cannot_be_rewritten_after_observing_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);features=root/'features';labels=root/'labels'
            features.write_bytes(b'features');labels.write_bytes(b'labels')
            result=freeze_recipe(root/'out',features,labels)
            self.assertEqual(result['new_variant_count'],1)
            self.assertFalse(result['parameter_search_performed'])
            with self.assertRaises(FileExistsError):freeze_recipe(root/'out',features,labels)

    def test_partial_pdf_scan_never_claims_entire_document_or_certified_quantities(self):
        writer=PdfWriter();writer.add_blank_page(100,100);writer.add_blank_page(100,100)
        b=io.BytesIO();writer.write(b)
        with tempfile.TemporaryDirectory() as tmp, patch('v11_4_catalyst_ocr_recovery.subprocess.run') as run:
            def fake(args,**kwargs):
                if args[0]=='tesseract':
                    Path(args[2]+'.txt').write_text('Tata Consultancy Services disclosed information. '*5)
                    Path(args[2]+'.tsv').write_text('conf\ttext\n95\tTata\n')
            run.side_effect=fake
            _,proof=recover_pdf(b.getvalue(),tmp,max_pages=1)
            self.assertFalse(proof['entire_document_read'])
            self.assertTrue(proof['any_OCR_used'])
            self.assertFalse(proof['quantified_OCR_evidence_visually_verified'])
            self.assertFalse(proof['complete_causal_chain_verified'])


if __name__=='__main__':unittest.main()
