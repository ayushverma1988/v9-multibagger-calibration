import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from v12_dated_source_collection import Store, source_response


class NSEDomainRedirectTests(unittest.TestCase):
    api='https://www.nseindia.com/api/corporate-announcements?index=equities'
    alias='https://www.nseindia.in/api/corporate-announcements?index=equities'

    def response(self,url,status=200,location=None,body=b'[]'):
        response=Mock(url=url,status_code=status,content=body)
        response.headers={} if location is None else {'Location':location}
        return response

    def test_observed_api_alias_redirect_preserves_route_hash_and_clock(self):
        first=self.response(self.api,301,self.alias)
        final=self.response(self.alias,body=b'[{"symbol":"TEST"}]')
        session=Mock();session.get.side_effect=[first,final]
        with tempfile.TemporaryDirectory() as folder:
            store=Store(folder);store.local.session=session
            raw,proof=store.get(self.api)
            self.assertEqual(proof['requested_url'],self.api)
            self.assertEqual(proof['final_url'],self.alias)
            self.assertEqual(proof['sha256'],hashlib.sha256(raw).hexdigest())
            self.assertIn('+00:00',proof['first_retrieved_utc'])
            self.assertEqual(proof['redirect_chain'],[{'url':self.api,'status':301,'location':self.alias}])
            before={p.name:p.read_bytes() for p in Path(folder).iterdir()}
            self.assertEqual(store.get(self.api),(raw,proof))
            self.assertEqual(before,{p.name:p.read_bytes() for p in Path(folder).iterdir()})
            self.assertEqual(session.get.call_count,2)
            self.assertTrue(all(call.kwargs['allow_redirects'] is False for call in session.get.call_args_list))

    def test_archive_alias_and_direct_in_sources(self):
        before='https://nsearchives.nseindia.com/corporate/result.xml'
        after='https://nsearchives.nseindia.in/corporate/result.xml'
        session=Mock();session.get.side_effect=[self.response(before,308,after),self.response(after)]
        response,chain=source_response(session,before)
        self.assertEqual(response.url,after);self.assertEqual(len(chain),1)
        direct=Mock();direct.get.return_value=self.response(self.alias)
        self.assertEqual(source_response(direct,self.alias)[1],[])

    def test_current_primary_financial_urls_are_retained_without_rewriting(self):
        from v12_dated_training_inputs import normalize_financial_catalog
        url='https://nsearchives.nseindia.in/corporate/xbrl/RESULT_17112026.xml'
        item={'schema':'integrated','venue_index':'equities','catalog_sha256':'a'*64,
              'record':{'symbol':'TEST','broadcast_Date':'17-Nov-2026 19:48:43',
                        'creation_Date':'17-Nov-2026 19:48:43','qe_Date':'30-SEP-2026',
                        'consolidated':'Consolidated','seq_Id':'123','xbrl':url}}
        frame=normalize_financial_catalog([item],'2026-11-17')
        self.assertEqual(frame.iloc[0]['source_url'],url)
        self.assertEqual(frame.iloc[0]['available_at_utc'],'2026-11-17T14:18:43+00:00')
        item['record']['xbrl']=url.replace('nseindia.in','nseindia.in.evil.example')
        with self.assertRaisesRegex(ValueError,'No dated primary'):normalize_financial_catalog([item],'2026-11-17')

    def test_reject_untrusted_or_changed_alias_before_fetching_destination(self):
        for target in ['https://www.nseindia.in.evil.example/api/x','http://www.nseindia.in/api/x',
                       'https://www.nseindia.in:444/api/x','https://user:pass@www.nseindia.in/api/x',
                       'https://nsearchives.nseindia.in/api/x','https://query1.finance.yahoo.com/api/x',
                       'https://www.nseindia.in/api/other?index=equities',
                       'https://www.nseindia.in/api/corporate-announcements?index=sme']:
            with self.subTest(target=target):
                session=Mock();session.get.return_value=self.response(self.api,301,target)
                with self.assertRaises(ValueError):source_response(session,self.api)
                self.assertEqual(session.get.call_count,1)

    def test_reject_silent_redirect_and_loops(self):
        silent=Mock();silent.get.return_value=self.response(self.alias)
        with self.assertRaisesRegex(ValueError,'Unexpected source'):source_response(silent,self.api)
        loop=Mock();loop.get.side_effect=[self.response(self.api,301,self.alias),self.response(self.alias,302,self.api)]
        with self.assertRaisesRegex(ValueError,'loop'):source_response(loop,self.api)
        self.assertEqual(loop.get.call_count,2)

    def test_legacy_cache_is_read_without_rewriting_original_clock(self):
        with tempfile.TemporaryDirectory() as folder:
            raw=b'[]';key=hashlib.sha256(self.api.encode()).hexdigest()
            meta={'requested_url':self.api,'status':200,'first_retrieved_utc':'2026-10-14T16:30:00+00:00',
                  'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'body_file':key+'.bin'}
            base=Path(folder);(base/(key+'.bin')).write_bytes(raw);(base/(key+'.json')).write_text(json.dumps(meta))
            before={p.name:p.read_bytes() for p in base.iterdir()}
            store=Store(folder);store.local.session=Mock()
            self.assertEqual(store.get(self.api),(raw,meta))
            store.local.session.get.assert_not_called()
            self.assertEqual(before,{p.name:p.read_bytes() for p in base.iterdir()})

    def test_non_success_remains_error_and_never_becomes_empty_success(self):
        session=Mock();final=self.response(self.alias,429,body=b'Too many requests')
        final.raise_for_status.side_effect=RuntimeError('HTTP 429')
        session.get.side_effect=[self.response(self.api,301,self.alias),final]
        with tempfile.TemporaryDirectory() as folder:
            store=Store(folder);store.local.session=session
            with self.assertRaisesRegex(RuntimeError,'429'):store.get(self.api)
            with self.assertRaisesRegex(ValueError,'Cached source error'):store.get(self.api)
            self.assertEqual(session.get.call_count,2)

    def event_fixture(self,folder):
        import pandas as pd
        source=Path(folder)/'sources/2026-11-17';source.mkdir(parents=True)
        url='https://nsearchives.nseindia.in/corporate/TEST_17112026.pdf'
        row={'attachment_url':url,'available_at_utc':'2026-11-17T12:00:00+00:00',
             'symbol':'TEST','isin':'INE000000001','issuer_name':'Test Limited',
             'headline':'Bonus allotment and capacity update','text':'Bonus allotment and capacity update'}
        pd.DataFrame([row]).to_parquet(source/'events.parquet',index=False)
        raw=b'%PDF original';receipt={'sha256':hashlib.sha256(raw).hexdigest(),
            'first_retrieved_utc':'2026-11-18T03:00:00+00:00'}
        return url,raw,receipt

    def test_in_notice_is_collected_without_certifying_an_action_interval(self):
        from v13_action_sources import collect
        with tempfile.TemporaryDirectory() as folder:
            url,raw,receipt=self.event_fixture(folder)
            with patch('v13_action_sources.Store.get',return_value=(raw,receipt)) as get, \
                 patch('v13_action_sources.extract_pdf',return_value='Test Limited bonus record date pending'):
                result=collect(folder,'2026-11-17',budget=1)
            get.assert_called_once_with(url)
            self.assertEqual(result['new_notices'],1)
            self.assertEqual(result['complete_intervals_certified'],0)
            notice=json.loads((Path(folder)/'action_audit/original_notice_candidates.json').read_text())[0]
            self.assertEqual(notice['attachment_url'],url)
            self.assertEqual(notice['first_retrieved_utc'],receipt['first_retrieved_utc'])
            self.assertFalse(notice['all_primary_notices_reviewed'])

    def test_in_catalyst_is_collected_without_automatic_future_profit_claim(self):
        from v13_catalyst_execution import collect
        with tempfile.TemporaryDirectory() as folder:
            url,raw,receipt=self.event_fixture(folder)
            with patch('v13_catalyst_execution.Store.get',return_value=(raw,receipt)) as get, \
                 patch('v13_catalyst_execution.extract_pdf',return_value='Test Limited capacity planned'), \
                 patch('v13_catalyst_execution.classify_document',return_value={'issuer_identity_read':True}), \
                 patch('v13_catalyst_execution.enrich',side_effect=lambda text,info:info):
                result=collect(folder,'2026-11-17',budget=1,workers=1)
            get.assert_called_once_with(url)
            self.assertEqual(result['documents'],1)
            self.assertEqual(result['automatic_future_profit_claims'],0)
            self.assertFalse(result['catalogue_complete'])
            document=json.loads((Path(folder)/'catalyst_evidence/document_evidence.json').read_text())[0]
            self.assertEqual(document['attachment_url'],url)
            self.assertEqual(document['review_status'],'NEEDS_QUANTIFIED_EXECUTION_REVIEW')


if __name__=='__main__':unittest.main()
