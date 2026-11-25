import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from v13_collection import collect_financials


class FilingRestart(unittest.TestCase):
    """Exercise the real XBRL parsing, queue, parquet commit and restart boundary."""

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.day='2026-11-23'
        self.docs={};catalog=[];master=[]
        for symbol,start in [('GOOD','2026-07-01'),('HELD','2026-07-03')]:
            isin='INE00000000'+('1' if symbol=='GOOD' else '2')
            url='https://nsearchives.nseindia.com/corporate/xbrl/'+symbol+'.xml'
            self.docs[url]=f'''<xbrl xmlns="http://www.xbrl.org/2003/instance" xmlns:f="urn:test">
              <context id="q"><period><startDate>2026-07-01</startDate><endDate>2026-09-30</endDate></period></context>
              <context id="p"><period><startDate>{start}</startDate><endDate>2026-09-30</endDate></period></context>
              <unit id="INR"><measure>iso4217:INR</measure></unit>
              <f:Symbol>{symbol}</f:Symbol><f:ISIN>{isin}</f:ISIN>
              <f:NatureOfReportStandaloneConsolidated>Consolidated</f:NatureOfReportStandaloneConsolidated>
              <f:RevenueFromOperations contextRef="q" unitRef="INR">100</f:RevenueFromOperations>
              <f:ProfitLossForPeriod contextRef="p" unitRef="INR">10</f:ProfitLossForPeriod></xbrl>'''.encode()
            catalog.append(dict(symbol=symbol,catalog_isin=isin,reporting_mode='consolidated',
                                source_url=url,available_at_utc='2026-11-10T09:00:00Z',
                                filed_period='Quarterly',period_end='2026-09-30'))
            master.append(dict(nse_current_symbol=symbol,isin=isin))
        pd.DataFrame(catalog).to_parquet(self.root/'catalog.parquet')
        pd.DataFrame(master).to_parquet(self.root/'universe.parquet')

    def collect(self):
        def get(url):
            raw=self.docs[url]
            return raw,dict(sha256=hashlib.sha256(raw).hexdigest(),first_retrieved_utc='2026-11-23T10:00:00Z')
        with patch('v13_symbol_identity.collect',return_value=[]),patch('v13_collection.Store.get',side_effect=get):
            return collect_financials(self.root,self.day,budget=8,workers=1,seconds=30)

    def test_one_disputed_source_is_retryable_and_never_marks_the_good_batch_failed(self):
        report=self.collect()
        self.assertEqual((report['validated_filing_versions'],report['quarantined_filing_versions']),(1,1))
        self.assertEqual(report['filing_batch_status'],'PARTIAL_SOURCE_CONFLICT')
        path=self.root/'financial_wide.parquet';initial=path.read_bytes()
        self.assertEqual(pd.read_parquet(path).symbol.tolist(),['GOOD'])
        journal=json.loads((self.root/'evidence/financial_retry_queue.json').read_text())
        key=next(iter(journal));self.assertIn('HELD.xml',key)
        self.assertEqual(journal[key]['category'],'DOCUMENT_REVIEW')
        self.assertEqual(journal[key]['contexts'][0]['conflicting_metadata'],
                         {'period_start':['2026-07-01','2026-07-03']})
        for name in ('accepted_filings.json','financial_parser_owner_v2.json'):
            markers=json.loads((self.root/name).read_text())
            self.assertEqual(len(markers),1);self.assertIn('GOOD.xml',markers[0])
        again=self.collect()
        self.assertEqual(again['requested'],0)
        self.assertEqual(again['retry_queue']['deferred_by_reason'],{'DOCUMENT_REVIEW':1})
        self.assertEqual(path.read_bytes(),initial)

    def test_failed_parquet_write_preserves_previous_bytes_and_does_not_advance_markers(self):
        path=self.root/'financial_wide.parquet'
        self.collect();old=path.read_bytes()
        for name in ('accepted_filings.json','financial_parser_owner_v2.json'):
            (self.root/name).write_text('[]')
        def fail(frame,destination,**kwargs):
            Path(destination).write_bytes(b'partial');raise OSError('injected write failure')
        with patch.object(pd.DataFrame,'to_parquet',new=fail),self.assertRaisesRegex(OSError,'injected write failure'):
            self.collect()
        self.assertEqual(path.read_bytes(),old)
        self.assertFalse(path.with_suffix('.parquet.tmp').exists())
        for name in ('accepted_filings.json','financial_parser_owner_v2.json'):
            self.assertEqual(json.loads((self.root/name).read_text()),[])
