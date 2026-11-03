import json,unittest,hashlib,tempfile
from pathlib import Path
import pandas as pd
from v12_3_BSE_current_refresh import merge_recent,run


class BSERefreshTests(unittest.TestCase):
    def test_previous_normalized_history_cannot_leak_into_a_rerun(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'securities';p.mkdir();(p/'old.parquet').write_bytes(b'old')
            with self.assertRaises(FileExistsError):run('unused','unused','unused','unused',d,'2026-10-09')

    def fixture(self):
        dates=pd.date_range('2026-10-05',periods=5)
        ref={'isin':'INE000000001','provider_symbol':'EXAMPLE.BO','provider_long_name':'Example Limited'}
        meta={'symbol':'EXAMPLE.BO','longName':'Example Limited','exchangeName':'BSE','currency':'INR','instrumentType':'EQUITY','exchangeTimezoneName':'Asia/Kolkata'}
        quote={'open':[100]*5,'high':[102]*5,'low':[99]*5,'close':[101]*5,'volume':[1000]*5}
        obj={'chart':{'result':[{'meta':meta,'timestamp':[int(d.tz_localize('Asia/Kolkata').timestamp()) for d in dates],'indicators':{'quote':[quote],'adjclose':[{'adjclose':[101]*5}]}}]}}
        old=pd.DataFrame({k:v[:4] for k,v in quote.items()});old['adj_close']=101;old['date']=dates[:4];old['symbol']='EXAMPLE.BO';old['isin']=ref['isin'];old['source_sha256']='old'
        return obj,ref,old

    def merge(self,obj,ref,old):
        raw=json.dumps(obj).encode();receipt={'sha256':hashlib.sha256(raw).hexdigest(),'first_retrieved_utc':'2026-10-11T00:00:00Z'}
        return merge_recent(raw,receipt,ref,old,'2026-10-09')

    def test_appends_only_overlap_verified_current_bar(self):
        obj,ref,old=self.fixture();merged,info=self.merge(obj,ref,old)
        self.assertEqual(len(merged),5);self.assertEqual(info['new_bars'],1)
        self.assertEqual(merged.source_sha256.iloc[0],'old')
        self.assertNotEqual(merged.source_sha256.iloc[-1],'old')

    def test_adjustment_revision_is_rejected(self):
        obj,ref,old=self.fixture();obj['chart']['result'][0]['indicators']['adjclose'][0]['adjclose'][1]=50
        with self.assertRaisesRegex(ValueError,'revision'):self.merge(obj,ref,old)

    def test_wrong_venue_is_rejected(self):
        obj,ref,old=self.fixture();obj['chart']['result'][0]['meta']['exchangeName']='NSE'
        with self.assertRaisesRegex(ValueError,'identity'):self.merge(obj,ref,old)

    def test_volume_revision_is_rejected(self):
        obj,ref,old=self.fixture();obj['chart']['result'][0]['indicators']['quote'][0]['volume'][0]=999
        with self.assertRaisesRegex(ValueError,'volume'):self.merge(obj,ref,old)

    def test_no_current_bar_does_not_become_a_current_quote(self):
        obj,ref,old=self.fixture();c=obj['chart']['result'][0];c['timestamp']=c['timestamp'][:4]
        c['indicators']['quote'][0]={k:v[:4] for k,v in c['indicators']['quote'][0].items()};c['indicators']['adjclose'][0]['adjclose']=[101]*4
        with self.assertRaisesRegex(ValueError,'unavailable'):self.merge(obj,ref,old)


if __name__=='__main__':unittest.main()
