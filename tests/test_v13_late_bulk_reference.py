import json,tempfile,zipfile,io,hashlib,unittest
from pathlib import Path
from datetime import datetime,timezone
import pandas as pd
from v13_breadth_primary import url
from v13_late_bulk_reference import collect
class LateCollection(unittest.TestCase):
    def fixture(self,p,wanted,volume=10):
        day='2026-11-26';key='INE000000001'
        payload='TradDt,BizDt,Src,Sgmt,ISIN,TckrSymb,SctySrs,OpnPric,HghPric,LwPric,ClsPric,TtlTradgVol\n'+f'{day},{day},NSE,CM,{key},ABC,EQ,99,103,99,101,10\n'
        with zipfile.ZipFile(p/'nse.zip','w') as z:z.writestr('market.csv',payload)
        raw=(p/'nse.zip').read_bytes();(p/'primary.json').write_text(json.dumps(dict(source_asof=day,source_url=url(day),scope='NSE_LATE_REFERENCE_ONLY',sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),nse_rows=1,first_retrieved_utc='2026-11-27T01:00:00Z')))
        frame=pd.DataFrame([dict(date=wanted,isin=key,symbol='ABC',series='EQ',open=99,high=103,low=99,close=101,volume=volume,adj_close=101,adj_factor_cumulative=1)])
        buf=io.BytesIO();frame.to_parquet(buf,index=False);b=buf.getvalue();meta=[{'path':'prices_adjusted/nse_2026.parquet','lastCommit':{'id':'f'*40,'date':'2026-11-27T00:00:00Z'},'lfs':{'oid':hashlib.sha256(b).hexdigest()},'size':len(b)}]
        return lambda u,lim:json.dumps(meta).encode() if '/api/' in u else b
    def test_same_day_or_conflict(self):
        for volume,expected in [(10,1),(11,0)]:
            with tempfile.TemporaryDirectory() as d:
                p=Path(d);r=collect(p,self.fixture(p,'2026-11-26',volume))
                self.assertEqual((r['corroborated'],r['formal_scored_rows_added']),(expected,0))
    def test_stale_kept_on_own_date_and_missing_alternate_retains_raw(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);r=collect(p,self.fixture(p,'2026-11-25'));self.assertEqual(r['corroborated'],0)
            def unavailable(*args):raise OSError('unavailable')
            r=collect(p,unavailable);self.assertEqual(r['valid_primary'],1);self.assertEqual(r['status'],'ALTERNATE_UNAVAILABLE')
