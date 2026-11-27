import base64,hashlib,io,json,tempfile,unittest,zipfile
from pathlib import Path
from unittest.mock import patch
import pandas as pd
from v13_breadth_handoff import recover,PUBLIC
from v13_breadth_fallback import SOURCE_BASE,NSE_CALENDAR_URL,NSE_ACTIONS,load
from v13_breadth_primary import url
import test_v13_breadth_fallback as fixtures

class PublicHandoff(unittest.TestCase):
    def setUp(self):
        self.day='2026-11-18'; self.time='2026-11-18T14:30:00Z';self.raw=b'test adjusted parquet'
        def obj(b,u):return dict(source_url=u,sha256=hashlib.sha256(b).hexdigest(),bytes=len(b),first_retrieved_utc=self.time)
        raw=io.BytesIO()
        with zipfile.ZipFile(raw,'w') as z:z.writestr('market.csv','TradDt,BizDt,Src,Sgmt,ISIN,TckrSymb,SctySrs,OpnPric,HghPric,LwPric,ClsPric,TtlTradgVol\n2026-11-18,2026-11-18,NSE,CM,INE000000001,ABC,EQ,95,101,94,100,1000\n')
        orig={'nse_original':raw.getvalue(),'nse_calendar':b'{"CM":[{"tradingDate":"26-Jan-2026"}]}','nse_corporate_actions':b'[]'}
        rev='1'*40;r={**obj(self.raw,SOURCE_BASE+rev+'/prices_adjusted/nse_2026.parquet?download=true'),'source_asof':self.day,'adjusted_through_asof':self.day,'publisher_commit_utc':self.time,'publisher_revision':rev}
        for k,u in zip(PUBLIC,[url(self.day),NSE_CALENDAR_URL,NSE_ACTIONS.format(day='18-11-2026')]):r[k]=obj(orig[k],u)
        r['nse_corporate_actions'].update(no_action_interval_certified=False,rows=0)
        self.packet={'contract':'V13_PUBLIC_CAPTURE_HANDOFF_1','receipt':r,'originals_b64':{k:base64.b64encode(v).decode() for k,v in orig.items()}}
    def attempt(self,folder,witness='2026-11-18T15:30:00Z',fetch=None):
        return recover(self.packet,witness,'2'*40,folder,fetch or (lambda *args:self.raw),clock=lambda:pd.Timestamp('2026-11-19T04:00:00Z'))
    def test_delayed_runner_preserves_actual_earlier_clock(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'out';r=self.attempt(p)
            self.assertEqual(json.loads((p/'receipt.json').read_text()),self.packet['receipt'])
            self.assertGreater(pd.Timestamp(r['reconstructed_utc']),pd.Timestamp(r['github_run_created_utc']))
            with self.assertRaisesRegex(ValueError,'replace'):self.attempt(p)
    def test_missing_late_early_or_future_witness_is_rejected(self):
        for stamp in ['2026-11-18T15:15:00','2026-11-18T16:31:00Z','2026-11-18T12:00:00Z','2026-11-19T05:00:00Z']:
            with self.subTest(stamp=stamp),tempfile.TemporaryDirectory() as tmp,self.assertRaises(ValueError):
                self.attempt(Path(tmp)/'out',stamp)
    def test_changed_public_bytes_never_pass(self):
        with tempfile.TemporaryDirectory() as tmp,self.assertRaisesRegex(ValueError,'data changed'):
            self.attempt(Path(tmp)/'out',fetch=lambda *a:b'changed')
    def test_edited_actions_and_arbitrary_host_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.packet['originals_b64']['nse_corporate_actions']='e30='
            with self.assertRaisesRegex(ValueError,'Wrong public'):self.attempt(Path(tmp)/'out')
            self.setUp();self.packet['receipt']['source_url']='https://invalid.local/foreign'
            with self.assertRaisesRegex(ValueError,'Unexpected public data URL'):self.attempt(Path(tmp)/'out')

class TimelyReplacement(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.BreadthFallbackTests();self.fixture.setUp()
    def scan(self,tmp,conflict=False):
        from v13_runner import current_frame
        from v13_registration_time import registration_time_gate
        base=Path(tmp);root=base/'state';p=root/'history/securities';p.mkdir(parents=True);off=root/'official';off.mkdir()
        f=self.fixture;bars=f.alt.copy();bars['source_sha256']='a'*64
        if conflict:bars.loc[bars.index[-1],'volume']+=2
        bars.to_parquet(p/'ABC.parquet');pd.DataFrame([{'isin':'INE000000001','nse_current_symbol':'ABC','nse_current_series':'EQ'}]).to_parquet(root/'universe.parquet')
        snap=base/'v13_public_breadth_snapshot';snap.mkdir();data=snap/'nse_adjusted.parquet';f.alt.to_parquet(data,index=False)
        t='2026-11-18T14:20:00Z';rev='1'*40;r={'source_asof':'2026-11-18','publisher_revision':rev,'publisher_commit_utc':'2026-11-18T14:00:00Z','first_retrieved_utc':t,'source_url':SOURCE_BASE+rev+'/prices_adjusted/nse_2026.parquet?download=true','bytes':data.stat().st_size,'sha256':hashlib.sha256(data.read_bytes()).hexdigest()}
        f.add_primary(snap,r,'2026-11-18',t);f.official.assign(source_sha256=r['nse_original']['sha256']).to_parquet(off/'2026-11-18_official_market.parquet',index=False);(snap/'receipt.json').write_text(json.dumps(r))
        frame,h=current_frame(root,'2026-11-18')
        self.assertEqual(h['INE000000001'].iloc[-1]['source_sha256'],'a'*64)
        if conflict:self.assertTrue(frame.empty)
        else:
            self.assertTrue(registration_time_gate(frame)['passed']);self.assertEqual(len(frame),1)
            self.assertEqual(frame.iloc[0].scoring_price_history_source,'SINGLE_SOURCE_ADJUSTED_SCORING_ONLY')
    def test_late_current_provider_does_not_crowd_out_timely_price(self):
        with tempfile.TemporaryDirectory() as tmp:self.scan(tmp)
    def test_known_current_price_conflict_held(self):
        with tempfile.TemporaryDirectory() as tmp:self.scan(tmp,True)
