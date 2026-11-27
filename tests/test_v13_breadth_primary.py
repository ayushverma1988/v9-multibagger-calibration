import io,json,tempfile,unittest,zipfile
from pathlib import Path
from datetime import datetime,timezone
from unittest.mock import Mock
from urllib.error import HTTPError
from v13_breadth_primary import retain,read_primary,late_reference,url
def clock(h,m=0):return datetime(2026,11,26,h,m,tzinfo=timezone.utc)
def bar(day='2026-11-26'):
    f=io.BytesIO()
    with zipfile.ZipFile(f,'w') as z:
        z.writestr('day.csv','TradDt,BizDt,Sgmt,Src,ISIN,TckrSymb,SctySrs,OpnPric,HghPric,LwPric,ClsPric,TtlTradgVol\n'+day+','+day+',CM,NSE,INE000000001,A,EQ,10,11,9,10,18\n')
    return f.getvalue()
class TestPrimaryEvidence(unittest.TestCase):
 def setUp(self):
    self.t=tempfile.TemporaryDirectory();self.addCleanup(self.t.cleanup);self.p=Path(self.t.name);self.day='2026-11-26'
 def test_primary_survives_provider_failure_and_no_second_exchange_fetch(self):
    fetch=Mock(return_value=bar());one=retain(self.day,self.p,fetch,clock=lambda:clock(11))
    two=retain(self.day,self.p,fetch,clock=lambda:clock(14))
    self.assertEqual(one,two);self.assertEqual(fetch.call_count,1);self.assertFalse((self.p/'receipt.json').exists())
 def test_fetch_crossing_cutoff_never_gets_in_window_receipt(self):
    with self.assertRaisesRegex(ValueError,'outside'):retain(self.day,self.p,lambda *_:bar(),clock=Mock(side_effect=[clock(16,29),clock(16,31)]))
    self.assertFalse((self.p/'primary.json').exists())
 def test_wrong_day_or_tampered_bytes_are_not_reused(self):
    with self.assertRaisesRegex(ValueError,'date mismatch'):retain(self.day,self.p,lambda *_:bar('2026-11-25'),clock=lambda:clock(11))
    retain(self.day,self.p,lambda *_:bar(),clock=lambda:clock(11));(self.p/'nse.zip').write_bytes(bar('2026-11-25'))
    with self.assertRaises(ValueError):read_primary(self.day,self.p,lambda:clock(12))
 def test_late_never_eligible_and_cannot_overwrite(self):
    late=late_reference(self.p,lambda *_:bar(),clock=lambda:clock(19));self.assertEqual(late['scope'],'NSE_LATE_REFERENCE_ONLY')
    with self.assertRaises(ValueError):read_primary(self.day,self.p,lambda:clock(20))
    with self.assertRaisesRegex(ValueError,'replace'):retain(self.day,self.p,lambda *_:bar(),timely=False,clock=lambda:clock(20))
 def test_404_can_move_back_but_403_and_wrong_session_cannot(self):
    fetch=Mock(side_effect=[HTTPError(url(self.day),404,'not found',{},None),bar('2026-11-25')])
    self.assertEqual(late_reference(self.p,fetch,clock=lambda:clock(19))['source_asof'],'2026-11-25')
    failure=Mock(side_effect=HTTPError(url(self.day),403,'blocked',{},None))
    with self.assertRaises(HTTPError):late_reference(self.p,failure,clock=lambda:clock(19))
    self.assertEqual(failure.call_count,1)
 def test_future_cache_clock_rejected(self):
    retain(self.day,self.p,lambda *_:bar(),clock=lambda:clock(13))
    with self.assertRaises(ValueError):read_primary(self.day,self.p,lambda:clock(12))
 def test_full_receipt_keeps_original_clock_after_provider_recovery(self):
    import pandas as pd
    from unittest.mock import patch
    from v13_breadth_capture import capture
    raw=bar();retain(self.day,self.p/'partial_primary',lambda *_:raw,clock=lambda:clock(11))
    rev='1'*40
    tree=json.dumps([{'path':'prices_adjusted/nse_2026.parquet','lastCommit':{'id':rev,'date':clock(12).isoformat()}}]).encode()
    parquet=pd.DataFrame({'date':pd.bdate_range(end=self.day,periods=170)}).to_parquet(index=False)
    def get(address,bound):
        if '/tree/' in address:return tree
        if '/resolve/' in address:return parquet
        if 'holiday-master' in address:return b'{"CM":[{"tradingDate":"26-Jan-2026"}]}'
        if 'corporateActions?' in address:return b'[]'
        raise AssertionError('Unexpected second download of cached original: '+address)
    with patch('v13_breadth_capture.get',get),patch('v13_breadth_capture.datetime') as cl,patch('v13_breadth_capture.read_primary',side_effect=lambda d,p:read_primary(d,p,clock=lambda:clock(14))):
        cl.now.return_value=clock(14)
        rec=capture(self.day,self.p)
    self.assertEqual(rec['nse_original']['first_retrieved_utc'],clock(11).isoformat())
    self.assertEqual(rec['first_retrieved_utc'],clock(14).isoformat())
    self.assertEqual((self.p/'nse_source_session.zip').read_bytes(),raw)
