import hashlib,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
import requests
from v13_checkpoint import VerifiedObjectDownloader,object_path

class Response:
    def __init__(self,status,data=b'original'):self.status_code=status;self.headers={};self.data=data
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def iter_content(self,size):yield self.data

class TransientArchiveRead(unittest.TestCase):
    def download(self,outcomes):
        class Session:
            calls=0
            def get(self,*args,**kwargs):
                self.calls+=1;value=outcomes.pop(0)
                if isinstance(value,Exception):raise value
                return value
        session=Session();raw=b'original';digest=hashlib.sha256(raw).hexdigest()
        stub=types.ModuleType('huggingface_hub');stub.hf_hub_url=lambda *a,**k:'https://example.invalid/object'
        with tempfile.TemporaryDirectory() as t,patch.dict(sys.modules,{'huggingface_hub':stub}),patch.object(requests,'Session',return_value=session),patch('v13_checkpoint.time.sleep'):
            item=dict(sha256=digest,bytes=len(raw),object_path=object_path(digest))
            loader=VerifiedObjectDownloader([item],Path(t),'test-token',requests_per_second=1e6)
            try:return loader(item['object_path']).read_bytes(),session.calls,None
            except Exception as exc:return None,session.calls,exc
    def test_temporary_gateway_failure_retries_and_verifies_original(self):
        data,calls,error=self.download([Response(504),Response(200)])
        self.assertIsNone(error);self.assertEqual(data,b'original');self.assertEqual(calls,2)
    def test_connection_failure_retries_without_accepting_partial_source(self):
        data,calls,error=self.download([requests.ConnectionError('temporary'),Response(200)])
        self.assertIsNone(error);self.assertEqual(data,b'original');self.assertEqual(calls,2)
    def test_permission_and_content_integrity_failures_never_retry_or_pass(self):
        for response,kind in [(Response(403),RuntimeError),(Response(200,b'tampered'),ValueError)]:
            data,calls,error=self.download([response])
            self.assertIsNone(data);self.assertEqual(calls,1);self.assertIsInstance(error,kind)

if __name__=='__main__':unittest.main()
