import hashlib
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import types
import unittest
from unittest.mock import patch

import v13_checkpoint as checkpoint
import v13_checkpoint_writes as writes


class HttpFailure(Exception):
    def __init__(self, code, headers=None):
        self.response=types.SimpleNamespace(status_code=code, headers=headers or {})
        super().__init__('test HTTP '+str(code))


class Operation:
    instances=[]
    def __init__(self,path_in_repo,path_or_fileobj):
        self.path_in_repo=path_in_repo;self.path_or_fileobj=path_or_fileobj
        self.used=False;self.instances.append(self)


class Api:
    def __init__(self,remote=None,failures=None,lfs=False):
        self.remote=remote if remote is not None else {}
        self.failures=list(failures or []);self.calls=[];self.inspections=0;self.lfs=lfs
    def repo_info(self,*a,**k):return types.SimpleNamespace(private=True)
    def get_paths_info(self,**kwargs):
        self.inspections+=1;out=[]
        for p in kwargs['paths']:
            if p not in self.remote:continue
            b=self.remote[p]
            out.append(types.SimpleNamespace(path=p,size=len(b),
                blob_id=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest(),
                lfs=types.SimpleNamespace(sha256=hashlib.sha256(b).hexdigest()) if self.lfs else None))
        return out
    def create_commit(self,**kwargs):
        ops=kwargs['operations'];self.calls.append([o.path_in_repo for o in ops])
        for o in ops:
            if o.used:raise AssertionError('Mutated operation reused')
            o.used=True
        mode=self.failures.pop(0) if self.failures else 'success'
        if mode=='split' and len(ops)>1:
            self.failures.insert(0,'split');raise HttpFailure(504)
        if mode=='post504' or mode=='success' or mode=='split':
            for o in ops:self.remote[o.path_in_repo]=Path(o.path_or_fileobj).read_bytes()
        if mode=='post504':raise HttpFailure(504)
        if isinstance(mode,int):raise HttpFailure(mode,{'Retry-After':'5'} if mode==429 else {})


class CheckpointWrites(unittest.TestCase):
    def file_spec(self,folder,name='a',raw=b'new evidence'):
        p=Path(folder)/name;p.write_bytes(raw)
        digest=hashlib.sha256(raw).hexdigest()
        return writes.spec(checkpoint.object_path(digest),p)
    def write(self,api,specs,**kwargs):
        with patch.object(writes.time,'sleep'):
            writes.write(api,checkpoint.ARCHIVE,specs,'test checkpoint',Operation,**kwargs)
    def hf(self,api):
        stub=types.ModuleType('huggingface_hub')
        stub.HfApi=lambda **k:api;stub.CommitOperationAdd=Operation
        return patch.dict(sys.modules,{'huggingface_hub':stub})

    def test_gateway_before_commit_retries_with_new_operations(self):
        with tempfile.TemporaryDirectory() as t:
            s=self.file_spec(t);api=Api(failures=[504])
            self.write(api,[s]);self.assertEqual(len(api.calls),2)
            self.assertEqual(api.remote[s['remote']],b'new evidence')

    def test_lost_success_reply_is_verified_without_duplicate_commit(self):
        for lfs in (False,True):
            with self.subTest(lfs=lfs),tempfile.TemporaryDirectory() as t:
                s=self.file_spec(t);api=Api(failures=['post504'],lfs=lfs)
                confirmed=[];self.write(api,[s],confirmed=lambda xs:confirmed.extend(xs))
                self.assertEqual(len(api.calls),1);self.assertEqual(len(confirmed),1)

    def test_gateway_batch_is_split_and_all_original_bytes_arrive(self):
        with tempfile.TemporaryDirectory() as t:
            specs=[self.file_spec(t,str(i),str(i).encode()) for i in range(5)]
            api=Api(failures=['split']);self.write(api,specs)
            self.assertEqual(len(api.remote),5)
            self.assertGreater(len(api.calls),1)
            for s in specs:self.assertEqual(hashlib.sha256(api.remote[s['remote']]).hexdigest(),s['sha256'])

    def test_rate_limit_honors_retry_after(self):
        with tempfile.TemporaryDirectory() as t,patch.object(writes.time,'sleep') as wait:
            s=self.file_spec(t);api=Api(failures=[429])
            writes.write(api,checkpoint.ARCHIVE,[s],'test',Operation)
            wait.assert_called_once_with(5.);self.assertEqual(len(api.calls),2)

    def test_permission_failure_is_never_retried(self):
        with tempfile.TemporaryDirectory() as t:
            s=self.file_spec(t);api=Api(failures=[403])
            with self.assertRaises(HttpFailure):self.write(api,[s])
            self.assertEqual(len(api.calls),1);self.assertEqual(api.inspections,0)

    def test_persistent_gateway_has_bounded_attempts(self):
        with tempfile.TemporaryDirectory() as t:
            s=self.file_spec(t);api=Api(failures=[504]*10)
            with self.assertRaises(writes.CheckpointTemporaryError):self.write(api,[s])
            self.assertEqual(len(api.calls),writes.MAX_ATTEMPTS)

    def test_source_change_is_not_retried_or_uploaded(self):
        with tempfile.TemporaryDirectory() as t:
            s=self.file_spec(t);Path(s['file']).write_bytes(b'changed source')
            api=Api()
            with self.assertRaises(ValueError):self.write(api,[s])
            self.assertEqual(api.calls,[])

    def test_atomic_manifest_publication_is_never_split(self):
        with tempfile.TemporaryDirectory() as t:
            a=self.file_spec(t,'a',b'manifest');b=dict(a,remote='v13/state/latest_manifest.json')
            api=Api(failures=[504]);self.write(api,[a,b],split=False)
            self.assertEqual([len(c) for c in api.calls],[2,2])

    def test_unreadable_postcondition_cannot_be_accepted_as_success(self):
        with tempfile.TemporaryDirectory() as t:
            a=self.file_spec(t);api=Api(failures=[504])
            api.get_paths_info=lambda **k:(_ for _ in ()).throw(HttpFailure(504))
            with self.assertRaises(writes.CheckpointTemporaryError):self.write(api,[a])
            self.assertEqual(len(api.calls),1)

    def test_operation_count_is_bounded_and_frozen_bytes_preserved(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)/'state';root.mkdir()
            frozen=b'fixed probabilities unchanged';(root/'frozen.json').write_bytes(frozen)
            for i in range(122):(root/str(i)).write_bytes(str(i).encode())
            api=Api()
            with self.hf(api),patch.dict('os.environ',{'HF_ARCHIVE_TOKEN':'test'}):
                receipt=checkpoint.save(root)
            self.assertLessEqual(max(map(len,api.calls)),50)
            self.assertEqual((root/'frozen.json').read_bytes(),frozen)
            self.assertEqual(receipt['maximum_object_commit_files'],50)

    def test_failed_save_keeps_old_pointer_and_preserves_complete_delta(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)/'state';root.mkdir();(root/'source.bin').write_bytes(b'original new source')
            old=json.dumps({'format':'V13-content-addressed-checkpoint-1','files':[]}).encode()
            (root/'remote_manifest.json').write_bytes(old)
            api=Api(remote={'v13/state/latest_manifest.json':old},failures=[403])
            with self.hf(api),patch.dict('os.environ',{'HF_ARCHIVE_TOKEN':'test'}):
                with self.assertRaises(HttpFailure):checkpoint.save(root)
            self.assertEqual(api.remote['v13/state/latest_manifest.json'],old)
            self.assertEqual((root/'remote_manifest.json').read_bytes(),old)
            p=Path(t)/'v13_checkpoint_recovery.tar.gz'
            desc=json.loads((Path(t)/'v13_checkpoint_recovery.json').read_text())
            self.assertEqual(desc['sha256'],hashlib.sha256(p.read_bytes()).hexdigest())
            with tarfile.open(p,'r:gz') as tar:
                self.assertEqual(tar.extractfile('base_manifest.json').read(),old)
                plan=json.load(tar.extractfile('planned_manifest.json'))
                self.assertEqual(len(plan['files']),1)
                for item in plan['files']:
                    raw=tar.extractfile('objects/'+item['sha256']+'.bin').read()
                    self.assertEqual(hashlib.sha256(raw).hexdigest(),item['sha256'])

    def test_confirmed_batches_resume_without_uploading_them_again(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)/'state';root.mkdir()
            for i in range(51):(root/str(i)).write_bytes(str(i).encode())
            api=Api(failures=['success',403])
            with self.hf(api),patch.dict('os.environ',{'HF_ARCHIVE_TOKEN':'test'}):
                with self.assertRaises(HttpFailure):checkpoint.save(root)
                previous=len(api.calls)
                receipt=checkpoint.save(root)
            self.assertEqual(receipt['resumed_confirmed_objects'],50)
            self.assertEqual(len(api.calls)-previous,3)
            self.assertFalse((Path(t)/'v13_checkpoint_recovery.tar.gz').exists())

    def test_recovery_failure_does_not_hide_archive_error(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)/'state';root.mkdir();(root/'new').write_bytes(b'new')
            api=Api(failures=[403])
            original=writes.recovery_delta
            calls=[0]
            def fail_second(*args):
                calls[0]+=1
                if calls[0]>1:raise OSError
                return original(*args)
            with self.hf(api),patch.dict('os.environ',{'HF_ARCHIVE_TOKEN':'test'}),patch.object(writes,'recovery_delta',side_effect=fail_second):
                with self.assertRaises(HttpFailure):checkpoint.save(root)
            error=json.loads((Path(t)/'v13_checkpoint_recovery_error.json').read_text())
            self.assertEqual(error['archive_error_type'],'HttpFailure')

    def test_upload_journal_from_another_archive_is_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'progress.json';p.write_text(json.dumps({'format':'V13-confirmed-object-uploads-1','archive':'other'}))
            with self.assertRaises(ValueError):writes.UploadProgress(p,checkpoint.ARCHIVE)


if __name__=='__main__':unittest.main()
