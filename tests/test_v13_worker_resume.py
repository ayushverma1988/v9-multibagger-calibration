import hashlib,json,tempfile,unittest,zipfile
from pathlib import Path
from v13_worker_resume import select_recovery

class WorkerRecoverySelection(unittest.TestCase):
    def scenario(self,t,contents,bad=False,floor=0):
        p=Path(t);payloads={};artifacts={};downloads=[]
        for run_id,members in contents:
            path=p/f'{run_id}.zip'
            with zipfile.ZipFile(path,'w') as z:
                for name,body in members.items():z.writestr(name,json.dumps(body))
            raw=path.read_bytes();payloads[run_id]=raw
            artifacts[run_id]=[{'id':run_id,'name':f'v13-private-research-state-{run_id}',
              'expired':False,'digest':'sha256:'+('a'*64 if bad else hashlib.sha256(raw).hexdigest())}]
        def download(a,path):downloads.append(a['id']);Path(path).write_bytes(payloads[a['id']])
        runs=[{'id':n,'head_branch':'main'} for n,_ in contents]
        result=select_recovery(runs,lambda r:artifacts[r],download,p/'out',floor)
        return result,downloads,p/'out'/'previous_worker.zip'

    def test_published_newest_checkpoint_stops_before_older_delta(self):
        with tempfile.TemporaryDirectory() as t:
            r,calls,path=self.scenario(t,[(3,{'archive_receipt.json':{'snapshot':'new','manifest_sha256':'hash'}}),(2,{'v13_checkpoint_recovery.json':{}})])
            self.assertEqual(calls,[3]);self.assertFalse(path.exists());self.assertIn('PUBLISHED',r['status'])

    def test_pending_newest_delta_is_preserved(self):
        with tempfile.TemporaryDirectory() as t:
            r,calls,path=self.scenario(t,[(3,{'v13_checkpoint_recovery.json':{}}),(2,{'v13_checkpoint_recovery.json':{}})])
            self.assertEqual(calls,[3]);self.assertTrue(path.exists())

    def test_incomplete_artifact_is_skipped(self):
        with tempfile.TemporaryDirectory() as t:
            r,calls,path=self.scenario(t,[(3,{'latest_run.json':{}}),(2,{'v13_checkpoint_recovery.json':{}})])
            self.assertEqual(calls,[3,2]);self.assertEqual(r['run_id'],2)

    def test_corrupt_artifact_and_published_maintenance_floor(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaisesRegex(ValueError,'hash mismatch'):self.scenario(t,[(3,{'v13_checkpoint_recovery.json':{}})],bad=True)
            r,calls,path=self.scenario(t,[(3,{'v13_checkpoint_recovery.json':{}})],floor=3)
            self.assertEqual(calls,[]);self.assertFalse(path.exists())
