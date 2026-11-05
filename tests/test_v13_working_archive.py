import hashlib,json,os,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
import v13_checkpoint as checkpoint
from v13_archive_codec import archive_download
from v13_working_archive import FORMAT,CODEC,validate_core,identities

class MemoryArchive:
    def __init__(self):self.remote={};self.head=0;self.fail_pointer=False;self.calls=[]
    def repo_info(self,*a,**k):return types.SimpleNamespace(private=True,sha=str(self.head))
    def create_commit(self,*a,**kw):
        assert kw.get('parent_commit',str(self.head))==str(self.head)
        ops=list(kw['operations']);self.calls.append([o.path_in_repo for o in ops])
        if self.fail_pointer and any(o.path_in_repo=='v13/state/latest_manifest.json' for o in ops):raise ValueError('Pointer rejected')
        for o in ops:self.remote[o.path_in_repo]=Path(o.path_or_fileobj).read_bytes()
        self.head+=1;return types.SimpleNamespace(oid=str(self.head))
    def get_paths_info(self,*a,**k):
        result=[]
        for key in k['paths']:
            if key not in self.remote:continue
            raw=self.remote[key]
            result.append(types.SimpleNamespace(path=key,size=len(raw),lfs=types.SimpleNamespace(sha256=hashlib.sha256(raw).hexdigest()),blob_id='fixture'))
        return result

class WorkingArchiveContracts(unittest.TestCase):
    def setup_state(self,p):
        root=p/'state';root.mkdir();(root/'filing_raw').mkdir()
        values={'active_heads.joblib':b'unchanged frozen model\x00'*100,
          'challenger/research_heads.joblib':b'unchanged challenger\x00'*100,
          'prospective_observations.parquet':b'exact probability 0.12345678901234566'*100,
          'latest_run.json':b'{"asof":"2026-11-04","workflow_run":"41921839880"}',
          'filing_raw/original.bin':b'approved retired original body'}
        for name,raw in values.items():
            target=root/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
        old=checkpoint.manifest(root);(root/'remote_manifest.json').write_text(json.dumps(old,sort_keys=True,indent=2))
        return root,values,old
    def download(self,api,p):
        def fetch(repo,key,**kw):
            if key not in api.remote:
                from huggingface_hub.errors import EntryNotFoundError
                raise EntryNotFoundError('missing fixture')
            path=p/'http'/str(api.head)/hashlib.sha256(key.encode()).hexdigest();path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(api.remote[key]);return str(path)
        return fetch
    def environment(self,p):
        return patch.dict(os.environ,{'HF_ARCHIVE_TOKEN':'fixture','V13_COMPACT_ARCHIVE':'1','V13_WORKING_CORE_ARCHIVE':'1',
          'V13_STORAGE_STAGING_DIR':str(p/'encoded'),'V13_DECODED_CACHE':str(p/'decoded'),'V13_CORE_CACHE':str(p/'members')})

    def test_successful_save_restores_every_working_byte_without_uploading_raw_bodies(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);root,values,old=self.setup_state(p);api=MemoryArchive()
            with self.environment(p),patch('huggingface_hub.HfApi',return_value=api),patch('huggingface_hub.hf_hub_download',side_effect=self.download(api,p)):
                receipt=checkpoint.save(root)
                data=json.loads((root/'remote_manifest.json').read_text())
                self.assertEqual(data['format'],FORMAT);self.assertFalse(any(r['raw_source'] for r in data['files']))
                self.assertFalse(any('/object_shards/' in k for k in api.remote))
                self.assertTrue(checkpoint.restore(p/'fresh'))
                for name,raw in values.items():
                    if name.startswith('filing_raw/'):
                        self.assertFalse((p/'fresh'/name).exists());self.assertEqual((root/name).read_bytes(),raw)
                    else:self.assertEqual((p/'fresh'/name).read_bytes(),raw)
                self.assertTrue((p/'fresh'/'archive_provenance'/'retention_policy.json').exists())
                self.assertLess(receipt['derived_bundle_bytes'],1_000_000_000)
                item=next(r for r in data['files'] if r['path']=='active_heads.joblib')
                self.assertEqual(checkpoint.object_bytes(item,lambda key:self.download(api,p)('',key)),values['active_heads.joblib'])

    def test_failed_new_pointer_retains_old_checkpoint_and_recoverable_new_values(self):
        from v13_recovery_restore import restore_delta
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);root,values,old=self.setup_state(p);api=MemoryArchive();download=self.download(api,p)
            with self.environment(p),patch('huggingface_hub.HfApi',return_value=api),patch('huggingface_hub.hf_hub_download',side_effect=download):
                checkpoint.save(root);old_pointer=api.remote['v13/state/latest_manifest.json']
                (root/'prospective_observations.parquet').write_bytes(b'new frozen records with exact values')
                api.fail_pointer=True
                with self.assertRaisesRegex(ValueError,'Pointer rejected'):checkpoint.save(root)
                self.assertEqual(api.remote['v13/state/latest_manifest.json'],old_pointer)
                self.assertTrue((p/'v13_checkpoint_recovery.tar.gz').exists())
                restore_delta(p/'v13_checkpoint_recovery.tar.gz',p/'recovered',lambda key:download('',key))
                self.assertEqual((p/'recovered'/'prospective_observations.parquet').read_bytes(),b'new frozen records with exact values')
                self.assertEqual((p/'recovered'/'active_heads.joblib').read_bytes(),values['active_heads.joblib'])

    def test_invalid_or_oversized_core_reference_is_rejected(self):
        valid={'sha256':'a'*64,'object_path':'v13/working_core/'+'a'*64+'.zip','bytes':400,'source_manifest_sha256':'b'*64}
        validate_core(valid)
        for change in ({'bytes':1_000_000_001},{'object_path':'../outside.zip'},{'source_manifest_sha256':'bad'}):
            with self.assertRaises(ValueError):validate_core({**valid,**change})

    def test_core_manifest_filter_does_not_remove_local_source_inputs(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);root,values,old=self.setup_state(p)
            with self.environment(p):new=checkpoint.manifest(root)
            self.assertEqual(set(identities(new)),set(name for name in values if not name.startswith('filing_raw/')))
            self.assertEqual((root/'filing_raw'/'original.bin').read_bytes(),values['filing_raw/original.bin'])

    def test_retirement_plan_keeps_current_core_and_latest_snapshot(self):
        from v13_working_archive import retirement_plan
        class Folder:
            def __init__(self,path):self.path=path
        class File:
            def __init__(self,path):self.path=path
        data={'files':[],'derived_state_bundle':{'codec':CODEC,'sha256':'a'*64,
          'object_path':'v13/working_core/'+'a'*64+'.zip','bytes':1,'source_manifest_sha256':'b'*64}}
        current=hashlib.sha256(json.dumps(data,sort_keys=True,indent=2).encode()).hexdigest()
        paths={'v13':[Folder('v13/object_shards'),Folder('v13/working_core'),Folder('v13/state'),Folder('v13/core_dates')],
          'v13/working_core':[File(data['derived_state_bundle']['object_path']),File('v13/working_core/old.zip')],
          'v13/state':[File('v13/state/latest_manifest.json'),File('v13/state/snapshots/clock_'+current[:12]+'.json'),File('v13/state/snapshots/old.json')]}
        api=types.SimpleNamespace(list_repo_tree=lambda *a,**k:paths[k['path_in_repo']])
        with patch('huggingface_hub.hf_api.RepoFolder',Folder):remove=retirement_plan(api,data)
        self.assertIn(('v13/object_shards',True),remove)
        self.assertIn(('v13/working_core/old.zip',False),remove)
        self.assertIn(('v13/state/snapshots/old.json',False),remove)
        self.assertFalse(any(x[0]==data['derived_state_bundle']['object_path'] or x[0]=='v13/state/latest_manifest.json' for x in remove))

class WorkingMigrationBridge(unittest.TestCase):
    def test_migration_bridge_requires_exact_original_working_bytes(self):
        from v13_working_archive import migration_matches
        row={'path':'frozen.parquet','sha256':'a'*64,'bytes':15,'raw_source':False}
        source=json.dumps({'files':[row,{'path':'raw/document','sha256':'b'*64,'bytes':99,'raw_source':True}]}).encode()
        current={'format':FORMAT,'migration_source_manifest_sha256':hashlib.sha256(source).hexdigest(),'files':[row,
          {'path':'archive_provenance/policy.json','sha256':'c'*64,'bytes':20,'raw_source':False}]}
        self.assertTrue(migration_matches(source,current))
        self.assertFalse(migration_matches(source,{**current,'files':[{**row,'sha256':'d'*64}]}))
        self.assertFalse(migration_matches(source,{**current,'migration_source_manifest_sha256':'e'*64}))
        self.assertFalse(migration_matches(source,{**current,'files':[row,{'path':'unreviewed_model.joblib','sha256':'c'*64,'bytes':20,'raw_source':False}]}))

class WorkingRetirementContract(unittest.TestCase):
    setup_state=WorkingArchiveContracts.setup_state
    download=WorkingArchiveContracts.download
    environment=WorkingArchiveContracts.environment
    def test_retirement_purges_old_storage_but_never_live_core_or_metadata(self):
        from v13_working_archive import retire
        from v13_archive_cleanup import OWNER
        class Folder:
            def __init__(self,path):self.path=path
        class Archive(MemoryArchive):
            def __init__(self):super().__init__();self.lfs={};self.purged=[]
            def repo_info(self,*a,**k):return types.SimpleNamespace(private=True,sha=str(self.head),used_storage=sum(len(x) for x in self.lfs.values()))
            def whoami(self):return {'name':OWNER}
            def list_repo_refs(self,*a,**k):return types.SimpleNamespace(branches=[types.SimpleNamespace(name='main')],tags=[])
            def create_commit(self,*a,**kw):
                assert kw.get('parent_commit',str(self.head))==str(self.head)
                for o in kw['operations']:
                    key=o.path_in_repo.rstrip('/')
                    if hasattr(o,'path_or_fileobj'):
                        raw=Path(o.path_or_fileobj).read_bytes();self.remote[key]=raw;self.lfs[hashlib.sha256(raw).hexdigest()]=raw
                    else:
                        for old in list(self.remote):
                            if old==key or o.is_folder and old.startswith(key+'/'):del self.remote[old]
                self.head+=1;return types.SimpleNamespace(oid=str(self.head))
            def list_repo_tree(self,*a,**k):
                prefix=k.get('path_in_repo','');prefix=prefix+'/' if prefix else '';out=[];dirs=set()
                for key,raw in sorted(self.remote.items()):
                    if not key.startswith(prefix):continue
                    rel=key[len(prefix):]
                    if '/' in rel and not k.get('recursive'):
                        dirs.add(prefix+rel.split('/')[0]);continue
                    out.append(types.SimpleNamespace(path=key,size=len(raw),blob_id='fixture',lfs=types.SimpleNamespace(sha256=hashlib.sha256(raw).hexdigest())))
                return out+[Folder(n) for n in sorted(dirs)]
            def super_squash_history(self,*a,**k):self.head+=1
            def list_lfs_files(self,*a,**k):return [types.SimpleNamespace(file_oid='sha256:'+sha,size=len(raw)) for sha,raw in self.lfs.items()]
            def permanently_delete_lfs_files(self,repo,items,**kwargs):
                live={hashlib.sha256(raw).hexdigest() for raw in self.remote.values()}
                for item in items:
                    sha=item.file_oid.removeprefix('sha256:');assert sha not in live
                    self.lfs.pop(sha,None);self.purged.append(sha)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);root,values,old=self.setup_state(p);api=Archive();download=self.download(api,p)
            for key,raw in {'v13/object_shards/old/raw.bin':b'approved retired evidence',
              'v13/storage_catalog/old/index.json':b'old catalog','v13/state/snapshots/old.json':b'old snapshot'}.items():
                api.remote[key]=raw;api.lfs[hashlib.sha256(raw).hexdigest()]=raw
            with self.environment(p),patch('huggingface_hub.HfApi',return_value=api),patch('huggingface_hub.hf_hub_download',side_effect=download),patch('huggingface_hub.hf_api.RepoFolder',Folder):
                checkpoint.save(root);pointer=api.remote['v13/state/latest_manifest.json']
                result=retire(p/'receipt.json',purge=True)
                self.assertTrue(result['fresh_restart_before_and_after_retirement_verified'])
                self.assertGreater(result['retired_lfs_objects_purged'],0)
                self.assertFalse(any(n.startswith('v13/object_shards/') for n in api.remote))
                self.assertFalse(any(n.startswith('v13/storage_catalog/') for n in api.remote))
                self.assertEqual(api.remote['v13/state/latest_manifest.json'],pointer)
                self.assertTrue(checkpoint.restore(p/'after_retirement'))
                self.assertEqual((p/'after_retirement'/'active_heads.joblib').read_bytes(),values['active_heads.joblib'])

class WorkingCoreBudget(unittest.TestCase):
    setup_state=WorkingArchiveContracts.setup_state
    download=WorkingArchiveContracts.download
    environment=WorkingArchiveContracts.environment
    def test_budget_overflow_keeps_model_values_and_creates_verified_recovery(self):
        from v13_checkpoint_writes import CheckpointTemporaryError
        from v13_artifact_checkpoint import save_with_quota_fallback
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);root,values,old=self.setup_state(p);api=MemoryArchive()
            with self.environment(p),patch.dict(os.environ,{'V13_ARTIFACT_CHECKPOINT_FALLBACK':'1'}),patch('v13_working_archive.BUDGET',10_000_010),patch('huggingface_hub.HfApi',return_value=api):
                result=save_with_quota_fallback(root,checkpoint.save)
            self.assertFalse(result['remote_publication_confirmed'])
            self.assertTrue(result['github_artifact_upload_required'])
            self.assertTrue((p/'v13_checkpoint_recovery.tar.gz').exists())
            self.assertEqual((root/'active_heads.joblib').read_bytes(),values['active_heads.joblib'])
            self.assertEqual(api.calls,[])
