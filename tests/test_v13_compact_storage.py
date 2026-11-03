import hashlib,json,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from v13_archive_codec import encode_file,logical_download
from v13_storage_migrate import catalog_rows,verified_pack_members
from v13_archive_catalog import validate_row
from test_v13_checkpoint_writes import Api,Operation

class CompactStorageIntegration(unittest.TestCase):
    def test_shared_catalog_pack_is_downloaded_and_decoded_only_once(self):
        from v13_archive_catalog import fetch_member
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);values=[b'first exact source'*100,b'second exact source'*100]
            (p/'raw').write_bytes(b''.join(values));s=encode_file(p/'raw',p/'encoded')
            groups=[{'sha':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)} for raw in values]
            key=f"v13/compact_packs/{s['sha256'][:2]}/{s['sha256']}.zst"
            rows=catalog_rows(groups,key,sum(map(len,values)),s['original_sha256'],s['sha256'],s['bytes'])
            mapping={r['original_sha256']:r for r in rows}
            with patch('v13_archive_catalog.prefix_rows',return_value=mapping),patch('huggingface_hub.hf_hub_download',return_value=str(p/'encoded')) as download:
                for group,raw in zip(groups,values):
                    found=fetch_member('fixture',group['sha'],cache=p/'cache');self.assertEqual(Path(found).read_bytes(),raw)
                self.assertEqual(download.call_count,1)

    def test_parallel_preparation_reads_and_verifies_without_publishing(self):
        from concurrent.futures import ThreadPoolExecutor
        from v13_storage_migrate import prepare_batch
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);values=[b'source A '*100,b'source B '*100]
            by_sha={hashlib.sha256(raw).hexdigest():raw for raw in values}
            group=[{'sha':sha,'bytes':len(raw),'paths':['source/'+sha],
              'physical':{'lfs_sha256':sha,'bytes':len(raw)}} for sha,raw in by_sha.items()]
            def downloaded(action,path,sha,size):Path(path).write_bytes(by_sha[sha])
            with ThreadPoolExecutor(max_workers=2) as pool,patch('v13_storage_migrate.batch_actions',return_value={sha:{} for sha in by_sha}),patch('v13_storage_migrate.download_action',side_effect=downloaded):
                result=prepare_batch(group,p/'prepared',pool,object())
            self.assertEqual(result['size'],sum(map(len,values)))
            verified_pack_members(result['encoded'],result['rows'],p)

    def test_catalog_and_remote_pack_preserve_every_member_byte(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);values=[b'exact frozen probability 0.123\x00'*100,b'original source clock 2026-11-02'*1000]
            groups=[{'sha':hashlib.sha256(b).hexdigest(),'bytes':len(b)} for b in values]
            (p/'pack').write_bytes(b''.join(values));s=encode_file(p/'pack',p/'pack.zst')
            key=f"v13/compact_packs/{s['sha256'][:2]}/{s['sha256']}.zst"
            rows=catalog_rows(groups,key,sum(map(len,values)),s['original_sha256'],s['sha256'],s['bytes'])
            for row in rows:validate_row(row)
            verified_pack_members(p/'pack.zst',rows,p)
            with self.assertRaises(ValueError):validate_row({**rows[-1],'offset':10**9})
            with self.assertRaises(ValueError):verified_pack_members(p/'pack.zst',[{**rows[0],'original_sha256':'a'*64}],p)

    def test_compact_checkpoint_and_legacy_member_interface_restore_exactly(self):
        import v13_checkpoint as checkpoint
        from huggingface_hub.errors import EntryNotFoundError
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);root=p/'state';(root/'filing_raw').mkdir(parents=True)
            (root/'frozen.json').write_bytes(b'{"p6":0.12345678901234566,"missing":null}')
            (root/'latest_run.json').write_text(json.dumps({'asof':'2026-11-02'}))
            for i in range(101):(root/'filing_raw'/str(i)).write_bytes((str(i)+' exact original source '*200).encode())
            before={str(q.relative_to(root)):q.read_bytes() for q in root.rglob('*') if q.is_file()}
            api=Api(lfs=True)
            def download(repo,key,**kwargs):
                if key not in api.remote:raise EntryNotFoundError('missing fixture key')
                path=p/'downloads'/hashlib.sha256(key.encode()).hexdigest();path.parent.mkdir(exist_ok=True)
                path.write_bytes(api.remote[key]);return str(path)
            env={'HF_ARCHIVE_TOKEN':'test','V13_COMPACT_ARCHIVE':'1','V13_STORAGE_STAGING_DIR':str(p/'uploads'),
              'V13_DECODED_CACHE':str(p/'decoded')}
            with patch('huggingface_hub.HfApi',return_value=api),patch('huggingface_hub.CommitOperationAdd',Operation),patch('huggingface_hub.hf_hub_download',side_effect=download),patch.dict(os.environ,env):
                receipt=checkpoint.save(root)
                latest=logical_download(download('', 'v13/state/latest_manifest.json'),'v13/state/latest_manifest.json',p/'cache')
                data=json.loads(Path(latest).read_text());self.assertEqual(data['format'],'V13-content-addressed-checkpoint-3')
                def fetch(key):return logical_download(download('',key),key,p/'cache')
                for item in data['files']:checkpoint.materialize_one(item,p/'restored',fetch)
                self.assertEqual({name:(p/'restored'/name).read_bytes() for name in before},before)
            self.assertEqual(receipt['new_objects'],len(before))
            indexes=[k for k in api.remote if k.startswith('v13/date_index/2026/11/02/')]
            self.assertEqual(len(indexes),1);self.assertFalse(json.loads(api.remote[indexes[0]])['full_daily_data_copy_created'])

    def test_lost_upload_reply_confirms_original_not_compressed_hash(self):
        from v13_checkpoint import ARCHIVE,object_path
        from v13_checkpoint_writes import spec,write
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);raw=b'original immutable source '*10000;(p/'raw').write_bytes(raw)
            sha=hashlib.sha256(raw).hexdigest();logical=spec(object_path(sha),p/'raw');api=Api(failures=['post504'],lfs=True);confirmed=[]
            with patch.dict(os.environ,{'V13_COMPACT_ARCHIVE':'1','V13_STORAGE_STAGING_DIR':str(p/'uploads')}):
                write(api,ARCHIVE,[logical],'test compressed retry',Operation,confirmed=lambda xs:confirmed.extend(xs))
            self.assertEqual(len(api.calls),1);self.assertEqual(confirmed,[logical]);self.assertNotEqual(api.remote[logical['remote']],raw)

class CompactMigrationTransactions(unittest.TestCase):
    def test_provider_commit_limit_waits_for_reset_in_interruptible_chunks(self):
        from types import SimpleNamespace
        from v13_storage_migrate import commit,retry_seconds
        from test_v13_checkpoint_writes import HttpFailure
        failure=HttpFailure(429);failure.args=('128 repo commits per hour limit exceeded',)
        self.assertEqual(retry_seconds(failure,0),3600)
        self.assertEqual(retry_seconds(HttpFailure(429,{'Retry-After':'95'}),0),95)
        self.assertEqual(retry_seconds(HttpFailure(429,{'RateLimit':'"api";r=0;t=125'}),0),125)
        class Api:
            calls=0
            def create_commit(self,*a,**k):
                self.calls+=1
                if self.calls==1:raise HttpFailure(429,{'Retry-After':'95'})
                return SimpleNamespace(oid='new-head')
            def repo_info(self,*a,**k):return SimpleNamespace(sha='old-head')
            def get_paths_info(self,*a,**k):return []
        with tempfile.TemporaryDirectory() as t:
            from v13_checkpoint_writes import spec
            p=Path(t)/'data';p.write_bytes(b'verified source')
            waited=[]
            with patch('v13_storage_migrate.time.sleep') as sleep:
                head=commit(Api(),'old-head',adds=[spec('data',p)],on_wait=waited.append)
            self.assertEqual(head,'new-head');self.assertEqual(sum(c.args[0] for c in sleep.call_args_list),95)
            self.assertLessEqual(max(c.args[0] for c in sleep.call_args_list),30)
            self.assertEqual(waited[0]['phase'],'WAIT_PROVIDER_RATE_LIMIT')

    def test_upload_hashes_are_streamed_without_loading_large_files(self):
        from v13_checkpoint_writes import spec
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'data';raw=b'exact bytes\x00'*300000;p.write_bytes(raw)
            with patch.object(Path,'read_bytes',side_effect=AssertionError('unbounded read')):record=spec('test',p)
            self.assertEqual(record['sha256'],hashlib.sha256(raw).hexdigest())
            self.assertEqual(record['git_blob'],hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest())

    def batch_fixture(self,folder):
        out=[]
        for n in range(3):
            p=folder/str(n);p.mkdir();raw=(str(n)+' source data '*100).encode();(p/'raw').write_bytes(raw)
            stats=encode_file(p/'raw',p/'pack.zst');sha=stats['sha256'];remote=f'v13/compact_packs/{sha[:2]}/{sha}.zst'
            batch=[{'sha':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'paths':['original/'+str(n)]}]
            out.append({'folder':p,'encoded':p/'pack.zst','pack_path':remote,'pack_sha':sha,'stats':stats,'prefix':'ab',
              'batch':batch,'rows':catalog_rows(batch,remote,len(raw),stats['original_sha256'],sha,stats['bytes'])})
        return out

    def test_many_packs_switch_in_two_commits_and_one_prefix_index(self):
        from concurrent.futures import ThreadPoolExecutor
        from v13_storage_migrate import publish_prepared
        with tempfile.TemporaryDirectory() as t:
            prepared=self.batch_fixture(Path(t));calls=[];remote={};verified=[]
            indexes={'ab':{'format':'V13-compressed-object-catalog-1','prefix':'ab','batches':[]}}
            def write(api,head,adds=(),deletes=(),**k):
                if deletes:self.assertEqual(len(verified),3)
                calls.append((list(adds),list(deletes)))
                for item in adds:remote[item['sha256']]=Path(item['file']).read_bytes()
                return str(len(calls))
            def download(action,path,sha,size):Path(path).write_bytes(remote[sha]);verified.append(sha)
            with ThreadPoolExecutor(max_workers=3) as pool,patch('v13_storage_migrate.batch_actions',return_value={p['pack_sha']:{} for p in prepared}),patch('v13_storage_migrate.download_action',side_effect=download),patch('v13_checkpoint_writes.committed',return_value=True):
                head,paths=publish_prepared(object(),'initial',prepared,indexes,write,pool)
            self.assertEqual(head,'2');self.assertEqual(len(calls),2);self.assertEqual(len(paths),3)
            targets=[x['remote'] for x in calls[1][0]]
            self.assertEqual(targets.count('v13/storage_catalog/ab/index.json'),1)
            self.assertEqual(len(indexes['ab']['batches']),3)

    def test_one_corrupt_pack_prevents_all_original_deletions(self):
        from concurrent.futures import ThreadPoolExecutor
        from v13_storage_migrate import publish_prepared
        with tempfile.TemporaryDirectory() as t:
            prepared=self.batch_fixture(Path(t));calls=[];indexes={'ab':{'batches':[]}}
            def write(api,head,**kwargs):calls.append(kwargs);return 'staged'
            def corrupt(action,path,sha,size):Path(path).write_bytes(b'corrupted source')
            with ThreadPoolExecutor(max_workers=3) as pool,patch('v13_storage_migrate.batch_actions',return_value={p['pack_sha']:{} for p in prepared}),patch('v13_storage_migrate.download_action',side_effect=corrupt):
                with self.assertRaises(ValueError):publish_prepared(object(),'initial',prepared,indexes,write,pool)
            self.assertEqual(len(calls),1);self.assertFalse(calls[0].get('deletes'));self.assertEqual(indexes['ab']['batches'],[])
