import hashlib,json,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from v13_archive_codec import encode_file,logical_download
from v13_storage_migrate import catalog_rows,verified_pack_members
from v13_archive_catalog import validate_row
from test_v13_checkpoint_writes import Api,Operation

class CompactStorageIntegration(unittest.TestCase):
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
