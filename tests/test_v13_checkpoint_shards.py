"""Regression for the live Hub directory-limit failure; no remote credentials."""
import hashlib,json,tempfile,unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from v13_checkpoint import manifest,materialize_one,object_path,save

class CheckpointShards(unittest.TestCase):
    def test_mixed_manifest_preserves_legacy_addresses_and_forecasts(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);frozen=b'frozen probabilities';digest=hashlib.sha256(frozen).hexdigest()
            item={'path':'registry/frozen.json','sha256':digest,'bytes':len(frozen),
                  'object_path':f'v13/objects/{digest}.bin','raw_source':False}
            (root/'remote_manifest.json').write_text(json.dumps({'files':[item]}))
            (root/'registry').mkdir();(root/item['path']).write_bytes(frozen)
            (root/'duplicate.json').write_bytes(frozen);(root/'new.json').write_bytes(b'new evidence')
            entries={r['path']:r for r in manifest(root)['files']}
            self.assertEqual(entries[item['path']],item)
            self.assertEqual(entries['duplicate.json']['object_path'],item['object_path'])
            self.assertEqual(entries['new.json']['object_path'],object_path(entries['new.json']['sha256']))

    def test_shard_restore_checks_digest_and_address(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);source=root/'payload';source.write_bytes(b'original evidence')
            digest=hashlib.sha256(source.read_bytes()).hexdigest()
            item={'path':'raw/disclosure.bin','sha256':digest,'bytes':source.stat().st_size,'object_path':object_path(digest)}
            materialize_one(item,root/'restored',lambda p:source)
            self.assertEqual((root/'restored/raw/disclosure.bin').read_bytes(),source.read_bytes())
            for bad in (f'v13/object_shards/ff/ff/{digest}.bin','../outside'):
                with self.subTest(bad=bad):
                    with self.assertRaises(ValueError):materialize_one(dict(item,object_path=bad),root/'restored',lambda p:source)
            source.write_bytes(b'altered evidence')
            with self.assertRaises(ValueError):materialize_one(item,root/'restored',lambda p:source)

    def test_save_succeeds_with_full_legacy_directory_and_restores_every_byte(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)/'state';root.mkdir();remote={}
            # Model the actual failure boundary, including archived raw bytes
            # which are intentionally absent from the local restored state.
            digest=hashlib.sha256(b'old raw').hexdigest();legacy=f'v13/objects/{digest}.bin'
            remote[legacy]=b'old raw'
            remote.update({f'v13/objects/{i:064x}.bin':b'archived' for i in range(9999)})
            old={'files':[{'path':'raw/old.bin','sha256':digest,'bytes':7,'object_path':legacy,'raw_source':True}]}
            (root/'remote_manifest.json').write_text(json.dumps(old))
            for i in range(401):(root/f'evidence_{i}.json').write_bytes(f'dated evidence {i}'.encode())
            def commit(**kwargs):
                additions={op.path_in_repo:Path(op.path_or_fileobj).read_bytes() for op in kwargs['operations']}
                counts=Counter(str(Path(p).parent) for p in set(remote)|set(additions))
                if max(counts.values())>10000:raise RuntimeError('too many files per directory')
                remote.update(additions)
            api=SimpleNamespace(repo_info=lambda *a,**k:SimpleNamespace(private=True),create_commit=commit)
            with patch.dict('os.environ',{'HF_ARCHIVE_TOKEN':'test-placeholder'}),patch('huggingface_hub.HfApi',return_value=api):
                receipt=save(root)
            self.assertEqual(receipt['new_objects'],401)
            self.assertEqual(Counter(str(Path(p).parent) for p in remote)['v13/objects'],10000)
            data=json.loads(remote['v13/state/latest_manifest.json'])
            self.assertEqual(next(r for r in data['files'] if r['path']=='raw/old.bin')['object_path'],legacy)
            def fetch(key):
                p=Path(t)/'download';p.write_bytes(remote[key]);return p
            for entry in data['files']:materialize_one(entry,Path(t)/'recovered',fetch)
            self.assertEqual((Path(t)/'recovered/raw/old.bin').read_bytes(),b'old raw')
            self.assertEqual(hashlib.sha256(remote['v13/state/latest_manifest.json']).hexdigest(),receipt['manifest_sha256'])

if __name__=='__main__':unittest.main()
