import hashlib
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from v13_checkpoint import VerifiedObjectDownloader, manifest, materialize_one
from v13_checkpoint_packs import build_packs


class PackedDownload(unittest.TestCase):
    def test_mixed_old_objects_and_shared_pack_restore_exact_original_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder);root=base/'state';root.mkdir()
            expected={'one':b'first original','two':b'second original','empty':b''}
            for name,raw in expected.items():(root/name).write_bytes(raw)
            records=manifest(root)['files']
            packs,members=build_packs(root,records,base/'packs')
            for item in records:item.update(members[item['sha256']])
            (root/'legacy').write_bytes(b'legacy direct object')
            direct=next(r for r in manifest(root)['files'] if r['path']=='legacy')
            records.append(direct);expected['legacy']=b'legacy direct object'
            cache=base/'cache';cache.mkdir()
            for pack in packs:(cache/(pack['sha256']+'.bin')).write_bytes(Path(pack['file']).read_bytes())
            (cache/(direct['sha256']+'.bin')).write_bytes(expected['legacy'])
            downloader=VerifiedObjectDownloader(records,cache,'unused')
            self.assertEqual(len(downloader.items),2)
            self.assertEqual(len(downloader.object_locks),2)
            # Cached exact bytes need no network or optional runtime modules.
            with patch.dict(sys.modules,{'huggingface_hub':types.SimpleNamespace(hf_hub_url=None),'requests':types.SimpleNamespace()}):
                for item in records:materialize_one(item,base/'restored',downloader)
            for name,raw in expected.items():self.assertEqual((base/'restored'/name).read_bytes(),raw)

    def test_member_with_tampered_pack_identity_rejected_before_network(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'one').write_bytes(b'first original')
            records=manifest(root)['files'];packs,members=build_packs(root,records,root/'packs')
            item=records[0];item.update(members[item['sha256']]);item['storage_pack']['object_path']='v13/unsafe.bin'
            with self.assertRaises(ValueError):VerifiedObjectDownloader([item],root/'cache','unused')
