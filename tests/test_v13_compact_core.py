import hashlib,json,tempfile,unittest,zipfile
from pathlib import Path
from v13_compact_core import build_core
from v13_core_restore import restore_package

class CompactWorkingCore(unittest.TestCase):
    def fixture(self,folder):
        root=folder/'original';root.mkdir()
        values={'active_heads.joblib':b'original fixed model\x00'*200,
          'prospective_observations.parquet':b'exact immutable probability 0.12345678901234566'*100,
          'history/issuer.json':b'{"source_asof":"2026-11-04","close":12.3456789}',
          'filing_raw/original.bin':b'original retained evidence outside core'}
        rows=[]
        for name,raw in values.items():
            p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
            rows.append({'path':name,'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'raw_source':name.startswith('filing_raw/')})
        return root,values,json.dumps({'format':'fixture','files':rows}).encode()

    def test_every_working_byte_restores_without_raw_pool_or_numeric_rounding(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);root,values,manifest=self.fixture(p)
            receipt=build_core(root,manifest,p/'out','2026-11-04',41921839880)
            self.assertTrue(receipt['under_one_gb']);self.assertEqual(receipt['state_files'],3)
            restored=p/'out'/'fresh_verified_state'
            for name,raw in values.items():
                if name.startswith('filing_raw/'):
                    self.assertFalse((restored/name).exists());self.assertEqual((root/name).read_bytes(),raw)
                else:self.assertEqual((restored/name).read_bytes(),raw)
            self.assertEqual((restored/'remote_manifest.json').read_bytes(),manifest)
            self.assertFalse(receipt['historical_evidence_archive_deleted'])

    def test_changed_model_cannot_be_packaged_as_original(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);root,values,manifest=self.fixture(p)
            (root/'active_heads.joblib').write_bytes(b'different model')
            with self.assertRaises(ValueError):build_core(root,manifest,p/'out','2026-11-04',1)

    def test_corrupted_payload_fails_without_overwriting_a_working_state(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);root,values,manifest=self.fixture(p)
            receipt=build_core(root,manifest,p/'out','2026-11-04',1)
            package=p/'out'/receipt['package_file'];bad=p/'corrupt.zip'
            with zipfile.ZipFile(package) as source,zipfile.ZipFile(bad,'w') as out:
                for name in source.namelist():
                    raw=source.read(name)
                    if name=='v13_state.tar.zst':raw=raw[:-1]+bytes([raw[-1]^1])
                    out.writestr(name,raw)
            with self.assertRaises(ValueError):restore_package(bad,p/'new')
            self.assertFalse((p/'new').exists())
            (p/'existing').mkdir();(p/'existing'/'keep').write_text('user data')
            with self.assertRaises(ValueError):restore_package(package,p/'existing')
            self.assertEqual((p/'existing'/'keep').read_text(),'user data')

    def test_size_target_is_measured_not_assumed(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);root,values,manifest=self.fixture(p)
            with self.assertRaisesRegex(ValueError,'exceeds requested'):build_core(root,manifest,p/'out','2026-11-04',1,target_bytes=10)
            receipt=json.loads((p/'out'/'core_build_receipt.json').read_text())
            self.assertFalse(receipt['under_one_gb']);self.assertGreater(receipt['package_bytes'],10)
