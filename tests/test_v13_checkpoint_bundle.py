import hashlib,json,tempfile,unittest,os
from pathlib import Path
from v13_checkpoint import manifest,build_derived_bundle,materialize_bundle,object_path

class DerivedBundle(unittest.TestCase):
    def fixture(self,folder):
        root=Path(folder)/'state';root.mkdir();(root/'history').mkdir();(root/'filing_raw').mkdir()
        (root/'history/daily.parquet').write_bytes(b'exact source-bound prices')
        (root/'frozen.json').write_bytes(b'{"p6":0.123}')
        (root/'filing_raw/original.bin').write_bytes(b'original filing, kept separately')
        return root,manifest(root)
    def test_bundle_restores_every_derived_byte_and_leaves_raw_in_archive(self):
        with tempfile.TemporaryDirectory() as t:
            root,data=self.fixture(t);pack=Path(t)/'pack.gz';item=build_derived_bundle(root,data,pack)
            out=Path(t)/'out';n=materialize_bundle(item,data,out,lambda p:pack)
            self.assertEqual(n,2);self.assertEqual((out/'frozen.json').read_bytes(),(root/'frozen.json').read_bytes())
            self.assertEqual((out/'history/daily.parquet').read_bytes(),b'exact source-bound prices')
            self.assertFalse((out/'filing_raw/original.bin').exists())
    def test_bundle_is_deterministic_and_does_not_depend_on_file_mtime(self):
        with tempfile.TemporaryDirectory() as t:
            root,data=self.fixture(t);a=build_derived_bundle(root,data,Path(t)/'a.gz')
            os.utime(root/'frozen.json',(1,1));b=build_derived_bundle(root,data,Path(t)/'b.gz')
            self.assertEqual(a,b)
    def test_outer_and_member_hashes_and_omissions_are_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root,data=self.fixture(t);pack=Path(t)/'pack.gz';item=build_derived_bundle(root,data,pack)
            bad=dict(item,sha256='0'*64,object_path=object_path('0'*64))
            with self.assertRaises(ValueError):materialize_bundle(bad,data,Path(t)/'a',lambda p:pack)
            changed=json.loads(json.dumps(data));entry=next(r for r in changed['files'] if not r['raw_source']);entry['sha256']='0'*64;entry['object_path']=object_path('0'*64)
            with self.assertRaises(ValueError):materialize_bundle(item,changed,Path(t)/'b',lambda p:pack)
            with self.assertRaises(ValueError):materialize_bundle(dict(item,files=99),data,Path(t)/'c',lambda p:pack)
    def test_changed_local_input_cannot_enter_checkpoint_bundle(self):
        with tempfile.TemporaryDirectory() as t:
            root,data=self.fixture(t);(root/'frozen.json').write_bytes(b'changed forecast')
            with self.assertRaises(ValueError):build_derived_bundle(root,data,Path(t)/'pack.gz')

if __name__=='__main__':unittest.main()
