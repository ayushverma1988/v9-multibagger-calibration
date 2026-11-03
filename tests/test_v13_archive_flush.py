import unittest
from v13_archive_flush import assert_same_members,recovery_raw_aliases

class PublicationGuards(unittest.TestCase):
    def setUp(self):
        self.original={'files':[{'path':'frozen.parquet','sha256':'a'*64,'bytes':123,'raw_source':False}]}

    def test_changed_forecast_hash_is_blocked(self):
        other={'files':[{**self.original['files'][0],'sha256':'b'*64}]}
        with self.assertRaisesRegex(ValueError,'change original'):assert_same_members(self.original,other)

    def test_duplicate_paths_cannot_hide_a_member(self):
        other={'files':self.original['files']*2}
        with self.assertRaisesRegex(ValueError,'Duplicate'):assert_same_members(self.original,other)

    def test_storage_repacking_does_not_change_forecast_identity(self):
        other={'files':[{**self.original['files'][0],'storage_pack':{'object_path':'new-pack'}}]}
        assert_same_members(self.original,other)

    def test_only_existing_raw_hash_aliases_can_be_carried(self):
        raw={'path':'sources/raw/old.json','sha256':'a'*64,'bytes':123,'raw_source':True}
        base={'files':[raw]};planned={'files':[raw,{**raw,'path':'sources/raw/new.json'}]}
        self.assertEqual(recovery_raw_aliases(planned,base,base),[planned['files'][1]])
        with self.assertRaisesRegex(ValueError,'not an existing'):
            recovery_raw_aliases({'files':[{**raw,'sha256':'b'*64}]},{'files':[]},base)

    def test_existing_model_changes_cannot_be_masked_as_aliases(self):
        with self.assertRaisesRegex(ValueError,'changed members'):
            recovery_raw_aliases(self.original,{'files':[{**self.original['files'][0],'bytes':124}]},{'files':[]})

    def test_alias_manifest_reuses_only_a_preserved_remote_original(self):
        import tempfile,json
        from pathlib import Path
        from v13_checkpoint import manifest,object_path
        raw={'path':'sources/raw/old.json','sha256':'a'*64,'bytes':123,'raw_source':True,'object_path':object_path('a'*64)}
        alias={**raw,'path':'sources/raw/new.json'}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'remote_manifest.json').write_text(json.dumps({'files':[raw]}))
            actual=manifest(root,preserved_raw_aliases=[alias])
            assert_same_members({'files':[raw,alias]},actual)
            self.assertEqual(next(r for r in actual['files'] if r['path']==alias['path'])['object_path'],raw['object_path'])
            with self.assertRaisesRegex(ValueError,'lacks a preserved'):
                manifest(root,preserved_raw_aliases=[{**alias,'bytes':124}])
