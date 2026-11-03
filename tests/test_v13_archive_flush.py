import unittest
from v13_archive_flush import assert_same_members

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
