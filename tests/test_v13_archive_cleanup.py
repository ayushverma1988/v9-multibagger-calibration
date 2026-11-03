import hashlib,json,tempfile,unittest
from pathlib import Path
from v13_archive_cleanup import inspect_archive,reference,record_file,deletion_plan,OWNER,ARCHIVE
from v13_checkpoint import object_path
from v13_checkpoint_packs import pack_path

class ArchiveCleanup(unittest.TestCase):
    def plan(self):
        return {'owner':OWNER,'protected_archive':ARCHIVE,'archive_head':'a'*40,
          'archive_dependencies':{'protected_paths':['v13/model.bin'],'reference_problems':[],
            'delete_candidates':[{'path':'v11/old.bin','bytes':10}]},
          'other_repository_candidates':[{'repo_id':OWNER+'/old-model','repo_type':'model','head':'b'*40}]}

    def test_authorised_plan_is_exact_and_does_not_include_v13(self):
        p=deletion_plan(self.plan())
        self.assertEqual(p['delete_paths'],['v11/old.bin'])
        self.assertEqual(p['delete_repositories'][0]['repo_id'],OWNER+'/old-model')

    def test_other_owners_required_archive_and_v13_repositories_cannot_be_deleted(self):
        for name in ['another-user/model',ARCHIVE,OWNER+'/v13-model']:
            p=self.plan();p['other_repository_candidates'][0]['repo_id']=name
            with self.assertRaises(ValueError):deletion_plan(p)

    def test_required_files_unsafe_paths_and_unresolved_references_block_deletion(self):
        for name in ['v13/model.bin','v13/new.bin','../outside','/absolute']:
            p=self.plan();p['archive_dependencies']['delete_candidates'][0]['path']=name
            with self.assertRaises(ValueError):deletion_plan(p)
        p=self.plan();p['archive_dependencies']['reference_problems']=[{'path':'required'}]
        with self.assertRaises(ValueError):deletion_plan(p)

    def test_only_unrelated_namespaces_are_candidates_and_packed_dependencies_stay(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);raw=b'V13 data';h=hashlib.sha256(raw).hexdigest()
            pack={'sha256':'b'*64,'object_path':pack_path('b'*64),'bytes':50}
            item={'path':'raw/source.xml','sha256':h,'bytes':len(raw),'raw_source':True,
              'object_path':object_path(h),'storage_pack':pack,'pack_offset':0}
            data={'format':'V13-content-addressed-checkpoint-2','files':[item]}
            manifest=root/'manifest.json';manifest.write_text(json.dumps(data))
            paths=['v13/state/latest_manifest.json',pack['object_path'],'v13/unreferenced_but_preserved.bin',
              '.gitattributes','README.md','v10/old.tar.gz','v11_4/old.csv']
            files=[{'path':p,'bytes':50 if p==pack['object_path'] else 1,
              'lfs_sha256':'b'*64 if p==pack['object_path'] else None,'blob_id':'x'} for p in paths]
            d=inspect_archive({},files,lambda p:manifest,root)
            self.assertEqual({r['path'] for r in d['delete_candidates']},{'v10/old.tar.gz','v11_4/old.csv'})
            self.assertFalse(d['reference_problems'])
            self.assertIn('v13/unreferenced_but_preserved.bin',d['protected_paths'])

    def test_missing_or_wrong_size_referenced_bytes_are_never_certified(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);h='a'*64;path=object_path(h)
            p=root/'manifest.json';p.write_text(json.dumps({'format':'V13-content-addressed-checkpoint-1',
              'files':[{'path':'active_heads.joblib','sha256':h,'bytes':9,'object_path':path,'raw_source':False}]}))
            base=[{'path':'v13/state/latest_manifest.json','bytes':1}]
            d=inspect_archive({},base,lambda x:p,root)
            self.assertEqual(d['reference_problems'][0]['reason'],'MISSING_REFERENCED_OBJECT')
            d=inspect_archive({},base+[{'path':path,'bytes':8,'lfs_sha256':h}],lambda x:p,root)
            self.assertEqual(d['reference_problems'][0]['reason'],'REFERENCED_OBJECT_SIZE_MISMATCH')

if __name__=='__main__':unittest.main()
