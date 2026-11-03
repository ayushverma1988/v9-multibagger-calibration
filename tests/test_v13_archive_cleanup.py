import hashlib,json,tempfile,unittest
from pathlib import Path
from v13_archive_cleanup import inspect_archive,reference,record_file,deletion_plan,OWNER,ARCHIVE,root_fingerprint,verify_retained_root,git_tree_hash
from v13_checkpoint import object_path
from v13_checkpoint_packs import pack_path

class ArchiveCleanup(unittest.TestCase):
    def test_merkle_identity_proves_retained_v13_tree_and_detects_changes_or_leftovers(self):
        before={'v13':'a'*40,'.gitattributes':'b'*40,'v10':'c'*40}
        verify_retained_root(before,{'v13':'a'*40,'.gitattributes':'b'*40})
        for after in [{'v13':'d'*40,'.gitattributes':'b'*40},before,{'v13':'a'*40}]:
            with self.assertRaises(ValueError):verify_retained_root(before,after)

    def test_missing_root_identity_blocks_a_storage_mutation(self):
        with self.assertRaises(ValueError):root_fingerprint([{'path':'v13','tree_id':None}])
        with self.assertRaises(ValueError):root_fingerprint([{'path':'v10','tree_id':'a'*40}])
        self.assertEqual(root_fingerprint([{'path':'v13','tree_id':'a'*40}]),{'v13':'a'*40})
    def plan(self):
        return {'owner':OWNER,'protected_archive':ARCHIVE,'archive_head':'a'*40,
          'archive_dependencies':{'protected_paths':['v13/model.bin'],'reference_problems':[],
            'referenced_objects':[{'path':'v13/model.bin'}],
            'delete_candidates':[{'path':'v11/old.bin','bytes':10}]},
          'other_repository_candidates':[{'repo_id':OWNER+'/old-model','repo_type':'model','head':'b'*40}]}

    def test_git_tree_hash_matches_git_for_nested_binary_files_and_sorting(self):
        import subprocess
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);subprocess.run(['git','init','-q',t],check=True)
            mapping={}
            for name,raw in [('b/file.bin',b'\0binary'),('b.c',b'sorting'),('B',b'upper')]:
                p=root/name;p.parent.mkdir(exist_ok=True);p.write_bytes(raw)
                mapping[name]=subprocess.check_output(['git','hash-object',str(p)],text=True).strip()
            subprocess.run(['git','-C',t,'add','.'],check=True)
            expected=subprocess.check_output(['git','-C',t,'write-tree'],text=True).strip()
            self.assertEqual(git_tree_hash(mapping),expected)

    def test_optional_cache_must_never_be_an_original_member_or_active_restart(self):
        p=self.plan();path=object_path('c'*64)
        p['optional_bundle_caches']=[{'path':path,'sha256':'c'*64,'bytes':20,'reconstructable_from_retained_members':True}]
        p['archive_dependencies']['delete_candidates'].append({'path':path,'bytes':20})
        self.assertIn(path,deletion_plan(p)['delete_paths'])
        p['active_bundle_paths']=[path]
        with self.assertRaises(ValueError):deletion_plan(p)
        p['active_bundle_paths']=[];p['archive_dependencies']['referenced_objects'].append({'path':path})
        with self.assertRaises(ValueError):deletion_plan(p)

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
