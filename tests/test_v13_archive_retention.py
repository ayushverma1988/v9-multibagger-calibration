import hashlib,json,unittest
from copy import deepcopy
from v13_archive_cleanup import ARCHIVE
from v13_archive_retention import cache_plan

class CacheRetention(unittest.TestCase):
    def test_old_caches_are_optional_but_current_pending_and_member_bytes_stay(self):
        bundles=[{'object_path':'v13/objects/'+str(i)*64+'.bin','sha256':str(i)*64,'bytes':100+i} for i in range(1,5)]
        manifests=[];metadata=[]
        for i,b in enumerate(bundles):
            m={'format':'V13-content-addressed-checkpoint-1','derived_state_bundle':b,
              'files':[{'object_path':bundles[3]['object_path'],'sha256':bundles[3]['sha256'],'bytes':104}]}
            raw=json.dumps(m).encode();name='latest_manifest.json' if i==0 else 'old'+str(i)+'.json'
            manifests.append((name,raw));metadata.append({'path':'v13/state/'+name,'sha256':hashlib.sha256(raw).hexdigest()})
        refs=[{'path':b['object_path'],'sha256':b['sha256'],'bytes':b['bytes']} for b in bundles]
        physical=[{'path':b['object_path'],'lfs_sha256':b['sha256'],'bytes':b['bytes']} for b in bundles]
        d={'repositories':[{'repo_id':ARCHIVE,'files':physical}],
          'archive_dependencies':{'manifests':metadata,'protected_paths':[r['path'] for r in refs],
            'referenced_objects':refs,'delete_candidates':[]}}
        result=cache_plan(d,manifests,pending_raw=manifests[1][1])
        self.assertEqual([r['path'] for r in result['optional_bundle_caches']],[bundles[2]['object_path']])
        self.assertEqual(set(result['active_bundle_paths']),{bundles[0]['object_path'],bundles[1]['object_path']})
        self.assertIn(bundles[3]['object_path'],result['archive_dependencies']['protected_paths'])

    def test_unverified_pending_base_stops_retention_planning(self):
        with self.assertRaises(ValueError):
            cache_plan({'repositories':[{'repo_id':ARCHIVE,'files':[]}],
              'archive_dependencies':{'manifests':[]}},[],b'unknown recovery')

    def test_retired_restart_cache_can_be_reconstructed_byte_for_byte(self):
        import tempfile
        from pathlib import Path
        from v13_checkpoint import manifest,build_derived_bundle
        from v13_archive_retention import reconstruct_restart_cache
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);state=root/'state';state.mkdir()
            (state/'active_heads.joblib').write_bytes(b'fixed original model')
            (state/'registries').mkdir();(state/'registries/forecast.json').write_bytes(b'original forecast')
            data=manifest(state);data['derived_state_bundle']=build_derived_bundle(state,data,root/'original.tar.gz')
            addresses={r['object_path']:state/r['path'] for r in data['files']}
            rebuilt=reconstruct_restart_cache(data,root/'restored',lambda p:addresses[p])
            self.assertEqual(rebuilt.read_bytes(),(root/'original.tar.gz').read_bytes())
