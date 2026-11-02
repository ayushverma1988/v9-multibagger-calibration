import hashlib, json, tempfile, unittest
from pathlib import Path
import numpy as np
import pandas as pd
from v13_checkpoint import manifest
from v13_checkpoint_writes import recovery_delta
from v13_recovery_restore import restore_delta
from v13_history_views import histories,canonical_histories
from v13_redevelopment import attach

class AdoptionIntegrity(unittest.TestCase):
    def bars(self,symbol='A',isin='INE000000001'):
        p=100+np.arange(260)/10
        return pd.DataFrame({'date':pd.bdate_range('2025-01-01',periods=260),
          'symbol':symbol,'isin':isin,'open':p,'close':p,'adj_close':p,
          'high':p+1,'low':p-1,'volume':10000.})

    def test_aliases_do_not_contaminate_same_isin(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);a=self.bars();a.to_parquet(p/'a.parquet');b=self.bars('OLD');b['adj_close']*=5;b.to_parquet(p/'old.parquet')
            f=a.tail(1)[['date','symbol','isin']]
            x=attach(f,p)
            self.assertEqual(x.rd_history_status.iloc[0],'MATCHED_SYMBOL_ISIN_DATE')
            self.assertAlmostEqual(x.rd_return20.iloc[0],a.adj_close.iloc[-1]/a.adj_close.iloc[-21]-1)

    def test_duplicate_identical_history_uses_one_whole_superset(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);a=self.bars();a.to_parquet(p/'a.parquet');a.iloc[:200].to_parquet(p/'b.parquet')
            h,audit=histories(p)
            self.assertEqual(len(h),1);self.assertEqual(audit['redundant_files'],1)
            self.assertEqual(len(next(iter(h.values()))[1]),260)

    def test_conflict_quarantines_only_affected_security(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);a=self.bars();a.to_parquet(p/'a.parquet');b=a.copy();b.loc[20,'adj_close']+=1;b.to_parquet(p/'b.parquet')
            c=self.bars('C','INE000000002');c.to_parquet(p/'c.parquet')
            f=pd.concat([a.tail(1),c.tail(1)])[['date','symbol','isin']]
            x=attach(f,p).set_index('symbol')
            self.assertEqual(x.loc['A','rd_history_status'],'CONFLICTING_HISTORY_VINTAGES')
            self.assertTrue(np.isnan(x.loc['A','rd_return20']))
            self.assertEqual(x.loc['C','rd_history_status'],'MATCHED_SYMBOL_ISIN_DATE')

    def test_partial_non_superset_histories_are_not_spliced(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);a=self.bars();a.iloc[:200].to_parquet(p/'a.parquet');a.iloc[60:].to_parquet(p/'b.parquet')
            h,audit=histories(p)
            self.assertFalse(h);self.assertEqual(len(audit['quarantined']),1)

    def test_refit_aliases_require_unique_current_symbol_without_dropping_other_isins(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)
            self.bars().to_parquet(p/'a.parquet')
            self.bars('OLD').to_parquet(p/'old.parquet')
            self.bars('HISTORICAL','INE000000002').to_parquet(p/'historical.parquet')
            chosen,audit=canonical_histories(p,{('INE000000001','A')})
            self.assertEqual(set(chosen),{('INE000000001','A'),('INE000000002','HISTORICAL')})
            ambiguous,audit=canonical_histories(p)
            self.assertEqual(set(ambiguous),{('INE000000002','HISTORICAL')})
            self.assertEqual(len(audit['quarantined']),2)

    def recovery(self,p):
        root=p/'original';root.mkdir();(root/'active_heads.joblib').write_bytes(b'fixed-head')
        base=manifest(root);base_raw=json.dumps(base).encode()
        objects={r['object_path']:root/r['path'] for r in base['files']}
        (root/'remote_manifest.json').write_bytes(base_raw)
        (root/'new.json').write_text('new source data')
        planned=manifest(root);known={r['sha256'] for r in base['files']};new=[r for r in planned['files'] if r['sha256'] not in known]
        receipt=recovery_delta(root,planned,new,base_raw)
        return root,base_raw,objects,receipt

    def test_recovery_preserves_fixed_head_and_does_not_claim_remote_commit(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);original,base,objects,receipt=self.recovery(p);target=p/'restored'
            r=restore_delta(p/'v13_checkpoint_recovery.tar.gz',target,lambda k:objects[k],receipt['sha256'])
            self.assertEqual((target/'active_heads.joblib').read_bytes(),b'fixed-head')
            self.assertEqual((target/'new.json').read_text(),'new source data')
            self.assertEqual((target/'remote_manifest.json').read_bytes(),base)
            self.assertFalse(r['remote_publication_confirmed'])
            self.assertTrue(r['restored_local_state_verified'])

    def test_wrong_recovery_hash_fails_before_destination_write(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);original,base,objects,receipt=self.recovery(p);target=p/'restored'
            with self.assertRaisesRegex(ValueError,'archive hash'):
                restore_delta(p/'v13_checkpoint_recovery.tar.gz',target,lambda k:objects[k],'0'*64)
            self.assertFalse(target.exists())

    def test_known_object_reused_at_new_path_is_restored(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);root=p/'original';root.mkdir();(root/'old.json').write_bytes(b'unchanged')
            base=manifest(root);base_raw=json.dumps(base).encode()
            objects={r['object_path']:root/r['path'] for r in base['files']}
            (root/'remote_manifest.json').write_bytes(base_raw);(root/'new.json').write_bytes(b'unchanged')
            planned=manifest(root)
            receipt=recovery_delta(root,planned,[],base_raw)
            restore_delta(p/'v13_checkpoint_recovery.tar.gz',p/'restored',lambda k:objects[k],receipt['sha256'])
            self.assertEqual((p/'restored/new.json').read_bytes(),b'unchanged')

if __name__=='__main__':unittest.main()
