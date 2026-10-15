import unittest,tempfile,json,hashlib,io
from pathlib import Path
import pandas as pd,numpy as np
from v13_history_integrity import validate_refresh,cache_digest,validated_cached_entry,recover_known_entries
from v13_checkpoint import object_path

class HistoryIntegrity(unittest.TestCase):
    def history(self):
        return pd.DataFrame({'date':pd.to_datetime(['2026-10-12','2026-10-13','2026-10-14']),
          'symbol':'A','isin':'INE000000001','open':[100.,101.,102.],'close':[100.,101.,102.],'source_sha256':'b'*64})
    def official(self):
        d=self.history();d['source_status']='VALID_POSITIVE_VOLUME_SESSION';d['source_sha256']='a'*64;return d
    def evidence(self):
        return {'isin':'INE000000001','symbol':'A','entry_date':'2026-10-14','entry_open_INR':102.,
          'entry_official_sha256':'a'*64,'entry_provider_sha256':'b'*64,'entry_evidence_recorded_at_utc':'2026-10-14T20:00:00Z'}

    def test_thinner_provider_response_cannot_erase_an_accepted_session(self):
        h=self.history()
        for thin in [h.iloc[:2],h.iloc[[0,2]]]:
            with self.assertRaisesRegex(ValueError,'omitted'):validate_refresh(h,thin,'2026-10-14','A','INE000000001')
        validate_refresh(h,h,'2026-10-14','A','INE000000001')

    def test_only_expired_four_year_rows_may_leave_a_refresh(self):
        h=self.history();old={**h.iloc[0].to_dict(),'date':pd.Timestamp('2022-01-01')}
        validate_refresh(pd.concat([pd.DataFrame([old]),h]),h,'2026-10-14','A','INE000000001')
        with self.assertRaisesRegex(ValueError,'identity'):validate_refresh(h,h,'2026-10-14','A','INE000000002')

    def test_cached_entry_requires_original_hash_price_identity_and_first_date(self):
        e=self.evidence();c={'evidence':e,'evidence_sha256':cache_digest(e)};o=self.official()
        self.assertEqual(validated_cached_entry(c,o,'2026-10-14','A',e['isin'])['entry_open_INR'],102)
        self.assertIsNone(validated_cached_entry(c,o,'2026-10-13','A',e['isin']))
        self.assertIsNone(validated_cached_entry(c,o,'2026-10-14','B',e['isin']))
        bad=o.copy();bad.loc[2,'open']=999;self.assertIsNone(validated_cached_entry(c,bad,'2026-10-14','A',e['isin']))
        bad=o.copy();bad['source_sha256']='c'*64;self.assertIsNone(validated_cached_entry(c,bad,'2026-10-14','A',e['isin']))
        tampered=json.loads(json.dumps(c));tampered['evidence']['entry_open_INR']=999
        self.assertIsNone(validated_cached_entry(tampered,o,'2026-10-14','A',e['isin']))

    def test_action_audit_rechecks_cached_entry_but_does_not_certify_returns(self):
        from v13_lifecycle import register
        from v13_action_audit import audit
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);(root/'action_audit').mkdir()
            (root/'action_audit/calendar.json').write_text(json.dumps({'year':2026,'holidays':[]}))
            frame=pd.DataFrame([{'date':pd.Timestamp('2026-10-13'),'symbol':'A','isin':'INE000000001','p6':.1,'p12':.2,'loss12':.3,'joint12':.1}])
            reg=register(frame,'f'*64,root/'registries/champion','2026-10-13T16:00:00Z')
            proof,_=audit(root,self.official(),{'INE000000001':self.history()},'2026-10-14')
            self.assertTrue(proof[(reg.stem,'INE000000001')]['entry_verified'])
            # A later provider response drops the actual entry session.
            proof,summary=audit(root,self.official(),{'INE000000001':self.history().iloc[:2]},'2026-10-14')
            p=proof[(reg.stem,'INE000000001')]
            self.assertTrue(p['entry_verified']);self.assertFalse(p['independent_outcome_verified'])
            self.assertFalse(p['confirmed_early_2x']);self.assertEqual(summary['prior_entry_receipts_primary_rechecked'],1)

    def test_recovery_uses_hash_checked_original_checkpoint_without_forecast_writes(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);(root/'official').mkdir();(root/'history/securities').mkdir(parents=True)
            self.official().to_parquet(root/'official/2026-10-14_official_market.parquet')
            pd.DataFrame([{'registry_sha256':'r','isin':'INE000000001','symbol':'A'}]).to_parquet(root/'prospective_observations.parquet')
            ledger_before=(root/'prospective_observations.parquet').read_bytes()
            name=hashlib.sha256(b'A').hexdigest()+'.parquet';self.history().iloc[:2].to_parquet(root/'history/securities'/name)
            buf=io.BytesIO();self.history().to_parquet(buf,index=False)
            oldproof={**self.evidence(),'registry_sha256':'r','entry_verified':True,'independent_outcome_verified':False}
            objects={};files=[]
            for rel,data in [('action_audit/2026-10-14_audit.json',json.dumps({'records':[oldproof]}).encode()),('history/securities/'+name,buf.getvalue())]:
                digest=hashlib.sha256(data).hexdigest();obj=object_path(digest);file=root/(digest+'.bin');file.write_bytes(data);objects[obj]=file
                files.append({'path':rel,'sha256':digest,'bytes':len(data),'object_path':obj,'raw_source':False})
            raw=json.dumps({'files':files}).encode();manifest=root/'manifest.json';manifest.write_bytes(raw)
            config={'snapshot':'snapshot','manifest_sha256':hashlib.sha256(raw).hexdigest()};objects['v13/state/snapshots/snapshot.json']=manifest
            r=recover_known_entries(root,config,lambda p:objects[p])
            self.assertEqual(r['history_files_restored'],1);self.assertEqual(r['entry_receipts_recovered'],1)
            self.assertEqual((root/'prospective_observations.parquet').read_bytes(),ledger_before)
            self.assertEqual(len(pd.read_parquet(root/'history/securities'/name)),3)
            self.assertFalse(r['outcomes_or_action_intervals_certified'])
            # Restart is idempotent and does not fetch the archive again.
            self.assertTrue(recover_known_entries(root,config,lambda p: (_ for _ in ()).throw(AssertionError()))['complete'])

    def test_wrong_recovery_manifest_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'m.json';p.write_text('{}')
            with self.assertRaisesRegex(ValueError,'manifest hash'):
                recover_known_entries(Path(t),{'snapshot':'x','manifest_sha256':'0'*64},lambda name:p)

if __name__=='__main__':unittest.main()
