import tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np,pandas as pd
from v13_policy_evaluation import eligible,four_date_parts,probability_metrics,ranked,stable_context
from v13_priority_research import rank
from v13_artifact_checkpoint import save_with_quota_fallback

class PolicyEvaluation(unittest.TestCase):
    def frame(self):
        return pd.DataFrame({'isin':['A','B','C'],'close':[5.,3000.,100.],
          'distance_ma50':[.1,.1,.1],'log_ret63':[.1,.1,.1],'rsi14':[71.,71.,70.],
          'feature_eligible':[True]*3,'available_financial_tier':['AVAILABLE_CURRENT_FINANCIALS_RESEARCH']*3,
          'historical_official_identity_match':[True]*3,'p6':[.3,.1,.2],'joint12':[.1,.3,.2],
          'loss12':[.5,.5,.5],'distance_ma200':[.8,.8,.8],'available_financial_evidence_score':[.5]*3})

    def test_no_old_price_band_and_rsi_strict_and_primary_identity(self):
        f=self.frame();self.assertEqual(eligible(f).tolist(),[True,True,False])
        f.loc[0,'historical_official_identity_match']=False
        self.assertFalse(eligible(f).iloc[0])

    def test_six_month_preference_changes_rank_without_changing_probabilities(self):
        f=self.frame().iloc[:2].copy();before=f[['p6','joint12','loss12']].copy()
        x=rank(f,pd.Series(True,index=f.index))
        self.assertEqual(x.sort_values('priority_rank')['isin'].iloc[0],'A')
        pd.testing.assert_frame_equal(before,x[before.columns])

    def test_calibration_bins_count_boundaries_and_one_once(self):
        f=pd.DataFrame({'date':pd.Timestamp('2025-01-01'),'mature_date_6':pd.Timestamp('2025-07-01'),
          'hit2_6':[0,1,0,1],'p6':[0.,.1,.2,1.]})
        m=probability_metrics(f,'hit2_6','p6','mature_date_6','2026-01-01')
        self.assertEqual(sum(b['rows'] for b in m['bins']),4)
        self.assertEqual(m['rows'],4)

    def test_unmatured_labels_are_excluded_from_metrics(self):
        f=pd.DataFrame({'mature_date_6':[pd.Timestamp('2027-01-01')],'hit2_6':[1],'p6':[.9]})
        self.assertEqual(probability_metrics(f,'hit2_6','p6','mature_date_6','2026-01-01')['rows'],0)

    def test_future_rows_do_not_enter_calibration_or_training(self):
        dates=pd.date_range('2020-01-01',periods=30,freq='MS')
        f=pd.DataFrame({'date':np.repeat(dates,20),'mature_date_6':np.repeat(dates+pd.DateOffset(months=6),20),'hit2_6':np.tile(np.arange(20)%3==0,30)})
        train,cal=four_date_parts(f,pd.Timestamp('2022-07-01'),'hit2_6')
        self.assertTrue(train.mature_date_6.lt(cal.date.min()).all())
        self.assertTrue(cal.mature_date_6.lt(pd.Timestamp('2022-07-01')).all())
        self.assertEqual(cal.date.nunique(),4)

    def test_quota_fallback_is_explicit_and_other_errors_are_not_swallowed(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)/'state';root.mkdir();(root/'remote_manifest.json').write_text('{"format":"V13-content-addressed-checkpoint-1","files":[]}')
            (root/'x').write_text('observed')
            def quota(_):raise RuntimeError('Private storage limit reached for user')
            with patch.dict('os.environ',{'V13_ARTIFACT_CHECKPOINT_FALLBACK':'1'}):
                result=save_with_quota_fallback(root,quota)
                self.assertFalse(result['remote_publication_confirmed'])
                self.assertTrue(result['github_artifact_upload_required'])
            (Path(t)/'v13_archive_capacity.json').unlink()
            def other(_):raise RuntimeError('identity failure')
            with patch.dict('os.environ',{'V13_ARTIFACT_CHECKPOINT_FALLBACK':'1'}):
                with self.assertRaisesRegex(RuntimeError,'identity failure'):save_with_quota_fallback(root,other)

    def test_stable_context_preserves_missing_values_and_ignores_new_members(self):
        base=pd.DataFrame({'date':pd.Timestamp('2024-01-31'),'isin':[str(i) for i in range(100)],
          'log_ret63':np.arange(100)/100,'log_ret20':np.arange(100)/100})
        newer=base.copy();newer['date']=pd.Timestamp('2024-02-29')
        newer.loc[0,'log_ret63']=np.nan
        newcomer=pd.DataFrame({'date':[pd.Timestamp('2024-02-29')],'isin':['NEW'],
          'log_ret63':[-100.],'log_ret20':[-100.]})
        source=pd.concat([base,newer,newcomer],ignore_index=True)
        transformed,reference,_=stable_context(source)
        self.assertEqual(len(reference),100)
        self.assertTrue(pd.isna(transformed.loc[100,'relative_strength63']))
        self.assertAlmostEqual(transformed.loc[101,'market_median_ret63'],.50)
        self.assertEqual(transformed.loc[200,'relative_strength63'],0.)

    def test_recovery_metadata_does_not_recursively_enter_payload(self):
        from v13_checkpoint import manifest
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            (root/'pending_checkpoint_manifest.json').write_text('previous manifest')
            (root/'recovery_restore_receipt.json').write_text('local restore receipt')
            (root/'active_heads.joblib').write_bytes(b'original frozen head')
            self.assertEqual([x['path'] for x in manifest(root)['files']],['active_heads.joblib'])

    def test_sector_evidence_cannot_be_backfilled_before_its_observation(self):
        import json
        from v13_dated_sector import attach
        with tempfile.TemporaryDirectory() as t:
            folder=Path(t)/'master_data';folder.mkdir()
            (folder/'sector_master.json').write_text(json.dumps([{'isin':'A','sources':[{
              'isin':'A','symbol':'A','provider_family':'NSE','identity_proof':'EXACT_ISIN',
              'source_sha256':'a'*64,'first_observed_utc':'2026-11-02T09:00:00+00:00',
              'taxonomy':['Banking']}]}]))
            frame=pd.DataFrame({'isin':['A','A'],'symbol':['A','A'],
              'date':pd.to_datetime(['2026-10-30','2026-11-02'])})
            result=attach(frame,t)
            self.assertEqual(result.available_sector_kind.tolist(),['UNKNOWN','BANK'])

if __name__=='__main__':unittest.main()
