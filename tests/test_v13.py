import unittest,tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np,pandas as pd
import v13_model as model
from v13_features import daily_features
from v13_lifecycle import register,validate_registry,promotion_gate,adjudicate
from v13_runner import initialize,verify_manifest
from v13_collection import collect_histories
from v13_checkpoint import manifest,materialize_one
import base64,hashlib,json

class V13Contract(unittest.TestCase):
    def test_conditional_coherence(self):
        heads={t:{'value':v} for t,v in zip(model.TARGETS,[.4,.5,.7,.9])}
        with patch.object(model,'score',side_effect=lambda p,d:np.repeat(p['value'],len(d))):
            p=model.probabilities(heads,pd.DataFrame(index=range(8)))
        self.assertTrue((p['p6']<=p['p12']).all());self.assertTrue((p['joint12']<=p['p12']).all())
        self.assertTrue((p['joint12']+p['loss12']<=1+1e-12).all())
        self.assertAlmostEqual(p['p12'][0],.7)

    def test_chronological_conditional_split(self):
        dates=pd.date_range('2022-11-30',periods=30,freq='ME');rows=[]
        for date in dates:
            for j in range(100):rows.append({'date':date,'mature_date_12':date+pd.DateOffset(months=12),'hit2_12':j%2,'loss_given_hit12':j%2})
        train,cal=model.parts(pd.DataFrame(rows),pd.Timestamp('2026-10-09'),model.HIT_LOSS)
        self.assertTrue(train.mature_date_12.lt(cal.date.min()).all());self.assertTrue(cal.mature_date_12.lt(pd.Timestamp('2026-10-09')).all())

    def test_future_prices_do_not_change_past_features(self):
        n=600;date=pd.bdate_range('2024-01-01',periods=n);p=np.linspace(100,200,n)
        bars=pd.DataFrame({'date':date,'open':p,'close':p,'adj_close':p,'high':p+1,'low':p-1,'volume':100000,'isin':'INE000000001','symbol':'X'})
        original=daily_features(bars,'2026-10-09');changed=bars.copy();changed.loc[500:,'close']*=3;changed.loc[500:,'adj_close']*=3
        altered=daily_features(changed,'2026-10-09')
        pd.testing.assert_frame_equal(original.iloc[:500],altered.iloc[:500])

    def test_missing_financials_are_discovery_only(self):
        row={c:np.nan for c in (*model.old.BASE_FIELDS,'four_profitable_quarters','annual_EBIT_proxy_interest_coverage')}
        row.update(close=100,distance_ma50=.1,log_ret63=.2)
        v,d,b=model.gates(pd.DataFrame([row]));self.assertFalse(v.iloc[0]);self.assertTrue(d.iloc[0]);self.assertFalse(b.iloc[0])
        row['quarter_pat_margin']=-.1;v,d,b=model.gates(pd.DataFrame([row]));self.assertTrue(b.iloc[0]);self.assertFalse(d.iloc[0])

    def test_registration_cannot_use_same_day_entry_or_be_modified(self):
        frame=pd.DataFrame([{'date':pd.Timestamp('2026-10-09'),'isin':'INE000000001','symbol':'X','p6':.1,'p12':.2,'loss12':.4,'joint12':.1}])
        with tempfile.TemporaryDirectory() as t:
            p=register(frame,'a'*64,t,'2026-10-11T15:30:00Z');obj=validate_registry(p)
            official=pd.DataFrame([{'date':'2026-10-09','isin':'INE000000001','symbol':'X','source_status':'VALID_POSITIVE_VOLUME_SESSION','open':100,'close':200}])
            self.assertEqual(adjudicate(p,official,{},'2026-10-11')[0]['status'],'UNKNOWN_ENTRY')
            obj['records'][0]['p12']=.99;p.write_text(__import__('json').dumps(obj))
            with self.assertRaises(ValueError):validate_registry(p)

    def test_unseen_source_gate_blocks_promotion(self):
        d=pd.DataFrame(columns=['date','registered_at_utc','mature_at_utc','y','champion','challenger','selected_champion','selected_challenger','independent_outcome_verified'])
        self.assertFalse(promotion_gate(d,'2026-10-11T15:30:00Z')['promote'])

    def test_seed_restore_checks_every_payload(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);seed=root/'seed';seed.mkdir();state=root/'state'
            raw=b'checked model bytes';(seed/'active_heads.joblib.b64').write_text(base64.b64encode(raw).decode())
            (seed/'manifest.json').write_text(json.dumps({'files':{'active_heads.joblib':hashlib.sha256(raw).hexdigest()}}))
            frame=pd.DataFrame([{'date':pd.Timestamp('2026-10-09'),'isin':'INE000000001','symbol':'X','p6':.1,'p12':.2,'loss12':.4,'joint12':.1}])
            reg=register(frame,'a'*64,root/'reg','2026-10-11T15:30:00Z')
            (seed/'initial_registry.json').write_text(reg.read_text());initialize(state,seed)
            self.assertEqual((state/'active_heads.joblib').read_bytes(),raw)
            (seed/'active_heads.joblib.b64').write_text(base64.b64encode(b'tampered').decode())
            with self.assertRaises(ValueError):initialize(root/'newstate',seed)

    def test_checkpoint_tamper_fails_closed(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);(root/'forecast.json').write_bytes(b'immutable')
            (root/'state_manifest.json').write_text(json.dumps({'files':[{'path':'forecast.json','sha256':hashlib.sha256(b'immutable').hexdigest()}]}))
            verify_manifest(root);(root/'forecast.json').write_bytes(b'changed')
            with self.assertRaises(ValueError):verify_manifest(root)

    def test_unavailable_history_does_not_starve_public_queue(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            pd.DataFrame([{'nse_current_symbol':s,'isin':'INE000000001'} for s in ['AAA','BBB','CCC']]).to_parquet(root/'universe.parquet',index=False)
            with patch('v13_collection.Store.get',side_effect=ValueError('temporarily unavailable')):
                first=collect_histories(root,'2026-10-09',budget=1)
                second=collect_histories(root,'2026-10-09',budget=1)
            self.assertNotEqual(first['errors'][0]['symbol'],second['errors'][0]['symbol'])
            self.assertEqual(first['requested'],1);self.assertEqual(second['requested'],1)

    def test_incremental_checkpoint_preserves_remote_raw_objects(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);data=b'original filing';sha=hashlib.sha256(data).hexdigest()
            old={'files':[{'path':'filing_raw/hash/original.bin','sha256':sha,'bytes':len(data),'object_path':'v13/objects/'+sha+'.bin','raw_source':True}]}
            (root/'remote_manifest.json').write_text(json.dumps(old));(root/'head.joblib').write_bytes(b'new model')
            new=manifest(root);self.assertEqual(len(new['files']),2)
            self.assertIn(sha,{r['sha256'] for r in new['files']})

    def test_object_restore_checks_hash_size_and_safe_path(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);source=root/'source';source.write_bytes(b'checked');sha=hashlib.sha256(b'checked').hexdigest()
            item={'path':'heads/model.joblib','sha256':sha,'bytes':7,'object_path':'v13/objects/'+sha+'.bin'}
            materialize_one(item,root/'state',lambda p:source)
            self.assertEqual((root/'state/heads/model.joblib').read_bytes(),b'checked')
            source.write_bytes(b'tampered')
            with self.assertRaises(ValueError):materialize_one(item,root/'state',lambda p:source)
            item['path']='../escape'
            with self.assertRaises(ValueError):materialize_one(item,root/'state',lambda p:source)

    def test_repeated_daily_cohorts_are_not_independent_months(self):
        rows=[]
        for date in pd.bdate_range('2024-06-01',periods=8):
            for j in range(600):
                y=j%2;rows.append({'date':date,'registered_at_utc':date.tz_localize('UTC'),
                  'mature_at_utc':date.tz_localize('UTC')+pd.DateOffset(months=12),'y':y,
                  'champion':.5,'challenger':float(y),'selected_champion':True,
                  'selected_challenger':True,'independent_outcome_verified':True})
        r=promotion_gate(pd.DataFrame(rows),'2024-05-01T00:00:00Z',source_complete=True)
        self.assertFalse(r['promote']);self.assertIn('fewer_than_six_mature_decision_months',r['reasons'])

if __name__=='__main__':unittest.main()
