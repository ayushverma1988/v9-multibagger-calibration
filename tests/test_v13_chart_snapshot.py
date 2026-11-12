import hashlib,io,json,tarfile,tempfile,unittest,zipfile
from pathlib import Path
import pandas as pd
from v13_chart_snapshot import read_snapshot, verify_forecast_export


def snapshot(path,recovery=False,wrong_model=False,mixed_dates=False):
    head=b'frozen model bytes';replay={'model_sha256':hashlib.sha256(head).hexdigest()}
    if wrong_model:replay['model_sha256']='0'*64
    records=[{'date':'2026-10-22' if mixed_dates else '2026-10-23','isin':'INE000000001',
        'symbol':'ALPHA','close':100.,'p6':.04,'p12':.1,'loss12':.3}]
    raw=json.dumps({'source_date':'2026-10-23','prediction_replay':replay,'records':records}).encode()
    members={'redevelopment/latest.json':raw,'active_heads.joblib':head}
    manifest=json.dumps({'files':[{'path':k,'sha256':hashlib.sha256(v).hexdigest(),'bytes':len(v)} for k,v in members.items()]}).encode()
    buf=io.BytesIO()
    with tarfile.open(fileobj=buf,mode='w:gz') as t:
        def put(n,v):
            item=tarfile.TarInfo(n);item.size=len(v);t.addfile(item,io.BytesIO(v))
        if recovery:
            base=b'{"files":[]}'
            meta={'format':'V13-checkpoint-recovery-delta-1','base_manifest_sha256':hashlib.sha256(base).hexdigest(),'planned_manifest_sha256':hashlib.sha256(manifest).hexdigest()}
            put('recovery.json',json.dumps(meta).encode());put('base_manifest.json',base);put('planned_manifest.json',manifest)
        else:put('v13_checkpoint_manifest.json',manifest)
    archive=buf.getvalue()
    with zipfile.ZipFile(path,'w') as z:
        if recovery:
            desc={'bytes':len(archive),'sha256':hashlib.sha256(archive).hexdigest()}
            z.writestr('archive_receipt.json',json.dumps({'remote_publication_confirmed':False,'recovery':desc}))
            z.writestr('v13_checkpoint_recovery.json',json.dumps(desc));z.writestr('v13_checkpoint_recovery.tar.gz',archive)
        else:
            z.writestr('archive_receipt.json',json.dumps({'snapshot':'test','manifest_sha256':hashlib.sha256(manifest).hexdigest()}));z.writestr('v13_state.tar.gz',archive)
        for n,v in members.items():z.writestr('v13_state/'+n,v)


class ChartSnapshot(unittest.TestCase):
    def export_round_trip(self, ranks):
        frame = pd.DataFrame({'isin':['INE000000001','INE000000002'],
            'p6':[.04,.07], 'p12':[.1,.2], 'loss12':[.3,.4],
            'joint12':[.05,.1], 'research_selected':[False,False],
            'research_rank':ranks})
        result = pd.read_csv(io.StringIO(frame.to_csv(index=False)), float_precision='round_trip')
        return frame, result

    def test_no_selection_all_null_rank_survives_csv_round_trip(self):
        frame, result = self.export_round_trip([None,None])
        self.assertNotEqual(frame.research_rank.dtype, result.research_rank.dtype)
        original = frame.copy(deep=True)
        verify_forecast_export(frame, result)
        pd.testing.assert_frame_equal(frame, original, check_exact=True)

    def test_selected_and_nullable_ranks_stay_exact(self):
        for ranks in ([1,None], [1,2]):
            frame, result = self.export_round_trip(ranks)
            verify_forecast_export(frame, result.iloc[::-1])

    def test_probability_selection_identity_and_rank_changes_still_fail(self):
        frame, result = self.export_round_trip([1,None])
        for field, value in [('p6',.04000000000001), ('p12',.21),
                ('loss12',.31), ('joint12',.11), ('research_selected',True),
                ('isin','INE000000003'), ('research_rank',2)]:
            with self.subTest(field=field):
                changed = result.copy(deep=True)
                changed.loc[0,field] = value
                with self.assertRaises(AssertionError):verify_forecast_export(frame, changed)
        changed = result.copy(deep=True)
        changed.loc[1,'research_rank'] = 2
        with self.assertRaises(AssertionError):verify_forecast_export(frame, changed)
        with self.assertRaises(KeyError):verify_forecast_export(frame, result.drop(columns='research_rank'))

    def test_non_numeric_exported_rank_fails_closed(self):
        frame, result = self.export_round_trip([None,None])
        result['research_rank'] = ['unknown','unknown']
        with self.assertRaises(ValueError):verify_forecast_export(frame, result)

    def test_both_formats_keep_source_prices_probabilities_and_true_publication_state(self):
        for rec in (False,True):
            with tempfile.TemporaryDirectory() as t:
                p=Path(t)/'s.zip';snapshot(p,recovery=rec);f,m,r,proof=read_snapshot(p)
                self.assertEqual(f.p6.iloc[0],.04);self.assertEqual(f.date.iloc[0],__import__('pandas').Timestamp('2026-10-23'))
                self.assertEqual(len(m['files']),2);self.assertEqual(len(proof['source_json_sha256']),64)
                self.assertEqual(r['snapshot'],None if rec else 'test')
                if rec:self.assertFalse(r['remote_publication_confirmed'])

    def test_model_or_source_date_corruption_fails_closed(self):
        for rec in (False,True):
            for kwargs in [{'wrong_model':True},{'mixed_dates':True}]:
                with tempfile.TemporaryDirectory() as t:
                    p=Path(t)/'s.zip';snapshot(p,recovery=rec,**kwargs)
                    with self.assertRaises(ValueError):read_snapshot(p)


if __name__=='__main__':unittest.main()
