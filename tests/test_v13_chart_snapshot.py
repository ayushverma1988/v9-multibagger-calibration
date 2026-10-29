import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
import zipfile

from v13_chart_snapshot import read_snapshot


def snapshot(path, wrong_manifest=False, wrong_model=False, mixed_dates=False):
    manifest=json.dumps({'files':[]}).encode()
    receipt={'snapshot':'test','manifest_sha256':hashlib.sha256(manifest).hexdigest()}
    if wrong_manifest:receipt['manifest_sha256']='0'*64
    head=b'frozen model bytes'
    replay={'model_sha256':hashlib.sha256(head).hexdigest()}
    if wrong_model:replay['model_sha256']='0'*64
    records=[{'date':'2026-10-23','isin':'INE000000001','symbol':'ALPHA',
        'close':100.,'p6':.04,'p12':.1,'loss12':.3}]
    if mixed_dates:records[0]['date']='2026-10-22'
    source={'source_date':'2026-10-23','prediction_replay':replay,'records':records}
    buf=io.BytesIO()
    with tarfile.open(fileobj=buf,mode='w:gz') as t:
        info=tarfile.TarInfo('v13_checkpoint_manifest.json');info.size=len(manifest)
        t.addfile(info,io.BytesIO(manifest))
    with zipfile.ZipFile(path,'w') as z:
        z.writestr('archive_receipt.json',json.dumps(receipt))
        z.writestr('v13_state.tar.gz',buf.getvalue())
        z.writestr('v13_state/redevelopment/latest.json',json.dumps(source))
        z.writestr('v13_state/active_heads.joblib',head)


class ChartSnapshot(unittest.TestCase):
    def test_source_dates_probabilities_and_snapshot_identity_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'source.zip';snapshot(p)
            frame,manifest,receipt,replay=read_snapshot(p)
            self.assertEqual(frame.iloc[0].p6,.04)
            self.assertEqual(str(frame.iloc[0].date.date()),'2026-10-23')
            self.assertEqual(receipt['snapshot'],'test')
            self.assertEqual(manifest,{'files':[]})
            self.assertEqual(len(replay['source_json_sha256']),64)

    def test_manifest_and_model_corruption_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'source.zip'
            for kwargs in [{'wrong_manifest':True},{'wrong_model':True}]:
                with self.subTest(kwargs=kwargs):
                    snapshot(p,**kwargs)
                    with self.assertRaises(ValueError):read_snapshot(p)

    def test_wrong_source_date_is_not_substituted(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'source.zip';snapshot(p,mixed_dates=True)
            with self.assertRaises(ValueError):read_snapshot(p)
