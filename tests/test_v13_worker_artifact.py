import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
import zipfile

from v13_worker_artifact import read_manifest, verified_direct, extract_delta_histories, audit_exports


class WorkerArtifact(unittest.TestCase):
    def make(self, root, recovery=False, bad_manifest=False, unsafe=False, changed_export=False):
        data = b'{"p6":0.123,"p12":0.234}'
        path = '../outside.json' if unsafe else 'forecast.json'
        item = {'path': path, 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
        hist = b'original history bytes'
        h = {'path': 'history/securities/a.parquet', 'sha256': hashlib.sha256(hist).hexdigest(), 'bytes': len(hist)}
        raw = json.dumps({'files': [item, h]}).encode();base = b'{"files":[]}'
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode='w:gz') as t:
            def put(name, value):
                m = tarfile.TarInfo(name);m.size = len(value);t.addfile(m, io.BytesIO(value))
            if recovery:
                meta = {'format': 'V13-checkpoint-recovery-delta-1',
                    'base_manifest_sha256': hashlib.sha256(base).hexdigest(),
                    'planned_manifest_sha256': '0'*64 if bad_manifest else hashlib.sha256(raw).hexdigest()}
                put('recovery.json', json.dumps(meta).encode());put('base_manifest.json', base);put('planned_manifest.json', raw)
                put('objects/'+h['sha256']+'.bin', hist)
            else:
                put('v13_checkpoint_manifest.json', raw)
        content = buf.getvalue();p = Path(root)/'worker.zip'
        with zipfile.ZipFile(p, 'w') as z:
            if recovery:
                desc = {'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest()}
                z.writestr('v13_checkpoint_recovery.json', json.dumps(desc));z.writestr('v13_checkpoint_recovery.tar.gz', content)
                z.writestr('archive_receipt.json', json.dumps({'remote_publication_confirmed': False, 'recovery': desc}))
            else:
                z.writestr('v13_state.tar.gz', content)
                z.writestr('archive_receipt.json', json.dumps({'snapshot': 'original', 'manifest_sha256': '0'*64 if bad_manifest else hashlib.sha256(raw).hexdigest()}))
            z.writestr('v13_state/forecast.json', b'changed' if changed_export else data)
        return p, h, hist

    def test_legacy_published_manifest_and_exact_export(self):
        with tempfile.TemporaryDirectory() as t:
            p, _, _ = self.make(t);m, r, _ = read_manifest(p)
            self.assertEqual(r['snapshot'], 'original')
            self.assertEqual(r['manifest_kind'], 'PUBLISHED_PRIVATE_CHECKPOINT')
            with zipfile.ZipFile(p) as z:self.assertEqual(json.loads(verified_direct(z, m, 'v13_state/forecast.json'))['p6'], .123)

    def test_recovery_without_legacy_tar_is_verified_but_never_remote_published(self):
        with tempfile.TemporaryDirectory() as t:
            p, _, _ = self.make(t, recovery=True);m, r, _ = read_manifest(p)
            self.assertFalse(r['remote_publication_confirmed']);self.assertIsNone(r['snapshot'])
            self.assertIn('RECOVERY', r['manifest_kind']);self.assertEqual(len(m['files']), 2)

    def test_changed_manifest_refuses_both_storage_formats(self):
        for rec in (False, True):
            with tempfile.TemporaryDirectory() as t:
                p, _, _ = self.make(t, recovery=rec, bad_manifest=True)
                with self.assertRaises(ValueError):read_manifest(p)

    def test_tampered_direct_export_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as t:
            p, _, _ = self.make(t, recovery=True, changed_export=True)
            with self.assertRaisesRegex(ValueError, 'integrity'):audit_exports(p, Path(t)/'out')

    def test_unsafe_manifest_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            p, _, _ = self.make(t, recovery=True, unsafe=True)
            with self.assertRaisesRegex(ValueError, 'Unsafe'):read_manifest(p)

    def test_requested_delta_history_is_exact_and_original_zip_unchanged(self):
        with tempfile.TemporaryDirectory() as t:
            p, item, raw = self.make(t, recovery=True);before=p.read_bytes();m, _, _=read_manifest(p)
            n=extract_delta_histories(p, m, [item], Path(t)/'restore')
            self.assertEqual(n, 1);self.assertEqual((Path(t)/'restore'/item['path']).read_bytes(), raw)
            self.assertEqual(before, p.read_bytes())

    def test_small_audit_export_preserves_source_probabilities_and_claim_scope(self):
        with tempfile.TemporaryDirectory() as t:
            p, _, _=self.make(t, recovery=True);r=audit_exports(p, Path(t)/'out')
            self.assertEqual(r['direct_exports_verified'], 1);self.assertEqual(r['manifest_references'], 2)
            self.assertFalse(r['all_manifest_references_individually_downloaded'])
            self.assertEqual(json.loads((Path(t)/'out/v13_state/forecast.json').read_text())['p12'], .234)


if __name__ == '__main__':unittest.main()
