"""Archive scale, interrupted publication and exact evidence-byte recovery."""
import hashlib
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import types
import unittest
from unittest.mock import patch

import v13_checkpoint as checkpoint
import v13_checkpoint_packs as packs
import v13_checkpoint_writes as writes
from test_v13_checkpoint_writes import Api, Operation, HttpFailure


class PackedCheckpoint(unittest.TestCase):
    def hf(self, api):
        stub = types.ModuleType('huggingface_hub')
        stub.HfApi = lambda **k: api
        stub.CommitOperationAdd = Operation
        return patch.dict(sys.modules, {'huggingface_hub': stub})

    def save(self, root, api):
        with self.hf(api), patch.dict('os.environ', {'HF_ARCHIVE_TOKEN': 'test'}):
            return checkpoint.save(root)

    def fetcher(self, folder, remote):
        folder = Path(folder)
        folder.mkdir(exist_ok=True)
        def fetch(key):
            value = remote[key]
            target = folder / hashlib.sha256(key.encode()).hexdigest()
            target.write_bytes(value)
            return target
        return fetch

    def populated(self, folder, count=101):
        root = Path(folder) / 'state'
        root.mkdir()
        (root / 'history_raw').mkdir()
        (root / 'frozen.json').write_bytes(b'{"p6":0.123,"unselected":true}')
        for i in range(count):
            (root / 'history_raw' / str(i)).write_bytes(f'original source {i}'.encode())
        return root

    def test_8250_objects_need_only_packs_bundle_and_atomic_publish(self):
        with tempfile.TemporaryDirectory() as t:
            root = self.populated(t, 8249)
            before = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()}
            api = Api()
            receipt = self.save(root, api)
            self.assertEqual(receipt['new_objects'], 8250)
            self.assertEqual(receipt['packed_objects'], 8250)
            self.assertEqual(receipt['archive_files_uploaded'], 17)
            self.assertEqual(len(api.calls), 3)
            data = json.loads(api.remote['v13/state/latest_manifest.json'])
            self.assertEqual(data['format'], 'V13-content-addressed-checkpoint-2')
            fetch = self.fetcher(Path(t) / 'download', api.remote)
            for item in data['files']:
                checkpoint.materialize_one(item, Path(t) / 'restored', fetch)
                self.assertEqual((Path(t) / 'restored' / item['path']).read_bytes(), before[item['path']])
            self.assertFalse((Path(t) / 'v13_checkpoint_recovery.tar.gz').exists())

    def test_packed_mappings_survive_later_save_and_duplicate_content(self):
        with tempfile.TemporaryDirectory() as t:
            root = self.populated(t)
            api = Api()
            self.save(root, api)
            first = json.loads(api.remote['v13/state/latest_manifest.json'])
            before = {r['path']: r for r in first['files']}
            (root / 'copy.json').write_bytes((root / 'frozen.json').read_bytes())
            receipt = self.save(root, api)
            after = {r['path']: r for r in json.loads(api.remote['v13/state/latest_manifest.json'])['files']}
            self.assertEqual(receipt['new_objects'], 0)
            for name, item in before.items():
                self.assertEqual(after[name], item)
            for key in ('sha256', 'object_path', 'storage_pack', 'pack_offset'):
                self.assertEqual(after['copy.json'][key], before['frozen.json'][key])

    def test_old_raw_address_absent_locally_remains_unchanged(self):
        with tempfile.TemporaryDirectory() as t:
            root = self.populated(t)
            raw = b'previous raw evidence'
            sha = hashlib.sha256(raw).hexdigest()
            old = {'path': 'filing_raw/old.bin', 'sha256': sha, 'bytes': len(raw),
                   'raw_source': True, 'object_path': 'v13/objects/' + sha + '.bin'}
            (root / 'remote_manifest.json').write_text(json.dumps({'format': 'V13-content-addressed-checkpoint-1', 'files': [old]}))
            api = Api(remote={old['object_path']: raw})
            self.save(root, api)
            result = json.loads(api.remote['v13/state/latest_manifest.json'])
            self.assertEqual(next(r for r in result['files'] if r['path'] == old['path']), old)

    def test_hard_interruption_keeps_complete_delta_before_first_upload(self):
        with tempfile.TemporaryDirectory() as t:
            root = self.populated(t)
            old = json.dumps({'format': 'V13-content-addressed-checkpoint-1', 'files': []}).encode()
            (root / 'remote_manifest.json').write_bytes(old)
            api = Api(remote={'v13/state/latest_manifest.json': old})
            def interrupt(**kwargs):
                p = Path(t) / 'v13_checkpoint_recovery.tar.gz'
                self.assertTrue(p.exists())
                raise KeyboardInterrupt('simulated hard process termination')
            api.create_commit = interrupt
            with self.assertRaises(KeyboardInterrupt):
                self.save(root, api)
            self.assertEqual(api.remote['v13/state/latest_manifest.json'], old)
            desc = json.loads((Path(t) / 'v13_checkpoint_recovery.json').read_text())
            self.assertFalse(desc['remote_publication_confirmed'])
            with tarfile.open(Path(t) / 'v13_checkpoint_recovery.tar.gz', 'r:gz') as tar:
                plan = json.load(tar.extractfile('planned_manifest.json'))
                for item in plan['files']:
                    raw = tar.extractfile('objects/' + item['sha256'] + '.bin').read()
                    self.assertEqual(len(raw), item['bytes'])
                    self.assertEqual(hashlib.sha256(raw).hexdigest(), item['sha256'])

    def test_confirmed_pack_resumes_after_bundle_failure(self):
        with tempfile.TemporaryDirectory() as t:
            root = self.populated(t)
            api = Api(failures=['success', 403])
            with self.assertRaises(HttpFailure):
                self.save(root, api)
            old_calls = len(api.calls)
            self.save(root, api)
            self.assertEqual(len(api.calls) - old_calls, 2)

    def test_lost_pack_success_reply_is_reconciled(self):
        with tempfile.TemporaryDirectory() as t:
            root = self.populated(t)
            api = Api(failures=['post504'])
            self.save(root, api)
            self.assertEqual(len(api.calls), 3)
            self.assertEqual(api.inspections, 1)

    def test_pack_and_member_tampering_are_both_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root = self.populated(t)
            data = checkpoint.manifest(root)
            uploads, members = packs.build_packs(root, data['files'], Path(t) / 'packs')
            item = dict(data['files'][0], **members[data['files'][0]['sha256']])
            source = Path(uploads[0]['file'])
            fetch = lambda _: source
            original = source.read_bytes()
            bad = dict(item, sha256='0' * 64)
            with self.assertRaises(ValueError):
                packs.read_member(bad, fetch)
            source.write_bytes(b'X' + original[1:])
            with self.assertRaises(ValueError):
                packs.read_member(item, fetch)

    def test_unsafe_offsets_and_pack_addresses_fail_before_fetch(self):
        base = {'bytes': 2, 'pack_offset': 0, 'storage_pack': {
            'sha256': 'a' * 64, 'bytes': 2, 'object_path': packs.pack_path('a' * 64)}}
        variants = [dict(base, pack_offset=-1), dict(base, pack_offset=2),
                    dict(base, pack_offset=True), dict(base, bytes=-1),
                    dict(base, storage_pack=dict(base['storage_pack'], object_path='../outside')),
                    dict(base, storage_pack=dict(base['storage_pack'], bytes=packs.MAX_PACK_BYTES + 1))]
        for item in variants:
            with self.subTest(item=item), self.assertRaises(ValueError):
                packs.read_member(item, lambda _: self.fail('must not fetch'))

    def test_changed_source_cannot_enter_pack(self):
        with tempfile.TemporaryDirectory() as t:
            root = self.populated(t, 1)
            data = checkpoint.manifest(root)
            (root / 'frozen.json').write_bytes(b'changed')
            with self.assertRaises(ValueError):
                packs.build_packs(root, data['files'], Path(t) / 'packs')

    def test_pack_boundaries_and_order_are_deterministic(self):
        with tempfile.TemporaryDirectory() as t:
            root = self.populated(t, 501)
            entries = checkpoint.manifest(root)['files']
            a, am = packs.build_packs(root, entries, Path(t) / 'a')
            b, bm = packs.build_packs(root, list(reversed(entries)), Path(t) / 'b')
            self.assertEqual(am, bm)
            self.assertEqual([r['sha256'] for r in a], [r['sha256'] for r in b])
            self.assertEqual(len(a), 2)

    def test_zero_byte_member_round_trips(self):
        with tempfile.TemporaryDirectory() as t:
            root = self.populated(t)
            (root / 'empty').write_bytes(b'')
            api = Api()
            self.save(root, api)
            d = json.loads(api.remote['v13/state/latest_manifest.json'])
            item = next(r for r in d['files'] if r['path'] == 'empty')
            self.assertEqual(checkpoint.object_bytes(item, self.fetcher(Path(t) / 'downloads', api.remote)), b'')

    def test_deadline_stops_new_writes_without_false_receipt(self):
        with tempfile.TemporaryDirectory() as t:
            file = Path(t) / 'a'
            file.write_bytes(b'original')
            item = writes.spec('remote/a', file)
            api = Api()
            with patch.object(writes.time, 'monotonic', return_value=100):
                with self.assertRaises(writes.CheckpointTemporaryError):
                    writes.write(api, 'archive', [item], 'test', Operation, deadline=99)
            self.assertEqual(api.calls, [])

    def test_member_without_pack_and_overlarge_object_are_rejected(self):
        with self.assertRaises(ValueError):
            packs.validate_pack_member({'pack_offset': 0})
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            with self.assertRaises(ValueError):
                packs.build_packs(root, [{'path': 'x', 'sha256': 'a' * 64,
                                        'bytes': packs.MAX_PACK_BYTES + 1}], root / 'packs')

    def test_version_two_restores_restart_bundle_and_original_review_sources(self):
        with tempfile.TemporaryDirectory() as t:
            root = self.populated(t)
            api = Api()
            self.save(root, api)
            fetch = self.fetcher(Path(t) / 'downloads', api.remote)
            import huggingface_hub
            restored = Path(t) / 'restored'
            with patch.dict('os.environ', {'HF_ARCHIVE_TOKEN': 'test'}), \
                 patch.object(huggingface_hub, 'HfApi', return_value=api), \
                 patch.object(huggingface_hub, 'hf_hub_download', side_effect=lambda archive, key, **k: fetch(key)):
                self.assertTrue(checkpoint.restore(restored))
                self.assertEqual((restored / 'frozen.json').read_bytes(), (root / 'frozen.json').read_bytes())
                self.assertFalse((restored / 'history_raw/0').exists())
                digest = hashlib.sha256((root / 'history_raw/0').read_bytes()).hexdigest()
                checkpoint.restore_review_originals(restored, additional_hashes=[digest])
                self.assertEqual((restored / 'history_raw/0').read_bytes(), (root / 'history_raw/0').read_bytes())

    def test_recovery_delta_can_reconstruct_exact_packed_objects(self):
        with tempfile.TemporaryDirectory() as t:
            root = self.populated(t)
            api = Api(failures=[403])
            with self.assertRaises(HttpFailure):
                self.save(root, api)
            with tarfile.open(Path(t) / 'v13_checkpoint_recovery.tar.gz', 'r:gz') as tar:
                plan = json.load(tar.extractfile('planned_manifest.json'))
                by_pack = {}
                for item in plan['files']:
                    p = item['storage_pack']
                    by_pack.setdefault(p['object_path'], (p, []))[1].append(item)
                for descriptor, items in by_pack.values():
                    raw = b''
                    for item in sorted(items, key=lambda r: r['pack_offset']):
                        self.assertEqual(len(raw), item['pack_offset'])
                        raw += tar.extractfile('objects/' + item['sha256'] + '.bin').read()
                    self.assertEqual(len(raw), descriptor['bytes'])
                    self.assertEqual(hashlib.sha256(raw).hexdigest(), descriptor['sha256'])


if __name__ == '__main__':
    unittest.main()
