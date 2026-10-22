import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from v13_collection import universe
from v11_4_exchange_reference_universe import NSE_MASTER, NSE_SME_MASTER


HEADER = 'SYMBOL,NAME OF COMPANY,SERIES,DATE OF LISTING,ISIN NUMBER\n'
MAIN = (HEADER + 'RELIANCE,Reliance Industries Limited,EQ,29-Nov-1995,INE002A01018\n').encode()
SME = (HEADER + 'HDFCBANK,HDFC Bank Limited,EQ,08-Nov-1995,INE040A01034\n').encode()
INVALID = MAIN + MAIN.splitlines(keepends=True)[1]


class UniverseRetryTests(unittest.TestCase):
    def fetcher(self, responses, folders):
        def fetch(store, url):
            folders.append(store.folder)
            raw = responses[url].pop(0)
            proof = {'sha256': hashlib.sha256(raw).hexdigest(),
                     'first_retrieved_utc': '2026-10-23T00:00:00+00:00'}
            (store.folder / ('main.bin' if url == NSE_MASTER else 'sme.bin')).write_bytes(raw)
            return raw, proof
        return fetch

    def test_corrected_reply_is_fetched_fresh_and_all_originals_are_retained(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); folders = []
            # The stale failed response must not be reused on a restart.
            old = root / 'sources/2026-10-22/raw'; old.mkdir(parents=True)
            (old / 'old.bin').write_bytes(INVALID)
            with patch('v13_collection.Store.get', self.fetcher(
                    {NSE_MASTER: [INVALID, INVALID, MAIN], NSE_SME_MASTER: [SME]}, folders)), \
                    patch('v13_collection.time.sleep'):
                frame = universe(root, '2026-10-22')
            self.assertEqual(len(frame), 2)
            self.assertEqual(len(set(folders[:3])), 3)
            self.assertEqual((old / 'old.bin').read_bytes(), INVALID)
            audit = json.loads((root / 'evidence/universe_source_audit.json').read_text())
            self.assertEqual([r['status'] for r in audit['attempts']],
                             ['REJECTED', 'REJECTED', 'ACCEPTED', 'ACCEPTED'])
            self.assertFalse(audit['identity_checks_relaxed'])
            self.assertEqual(audit['attempts'][0]['source_sha256'], hashlib.sha256(INVALID).hexdigest())

    def test_persistent_bad_identity_preserves_prior_universe_and_stops(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); prior = b'previous accepted universe'; folders = []
            (root / 'universe.parquet').write_bytes(prior)
            with patch('v13_collection.Store.get', self.fetcher(
                    {NSE_MASTER: [INVALID] * 3}, folders)), patch('v13_collection.time.sleep'), \
                    self.assertRaisesRegex(ValueError, 'Ambiguous current NSE equity identity'):
                universe(root, '2026-10-22')
            self.assertEqual(len(folders), 3)
            self.assertEqual((root / 'universe.parquet').read_bytes(), prior)
            audit = json.loads((root / 'evidence/universe_source_audit.json').read_text())
            self.assertEqual(audit['status'], 'INVALID_MASTER')
            self.assertFalse(audit['previous_universe_used_as_current'])

    def test_access_error_is_not_retried(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch('v13_collection.Store.get', side_effect=RuntimeError('HTTP 403')) as get, \
                    self.assertRaisesRegex(RuntimeError, 'HTTP 403'):
                universe(folder, '2026-10-22')
            self.assertEqual(get.call_count, 1)
            audit = json.loads((Path(folder) / 'evidence/universe_source_audit.json').read_text())
            self.assertEqual(audit['status'], 'SOURCE_FETCH_FAILED')

    def test_cross_master_symbol_conflict_preserves_prior_universe(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); prior = b'previous accepted universe'; folders = []
            (root / 'universe.parquet').write_bytes(prior)
            other = SME.replace(b'HDFCBANK,', b'RELIANCE,')
            with patch('v13_collection.Store.get', self.fetcher(
                    {NSE_MASTER: [MAIN], NSE_SME_MASTER: [other]}, folders)), \
                    self.assertRaisesRegex(ValueError, 'Public universe symbol conflict'):
                universe(root, '2026-10-22')
            self.assertEqual((root / 'universe.parquet').read_bytes(), prior)


if __name__ == '__main__':
    unittest.main()
