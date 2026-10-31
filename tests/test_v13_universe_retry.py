import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import requests

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
            if isinstance(raw, Exception):
                raise raw
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

    def http_error(self, status):
        response = requests.Response()
        response.status_code = status
        return requests.exceptions.HTTPError('public source HTTP error', response=response)

    def test_timeout_or_connection_failure_is_retried_for_either_master(self):
        for url in (NSE_MASTER, NSE_SME_MASTER):
            for error in (requests.exceptions.ReadTimeout('slow reply'),
                          requests.exceptions.ConnectTimeout('slow connection'),
                          requests.exceptions.ConnectionError('connection reset'),
                          self.http_error(503)):
                with self.subTest(url=url, error=type(error).__name__), tempfile.TemporaryDirectory() as folder:
                    root = Path(folder); folders = []
                    responses = {NSE_MASTER: [MAIN], NSE_SME_MASTER: [SME]}
                    responses[url].insert(0, error)
                    with patch('v13_collection.Store.get', self.fetcher(responses, folders)), \
                            patch('v13_collection.time.sleep') as sleep:
                        frame = universe(root, '2026-10-30')
                    self.assertEqual(len(frame), 2)
                    self.assertEqual(len(folders), 3)
                    retry_folders = folders[:2] if url == NSE_MASTER else folders[1:]
                    self.assertEqual(len(set(retry_folders)), 2)
                    sleep.assert_called_once_with(1)
                    audit = json.loads((root / 'evidence/universe_source_audit.json').read_text())
                    self.assertEqual(audit['status'], 'ACCEPTED')
                    failed = [r for r in audit['attempts'] if r['status'] == 'FETCH_FAILED']
                    self.assertEqual(len(failed), 1)
                    self.assertTrue(failed[0]['transient'])
                    self.assertTrue(failed[0]['will_retry'])
                    self.assertIn('observed_at_utc', failed[0])

    def test_retry_budget_is_shared_by_fetch_and_identity_failures(self):
        with tempfile.TemporaryDirectory() as folder:
            folders = []
            responses = {NSE_MASTER: [requests.exceptions.ReadTimeout(), INVALID, MAIN], NSE_SME_MASTER: [SME]}
            with patch('v13_collection.Store.get', self.fetcher(responses, folders)), \
                    patch('v13_collection.time.sleep'):
                self.assertEqual(len(universe(folder, '2026-10-30')), 2)
            self.assertEqual(len(folders), 4)
            self.assertEqual(len(set(folders[:3])), 3)
            audit = json.loads((Path(folder) / 'evidence/universe_source_audit.json').read_text())
            self.assertEqual([r['status'] for r in audit['attempts']],
                             ['FETCH_FAILED', 'REJECTED', 'ACCEPTED', 'ACCEPTED'])
            self.assertEqual([r['attempt'] for r in audit['attempts']], [1, 2, 3, 1])

    def test_exhausted_sme_timeouts_preserve_prior_universe_and_frozen_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); folders = []
            originals = {'universe.parquet': b'previous accepted universe',
                         'prospective_observations.parquet': b'frozen complete ledger',
                         'active_heads.joblib': b'frozen heads',
                         'champion_latest_watchlist.json': b'frozen probabilities'}
            for name, raw in originals.items():
                (root / name).write_bytes(raw)
            responses = {NSE_MASTER: [MAIN], NSE_SME_MASTER: [requests.exceptions.ReadTimeout()] * 3}
            with patch('v13_collection.Store.get', self.fetcher(responses, folders)), \
                    patch('v13_collection.time.sleep') as sleep, \
                    self.assertRaises(requests.exceptions.ReadTimeout):
                universe(root, '2026-10-30')
            self.assertEqual([call.args[0] for call in sleep.call_args_list], [1, 2])
            self.assertEqual(len(folders), 4)
            for name, raw in originals.items():
                self.assertEqual((root / name).read_bytes(), raw)
            audit = json.loads((root / 'evidence/universe_source_audit.json').read_text())
            self.assertEqual(audit['status'], 'SOURCE_FETCH_FAILED')
            self.assertFalse(audit['previous_universe_used_as_current'])
            self.assertFalse(audit['identity_checks_relaxed'])
            self.assertFalse(audit['attempts'][-1]['will_retry'])
            self.assertFalse((root / 'universe.parquet.tmp').exists())

    def test_access_missing_tls_proxy_and_unknown_failures_are_not_retried(self):
        errors = [self.http_error(s) for s in (401, 403, 404, 429, 501)]
        errors += [requests.exceptions.SSLError('certificate failure'),
                   requests.exceptions.ProxyError('proxy configuration'),
                   ValueError('cache integrity failure')]
        for error in errors:
            with self.subTest(error=str(error)), tempfile.TemporaryDirectory() as folder:
                with patch('v13_collection.Store.get', side_effect=error) as get, \
                        patch('v13_collection.time.sleep') as sleep, \
                        self.assertRaises(type(error)):
                    universe(folder, '2026-10-30')
                self.assertEqual(get.call_count, 1)
                sleep.assert_not_called()
                audit = json.loads((Path(folder) / 'evidence/universe_source_audit.json').read_text())
                self.assertFalse(audit['attempts'][0]['transient'])
                self.assertFalse(audit['attempts'][0]['will_retry'])

    def test_transient_http_replies_have_a_finite_budget(self):
        for status in (408, 500, 502, 503, 504):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as folder:
                with patch('v13_collection.Store.get', side_effect=self.http_error(status)) as get, \
                        patch('v13_collection.time.sleep') as sleep, \
                        self.assertRaises(requests.exceptions.HTTPError):
                    universe(folder, '2026-10-30')
                self.assertEqual(get.call_count, 3)
                self.assertEqual([call.args[0] for call in sleep.call_args_list], [1, 2])
                audit = json.loads((Path(folder) / 'evidence/universe_source_audit.json').read_text())
                self.assertEqual(audit['status'], 'SOURCE_FETCH_FAILED')
                self.assertFalse(audit['attempts'][-1]['will_retry'])

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

    def test_partial_write_cannot_replace_prior_universe_or_pollute_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); prior = b'previous accepted universe'; folders = []
            (root / 'universe.parquet').write_bytes(prior)
            def fail_write(frame, path, **kwargs):
                Path(path).write_bytes(b'partial parquet')
                raise OSError('disk write failed')
            with patch('v13_collection.Store.get', self.fetcher(
                    {NSE_MASTER: [MAIN], NSE_SME_MASTER: [SME]}, folders)), \
                    patch.object(pd.DataFrame, 'to_parquet', fail_write), \
                    self.assertRaisesRegex(OSError, 'disk write failed'):
                universe(root, '2026-10-22')
            self.assertEqual((root / 'universe.parquet').read_bytes(), prior)
            self.assertFalse((root / 'universe.parquet.tmp').exists())


if __name__ == '__main__':
    unittest.main()
