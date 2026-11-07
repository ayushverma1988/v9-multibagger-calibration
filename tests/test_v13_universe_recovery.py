import hashlib
import json
import tempfile
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch

import requests

from v13_collection import universe
from v11_4_exchange_reference_universe import NSE_MASTER, NSE_SME_MASTER
from v13_universe_recovery import captured_master, FORMAT


DAY = '2026-11-06'
NOW = datetime(2026, 11, 6, 16, 0, tzinfo=timezone.utc)
HEADER = 'SYMBOL,NAME OF COMPANY,SERIES,DATE OF LISTING,ISIN NUMBER\n'
MAIN = (HEADER + 'RELIANCE,Reliance Industries Limited,EQ,29-Nov-1995,INE002A01018\n').encode()
SME = (HEADER + 'HDFCBANK,HDFC Bank Limited,EQ,08-Nov-1995,INE040A01034\n').encode()


class UniverseRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.seed = self.root / 'seed'
        self.day = self.seed / DAY
        self.day.mkdir(parents=True)
        self.output = self.root / 'capture'
        self.install(SME)

    def install(self, raw):
        digest = hashlib.sha256(raw).hexdigest()
        (self.day / (digest + '.csv')).write_bytes(raw)
        self.manifest = {'format': FORMAT, 'source_asof': DAY, 'captures': [{
            'source_url': NSE_SME_MASTER, 'final_url': NSE_SME_MASTER,
            'http_status': 200, 'sha256': digest, 'bytes': len(raw),
            'body_file': digest + '.csv',
            'first_retrieved_utc': '2026-11-06T14:51:58+00:00',
            'independently_reverified_utc': '2026-11-06T15:00:00+00:00'}]}
        self.write_manifest()

    def write_manifest(self):
        (self.day / 'manifest.json').write_text(json.dumps(self.manifest))

    def recover(self, **kw):
        return captured_master(NSE_SME_MASTER, DAY, self.output,
                               seed_root=self.seed, now=kw.pop('now', NOW), **kw)

    def fetch(self, store, url):
        if url == NSE_SME_MASTER:
            raise requests.exceptions.ReadTimeout('NSE source timed out')
        return MAIN, {'sha256': hashlib.sha256(MAIN).hexdigest(),
                      'first_retrieved_utc': '2026-11-06T14:00:00+00:00'}

    def fixture_recovery(self, url, day, folder):
        return captured_master(url, day, folder, seed_root=self.seed, now=NOW)

    def test_original_bytes_clocks_and_transport_are_preserved(self):
        raw, proof = self.recover()
        self.assertEqual(raw, SME)
        self.assertEqual((self.output / proof['body_file']).read_bytes(), SME)
        self.assertEqual(proof['first_retrieved_utc'], '2026-11-06T14:51:58+00:00')
        self.assertEqual(proof['recovery_used_at_utc'], NOW.isoformat())
        self.assertFalse(proof['remote_refetched_by_this_worker'])
        self.assertTrue(proof['same_day_original_recovery'])
        self.assertEqual(proof['recovery_manifest_sha256'],
                         hashlib.sha256((self.day / 'manifest.json').read_bytes()).hexdigest())

    def test_missing_or_wrong_day_capture_is_not_current(self):
        for day in ('2026-11-05', '2026-11-07'):
            self.assertIsNone(captured_master(NSE_SME_MASTER, day, self.output,
                                             seed_root=self.seed, now=NOW))
        self.assertFalse(self.output.exists())

    def test_overnight_resume_matches_source_date_and_keeps_original_clock(self):
        now=datetime(2026,11,7,1,tzinfo=timezone.utc)
        raw,proof=self.recover(now=now)
        self.assertEqual(raw,SME);self.assertEqual(proof['source_asof'],DAY)
        self.assertTrue(proof['overnight_same_source_resume']);self.assertEqual(proof['runner_date_IST'],'2026-11-07')
        self.assertEqual(proof['first_retrieved_utc'],'2026-11-06T14:51:58+00:00')
        self.assertFalse(proof['remote_refetched_by_this_worker'])
        self.assertIsNone(captured_master(NSE_SME_MASTER,'2026-11-07',self.output,seed_root=self.seed,now=now))

    def test_first_capture_age_not_latest_reverification_bounds_recovery(self):
        first=datetime.fromisoformat(self.manifest['captures'][0]['first_retrieved_utc'])
        self.assertIsNotNone(self.recover(now=first+timedelta(hours=24)))
        self.manifest['captures'][0]['independently_reverified_utc']=(first+timedelta(hours=23)).isoformat();self.write_manifest()
        self.assertIsNone(self.recover(now=first+timedelta(hours=24,seconds=1)))

    def test_malformed_csv_can_recover_only_after_three_attempts_and_strict_parse(self):
        raw=b'not a csv\none column\n1,2,3\n';calls=[]
        def fetch(store,url):
            calls.append(url)
            if url==NSE_MASTER:return self.fetch(store,url)
            return raw,{'sha256':hashlib.sha256(raw).hexdigest(),'first_retrieved_utc':NOW.isoformat()}
        with patch('v13_collection.Store.get',fetch),patch('v13_collection.time.sleep'),patch('v13_universe_recovery.captured_master',self.fixture_recovery):
            out=universe(self.root/'state',DAY)
        self.assertEqual(len(out),2);self.assertEqual(calls.count(NSE_SME_MASTER),3)
        a=json.loads((self.root/'state/evidence/universe_source_audit.json').read_text())
        self.assertEqual(sum(r['status']=='MALFORMED_CSV_RESPONSE' for r in a['attempts']),3)
        self.assertTrue(a['same_day_original_recovery_used']);self.assertFalse(a['identity_checks_relaxed'])

    def test_malformed_then_identity_conflict_blocks_later_transport_recovery(self):
        raw=b'not a csv\none column\n1,2,3\n';replies=[raw,SME+SME.splitlines(keepends=True)[1],requests.ReadTimeout()]
        def fetch(store,url):
            if url==NSE_MASTER:return self.fetch(store,url)
            item=replies.pop(0)
            if isinstance(item,Exception):raise item
            return item,{'sha256':hashlib.sha256(item).hexdigest(),'first_retrieved_utc':NOW.isoformat()}
        with patch('v13_collection.Store.get',fetch),patch('v13_collection.time.sleep'),patch('v13_universe_recovery.captured_master') as recovery,self.assertRaises(requests.ReadTimeout):
            universe(self.root/'state',DAY)
        recovery.assert_not_called()

    def test_missing_source_never_substitutes_another_master(self):
        self.assertIsNone(captured_master(NSE_MASTER, DAY, self.output,
                                         seed_root=self.seed, now=NOW))

    def test_hash_and_length_tampering_fail_closed(self):
        item = self.manifest['captures'][0]
        path = self.day / item['body_file']
        for raw in (SME[:-1], b'X' + SME[1:]):
            with self.subTest(length=len(raw)):
                path.write_bytes(raw)
                with self.assertRaises(ValueError):
                    self.recover()
                self.assertFalse(self.output.exists())

    def test_unsafe_body_path_is_rejected(self):
        for body in ('../outside.csv', '/tmp/master.csv', 'other.csv'):
            self.manifest['captures'][0]['body_file'] = body
            self.write_manifest()
            with self.assertRaises(ValueError):
                self.recover()

    def test_wrong_format_and_source_date_are_rejected(self):
        for field, value in [('format', 'unknown'), ('source_asof', '2026-11-05')]:
            self.install(SME)
            self.manifest[field] = value
            self.write_manifest()
            with self.assertRaises(ValueError):
                self.recover()

    def test_duplicate_source_is_ambiguous(self):
        self.manifest['captures'].append(dict(self.manifest['captures'][0]))
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'Ambiguous'):
            self.recover()

    def test_future_stale_reversed_and_naive_clocks_are_rejected(self):
        cases = [
            ('first_retrieved_utc', '2026-11-05T14:00:00+00:00'),
            ('first_retrieved_utc', '2026-11-06T15:30:00+00:00'),
            ('first_retrieved_utc', '2026-11-06T14:00:00'),
            ('independently_reverified_utc', '2026-11-06T17:00:00+00:00'),
            ('independently_reverified_utc', '2026-11-06T14:00:00+00:00'),
        ]
        for key, value in cases:
            with self.subTest(key=key, value=value):
                self.install(SME)
                self.manifest['captures'][0][key] = value
                self.write_manifest()
                with self.assertRaises(ValueError):
                    self.recover()

    def test_wrong_source_redirect_status_and_length_are_rejected(self):
        cases = [('final_url', 'https://example.com/master.csv'),
                 ('http_status', 403), ('bytes', True), ('bytes', 6_000_000)]
        for key, value in cases:
            with self.subTest(key=key):
                self.install(SME)
                self.manifest['captures'][0][key] = value
                self.write_manifest()
                with self.assertRaises(ValueError):
                    self.recover()
        with self.assertRaises(ValueError):
            captured_master('https://example.com/master.csv', DAY, self.output, now=NOW)

    def test_universe_recovers_after_three_timeouts_without_touching_frozen_files(self):
        state = self.root / 'state'
        state.mkdir()
        original = {'active_heads.joblib': b'frozen head',
                    'prospective_observations.parquet': b'complete original ledger',
                    'champion_latest_watchlist.json': b'frozen probabilities'}
        for name, raw in original.items():
            (state / name).write_bytes(raw)
        with patch('v13_collection.Store.get', lambda store, url: self.fetch(store, url)) as get, \
                patch('v13_collection.time.sleep') as sleep, \
                patch('v13_universe_recovery.captured_master', self.fixture_recovery):
            frame = universe(state, DAY)
        self.assertEqual(len(frame), 2)
        self.assertEqual([c.args[0] for c in sleep.call_args_list], [1, 2])
        audit = json.loads((state / 'evidence/universe_source_audit.json').read_text())
        self.assertEqual(audit['status'], 'ACCEPTED')
        self.assertTrue(audit['same_day_original_recovery_used'])
        self.assertFalse(audit['identity_checks_relaxed'])
        self.assertFalse(audit['previous_universe_used_as_current'])
        self.assertFalse(audit['attempts'][-1]['remote_refetched_by_this_worker'])
        for name, raw in original.items():
            self.assertEqual((state / name).read_bytes(), raw)

    def test_recovered_identity_still_uses_the_original_strict_parser(self):
        self.install(SME + SME.splitlines(keepends=True)[1])
        state = self.root / 'state'
        state.mkdir()
        (state / 'universe.parquet').write_bytes(b'prior accepted master')
        with patch('v13_collection.Store.get', lambda store, url: self.fetch(store, url)), \
                patch('v13_collection.time.sleep'), \
                patch('v13_universe_recovery.captured_master', self.fixture_recovery), \
                self.assertRaisesRegex(ValueError, 'Ambiguous current NSE equity identity'):
            universe(state, DAY)
        self.assertEqual((state / 'universe.parquet').read_bytes(), b'prior accepted master')

    def test_access_error_cannot_activate_recovery(self):
        response = requests.Response()
        response.status_code = 403
        for status in (401, 403, 404, 429):
            response.status_code = status
            with self.subTest(status=status), \
                    patch('v13_collection.Store.get', side_effect=requests.HTTPError(response=response)), \
                    patch('v13_universe_recovery.captured_master') as recovery, \
                    self.assertRaises(requests.HTTPError):
                universe(self.root / str(status), DAY)
            recovery.assert_not_called()

    def test_identity_failure_before_a_timeout_cannot_activate_recovery(self):
        replies = [SME + SME.splitlines(keepends=True)[1],
                   requests.ReadTimeout(), requests.ReadTimeout()]
        def fetch(store, url):
            if url == NSE_MASTER:
                return self.fetch(store, url)
            item = replies.pop(0)
            if isinstance(item, Exception):
                raise item
            return item, {'sha256': hashlib.sha256(item).hexdigest(),
                          'first_retrieved_utc': '2026-11-06T14:00:00+00:00'}
        with patch('v13_collection.Store.get', fetch), patch('v13_collection.time.sleep'), \
                patch('v13_universe_recovery.captured_master') as recovery, \
                self.assertRaises(requests.ReadTimeout):
            universe(self.root / 'state', DAY)
        recovery.assert_not_called()


if __name__ == '__main__':
    unittest.main()
