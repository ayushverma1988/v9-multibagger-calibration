import hashlib
import io
import json
import tempfile
import unittest
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import requests

import v13_runner as runner
from v12_3_official_monitor import official_url
from v13_official_session import SourceUnavailable
from v13_working_archive import save as save_archive

DAY = '2026-11-27'
CLOCK = datetime(2026, 11, 28, 4, tzinfo=timezone.utc)
STAMP = '2026-11-27T15:17:16+00:00'


def http(code):
    response = requests.Response()
    response.status_code = code
    return requests.HTTPError(str(code), response=response)


def fixture(root, day=DAY, stamp=STAMP):
    folder = root.parent / 'v13_public_breadth_snapshot'
    folder.mkdir()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as archive:
        archive.writestr('bhav.csv',
            'TradDt,BizDt,Src,Sgmt,ISIN,TckrSymb,SctySrs,OpnPric,HghPric,LwPric,ClsPric,TtlTradgVol,TtlNbOfTxsExctd,NewBrdLotQty\n'
            f'{day},{day},NSE,CM,INE123456789,ABC,EQ,100,105,98,103,1000,5,1\n')
    raw = buf.getvalue()
    proof = {'source_url':official_url(day), 'sha256':hashlib.sha256(raw).hexdigest(),
             'bytes':len(raw), 'first_retrieved_utc':stamp}
    (folder/'nse_source_session.zip').write_bytes(raw)
    (folder/'receipt.json').write_text(json.dumps({'source_asof':day,'nse_original':proof}))
    return folder, proof


class OfficialSessionRecovery(unittest.TestCase):
    def test_same_date_original_recovers_without_network_and_preserves_clock(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)/'state'
            _, proof = fixture(root)
            with patch('v12_3_official_monitor.requests.get', side_effect=http(403)) as fetch:
                date, report, errors = runner.last_completed_session(root, '2026-11-28', now=CLOCK)
            fetch.assert_not_called()
            self.assertEqual((date, errors), (DAY, []))
            self.assertEqual(report['collection_source'], 'VERIFIED_TIMED_NSE_CAPTURE')
            self.assertEqual(report['source_sha256'], proof['sha256'])
            self.assertEqual(report['first_retrieved_utc'], STAMP)
            result = pd.read_parquet(root/'official'/f'{DAY}_official_market.parquet')
            self.assertEqual(result.first_retrieved_utc.tolist(), [STAMP])
            self.assertEqual(result.close.tolist(), [103])
            self.assertEqual(result.corporate_actions_verified.tolist(), [False])

    def test_blocked_current_date_does_not_probe_or_relabel_older_capture(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)/'state'
            fixture(root, '2026-11-26', '2026-11-26T15:00:00+00:00')
            with patch('v12_3_official_monitor.requests.get', side_effect=http(403)) as fetch:
                with self.assertRaises(SourceUnavailable) as caught:
                    runner.last_completed_session(root, DAY, now=CLOCK)
            fetch.assert_called_once()
            self.assertEqual(caught.exception.attempts,[{'date':DAY,'reason':'HTTP_403'}])

    def test_absent_file_can_use_correctly_dated_earlier_session(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)/'state'
            fixture(root, '2026-11-26', '2026-11-26T15:00:00+00:00')
            with patch('v12_3_official_monitor.requests.get', side_effect=http(404)) as fetch:
                date,_,attempts=runner.last_completed_session(root, DAY, now=CLOCK)
            fetch.assert_called_once()
            self.assertEqual(date,'2026-11-26')
            self.assertEqual(attempts,[{'date':DAY,'reason':'HTTP_404'}])

    def test_bad_clock_or_hash_cannot_be_published(self):
        for tamper in ['late','hash','future']:
            with self.subTest(tamper=tamper), tempfile.TemporaryDirectory() as t:
                root = Path(t)/'state'
                folder,proof=fixture(root, stamp='2026-11-27T17:00:00+00:00' if tamper=='late' else STAMP)
                if tamper=='hash':
                    proof['sha256']='f'*64
                    (folder/'receipt.json').write_text(json.dumps({'source_asof':DAY,'nse_original':proof}))
                now=datetime(2026,11,27,14,tzinfo=timezone.utc) if tamper=='future' else CLOCK
                with patch('v12_3_official_monitor.requests.get') as fetch, self.assertRaises(ValueError):
                    runner.last_completed_session(root, DAY, now=now)
                fetch.assert_not_called()
                self.assertFalse((root/'official'/f'{DAY}_official_market.parquet').exists())

    def test_outage_leaves_last_completed_report_and_forecast_byte_identical(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)/'state';root.mkdir()
            summary=b'{"asof":"2026-11-26","completed_at_utc":"2026-11-26T17:20:00Z","current_verified_market_rows":1457}'
            result=root/'latest_run.json';result.write_bytes(summary)
            forecast=root/'champion_latest_watchlist.json';forecast.write_bytes(b'[]')
            with patch('v12_3_official_monitor.requests.get',side_effect=http(403)),patch.object(runner,'save') as save:
                paused=runner.run(root, root/'seed', DAY, local=True)
            save.assert_not_called()
            self.assertEqual(paused['status'],'SOURCE_UNAVAILABLE')
            self.assertEqual(paused['previous_completed_source_asof'],'2026-11-26')
            self.assertEqual(result.read_bytes(),summary)
            self.assertEqual(forecast.read_bytes(),b'[]')
            self.assertEqual(json.loads((root.parent/'v13_worker_attempt.json').read_text())['phase'],'SOURCE_UNAVAILABLE')

    def test_non_availability_failure_is_not_swallowed_or_checkpointed(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)/'state';root.mkdir()
            prior=root/'latest_run.json';prior.write_bytes(b'{}')
            with patch.object(runner,'collect_official',side_effect=ValueError('Wrong official schema')),patch.object(runner,'save') as save:
                with self.assertRaisesRegex(ValueError,'Wrong official schema'):
                    runner.run(root, root/'seed', DAY, local=True)
            save.assert_not_called()
            self.assertEqual(prior.read_bytes(),b'{}')

    def test_dateless_archive_cannot_stage_or_publish_anything(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            (root/'latest_run.json').write_text('{"asof":null}')
            with patch('v13_storage_migrate.commit') as commit, patch('v13_compact_core.build_core') as build:
                with self.assertRaisesRegex(ValueError,'without a verified source session'):
                    save_archive(root, None)
            commit.assert_not_called()
            build.assert_not_called()
