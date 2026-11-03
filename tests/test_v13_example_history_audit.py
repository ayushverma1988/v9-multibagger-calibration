import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from v12_dated_source_collection import sha
from v13_example_history_audit import audit


class ExampleHistoryAuditTests(unittest.TestCase):
    def test_separates_stale_absent_and_outside_universe_without_inventing_a_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'history' / 'securities').mkdir(parents=True)
            pd.DataFrame([
                {'isin': 'INE01QM01018', 'nse_current_symbol': 'EMUDHRA',
                 'nse_current_series': 'EQ', 'nse_listing_date': '2022-06-01'},
                {'isin': 'INE417B01040', 'nse_current_symbol': 'DSSL',
                 'nse_current_series': 'EQ', 'nse_listing_date': '2000-01-01'},
            ]).to_parquet(root / 'universe.parquet', index=False)
            dates = pd.bdate_range('2025-01-02', '2026-10-15')
            prices = 500 + np.arange(len(dates)) * .03
            bars = pd.DataFrame({'date': dates, 'isin': 'INE01QM01018',
                                 'symbol': 'EMUDHRA', 'open': prices, 'high': prices * 1.01,
                                 'low': prices * .99, 'close': prices, 'adj_close': prices,
                                 'volume': 100000})
            # The pattern calculator drops a zero-volume day. Its preceding
            # day's pattern must never be attached to this later date.
            bars.loc[bars.index[-1], 'volume'] = 0
            bars.to_parquet(root / 'history' / 'securities' /
                            (sha(b'EMUDHRA') + '.parquet'), index=False)
            (root / 'history_attempts.json').write_text(json.dumps({'DSSL': '2026-10-18T02:00:00Z'}))
            (root / 'early_potential').mkdir()
            (root / 'early_potential' / 'latest.json').write_text(json.dumps({
                'source_date':'2026-10-16', 'scored_states':[
                    {'isin':'INE01QM01018', 'early_state':'CHART_WATCH',
                     'early_pattern':'DEEP_RESET_BASE', 'early_rank':None}]}))
            summary = audit(root, '2026-10-16', collection_report={'errors': [
                {'symbol': 'DSSL', 'error': 'Provider security unavailable'}]})
            records = json.loads((root / 'evidence' / 'example_history_audit.json').read_text())['records']
            by_name = {r['requested_symbol']: r for r in records}
            self.assertEqual(by_name['EMUDHRA']['current_status'], 'NO_SOURCE_DATE_BAR')
            self.assertGreater(by_name['EMUDHRA']['feature_eligible_dates'], 0)
            self.assertEqual(by_name['EMUDHRA']['last_bar'], '2026-10-15')
            self.assertEqual(by_name['EMUDHRA']['last_feature_date'], '2026-10-14')
            self.assertEqual(by_name['EMUDHRA']['nse_listing_date'], '2022-06-01')
            self.assertEqual(by_name['EMUDHRA']['early_potential_status'], 'CHART_WATCH')
            self.assertEqual(by_name['DSSL']['current_status'], 'NOT_SCORABLE_MISSING_HISTORY')
            self.assertEqual(by_name['DSSL']['early_potential_status'], 'NOT_SCORABLE_WITH_CURRENT_DATA')
            self.assertEqual(by_name['DSSL']['last_public_history_attempt_utc'], '2026-10-18T02:00:00Z')
            self.assertEqual(by_name['DSSL']['current_run_public_history_error'], 'Provider security unavailable')
            self.assertEqual(by_name['CHEMCRUX']['current_status'], 'OUTSIDE_NSE_COLLECTION')
            self.assertEqual(len(records), 9)
            self.assertTrue(summary['listing_to_date_history_not_claimed'])


if __name__ == '__main__':
    unittest.main()
