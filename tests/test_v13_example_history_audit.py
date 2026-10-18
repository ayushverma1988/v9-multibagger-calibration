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
                {'isin': 'INE01QM01018', 'nse_current_symbol': 'EMUDHRA'},
                {'isin': 'INE417B01040', 'nse_current_symbol': 'DSSL'},
            ]).to_parquet(root / 'universe.parquet', index=False)
            dates = pd.bdate_range('2025-01-02', '2026-10-15')
            prices = 500 + np.arange(len(dates)) * .03
            bars = pd.DataFrame({'date': dates, 'isin': 'INE01QM01018',
                                 'symbol': 'EMUDHRA', 'open': prices, 'high': prices * 1.01,
                                 'low': prices * .99, 'close': prices, 'adj_close': prices,
                                 'volume': 100000})
            bars.to_parquet(root / 'history' / 'securities' /
                            (sha(b'EMUDHRA') + '.parquet'), index=False)
            summary = audit(root, '2026-10-16')
            records = json.loads((root / 'evidence' / 'example_history_audit.json').read_text())['records']
            by_name = {r['requested_symbol']: r for r in records}
            self.assertEqual(by_name['EMUDHRA']['current_status'], 'NO_SOURCE_DATE_BAR')
            self.assertGreater(by_name['EMUDHRA']['feature_eligible_dates'], 0)
            self.assertEqual(by_name['DSSL']['current_status'], 'NOT_SCORABLE_MISSING_HISTORY')
            self.assertEqual(by_name['CHEMCRUX']['current_status'], 'OUTSIDE_NSE_COLLECTION')
            self.assertEqual(len(records), 8)
            self.assertTrue(summary['listing_to_date_history_not_claimed'])


if __name__ == '__main__':
    unittest.main()
