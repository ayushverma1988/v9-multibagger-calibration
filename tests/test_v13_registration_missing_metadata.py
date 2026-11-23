import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from v13_lifecycle import register, validate_registry


class RegistryMissingMetadataTests(unittest.TestCase):
    def frame(self):
        common = {'date': pd.Timestamp('2026-11-20'), 'p6': .1, 'p12': .2,
                  'loss12': .3, 'joint12': .1, 'selected_verified': False,
                  'selected_discovery': False, 'information_cutoff_utc': '2026-11-20T16:30:00Z'}
        return pd.DataFrame([
            {**common, 'isin': 'INE000000001', 'symbol': 'A',
             'scoring_price_history_source': 'SINGLE_SOURCE_ADJUSTED_SCORING_ONLY',
             'scoring_fallback_source_sha256': 'a' * 64,
             'scoring_fallback_action_interval_certified': False,
             'scoring_fallback_provider_sha256': None,
             'scoring_fallback_provider_first_retrieved_utc': None,
             'scoring_fallback_reference_status': 'NO_TIMELY_COMPARABLE_PROVIDER',
             'scoring_fallback_overlap_sessions': 0,
             'scoring_history_series_transition_unverified': False},
            {**common, 'isin': 'INE000000002', 'symbol': 'B',
             'scoring_price_history_source': 'SINGLE_SOURCE_ADJUSTED_SCORING_ONLY',
             'scoring_fallback_source_sha256': 'a' * 64,
             'scoring_fallback_action_interval_certified': False,
             'scoring_fallback_provider_sha256': None,
             'scoring_fallback_provider_first_retrieved_utc': None,
             'scoring_fallback_reference_status': 'NO_TIMELY_COMPARABLE_PROVIDER',
             'scoring_fallback_overlap_sessions': np.nan,
             'scoring_history_series_transition_unverified': False},
        ])

    def test_mixed_optional_missing_values_are_null_and_prior_forecast_is_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            frame = self.frame()
            old = register(frame.iloc[:1], 'b' * 64, folder, '2026-11-21T10:00:00Z')
            old_bytes = old.read_bytes()
            new = register(frame, 'b' * 64, folder, '2026-11-21T10:01:00Z')
            self.assertEqual(old.read_bytes(), old_bytes)
            records = validate_registry(new)['records']
            self.assertEqual(len(records), 2)
            self.assertEqual(records[0]['scoring_fallback_overlap_sessions'], 0)
            self.assertIsNone(records[1]['scoring_fallback_overlap_sessions'])
            self.assertIsNone(records[1]['scoring_fallback_provider_sha256'])
            self.assertEqual([r['p6'] for r in records], [.1, .1])
            self.assertEqual([r['p12'] for r in records], [.2, .2])
            self.assertNotIn('NaN', new.read_text())
            json.loads(new.read_text())

    def test_missing_forecast_and_selection_flags_are_rejected(self):
        for field in ('p6', 'p12', 'loss12', 'joint12', 'selected_verified', 'selected_discovery'):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                frame = self.frame()
                frame.loc[1, field] = np.nan
                with self.assertRaises(ValueError):
                    register(frame, 'b' * 64, tmp, '2026-11-21T10:00:00Z')
                self.assertEqual(list(Path(tmp).glob('*.json')), [])

    def test_infinite_optional_metadata_is_not_silently_treated_as_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            frame = self.frame()
            frame.loc[1, 'scoring_fallback_overlap_sessions'] = np.inf
            with self.assertRaises(ValueError):
                register(frame, 'b' * 64, tmp, '2026-11-21T10:00:00Z')
            self.assertEqual(list(Path(tmp).glob('*.json')), [])


if __name__ == '__main__':
    unittest.main()
