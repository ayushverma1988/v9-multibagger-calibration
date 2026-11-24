import unittest
import pandas as pd
from v13_registration_time import registration_time_gate


class RegistrationTimeTests(unittest.TestCase):
    def original(self):
        return {'date':pd.Timestamp('2026-11-20'),'information_cutoff_utc':'2026-11-20T16:30:00Z',
          'official_source_sha256':'a'*64,'official_first_retrieved_utc':'2026-11-20T13:00:00Z',
          'scoring_price_history_source':'ORIGINAL_ADJUSTED_PROVIDER','source_sha256':'b'*64,
          'scoring_original_provider_sha256':'b'*64,'scoring_original_provider_first_retrieved_utc':'2026-11-20T15:00:00Z'}

    def gate(self, rows):return registration_time_gate(pd.DataFrame(rows),'2026-11-23T12:00:00Z')

    def test_original_and_mixed_missing_proof(self):
        row=self.original();self.assertTrue(self.gate([row])['passed'])
        for change in ({'scoring_original_provider_first_retrieved_utc':None},
                       {'scoring_original_provider_first_retrieved_utc':'2026-11-23T12:00:00Z'},
                       {'scoring_original_provider_sha256':'c'*64},
                       {'official_first_retrieved_utc':'2026-11-23T12:00:00Z'}):
            with self.subTest(change=change):
                gate=self.gate([row,{**row,**change}]);self.assertFalse(gate['passed']);self.assertEqual(gate['rows_with_unproven_time'],1)

    def test_single_source_requires_bound_timely_snapshot_and_original_nse(self):
        row={**self.original(),'official_first_retrieved_utc':'2026-11-23T12:00:00Z',
          'scoring_price_history_source':'SINGLE_SOURCE_ADJUSTED_SCORING_ONLY',
          'scoring_fallback_source_sha256':'d'*64,'scoring_snapshot_sha256':'d'*64,
          'scoring_snapshot_ready_utc':'2026-11-20T15:00:00Z',
          'scoring_snapshot_nse_sha256':'a'*64,'scoring_snapshot_nse_retrieved_utc':'2026-11-20T14:00:00Z'}
        self.assertTrue(self.gate([row])['passed'])
        for change in ({'scoring_snapshot_nse_sha256':'c'*64},
                       {'scoring_snapshot_ready_utc':'2026-11-20T16:31:00Z'},
                       {'scoring_snapshot_sha256':'c'*64}):
            self.assertFalse(self.gate([{**row,**change}])['passed'])

    def test_missing_or_future_actual_cutoff_blocks(self):
        for value in (None,'2026-11-20T17:00:00Z','2026-11-20 16:30:00'):
            gate=self.gate([{**self.original(),'information_cutoff_utc':value}])
            self.assertFalse(gate['passed'])

    def test_empty_frame_blocks(self):self.assertFalse(self.gate([])['passed'])

if __name__=='__main__':unittest.main()
