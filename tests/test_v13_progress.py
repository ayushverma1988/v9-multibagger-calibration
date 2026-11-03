import unittest,tempfile,json
from pathlib import Path
import pandas as pd
from v13_progress import snapshot

class Progress(unittest.TestCase):
    def test_measured_denominators_and_accuracy_are_explicit(self):
        with tempfile.TemporaryDirectory() as t:
            r={'asof':'2026-10-13','current_verified_market_rows':161,'financial_complete_rows':75,
               'execution_chain_supported_rows':0,'source_bound_earnings_scenario_rows':0,'independently_verified_mature_outcomes':0}
            d=pd.DataFrame({'entry_verified':[True]*207+[False]*777})
            out=snapshot(t,r,d)
            self.assertEqual(out['areas'][0]['coverage_percent'],46.58)
            self.assertEqual(out['areas'][3]['denominator'],984)
            self.assertEqual(out['areas'][3]['coverage_percent'],21.04)
            self.assertFalse(out['accuracy_approved'])
            self.assertTrue(out['different_denominators_must_not_be_averaged'])
            self.assertTrue((Path(t)/'evidence/progress_status.json').exists())

    def test_empty_data_is_unknown_not_completed(self):
        with tempfile.TemporaryDirectory() as t:
            out=snapshot(t,{},pd.DataFrame())
            self.assertTrue(all(z['coverage_percent'] is None for z in out['areas']))

    def test_any_scenario_is_not_necessarily_doubling_support(self):
        with tempfile.TemporaryDirectory() as t:
            out=snapshot(t,{'current_verified_market_rows':10,'source_bound_earnings_scenario_rows':7,'source_bound_2x_scenario_rows':2},pd.DataFrame())
            self.assertEqual(out['areas'][2]['coverage_percent'],20)

if __name__=='__main__':unittest.main()
