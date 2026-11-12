import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import pandas as pd

import v13_available_financials as available
from v13_priority_research import export, unavailable
import test_v13_available_financials as fixtures


class DualRSI(unittest.TestCase):
    def population(self):
        fixture = fixtures.AvailableFinancials()
        frames, facts = [], []
        for n, rsi in enumerate((50., 50.001, 70., 70.001)):
            isin = f'INE00000000{n+1}'
            frames.append(fixture.frame().assign(isin=isin, symbol=f'A{n}', symbol_official=f'A{n}', rsi14=rsi))
            facts.append(fixture.wide().assign(security_isin=isin, symbol=f'A{n}'))
        return pd.concat(frames, ignore_index=True), pd.concat(facts, ignore_index=True)

    def test_strict_boundaries_nested_sets_and_unchanged_input_probabilities(self):
        frame, wide = self.population(); before = frame.copy(deep=True)
        with patch.object(available, 'enrich', wraps=available.enrich) as enrich:
            outputs = available.screen_many(frame, wide)
        self.assertEqual(enrich.call_count, 1)
        self.assertEqual(set(outputs), {70, 50})
        d70, rows70, audit70 = outputs[70]
        d50, rows50, audit50 = outputs[50]
        self.assertEqual(d70.loc[d70.available_research_eligible, 'symbol'].tolist(), ['A3'])
        self.assertEqual(d50.loc[d50.available_research_eligible, 'symbol'].tolist(), ['A1', 'A2', 'A3'])
        self.assertFalse(rows70[1]['eligible']); self.assertTrue(rows50[1]['eligible'])
        self.assertEqual(audit70['RSI_rule'], 'strict >70; simple rolling 14 sessions')
        self.assertEqual(audit50['RSI_rule'], 'strict >50; simple rolling 14 sessions')
        pd.testing.assert_frame_equal(frame, before)
        for d in (d70, d50):
            pd.testing.assert_frame_equal(d[['p6','p12','joint12','loss12']], before[['p6','p12','joint12','loss12']])
        self.assertEqual(available.RSI_MIN, 70)

    def test_lower_RSI_does_not_remove_financial_or_identity_exclusions(self):
        frame, wide = self.population()
        wide.loc[wide.period_kind.eq('annual'), 'cfo'] = -1.
        for d, rows, _ in available.screen_many(frame, wide).values():
            self.assertFalse(d.available_research_eligible.any())
            self.assertTrue(all(r['tier'] == 'OBSERVED_CURRENT_FINANCIAL_WEAKNESS' for r in rows))

    def test_both_lists_are_exported_even_when_only_the_lower_screen_has_names(self):
        frame, wide = self.population(); frame['rsi14'] = 60.
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)/'available_financial_research'; out.mkdir()
            receipt = {'replay_verified': True}
            (out/'latest.json').write_text(json.dumps({'prediction_replay':receipt}))
            with patch('v13_priority_research.validate_exact_replay'):
                result = export(frame, wide, folder, receipt)
            self.assertTrue(result['dual_RSI_required'])
            combined = json.loads((Path(folder)/'priority_research/latest_dual.json').read_text())
            self.assertEqual(combined['lists']['70']['selected'], 0)
            self.assertEqual(combined['lists']['50']['selected'], 4)
            self.assertTrue((Path(folder)/'priority_research/latest_RSI70.md').exists())
            self.assertTrue((Path(folder)/'priority_research/latest_RSI50.md').exists())
            self.assertTrue(combined['same_model_and_probabilities'])

    def test_unavailable_data_produces_two_explicit_blocked_lists(self):
        with tempfile.TemporaryDirectory() as folder:
            result = unavailable(folder, '2026-11-11', 'NO_CURRENT_VERIFIED_INPUTS')
            for key in ('70','50'):
                self.assertEqual(result['dual_RSI'][key]['records'], [])
                self.assertTrue(result['dual_RSI'][key]['not_a_no_opportunity_conclusion'])

    def test_explicit_threshold_does_not_mutate_default_and_invalid_threshold_fails(self):
        fixture = fixtures.AvailableFinancials()
        self.assertTrue(available.screen(fixture.frame().assign(rsi14=60), fixture.wide(), rsi_min=50)[0].available_research_eligible.iloc[0])
        self.assertEqual(available.RSI_MIN, 70)
        for invalid in (True, float('nan'), -1, 100, '50'):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                available.screen(fixture.frame(), fixture.wide(), rsi_min=invalid)


if __name__ == '__main__': unittest.main()
