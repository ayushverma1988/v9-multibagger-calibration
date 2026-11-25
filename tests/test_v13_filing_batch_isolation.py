import unittest
import pandas as pd
from v13_filing_batch import financial_wide, partition


def fact(version='a', **change):
    return dict(dict(symbol='A', security_isin='INE000000001', reporting_mode='consolidated',
                     period_kind='quarter', period_end='2026-09-30', period_start='2026-07-01',
                     available_at_utc='2026-11-10T09:00:00Z', source_sha256='a'*64,
                     first_retrieved_utc='2026-11-10T10:00:00Z',
                     source_url='https://nsearchives.nseindia.com/corporate/'+version+'.xml',
                     metric='pat', value=25., version_key=version), **change)


class FilingBatch(unittest.TestCase):
    def test_disagreeing_periods_reject_whole_filing_but_keep_unrelated_and_exact_duplicates(self):
        good=fact('good',symbol='B',security_isin='INE000000002',source_sha256='b'*64)
        d=pd.DataFrame([fact(),fact(metric='revenue',period_start='2026-07-03'),
                        fact(period_kind='annual'),good,good])
        with self.assertRaisesRegex(ValueError,'Conflicting financial source metadata'):
            financial_wide(d)
        out,accepted,held,count=partition(d)
        self.assertEqual((accepted,set(held),count),({'good'},{'a'},2))
        self.assertEqual(out.symbol.tolist(),['B'])
        self.assertEqual(held['a'][0]['conflicting_metadata'],{'period_start':['2026-07-01','2026-07-03']})

    def test_equal_fact_values_do_not_hide_disagreeing_source_or_retrieval_receipts(self):
        a,b=fact(),fact('alias',first_retrieved_utc='2026-11-11T10:00:00Z')
        for rows in ([a,b],[b,a]):
            out,accepted,held,count=partition(pd.DataFrame(rows))
            self.assertTrue(out.empty);self.assertFalse(accepted);self.assertEqual(count,0)
            self.assertEqual(set(held),{'a','alias'})
            self.assertEqual(set(held['a'][0]['conflicting_metadata']),{'source_url','first_retrieved_utc'})

    def test_conflicting_numeric_facts_do_not_become_a_chosen_value(self):
        bad=fact(value=26.)
        with self.assertRaisesRegex(ValueError,'Conflicting original facts'):
            financial_wide(pd.DataFrame([fact(),bad]))
        out,accepted,held,_=partition(pd.DataFrame([fact(),bad]))
        self.assertFalse(accepted);self.assertTrue(out.empty)
        self.assertEqual(held['a'][0]['conflicting_facts'],{'pat':['25.0','26.0']})

    def test_distinct_reporting_bases_periods_and_source_vintages_stay_separate(self):
        rows=[fact(metric='cfo',period_kind='annual',period_start='2025-10-01'),
              fact(metric='cfo',period_kind='ytd',period_start='2026-04-01'),
              fact(reporting_mode='standalone'),fact(),
              fact('later',available_at_utc='2026-11-12T09:00:00Z')]
        out,accepted,held,_=partition(pd.DataFrame(rows))
        self.assertEqual(len(out),5);self.assertEqual(accepted,{'a','later'});self.assertFalse(held)
        self.assertEqual(out[out.cfo.notna()].period_kind.tolist(),['annual','ytd'])
        self.assertEqual(out.available.nunique(),2)
