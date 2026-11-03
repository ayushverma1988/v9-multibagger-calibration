import unittest
import numpy as np
from v13_financial_evidence import financial_quality,financial_readiness,REQUIRED
import pandas as pd

class FinancialQualityAudit(unittest.TestCase):
    def good(self):
        return dict(zip(REQUIRED,[1,0,.1,.1,.4,1.5,1.5]))

    def test_complete_is_not_quality_and_threshold_boundaries(self):
        f=self.good();self.assertTrue(financial_quality(f)['quality_pass'])
        f['annual_CFO_PAT']=-.1
        q=financial_quality(f)
        self.assertTrue(q['complete']);self.assertFalse(q['quality_pass'])
        self.assertEqual(q['failed_checks'],['annual_CFO_PAT'])
        self.assertEqual(q['status'],'KNOWN_FINANCIAL_FAILURE')

    def test_missing_input_never_passes_and_known_failure_is_retained(self):
        f=self.good();f['quarter_pat_yoy']=np.nan
        q=financial_quality(f)
        self.assertEqual(q['status'],'UNKNOWN_FINANCIAL_INPUTS')
        self.assertFalse(q['complete']);self.assertFalse(q['quality_pass'])
        f['dated_debt_equity']=2
        q=financial_quality(f)
        self.assertEqual(q['unknown_fields'],['quarter_pat_yoy'])
        self.assertEqual(q['failed_checks'],['dated_debt_equity'])
        self.assertEqual(q['status'],'KNOWN_FINANCIAL_FAILURE')

    def test_no_data_or_nonfinite_data_is_not_profitability(self):
        self.assertEqual(financial_quality({})['unknown_fields'],list(REQUIRED))
        f=self.good();f['annual_EBIT_proxy_interest_coverage']=np.inf
        self.assertFalse(financial_quality(f)['quality_pass'])

    def test_undefined_growth_is_not_a_missing_original(self):
        date=pd.Timestamp('2026-10-13');rows=[]
        for p,pat in [('2026-06-30',10),('2026-03-31',10),('2025-12-31',10),('2025-09-30',10),('2025-06-30',-1)]:
            rows.append({'period':pd.Timestamp(p),'period_kind':'quarter','reporting_mode':'standalone','available':pd.Timestamp('2026-08-01T00:00:00Z'),
              'revenue':100,'pat':pat,'eps_diluted_continuing':1})
        rows.extend([{'period':pd.Timestamp('2026-03-31'),'period_kind':'annual','reporting_mode':'standalone','available':pd.Timestamp('2026-05-01T00:00:00Z'),'pat':40,'cfo':40,'pbt':40,'finance_cost':0},
          {'period':pd.Timestamp('2026-03-31'),'period_kind':'instant','reporting_mode':'standalone','available':pd.Timestamp('2026-05-01T00:00:00Z'),'equity':100,'borrowings_current':0,'borrowings_noncurrent':0}])
        f=self.good();f['quarter_pat_yoy']=np.nan;f['annual_EBIT_proxy_interest_coverage']=np.nan
        out=financial_readiness(pd.DataFrame(rows),date,f)
        self.assertTrue(out['underlying_required_facts_complete'])
        self.assertTrue(out['latest_reported_diluted_EPS_available'])
        self.assertTrue(all(v=='OBSERVED_NONPOSITIVE_DENOMINATOR_NOT_MISSING_SOURCE' for v in out['unknown_reasons'].values()))
        self.assertFalse(financial_quality(f)['quality_pass'])

if __name__=='__main__':unittest.main()
