import unittest
import numpy as np
from v13_financial_evidence import financial_quality,REQUIRED

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

if __name__=='__main__':unittest.main()
