import json,tempfile,unittest
from pathlib import Path
import numpy as np
import pandas as pd
from v12_3_prospective_comparison import comparison_records,validate_comparison
from v12_prospective_registry import digest


class ProspectiveComparisonTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame([dict(date=pd.Timestamp('2026-10-09'),symbol='EXAMPLE',isin='INE000000001',
            technical_pass=True,close=100.,relative_strength63=.4,rsi14=75.,quarter_sales_yoy=.2,
            quarter_pat_yoy=.3,quarter_pat_margin=.1,annual_CFO_PAT=.8,dated_debt_equity=.2,
            p_hit2_6=.2,p_hit2_12=.3,p_loss30_12=.4)])

    def test_rejects_future_or_stale_or_duplicate_snapshot(self):
        for now in ['2026-10-09T09:00:00Z','2026-10-20T09:00:00Z']:
            with self.assertRaises(ValueError):comparison_records(self.frame(),'2026-10-09',now)
        with self.assertRaises(ValueError):comparison_records(pd.concat([self.frame(),self.frame()]),'2026-10-09','2026-10-10T16:00:00Z')

    def test_rejects_missing_or_incoherent_probabilities(self):
        for p in [np.nan,-.1,1.1,.1]:
            f=self.frame();f['p_hit2_12']=p
            with self.assertRaises(ValueError):comparison_records(f,'2026-10-09','2026-10-10T16:00:00Z')

    def test_missing_financials_keep_abstention_in_registered_membership(self):
        f=self.frame();f['annual_CFO_PAT']=np.nan
        r=comparison_records(f,'2026-10-09','2026-10-10T16:00:00Z')[0]
        self.assertTrue(r['variant_membership']['enriched_top10']['12'])
        self.assertFalse(r['variant_membership']['enriched_selective_max5']['12'])
        self.assertIsNone(r['dated_financial_predictors']['annual_CFO_PAT'])
        self.assertFalse(r['trading_approved'])
        self.assertEqual(r['outcome12_status'],'PENDING')

    def test_tampered_future_record_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);protocol={'entry':'verified_actual','calendar_months':[6,12]}
            reg={'schema':'V12_3_PROSPECTIVE_VARIANT_COMPARISON_1','adjudication_protocol':protocol,
                 'adjudication_protocol_sha256':digest(protocol),'records':comparison_records(self.frame(),'2026-10-09','2026-10-10T16:00:00Z')}
            (root/'registration.json').write_text(json.dumps(reg))
            (root/'public_registration_receipt.json').write_text(json.dumps({'registration_sha256':digest(reg)}))
            validate_comparison(root)
            reg['records'][0]['p_hit2_12']=.8
            (root/'registration.json').write_text(json.dumps(reg))
            with self.assertRaises(ValueError):validate_comparison(root)


if __name__=='__main__':unittest.main()
