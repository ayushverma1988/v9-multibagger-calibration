import hashlib
import unittest
from io import BytesIO
import pandas as pd
from v11_4_lagged_company_pe_pit_gate import eligibility

def sample():
    data=pd.DataFrame({
       "date":["2026-10-08"]*2,
       "symbol":["ALPHA","BETA"],"isin":["INE000001","INE000002"],
       "company_pe":[21.5,None],
       "company_pe_source_file_sha256":["a"*64]*2,
       "pe_source_pit_at_1530_IST_verified":[False,False],
       "market_cap_crore":[None,None],
       "industry_pe_reference":[None,None],
    })
    original=data.to_csv(index=False).encode()
    m={
       "type":"EXCHANGE_ORIGINAL_COMPANY_PE_RESEARCH_COMPANION_NOT_A_MODEL_PREDICTION",
       "date_IST":"2026-10-08",
       "archived_at_utc":"2026-10-09T01:23:58+00:00",
       "original_NSE_company_PE_url":"https://archives.nseindia.com/content/equities/peDetail/PE_081026.csv",
       "original_company_PE_report_SHA256":"a"*64,
       "market_PE_stock_csv_SHA256":hashlib.sha256(original).hexdigest(),
       "matched_company_positive_PE_count":1,
       "research_only":True,
       "NOT_verified_at_1530_IST":True,
       "immutable_2026_10_08_forward_predictions_changed":False,
       "valuation_market_cap_and_peer_P_E_unverified":True,
    }
    return m,original

class LaggedPEPITPolicyTests(unittest.TestCase):
    def test_only_later_decision_after_actual_archival_first_seen(self):
        m,raw=sample()
        data,audit=eligibility(m,raw,"2026-10-09")
        self.assertEqual(audit["company_PE_usable"],1)
        self.assertEqual(audit["source_market_date"],"2026-10-08")
        self.assertTrue(audit["research_sidecar_not_approved_as_predictor"])
        self.assertEqual(data["company_pe_prior_original_close_only"].notna().sum(),1)

    def test_do_not_retroactively_assume_preclose_publication(self):
        m,raw=sample()
        with self.assertRaisesRegex(ValueError,"Same-day"):
            eligibility(m,raw,"2026-10-08")

    def test_do_not_backdate_archival_observation(self):
        m,raw=sample()
        m["archived_at_utc"]="2026-10-09T11:00:00Z"
        with self.assertRaisesRegex(ValueError,"not available by decision cutoff"):
            eligibility(m,raw,"2026-10-09")

    def test_reject_hash_tamper(self):
        m,raw=sample()
        with self.assertRaisesRegex(ValueError,"CSV bytes changed"):
            eligibility(m,raw+b"\n","2026-10-09")

    def test_reject_stale_and_missing_verification(self):
        m,raw=sample()
        with self.assertRaisesRegex(ValueError,"stale"):
            eligibility(m,raw,"2026-10-20")
        m["NOT_verified_at_1530_IST"]=False
        with self.assertRaisesRegex(ValueError,"unsupported PIT"):
            eligibility(m,raw,"2026-10-09")

if __name__=="__main__": unittest.main()
