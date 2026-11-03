"""Mocked official archive fallback contract: never confuse index PE with stock PE."""
import io
import unittest
from unittest.mock import Mock
import pandas as pd
import requests
from v11_4_direct_exchange_pe_valuation import fetch_original_company_pe

def _response(code, payload=b""):
    r=Mock()
    r.content=payload
    if code>=400:
        r.raise_for_status.side_effect=requests.HTTPError(f"HTTP {code}")
    else:
        r.raise_for_status.return_value=None
    return r

def _valid_company_report():
    f=pd.DataFrame({
        "SYMBOL":[f"ABC{i:04d}" for i in range(1700)],
        "SYMBOL P/E":[f"{11+i/100:.3f}" for i in range(1700)],
        "ADJUSTED P/E":[f"{12+i/100:.3f}" for i in range(1700)],
    })
    raw=f.to_csv(index=False).encode()
    assert len(raw)>20000
    return raw

class ArchiveFallbackContractTests(unittest.TestCase):
    def test_primary_genuine_company_report_preferred(self):
        get=Mock(side_effect=lambda url,**kwargs:_response(200,_valid_company_report()))
        raw,url,frame,audit=fetch_original_company_pe("2026-10-08",session_get=get)
        self.assertEqual(get.call_count,1)
        self.assertIn("archives.nseindia.com",url)
        self.assertEqual(len(frame),1700)
        self.assertEqual(len(audit),1)
        self.assertTrue(audit[0]["accepted_original_company_pe"])

    def test_after_primary_403_retry_only_official_company_archive(self):
        urls=[]
        def get(url,**kwargs):
            urls.append(url)
            return _response(403) if len(urls)==1 else _response(200,_valid_company_report())
        raw,url,frame,audit=fetch_original_company_pe("2026-10-08",session_get=get)
        self.assertEqual(len(urls),2)
        self.assertIn("nsearchives.nseindia.com",url)
        self.assertTrue(all("/content/equities/peDetail/PE_081026.csv" in x for x in urls))
        self.assertFalse(audit[0]["accepted_original_company_pe"])
        self.assertTrue(audit[1]["accepted_original_company_pe"])
        self.assertEqual(len(frame),1700)

    def test_wrong_index_file_does_not_pass_company_pe_semantics(self):
        fake=b"Index Name,P/E,P/B\nNIFTY 50,24,4\n"*1500
        get=Mock(return_value=_response(200,fake))
        with self.assertRaisesRegex(ValueError,"Both official NSE company PE archives unavailable"):
            fetch_original_company_pe("2026-10-08",session_get=get)
        self.assertEqual(get.call_count,2)

    def test_both_hosts_blocked_fail_closed_without_synthetic_valuation(self):
        get=Mock(return_value=_response(403))
        with self.assertRaisesRegex(ValueError,"no index PE substitution"):
            fetch_original_company_pe("2026-10-08",session_get=get)
        self.assertEqual(get.call_count,2)

if __name__=="__main__":
    unittest.main()
