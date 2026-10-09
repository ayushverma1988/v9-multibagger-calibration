import unittest
from v11_4_bse_FY2022_alternative_financial_source import candidate_row,parse_listing,clean_bse_url

class BSE2022SourceIntegrityTests(unittest.TestCase):
    def test_annual_row_with_published_IST_timestamp_is_only_candidate(self):
        html="""<html><head><title>Official BSE Financial Results</title></head><body>
        <table><tr><td>2021-2022</td><td>Consolidated-Mar-22</td>
        <td>Year</td><td>New</td><td>19-05-2022 16:20:00</td>
        <td><a href='/corporates/XBRL.aspx?id=10'>View</a></td></tr></table></body></html>"""
        summary=parse_listing(html,"506943","2026-10-09T00:00:00Z")
        self.assertEqual(summary["FY2021_2022_annual_rows"],1)
        self.assertEqual(summary["FY2021_2022_annual_date_verified"],1)
        item=summary["rows"][0]
        self.assertFalse(item["NSE_ISIN_BSE_ISIN_identity_verified"])
        self.assertFalse(item["XBRL_exact_context_and_numeric_units_verified"])
        self.assertEqual(item["bse_scrip_code"],"506943")
    def test_quarter_not_an_annual_fact(self):
        r=candidate_row(["2021-2022","Consolidated-Mar-22","Quarter","19-05-2022 16:20:00"],
                       [],"2026-10-09","506943")
        self.assertTrue(r["is_FY2021_2022"])
        self.assertFalse(r["row_type_year"])
    def test_late_published_2024_source_is_not_available_for_2023_close(self):
        r=candidate_row(["2021-2022","Consolidated-Mar-22","Year","30-01-2024 12:20:00"],
                        [],"2026-10-09","506943")
        self.assertFalse(r["filing_before_2023_Dec_29_close"])
    def test_no_fabricated_pub_time(self):
        r=candidate_row(["2021-2022","Consolidated-Mar-22","Year","-"],[],
                        "2026-10-09","506943")
        self.assertFalse(r["date_precision_verified"])
    def test_reject_third_party_xbrl_url(self):
        with self.assertRaisesRegex(ValueError,"HTTPS"):
            clean_bse_url("https://attacker.example.net/fiction.xml")
    def test_fy2023_not_mislabeled_as_2022(self):
        r=candidate_row(["2022-2023","Consolidated-Mar-23","Year","22-05-2023 11:20:00"],[],
                        "2026-10-09","506943")
        self.assertFalse(r["is_FY2021_2022"])

if __name__=="__main__":unittest.main()
