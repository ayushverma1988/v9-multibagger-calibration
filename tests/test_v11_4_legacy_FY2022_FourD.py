"""Yearly FourD fiscal metadata, unit and mode must be explicit for legacy NSE FY2022."""
import unittest
from v11_4_recover_legacy_FY2022_FourD import extract_2022_legacy_fourd
def sample(fy="2022-03-31",start="2021-04-01",yearly="Yearly",mode="Consolidated",currency="INR",duplicate=False):
 return (f'<xbrl><unit id="u"><measure>iso4217:{currency}</measure></unit>'
  f'<ReportingQuarter contextRef="OneD">{yearly}</ReportingQuarter>'
  f'<DateOfStartOfReportingPeriod contextRef="FourD">{start}</DateOfStartOfReportingPeriod>'
  f'<DateOfEndOfReportingPeriod contextRef="FourD">{fy}</DateOfEndOfReportingPeriod>'
  f'<NatureOfReportStandaloneConsolidated contextRef="FourD">{mode}</NatureOfReportStandaloneConsolidated>'
  f'<RevenueFromOperations contextRef="FourD" unitRef="u">1250000000</RevenueFromOperations>'
  f'<ProfitLossForPeriod contextRef="FourD" unitRef="u">150000000</ProfitLossForPeriod>'
  +(f'<RevenueFromOperations contextRef="FourD" unitRef="u">1</RevenueFromOperations>' if duplicate else "")
  +'</xbrl>').encode()
class OriginalLegacyFourDTests(unittest.TestCase):
 def test_explicit_full_year_INR_metadata_extracts(self):
  r,a=extract_2022_legacy_fourd(sample(),mode_expected="consolidated",source_is_annual=True)
  self.assertEqual(r["revenue"],1250000000.)
  self.assertEqual(r["pat"],150000000.)
  self.assertEqual(a["explicit_fiscal_start"],"2021-04-01")
 def test_reject_if_reporting_is_quarter_not_year(self):
  with self.assertRaisesRegex(ValueError,"Yearly"):
   extract_2022_legacy_fourd(sample(yearly="Quarterly"),source_is_annual=True)
 def test_reject_fy_start_from_Q4_not_full_year(self):
  with self.assertRaisesRegex(ValueError,"FourD fiscal period"):
   extract_2022_legacy_fourd(sample(start="2022-01-01"),source_is_annual=True)
 def test_reject_wrong_fy_end_even_if_index_claims_2022(self):
  with self.assertRaisesRegex(ValueError,"FourD fiscal period"):
   extract_2022_legacy_fourd(sample(fy="2023-03-31"),source_is_annual=True)
 def test_reject_non_INR(self):
  with self.assertRaisesRegex(ValueError,"currency"):
   extract_2022_legacy_fourd(sample(currency="USD"),source_is_annual=True)
 def test_reject_different_consolidation_mode(self):
  with self.assertRaisesRegex(ValueError,"mode changed"):
   extract_2022_legacy_fourd(sample(mode="Standalone"),mode_expected="Consolidated",source_is_annual=True)
 def test_reject_duplicate_fourD_numeric_concept(self):
  with self.assertRaisesRegex(ValueError,"Ambiguous repeated"):
   extract_2022_legacy_fourd(sample(duplicate=True),source_is_annual=True)
 def test_reject_quarterly_document_even_if_fourd_claims_full_year(self):
  with self.assertRaisesRegex(ValueError,"Quarterly XBRL"):
   extract_2022_legacy_fourd(sample(),mode_expected="Consolidated",source_is_annual=False)
if __name__=="__main__":unittest.main()
