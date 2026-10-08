"""Reject wrong-fiscal-year contexts, dimensional segments and non-INR units."""
import unittest
from v11_4_strict_annual_numeric_features import strict_annual_facts,fold_close

NS="http://www.xbrl.org/2003/instance"
def doc(currency="INR",revenue_period="2024-03-31",use_segment=False):
    segment=('<xbrli:segment><x:explicitMember dimension="x:OperatingSegment">'
             'x:Manufacturing</x:explicitMember></xbrli:segment>') if use_segment else ""
    return f'''<?xml version="1.0"?>
<xbrli:xbrl xmlns:xbrli="{NS}" xmlns:iso4217="http://www.xbrl.org/2003/iso4217"
 xmlns:x="http://example.com/indas">
<xbrli:context id="old"><xbrli:entity><xbrli:identifier scheme="e">CO</xbrli:identifier></xbrli:entity>
<xbrli:period><xbrli:startDate>2022-04-01</xbrli:startDate><xbrli:endDate>2023-03-31</xbrli:endDate></xbrli:period></xbrli:context>
<xbrli:context id="fy"><xbrli:entity><xbrli:identifier scheme="e">CO</xbrli:identifier>{segment}</xbrli:entity>
<xbrli:period><xbrli:startDate>2023-04-01</xbrli:startDate><xbrli:endDate>{revenue_period}</xbrli:endDate></xbrli:period></xbrli:context>
<xbrli:unit id="U"><xbrli:measure>iso4217:{currency}</xbrli:measure></xbrli:unit>
<x:RevenueFromOperations contextRef="old" unitRef="U">999999</x:RevenueFromOperations>
<x:ProfitLossForPeriod contextRef="old" unitRef="U">999999</x:ProfitLossForPeriod>
<x:RevenueFromOperations contextRef="fy" unitRef="U">2000000</x:RevenueFromOperations>
<x:ProfitLossForPeriod contextRef="fy" unitRef="U">110000</x:ProfitLossForPeriod>
</xbrli:xbrl>'''.encode()

class TestStrictAnnualFacts(unittest.TestCase):
    def test_exact_fy(self):
        r,a=strict_annual_facts(doc(),"2024-03-31")
        self.assertEqual(r["revenue"],2000000.0)
        self.assertEqual(r["pat"],110000.0)
        self.assertEqual(a["revenue"]["context"],"fy")
    def test_no_wrong_year_backfill(self):
        r,_=strict_annual_facts(doc(revenue_period="2023-03-31"),"2024-03-31")
        self.assertIsNone(r["revenue"])
        self.assertIsNone(r["pat"])
    def test_annual_fourd_ytd_only_when_annual_filing(self):
        xml=doc().decode().replace('id="fy"', 'id="FourD"').replace(
            "<xbrli:startDate>2023-04-01</xbrli:startDate>",
            "<xbrli:startDate>2024-01-01</xbrli:startDate>"
        ).replace('contextRef="fy"', 'contextRef="FourD"').encode()
        not_annual,_=strict_annual_facts(xml,"2024-03-31",source_is_annual=False)
        annual,audit=strict_annual_facts(xml,"2024-03-31",source_is_annual=True)
        self.assertIsNone(not_annual["revenue"])
        self.assertEqual(annual["revenue"],2000000.0)
        self.assertEqual(annual["pat"],110000.0)
        self.assertEqual(audit["revenue"]["period_interpretation"],"annual_FourD_YTD")

    def test_no_dollar_as_rupee(self):
        r,_=strict_annual_facts(doc(currency="USD"),"2024-03-31")
        self.assertIsNone(r["revenue"])
        self.assertIsNone(r["pat"])
    def test_no_segment_rollup(self):
        r,_=strict_annual_facts(doc(use_segment=True),"2024-03-31")
        self.assertIsNone(r["revenue"])
        self.assertIsNone(r["pat"])
    def test_market_close_cutoff(self):
        t=fold_close("2024-12-31")
        self.assertEqual(t.isoformat(),"2024-12-31T10:00:00+00:00")

if __name__=="__main__":
    unittest.main(verbosity=2)
