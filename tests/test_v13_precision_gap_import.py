import unittest
from v13_precision_gap_import import validate_xml_observation

class PreciseOriginalEquations(unittest.TestCase):
    def fixture(self,value='100',units='INR',extra=''):
        raw=f'''<xbrli:xbrl xmlns:xbrli="http://www.xbrl.org/2003/instance" xmlns:t="urn:source"><xbrli:context id="q"><xbrli:entity><xbrli:identifier scheme="urn:id">INE638A01035</xbrli:identifier></xbrli:entity><xbrli:period><xbrli:startDate>2026-04-01</xbrli:startDate><xbrli:endDate>2026-06-30</xbrli:endDate></xbrli:period></xbrli:context><xbrli:unit id="u"><xbrli:measure>{units}</xbrli:measure></xbrli:unit><t:Symbol>SINGERIND</t:Symbol><t:ISIN>INE638A01035</t:ISIN><t:NatureOfReportStandaloneConsolidated>Standalone</t:NatureOfReportStandaloneConsolidated><t:ProfitLossForThePeriod contextRef="q" unitRef="u">{value}</t:ProfitLossForThePeriod>{extra}</xbrli:xbrl>'''.encode()
        document={'record':{'symbol':'SINGERIND','period_end':'2026-06-30','reporting_mode':'standalone','source_url':'https://example.test/original.xml'}}
        observation={'isin':'INE638A01035','reporting_mode':'standalone','period_kind':'quarter','period_end':'2026-06-30','metric':'pat','value_INR':100.,'source_locator':{'context':'q','tags':['ProfitLossForThePeriod'],'period_start':'2026-04-01'}}
        return raw,document,observation
    def test_reported_value_allowed(self):
        self.assertTrue(validate_xml_observation(*self.fixture(),'INE638A01035'))
    def test_no_unreported_zero(self):
        raw,d,o=self.fixture('');o['value_INR']=0
        with self.assertRaises(ValueError):validate_xml_observation(raw,d,o,'INE638A01035')
    def test_currency_and_period_are_required(self):
        raw,d,o=self.fixture(units='USD')
        with self.assertRaises(ValueError):validate_xml_observation(raw,d,o,'INE638A01035')
        raw,d,o=self.fixture();o['period_end']='2025-06-30'
        with self.assertRaises(ValueError):validate_xml_observation(raw,d,o,'INE638A01035')
    def test_conflicting_fact_rejected(self):
        raw,d,o=self.fixture(extra='<t:ProfitLossForThePeriod contextRef="q" unitRef="u">101</t:ProfitLossForThePeriod>')
        with self.assertRaises(ValueError):validate_xml_observation(raw,d,o,'INE638A01035')
    def test_identity_and_transcription_rejected(self):
        raw,d,o=self.fixture()
        with self.assertRaises(ValueError):validate_xml_observation(raw,d,o,'INE0XAD01024')
        o['value_INR']=99
        with self.assertRaises(ValueError):validate_xml_observation(raw,d,o,'INE638A01035')
