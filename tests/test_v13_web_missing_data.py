import unittest
from v13_web_missing_data import parse_stockanalysis

class WebMissingInputsTests(unittest.TestCase):
    def setUp(self):
        self.issuer={'nse_current_symbol':'TEST','isin':'INE123456789'}
        self.receipt={'requested_url':'https://stockanalysis.com/quote/nse/TEST/financials/balance-sheet/',
            'sha256':'a'*64,'first_retrieved_utc':'2026-10-21T07:30:00Z'}
        self.raw=b'''<p>Test Limited (NSE:TEST)</p><p>Millions INR. S&amp;P Global</p><table>
        <tr><th>Fiscal Year</th><th id="TTM">TTM</th><th id="2026-03-31">FY 2026</th></tr>
        <tr><th>Period Ending</th><th>Jun 2026</th><th>Mar 2026</th></tr>
        <tr><td>Accounts Receivable</td><td>9999</td><td>123.4</td></tr>
        <tr><td>Short-Term Debt</td><td>-</td><td>12</td></tr>
        <tr><td>Current Portion of Long-Term Debt</td><td>-</td><td>3</td></tr></table>'''
    def test_one_page_supplies_dated_values_and_derived_current_debt(self):
        rows=parse_stockanalysis(self.raw,self.issuer,self.receipt,'balance-sheet')
        self.assertEqual(len(rows),2);self.assertEqual(rows[0]['value_INR'],123400000)
        self.assertEqual(rows[1]['value_INR'],15000000);self.assertIn('DERIVED',rows[1]['definition_status'])
        self.assertTrue(all(r['period_end']=='2026-03-31' for r in rows))
    def test_no_annual_amount_can_be_mislabeled_quarterly(self):
        with self.assertRaisesRegex(ValueError,'quarterly'):parse_stockanalysis(self.raw,self.issuer,self.receipt,'income-statement',True)
    def test_explicit_units_required(self):
        with self.assertRaisesRegex(ValueError,'units'):parse_stockanalysis(self.raw.replace(b'Millions INR',b'Financials'),self.issuer,self.receipt,'balance-sheet')
    def test_exact_ticker_identity_required(self):
        with self.assertRaisesRegex(ValueError,'identity'):parse_stockanalysis(self.raw.replace(b'NSE:TEST',b'NSE:OTHER'),self.issuer,self.receipt,'balance-sheet')

if __name__=='__main__':unittest.main()
