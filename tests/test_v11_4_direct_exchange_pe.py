import unittest
import pandas as pd
from v11_4_direct_exchange_pe_valuation import normalize_company_pe,parse_positive_pe,normalize_day_market
class DirectNSEOriginalPE(unittest.TestCase):
 def test_original_company_PE_not_index(self):
  good=pd.DataFrame({"SYMBOL":["A","B","C","D"],
                     "SYMBOL P/E":["14.5","--","-4","0"],
                     "ADJUSTED P/E":["16.3","12","nan","0"]})
  # Unit test small table without weakening production >1,200 rows:
  with self.assertRaisesRegex(ValueError,"Incomplete"):
   normalize_company_pe(good)
  self.assertEqual(parse_positive_pe(good["SYMBOL P/E"]).dropna().tolist(),[14.5])
 def test_blank_and_nonpositive_stay_unknown(self):
  p=parse_positive_pe(pd.Series(["29.11","","NA","-8","0","2,525.90"]))
  self.assertAlmostEqual(p.iloc[0],29.11)
  self.assertEqual(p.notna().sum(),2)
 def test_index_pe_cannot_be_misread_as_company(self):
  idx=pd.DataFrame({"Index Name":["NIFTY 50"],"P/E":[21.2],"P/B":[3.7]})
  with self.assertRaisesRegex(ValueError,"three-field company schema"):
   normalize_company_pe(idx)
 def test_nse_mainboard_series_priority(self):
  x=pd.DataFrame({"date":["2026-10-08"]*3,
   "symbol":["ABC","ABC","XYZ"],"series":["BE","EQ","EQ"],
   "isin":["INE1234","INE1234","INE5678"],"close":[201.,200.,88.]})
  result=normalize_day_market(x)
  self.assertEqual(len(result),2)
  self.assertEqual(result.loc[result.symbol.eq("ABC"),"series"].iloc[0],"EQ")
if __name__=="__main__":unittest.main(verbosity=2)
