import unittest
import pandas as pd
from v11_4_nse_valuation_peer_size_proxies import peer_reference,security_issued_shares_estimate
class NSEDerivedProxyTests(unittest.TestCase):
 def test_median_peer_not_official_index_PE(self):
  x=pd.DataFrame({"symbol":["A","B","C"],"isin":["INE1","INE2","INE3"],
     "company_pe":[10.,20.,30.],"close":[100,100,100]})
  m=pd.DataFrame({"symbol":["A","B","C"],"isin":["INE1","INE2","INE3"],
                  "industry":["Industrials"]*3})
  r=peer_reference(x,m,min_peers=2)
  self.assertEqual(r.loc[r.symbol.eq("A"),"industry_peer_median_PE_proxy"].iloc[0],20.)
  self.assertEqual(r["company_PE_below_own_current_sector_peer_median_proxy"].fillna(False).tolist(),[True,False,False])
 def test_unknown_for_unmatched_isin_and_sparse_peers(self):
  c=pd.DataFrame({"symbol":["A","B"],"isin":["INE1","INE2"],
       "company_pe":[10.,5.],"close":[100,100]})
  m=pd.DataFrame({"symbol":["A","B"],"isin":["INEWRONG","INE2"],"industry":["Chem","Chem"]})
  r=peer_reference(c,m,min_peers=5)
  self.assertTrue(r["industry_peer_median_PE_proxy"].isna().all())
  self.assertTrue(r["company_PE_below_own_current_sector_peer_median_proxy"].isna().all())
 def test_issued_capital_is_proxy_not_certified(self):
  c=pd.DataFrame({"symbol":["A"],"isin":["INE1"],"close":[100.],"company_pe":[12.]})
  m=pd.DataFrame({"TckrSymb":["A"],"ISIN":["INE1"],"SctySrs":["EQ"],
                  "IssdCptl":[1e8],"ParVal":[100]})
  v=security_issued_shares_estimate(m,c)
  self.assertAlmostEqual(v["issued_security_count_proxy_market_cap_crore"].iloc[0],1000.)
  self.assertFalse(v["issued_security_count_proxy_is_independent_certified_market_cap"].iloc[0])
if __name__=="__main__":unittest.main(verbosity=2)
