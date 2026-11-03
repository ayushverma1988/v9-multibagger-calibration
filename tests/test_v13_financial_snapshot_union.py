import unittest,pandas as pd,numpy as np
from v13_financial_snapshot_union import union

class FinancialSnapshotUnion(unittest.TestCase):
    def row(self,**kw):
        return dict(source_sha256='a'*64,period_kind='quarter',period_end='2026-06-30',security_isin='INE000000001',
          reporting_mode='consolidated',available_at_utc='2026-07-21T00:00:00Z',revenue=100.,pat=10.,**kw)
    def test_existing_known_facts_survive_a_thinner_snapshot_and_new_period_is_added(self):
        a=pd.DataFrame([self.row()]);b=a.copy();b['pat']=np.nan;b['cfo']=5.
        out,conflicts=union(a,b);self.assertEqual(out.pat.iloc[0],10);self.assertEqual(out.cfo.iloc[0],5);self.assertEqual(conflicts,[])
        b['source_sha256']='b'*64;b['period_end']='2025-06-30'
        out,_=union(a,b);self.assertEqual(len(out),2)
    def test_same_original_conflict_is_explicit_and_different_basis_is_not_spliced(self):
        a=pd.DataFrame([self.row()]);b=a.copy();b['revenue']=110.
        out,conflicts=union(a,b);self.assertTrue(np.isnan(out.revenue.iloc[0]));self.assertEqual(len(conflicts),1)
        repeated,_=union(out,a);self.assertTrue(np.isnan(repeated.revenue.iloc[0]))
        b['reporting_mode']='standalone';out,conflicts=union(a,b)
        self.assertEqual(len(out),2);self.assertEqual(conflicts,[])

if __name__=='__main__':unittest.main()
