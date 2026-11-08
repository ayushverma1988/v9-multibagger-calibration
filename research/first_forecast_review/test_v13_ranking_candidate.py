import unittest
import numpy as np
import pandas as pd
from v13_ranking_candidate import rank_candidate,WEIGHTS

class CandidateTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame({'isin':['A','B','C'],'p6':[.2,.2,.2],'p12':[.4,.4,.4],
            'joint12':[.25,.25,.25],'loss12':[.9,.4,.4],
            'log_ret20':[.05,.05,.40],'log_ret63':[.10,.10,.7],'distance_ma200':[.05,.05,.7]})
    def test_lower_risk_gets_higher_rank_without_rewriting_probabilities(self):
        d=self.frame();r=rank_candidate(d,pd.Series(True,index=d.index))
        self.assertGreater(r.loc[1,'candidate_score'],r.loc[0,'candidate_score'])
        pd.testing.assert_frame_equal(d[['p6','p12','joint12','loss12']],r[['p6','p12','joint12','loss12']])
    def test_extension_is_soft_penalty_not_exclusion(self):
        d=self.frame();r=rank_candidate(d,pd.Series(True,index=d.index))
        self.assertGreater(r.loc[1,'candidate_score'],r.loc[2,'candidate_score'])
        self.assertTrue(r.candidate_rank.notna().all())
    def test_unknown_extension_gets_no_positive_discipline_credit(self):
        d=self.frame();d.loc[0,'distance_ma200']=np.nan
        r=rank_candidate(d,pd.Series(True,index=d.index))
        self.assertEqual(r.loc[0,'candidate_price_discipline'],0)
        self.assertFalse(r.loc[0,'candidate_extension_known'])
    def test_original_exclusions_and_risk_ceiling_remain(self):
        d=self.frame();d.loc[2,'loss12']=.96
        r=rank_candidate(d,pd.Series([False,True,True]))
        self.assertTrue(r.candidate_rank.isna().iloc[[0,2]].all())
    def test_deterministic_top25_and_no_early_label_bonus(self):
        d=pd.concat([self.frame()]*10,ignore_index=True);d['isin']=[str(i).zfill(3) for i in d.index]
        d['rd_first_breakout']=0
        a=rank_candidate(d,pd.Series(True,index=d.index))
        d['rd_first_breakout']=1
        b=rank_candidate(d,pd.Series(True,index=d.index))
        self.assertEqual(a.candidate_rank.notna().sum(),25)
        pd.testing.assert_series_equal(a.candidate_score,b.candidate_score)
    def test_rejects_invalid_weights(self):
        w={**WEIGHTS,'p6_percentile':1.}
        with self.assertRaises(ValueError):rank_candidate(self.frame(),pd.Series([True]*3),w)

if __name__=='__main__':unittest.main(verbosity=2)
