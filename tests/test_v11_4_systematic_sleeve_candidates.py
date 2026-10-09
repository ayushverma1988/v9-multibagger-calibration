import unittest
from unittest.mock import patch
import numpy as np,pandas as pd
from v11_4_systematic_sleeve_candidates import rank_sleeve_views,original_global_observation_agrees

class SleeveCandidates(unittest.TestCase):
    def frame(self):
        rows=[]
        for sleeve,offset in (("PRE_OBVIOUS_LT12PCT",.01),
            ("SECOND_LEG_12_TO_100PCT",.03),("EXTENDED_GT100PCT",.8)):
            for i in range(12):rows.append({"date":"2026-10-09","symbol":f"{sleeve}{i:02d}",
                "p6_double_calibrated":offset+i*.001,"discovery_sleeve":sleeve,
                "primary_causal_chain_status":"UNKNOWN"})
        return pd.DataFrame(rows)
    def test_high_global_extended_scores_cannot_displace_early_views(self):
        x=self.frame();original=x.copy(deep=True);view=rank_sleeve_views(x)
        self.assertEqual(len(view),30)
        self.assertEqual(view["discovery_sleeve"].value_counts().to_dict(),
            {"PRE_OBVIOUS_LT12PCT":10,"SECOND_LEG_12_TO_100PCT":10,"EXTENDED_GT100PCT":10})
        self.assertTrue(view["primary_causal_chain_status"].eq("UNKNOWN").all())
        self.assertEqual(view.groupby("discovery_sleeve")["rank_in_sleeve"].max().tolist(),[10]*3)
        pd.testing.assert_frame_equal(x,original)
    def test_unknown_lookback_never_qualifies_as_early(self):
        x=self.frame();x.loc[0,"discovery_sleeve"]="UNKNOWN"
        self.assertNotIn(x.iloc[0]["symbol"],rank_sleeve_views(x)["symbol"].tolist())
    def test_duplicate_original_security_rejected(self):
        x=self.frame();x.loc[1,"symbol"]=x.loc[0,"symbol"]
        with self.assertRaisesRegex(ValueError,"Duplicate"):rank_sleeve_views(x)
    def test_invalid_probability_rejected(self):
        for value in (np.nan,0.,1.,-1.):
            x=self.frame();x.loc[0,"p6_double_calibrated"]=value
            with self.subTest(value=value),self.assertRaisesRegex(ValueError,"probabilities"):
                rank_sleeve_views(x)
    def test_probability_ties_have_stable_symbol_order(self):
        x=self.frame();x["p6_double_calibrated"]=.1
        a=rank_sleeve_views(x);b=rank_sleeve_views(x.sample(frac=1,random_state=31))
        self.assertEqual(a["symbol"].tolist(),b["symbol"].tolist())
    def test_source_replay_detects_any_change_to_global_top10(self):
        original=pd.DataFrame({"rank":range(1,11),"symbol":[f"X{i}" for i in range(10)],
            "close":[100.]*10,"p6_double_calibrated":[.02]*10})
        with patch("v11_4_systematic_sleeve_candidates.score_prospective",return_value=original.copy()):
            original_global_observation_agrees(None,None,None,None,original)
            changed=original.copy();changed.loc[0,"p6_double_calibrated"]+=.001
            with self.assertRaisesRegex(ValueError,"global Top 10"):
                original_global_observation_agrees(None,None,None,None,changed)
if __name__=="__main__":unittest.main()
