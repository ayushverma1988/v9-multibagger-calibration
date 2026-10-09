import json,unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock
import numpy as np,pandas as pd
from v11_4_systematic_run import screen_audit,validate_source_only,ADDITIONAL_CHECKS
from v11_4_required_news_sources import collect,parse_gdelt,parse_google
from v11_4_systematic_rsi_research import FEATURES,SPECIFIC_EVENTS
from v11_4_standalone_train_walkforward import partition_train_calibration

class SystematicRepair(unittest.TestCase):
    def test_exact_four_families_remain_independent_and_RSI_is_strict(self):
        config=json.loads(Path("config/v11_4_four_screener_families.json").read_text())
        x=pd.DataFrame({"date":["2026-10-09"]*3,"symbol":["A","B","C"],
            "historical_asof_utc":["2026-10-09T10:00:00Z"]*3,
            "rsi14_wilder":[70.,70.1,np.nan],"ret_60":[.119,.12,1.01],
            "ret_120":[.05,.10,.8],"ret_252":[0,.2,.4]})
        with TemporaryDirectory() as temp:
            table,report=screen_audit(x,config,temp)
        self.assertEqual(table["condition_4_RSI14_high_momentum_status"].tolist(),["FAIL","PASS","UNKNOWN"])
        self.assertEqual(table["condition_1_quality_compounder_status"].tolist(),["UNKNOWN"]*3)
        self.assertEqual(table["discovery_sleeve"].tolist(),[
            "PRE_OBVIOUS_LT12PCT","SECOND_LEG_12_TO_100PCT","EXTENDED_GT100PCT"])
        self.assertFalse(report["full_user_model_ready"])

    def test_missing_lookback_cannot_be_pre_obvious(self):
        config=json.loads(Path("config/v11_4_four_screener_families.json").read_text())
        x=pd.DataFrame({"date":["2026-10-09"],"symbol":["A"],"historical_asof_utc":["2026-10-09T10:00:00Z"],
            "rsi14_wilder":[85.],"ret_60":[0.],"ret_120":[0.],"ret_252":[np.nan]})
        with TemporaryDirectory() as temp:table,_=screen_audit(x,config,temp)
        self.assertEqual(table["discovery_sleeve"].iloc[0],"UNKNOWN")

    def test_historic_target_cannot_enter_source_only_audit(self):
        for name in ("y6","y12","y24","y6_mature_date","p_cal","research_rank"):
            with self.subTest(name=name),self.assertRaisesRegex(ValueError,"Future"):
                validate_source_only(pd.DataFrame({"date":["2025-12-31"],"symbol":["A"],
                    "historical_asof_utc":["2025-12-31T10:00:00Z"],name:[1]}))

    def test_new_recipe_excludes_generic_promoter_and_total_counts(self):
        self.assertEqual(len(FEATURES),29)
        self.assertEqual(len(FEATURES),len(set(FEATURES)))
        self.assertNotIn("nse_promoter_activity_180d_positive",FEATURES)
        self.assertNotIn("nse_catalyst_total_180d",FEATURES)
        self.assertIn("rsi14_wilder_fraction",FEATURES)
        self.assertIn("rsi14_gt70_indicator",FEATURES)
        self.assertEqual(len(SPECIFIC_EVENTS),6)

    def test_news_failure_cannot_be_replaced_with_other_provider(self):
        session=Mock()
        gdelt=Mock();gdelt.status_code=200;gdelt.content=b'{"error":"not articles"}'
        gdelt.json.return_value={"error":"not articles"}
        google=Mock();google.status_code=200
        google.content=b'<rss><channel><item><title>Order</title><link>https://news.google.com/item</link><pubDate>Fri, 09 Oct 2026 08:00:00 GMT</pubDate></item></channel></rss>'
        session.get.side_effect=[gdelt,google]
        with TemporaryDirectory() as temp:report=collect(temp,session)
        self.assertFalse(report["both_required_sources_collected"])
        self.assertEqual(report["providers"]["GDELT"]["status"],"BLOCKED")
        self.assertEqual(report["providers"]["Google News"]["status"],"COLLECTED")
        self.assertFalse(report["primary_corporate_verification_complete"])

    def test_gdelt_empty_is_valid_only_with_explicit_articles_list(self):
        response=Mock();response.json.return_value={"articles":[]}
        self.assertEqual(parse_gdelt(response),[])
        response.json.return_value={}
        with self.assertRaises(ValueError):parse_gdelt(response)

    def test_additional_checks_preserve_literal_receivables_profit_condition(self):
        metric=ADDITIONAL_CHECKS["receivables_below_10pct_profit"]["hard_rules"][0][0]
        self.assertEqual(metric,"receivables_lt_10pct_profit")

if __name__=="__main__":unittest.main()
