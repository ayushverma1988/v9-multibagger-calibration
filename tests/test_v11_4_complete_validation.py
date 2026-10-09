import json, unittest
from tempfile import TemporaryDirectory
from pathlib import Path
from unittest.mock import patch
import joblib
import numpy as np
import pandas as pd
import v11_4_robust_research_model as robust
from v11_4_validation_metrics import evaluate_ranked, clean_outcome_mask, summarize_folds
from v11_4_complete_validation import acceptance_report, partitions, normalize_keys
import v11_4_threeFY_market_catalyst_chronological_research as financial
from v11_4_systematic_rsi_research import cohort, FEATURES
from v11_4_threeFY_RSI70_promoter_source_ready import STRICT_STATUS
from v11_4_label_clock import maturity_utc
from v11_4_standalone_train_walkforward import partition_train_calibration


def ranked_fixture():
    return pd.DataFrame({"symbol":[f"X{i}" for i in range(12)],
        "probability":np.linspace(.2,.01,12),"y6":[1]+[0]*11,
        "integrity_y6_clean":[True]*12,
        "y6_mature_date":["2026-01-01T10:00:00Z"]*12})


class SelectionAndMetricsTests(unittest.TestCase):
    def test_unknown_winning_or_bad_corporate_action_stock_is_not_replaced(self):
        q=ranked_fixture();q.loc[0,"integrity_y6_clean"]=False
        q.loc[11,"y6"]=1
        report=evaluate_ranked(q,"probability","2026-10-09T00:00:00Z",.05)
        self.assertEqual(report["selected_unknown_or_unclean_outcomes"],1)
        self.assertEqual(report["top10_doublers"],0)
        self.assertIsNone(report["top10_precision"])
        self.assertEqual(report["top10_all_selected_precision_bounds"],[0,.1])
        self.assertFalse(report["unknown_selected_outcomes_replaced"])

    def test_future_or_missing_label_maturity_remains_unknown_after_selection(self):
        q=ranked_fixture();q.loc[0,"y6_mature_date"]="2027-01-01T10:00:00Z"
        q.loc[1,"y6_mature_date"]=None
        mask=clean_outcome_mask(q,"2026-10-09T00:00:00Z")
        self.assertEqual(mask.iloc[:3].tolist(),[False,False,True])
        with self.assertRaisesRegex(ValueError,"timezone"):
            clean_outcome_mask(q,"2026-10-09")

    def test_baseline_is_actual_prior_calibration_rate(self):
        q=ranked_fixture();q["probability"]=.05
        r=evaluate_ranked(q,"probability","2026-10-09T00:00:00Z",.05)
        self.assertAlmostEqual(r["brier_skill_vs_prior_calibration_rate"],0.)
        self.assertEqual(r["top10_precision"],.1)

    def test_bad_probabilities_and_duplicate_security_fail(self):
        for value in (np.nan,np.inf,1.1,-.1):
            q=ranked_fixture();q.loc[0,"probability"]=value
            with self.assertRaises(ValueError):evaluate_ranked(q,"probability","2026-10-09T00:00:00Z",.05)
        q=ranked_fixture();q.loc[1,"symbol"]=q.loc[0,"symbol"]
        with self.assertRaises(ValueError):evaluate_ranked(q,"probability","2026-10-09T00:00:00Z",.05)

    def test_high_average_stability_cannot_hide_failed_worst_date(self):
        folds=[{"selected_clean_mature_outcomes":10,"selected_unknown_or_unclean_outcomes":0,
                "top10_doublers":1,"top10_precision":.1} for _ in range(12)]
        stability=[{"date":str(i),"jaccard":1.} for i in range(12) for _ in range(12)]
        stability[0]["jaccard"]=0;stability[1]["jaccard"]=0
        result=summarize_folds(folds,stability)
        self.assertTrue(result["mean_jaccard_gate_pass"])
        self.assertFalse(result["selection_stability_gate_pass"])
        self.assertFalse(result["production_approved"])

    def test_missing_sources_and_seen_holdouts_prevent_promotion_even_when_jaccard_passes(self):
        improved={"minimum_12_folds_gate_pass":True,"mean_jaccard_gate_pass":True,
                  "worst_fold_p05_gate_pass":True,"new_blinded_evaluation_dates":0}
        r=acceptance_report({"original_corrected_reference_reproduced":True,
                            "fixed_training_tail_bounds_21input":improved},None,None,None,None)
        self.assertFalse(r["production_approved"])
        self.assertIn("unseen_independent_evaluation_complete",r["failed_gates"])
        self.assertIn("both_required_news_sources_collected",r["failed_gates"])


class TrainingBoundTests(unittest.TestCase):
    def test_unseen_extreme_value_cannot_update_training_bounds(self):
        train=np.array([[0,0],[1,1],[2,0],[3,1]],dtype=float)
        model=robust.TrainingTailBounds(lower=.25,upper=.75,n_market=1).fit(train)
        before=model.upper_bounds_.copy()
        result=model.transform([[1000,7]])
        np.testing.assert_array_equal(before,model.upper_bounds_)
        np.testing.assert_allclose(result,[[2.25,7]])
        self.assertEqual(model.support_audit([[1000,np.nan]])[0].tolist(),[1])

    def test_missing_source_stays_missing_until_training_imputer(self):
        b=robust.TrainingTailBounds(n_market=1).fit([[0,1],[1,1]])
        self.assertTrue(np.isnan(b.transform([[np.nan,1]])[0,0]))
        with self.assertRaisesRegex(ValueError,"no finite"):
            robust.TrainingTailBounds(n_market=1).fit([[np.nan,1],[np.nan,2]])

    def test_serialized_pipeline_produces_identical_future_scores(self):
        rng=np.random.default_rng(12);x=pd.DataFrame(rng.normal(size=(180,21)),columns=robust.FEATURES)
        y=np.array([0,1]*90);model=robust.make_model().fit(x,y)
        future=x.iloc[:12].copy();future.iloc[0,0]=1e6
        with TemporaryDirectory() as d:
            p=Path(d)/"model.joblib";joblib.dump(model,p);restored=joblib.load(p)
            np.testing.assert_array_equal(model.predict_proba(future),restored.predict_proba(future))

    def test_generic_promoter_filing_count_is_not_a_purchase_predictor(self):
        self.assertEqual(len(robust.FEATURES),21)
        self.assertNotIn("nse_promoter_activity_90d_positive",robust.FEATURES)

    def test_changed_feature_shape_and_infinity_fail(self):
        b=robust.TrainingTailBounds(n_market=1).fit([[0,1],[1,1]])
        for candidate in ([[0]],[[np.inf,1]]):
            with self.assertRaises(ValueError):b.transform(candidate)


class MarketMaturityClockTests(unittest.TestCase):
    def test_date_only_label_is_not_available_at_midnight(self):
        for values in (pd.Series(["2024-06-28",None]),pd.Series(pd.to_datetime(["2024-06-28",None]))):
            times=maturity_utc(values)
            self.assertEqual(times.iloc[0],pd.Timestamp("2024-06-28T10:00:00Z"))
            self.assertTrue(pd.isna(times.iloc[1]))

    def test_preserves_explicit_timestamp_but_rejects_ambiguous_intraday(self):
        self.assertEqual(maturity_utc(pd.Series(["2024-06-28T09:59:59Z"])).iloc[0],
                         pd.Timestamp("2024-06-28T09:59:59Z"))
        with self.assertRaisesRegex(ValueError,"explicit source timezone"):
            maturity_utc(pd.Series(["2024-06-28T11:00:00"]))

    def test_maturity_on_calibration_close_cannot_enter_base_training(self):
        x=pd.DataFrame({"date":["2023-12-29"]*3+["2024-06-28"],
            "symbol":["EARLIER","SAME_CLOSE","UNKNOWN","CAL"],
            "y6_mature_date":["2024-06-27","2024-06-28",None,"2024-12-30"]})
        base,cal=partition_train_calibration(x,"2024-06-28")
        self.assertEqual(base.symbol.tolist(),["EARLIER"])
        self.assertEqual(cal.symbol.tolist(),["CAL"])


class FinancialDecisionCohortTests(unittest.TestCase):
    def fixture(self):
        d="2025-06-30"
        x=pd.DataFrame({"date":[d]*40,"symbol":[f"X{i}" for i in range(40)],
            "has_strict_3FY_original_fiscal_source":[True]*40,
            "integrity_feature_clean":[True]*40,"close":[100.]*40,
            "avg_turnover_63":[100000.]*40,"source_verification":[STRICT_STATUS]*40,
            "y6":[1,0]*20,"integrity_y6_clean":[True]*40,
            "y6_mature_date":["2026-01-01T10:00:00Z"]*40})
        for f in set(financial.FEATURES)|set(FEATURES):
            if f not in x:x[f]=1.
        return x

    def test_financial_test_cohort_is_invariant_to_future_labels_and_integrity(self):
        x=self.fixture()
        with patch.object(financial,"MIN_TEST_ROWS",1):
            a=financial.eligible_at(x,"2025-06-30",require_mature_labels=False)
            x["y6"]=np.nan;x["integrity_y6_clean"]=False;x["y6_mature_date"]=None
            b=financial.eligible_at(x,"2025-06-30",require_mature_labels=False)
        self.assertEqual(a.symbol.tolist(),b.symbol.tolist())
        np.testing.assert_array_equal(a[list(financial.FEATURES)],b[list(financial.FEATURES)])

    def test_RSI_test_cohort_is_invariant_to_future_label_censoring(self):
        x=self.fixture();x["rsi14_wilder_source"]=71.;x["rsi14_strict_gt70_source_verified"]=True
        with patch.object(financial,"MIN_TEST_ROWS",1):
            a=cohort(x,"2025-06-30",require_mature_labels=False)
            x.loc[0,"y6"]=np.nan;x.loc[1,"integrity_y6_clean"]=False
            b=cohort(x,"2025-06-30",require_mature_labels=False)
        self.assertEqual(a.symbol.tolist(),b.symbol.tolist())

    def test_training_still_requires_clean_mature_labels(self):
        x=self.fixture();x.loc[0,"integrity_y6_clean"]=False
        x.loc[1,"y6_mature_date"]="2027-01-01T10:00:00Z"
        with patch.object(financial,"MIN_TEST_ROWS",1):
            fit=financial.eligible_at(x,"2025-06-30",pd.Timestamp("2026-10-09T00:00:00Z"))
        self.assertEqual(len(fit),38)
        with self.assertRaisesRegex(ValueError,"explicit maturity"):
            financial.eligible_at(x,"2025-06-30")


if __name__=="__main__":unittest.main()
