"""Evaluate an already frozen ranking without deleting unknown outcomes first.

Intervals are descriptive. Repeated issuers and adjacent market regimes make
pooled Bernoulli independence doubtful; date-block resampling is also reported.
No metric in this module chooses model features, thresholds or parameters.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from v11_4_label_clock import maturity_utc


def clean_outcome_mask(frame, cutoff):
    maturity = maturity_utc(frame["y6_mature_date"])
    moment = pd.Timestamp(cutoff)
    if moment.tzinfo is None:
        raise ValueError("Evaluation cutoff must be explicitly timezone-aware")
    return (frame["y6"].isin([0, 1]) & frame["integrity_y6_clean"].eq(True)
            & maturity.notna() & maturity.le(moment))


def wilson95(hits, count):
    if count == 0:
        return [None, None]
    z = 1.959963984540054
    p = hits / count
    den = 1 + z*z/count
    mid = (p + z*z/(2*count))/den
    half = z*np.sqrt(p*(1-p)/count + z*z/(4*count*count))/den
    return [float(max(0, mid-half)), float(min(1, mid+half))]


def evaluate_ranked(ranked, probability_column, cutoff, calibration_rate, top_k=10):
    """Do not sort or replace selections. Assess only after Top K is frozen."""
    if len(ranked) < top_k or ranked["symbol"].duplicated().any():
        raise ValueError("Ranking needs at least K distinct original securities")
    p = pd.to_numeric(ranked[probability_column], errors="coerce")
    if not np.isfinite(p).all() or not p.between(0, 1).all():
        raise ValueError("Invalid research probabilities")
    if not np.isfinite(calibration_rate) or not 0 <= calibration_rate <= 1:
        raise ValueError("Baseline must come from an independently matured calibration cohort")
    selected = ranked.head(top_k)
    mask = clean_outcome_mask(ranked, cutoff)
    assessed = ranked.loc[mask]
    known = selected.loc[mask.reindex(selected.index)]
    hits = int(known["y6"].sum())
    unknown = top_k-len(known)
    y = assessed["y6"].astype(int)
    pred = p.loc[assessed.index]
    rate = float(y.mean()) if len(y) else None
    brier = float(brier_score_loss(y, pred)) if len(y) else None
    baseline = float(np.mean((y.to_numpy()-calibration_rate)**2)) if len(y) else None
    report = {
        "decision_eligible_candidates": len(ranked),
        "evaluated_clean_mature_candidates": len(assessed),
        "candidates_with_unknown_or_unclean_outcomes": int((~mask).sum()),
        "actual_six_month_doublers": int(y.sum()),
        "cohort_base_rate": rate,
        "selected": top_k,
        "selected_clean_mature_outcomes": len(known),
        "selected_unknown_or_unclean_outcomes": unknown,
        "top10_doublers": hits,
        "top10_precision": hits/top_k if unknown == 0 else None,
        "top10_observed_label_precision": hits/len(known) if len(known) else None,
        "top10_all_selected_precision_bounds": [hits/top_k, (hits+unknown)/top_k],
        "top10_precision_Wilson95": wilson95(hits, top_k) if unknown == 0 else [None, None],
        "top10_lift": hits/top_k/rate if unknown == 0 and rate is not None and rate > 0 else None,
        "average_precision": float(average_precision_score(y, pred)) if len(y) and y.sum() > 0 else None,
        "roc_auc": float(roc_auc_score(y, pred)) if y.nunique() == 2 else None,
        "brier": brier,
        "known_prior_calibration_rate": float(calibration_rate),
        "prior_calibration_rate_baseline_brier": baseline,
        "brier_skill_vs_prior_calibration_rate": 1-brier/baseline if baseline and brier is not None else None,
        "selected_mean_model_probability": float(p.head(top_k).mean()),
        "test_selection_used_future_labels_or_integrity": False,
        "unknown_selected_outcomes_replaced": False,
        "interval_assumes_independent_trials": True,
        "drawdown_rate": None,
    }
    if "dd30_6m" in known:
        drawdown = pd.to_numeric(known["dd30_6m"], errors="coerce")
        report["drawdown_rate"] = float(drawdown.mean()) if drawdown.isin([0, 1]).all() else None
    return report


def summarize_folds(folds, stability):
    complete = [f for f in folds if f.get("selected_clean_mature_outcomes") == 10]
    n = sum(f["selected_clean_mature_outcomes"] for f in complete)
    hits = sum(f["top10_doublers"] for f in complete)
    # Resample whole dates, not individual correlated stock observations.
    block_ci = [None, None]
    if len(complete) >= 2:
        rates = np.array([f["top10_precision"] for f in complete])
        rng = np.random.default_rng(20261010)
        draws = rng.choice(rates, size=(4000, len(rates)), replace=True).mean(axis=1)
        block_ci = np.quantile(draws, [.025, .975]).tolist()
    s = pd.DataFrame(stability)
    mean = float(s["jaccard"].mean()) if len(s) else None
    by = s.groupby("date")["jaccard"].quantile(.05) if len(s) else pd.Series(dtype=float)
    worst = float(by.min()) if len(by) else None
    return {
        "scored_dates": len(folds), "fully_labelled_top10_dates": len(complete),
        "fully_labelled_selections": n, "confirmed_2x_hits": hits,
        "six_month_precision": hits/n if n else None,
        "pooled_precision_Wilson95_independence_assumption": wilson95(hits, n),
        "date_block_resampling_precision95": block_ci,
        "adjacent_dates_and_repeated_issuers_may_remain_dependent": True,
        "all_scored_selected": len(folds)*10,
        "all_scored_known_hits": sum(f["top10_doublers"] for f in folds),
        "all_scored_unknown_selected": sum(f["selected_unknown_or_unclean_outcomes"] for f in folds),
        "mean_top10_jaccard": mean, "worst_fold_p05_jaccard": worst,
        "worst_stability_date": str(by.idxmin()) if len(by) else None,
        "per_date_p05_jaccard": {str(k):float(v) for k,v in by.items()},
        "minimum_12_folds_gate_pass": len(complete) >= 12,
        "mean_jaccard_gate_pass": mean is not None and mean >= .8,
        "worst_fold_p05_gate_pass": worst is not None and worst >= .6,
        "selection_stability_gate_pass": mean is not None and worst is not None and mean >= .8 and worst >= .6,
        "unchanged_gates": {"valid_test_dates":12, "mean_top10_jaccard":.8, "worst_fold_p05_jaccard":.6},
        "new_blinded_evaluation_dates": 0,
        "production_approved": False,
    }
