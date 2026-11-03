"""One fixed, unapproved V11.4 improvement, not a hyperparameter search.

Cap continuous market tails using ONLY base-training 0.5/99.5 percentiles.
Retain measured, log-transformed specific exchange event counts. Drop generic
promoter filings: their count does not establish an actual promoter purchase.
Same C, seeds, source dates, target and calibration as the corrected core.
"""
from __future__ import annotations
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted
import v11_4_standalone_train_walkforward as core
from v11_4_forward_research_release import fit_calibration_params, apply_frozen_calibration

FEATURES = tuple(c for c in core.MODEL_FEATURES if c != "nse_promoter_activity_90d_positive")
VERSION = "V11.4-training-tail-bounds-specific-events-research-20261010"
LOW_QUANTILE = .005
HIGH_QUANTILE = .995


class TrainingTailBounds(TransformerMixin, BaseEstimator):
    def __init__(self, lower=LOW_QUANTILE, upper=HIGH_QUANTILE, n_market=len(core.PRICE_COLUMNS)):
        self.lower = lower
        self.upper = upper
        self.n_market = n_market

    def fit(self, X, y=None):
        a = np.asarray(X, dtype=float)
        if a.ndim != 2 or not 0 <= self.lower < self.upper <= 1:
            raise ValueError("Invalid training tail-bound configuration")
        if not 0 <= self.n_market <= a.shape[1]:
            raise ValueError("Market feature count exceeds input matrix")
        if not np.isfinite(a).any(axis=0).all():
            raise ValueError("A predictor has no finite base-training source observations")
        if np.isinf(a).any():
            raise ValueError("Infinite training predictor")
        self.n_features_in_ = a.shape[1]
        self.lower_bounds_ = np.full(a.shape[1], -np.inf)
        self.upper_bounds_ = np.full(a.shape[1], np.inf)
        self.lower_bounds_[:self.n_market] = np.nanquantile(a[:,:self.n_market], self.lower, axis=0)
        self.upper_bounds_[:self.n_market] = np.nanquantile(a[:,:self.n_market], self.upper, axis=0)
        return self

    def transform(self, X):
        check_is_fitted(self, "lower_bounds_")
        a = np.asarray(X, dtype=float)
        if a.ndim != 2 or a.shape[1] != self.n_features_in_ or np.isinf(a).any():
            raise ValueError("Prediction schema or numeric integrity changed")
        return np.clip(a, self.lower_bounds_, self.upper_bounds_)

    def support_audit(self, X):
        check_is_fitted(self, "lower_bounds_")
        a = np.asarray(X, dtype=float)
        self.transform(a)  # Validate shape and infinity handling.
        capped = (a < self.lower_bounds_) | (a > self.upper_bounds_)
        return capped.sum(axis=1), np.isnan(a).sum(axis=1)


def make_model():
    return Pipeline([
        ("training_tail_bounds", TrainingTailBounds()),
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(C=core.REGULARIZATION_C, max_iter=1300,
                                   class_weight=None, solver="lbfgs", random_state=31)),
    ])


def fit_once(train, cal, candidates):
    model = make_model()
    model.fit(train[list(FEATURES)], train["y6"].astype(int))
    cp = model.predict_proba(cal[list(FEATURES)])[:,1]
    params = fit_calibration_params(cp, cal["y6"], train["y6"])
    raw = model.predict_proba(candidates[list(FEATURES)])[:,1]
    return apply_frozen_calibration(raw, params), model, params
