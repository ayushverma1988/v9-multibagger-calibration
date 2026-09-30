from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression

EPS = 1e-6
POLICY_NAME = "rank_platt_anchor_r8"


class RankPlatt:
    def __init__(self):
        self.lr = LogisticRegression(C=2.0, max_iter=2000)

    def fit(self, rank_p, y):
        p = np.clip(np.asarray(rank_p, float), EPS, 1 - EPS)
        self.lr.fit(logit(p).reshape(-1, 1), np.asarray(y, int))
        return self

    def predict(self, rank_p):
        p = np.clip(np.asarray(rank_p, float), EPS, 1 - EPS)
        return self.lr.predict_proba(logit(p).reshape(-1, 1))[:, 1]


def add_fold_rank(df: pd.DataFrame, pcol: str = "p_fund_raw") -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["p_fund_rank"] = out.groupby("date")[pcol].rank(
        pct=True, method="average"
    )
    return out


def prior_fold_anchor(
    prior: pd.DataFrame,
    target: str = "y6",
    recent_folds: int = 8,
) -> float:
    q = prior.dropna(subset=["date", target]).copy()
    dates = sorted(pd.to_datetime(q["date"].unique()))
    if recent_folds > 0:
        dates = dates[-int(recent_folds):]
        q = q[q["date"].isin(dates)]
    rates = q.groupby("date")[target].mean().dropna()
    return float(rates.mean()) if len(rates) else np.nan


def shift_to_mean_probability(p, target_mean: float):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    target_mean = float(np.clip(target_mean, EPS, 1 - EPS))
    z = logit(p)
    lo, hi = -12.0, 12.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        pm = 1.0 / (1.0 + np.exp(-(z + mid)))
        if float(np.mean(pm)) < target_mean:
            lo = mid
        else:
            hi = mid
    delta = 0.5 * (lo + hi)
    return np.clip(1.0 / (1.0 + np.exp(-(z + delta))), EPS, 1 - EPS)


def apply_forward(
    oos: pd.DataFrame,
    cfg: dict,
    target: str = "y6",
    pcol: str = "p_fund_raw",
    outcol: str = "p_fund_cal",
):
    out = add_fold_rank(oos, pcol)
    out[outcol] = np.nan
    out[f"{outcol}_method"] = ""
    out[f"{outcol}_prior_folds"] = np.nan
    out[f"{outcol}_anchor"] = np.nan

    min_prior_folds = int(cfg.get("fund_cal_min_prior_folds", 8))
    min_rows = int(cfg.get("fund_cal_cv_min_train_rows", 1000))
    min_pos = int(cfg.get("fund_cal_cv_min_train_positives", 25))
    anchor_folds = int(cfg.get("fund_cal_anchor_recent_folds", 8))

    diag = []
    for td in sorted(pd.to_datetime(out["date"].unique())):
        prior = out[
            (out["date"] < td)
            & out[target].notna()
            & out["p_fund_rank"].notna()
        ].copy()
        idx = out.index[
            (out["date"] == td)
            & out[target].notna()
            & out["p_fund_rank"].notna()
        ]
        if len(idx) == 0:
            continue

        prior_folds = int(prior["date"].nunique())
        prior_pos = int(prior[target].sum()) if len(prior) else 0
        rec = {
            "date": td,
            "policy": POLICY_NAME,
            "prior_folds": prior_folds,
            "prior_rows": int(len(prior)),
            "prior_positives": prior_pos,
            "calibrated": False,
        }
        if (
            prior_folds < min_prior_folds
            or len(prior) < min_rows
            or prior_pos < min_pos
            or prior[target].nunique() < 2
        ):
            diag.append(rec)
            continue

        cal = RankPlatt().fit(
            prior["p_fund_rank"].to_numpy(float),
            prior[target].astype(int).to_numpy(),
        )
        pred = cal.predict(out.loc[idx, "p_fund_rank"].to_numpy(float))
        anchor = prior_fold_anchor(prior, target, anchor_folds)
        if not np.isfinite(anchor):
            diag.append(rec)
            continue
        pred = shift_to_mean_probability(pred, anchor)

        out.loc[idx, outcol] = pred
        out.loc[idx, f"{outcol}_method"] = POLICY_NAME
        out.loc[idx, f"{outcol}_prior_folds"] = prior_folds
        out.loc[idx, f"{outcol}_anchor"] = anchor

        rec.update({
            "calibrated": True,
            "anchor": float(anchor),
            "mean_probability": float(np.mean(pred)),
        })
        diag.append(rec)

    return out, pd.DataFrame(diag)


def fit_current(
    fund_oos: pd.DataFrame,
    raw_current,
    cfg: dict,
    target: str = "y6",
    pcol: str = "p_fund_raw",
):
    prior = add_fold_rank(
        fund_oos.dropna(subset=[target, pcol]).copy(),
        pcol,
    )
    min_prior_folds = int(cfg.get("fund_cal_min_prior_folds", 8))
    min_rows = int(cfg.get("fund_cal_cv_min_train_rows", 1000))
    min_pos = int(cfg.get("fund_cal_cv_min_train_positives", 25))
    anchor_folds = int(cfg.get("fund_cal_anchor_recent_folds", 8))

    raw_current = np.asarray(raw_current, float)
    if (
        prior["date"].nunique() < min_prior_folds
        or len(prior) < min_rows
        or int(prior[target].sum()) < min_pos
        or prior[target].nunique() < 2
    ):
        return raw_current

    cur_rank = pd.Series(raw_current).rank(
        pct=True, method="average"
    ).to_numpy(float)
    cal = RankPlatt().fit(
        prior["p_fund_rank"].to_numpy(float),
        prior[target].astype(int).to_numpy(),
    )
    pred = cal.predict(cur_rank)
    anchor = prior_fold_anchor(prior, target, anchor_folds)
    if not np.isfinite(anchor):
        return pred
    return shift_to_mean_probability(pred, anchor)
