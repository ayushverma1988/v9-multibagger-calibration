from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss


EPS = 1e-6


def add_fold_rank(df: pd.DataFrame, pcol: str = "p_fund_raw") -> pd.DataFrame:
    x = df.copy()
    x["date"] = pd.to_datetime(x["date"])
    x["p_fund_rank"] = x.groupby("date")[pcol].rank(
        pct=True, method="average"
    )
    return x


class Platt:
    def __init__(self):
        self.lr = LogisticRegression(C=2.0, max_iter=2000)

    def fit(self, p, y):
        p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
        self.lr.fit(logit(p).reshape(-1, 1), np.asarray(y, int))
        return self

    def predict(self, p):
        p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
        return self.lr.predict_proba(logit(p).reshape(-1, 1))[:, 1]


class Beta:
    def __init__(self):
        self.lr = LogisticRegression(C=2.0, max_iter=2000)

    def _x(self, p):
        p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
        return np.c_[np.log(p), -np.log1p(-p)]

    def fit(self, p, y):
        self.lr.fit(self._x(p), np.asarray(y, int))
        return self

    def predict(self, p):
        return self.lr.predict_proba(self._x(p))[:, 1]


class Iso:
    def __init__(self):
        self.iso = IsotonicRegression(out_of_bounds="clip")

    def fit(self, p, y):
        self.iso.fit(np.asarray(p, float), np.asarray(y, int))
        return self

    def predict(self, p):
        return np.clip(self.iso.predict(np.asarray(p, float)), EPS, 1 - EPS)


def fit_calibrator(method: str, p, y):
    if method == "platt":
        return Platt().fit(p, y)
    if method == "beta":
        return Beta().fit(p, y)
    if method == "isotonic":
        return Iso().fit(p, y)
    raise ValueError(method)


def calibration_slope_intercept(y, p):
    y = np.asarray(y, int)
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    if len(np.unique(y)) < 2:
        return np.nan, np.nan
    lr = LogisticRegression(C=1e5, max_iter=2000)
    lr.fit(logit(p).reshape(-1, 1), y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def metrics(df: pd.DataFrame, pcol: str) -> dict:
    q = df.dropna(subset=["y6", pcol]).copy()
    if q.empty or q["y6"].nunique() < 2:
        return {}
    y = q["y6"].astype(int).to_numpy()
    p = np.clip(q[pcol].to_numpy(float), EPS, 1 - EPS)
    br = float(y.mean())
    pr = float(average_precision_score(y, p))
    slope, intercept = calibration_slope_intercept(y, p)
    return {
        "n": int(len(q)),
        "positives": int(y.sum()),
        "folds": int(q["date"].nunique()),
        "base_rate": br,
        "pr_auc": pr,
        "pr_auc_lift": float(pr / br) if br > 0 else np.nan,
        "brier": float(brier_score_loss(y, p)),
        "logloss": float(log_loss(y, p)),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
    }


def within_fold_rank_metrics(df: pd.DataFrame, pcol: str) -> pd.DataFrame:
    rows = []
    for td, g in df.dropna(subset=["y6", pcol]).groupby("date"):
        if g["y6"].nunique() < 2:
            continue
        y = g["y6"].astype(int).to_numpy()
        p = g[pcol].to_numpy(float)
        br = float(y.mean())
        pr = float(average_precision_score(y, p))
        rows.append({
            "date": pd.Timestamp(td),
            "rows": int(len(g)),
            "positives": int(y.sum()),
            "base_rate": br,
            "pr_auc": pr,
            "pr_auc_lift": float(pr / br) if br > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def training_window(prior: pd.DataFrame, recent_folds: int | None):
    if recent_folds is None:
        return prior
    dates = sorted(pd.to_datetime(prior["date"].unique()))
    keep = dates[-int(recent_folds):]
    return prior[prior["date"].isin(keep)].copy()


def forward_fixed(
    df: pd.DataFrame,
    feature_col: str,
    method: str,
    cfg: dict,
    recent_folds: int | None = None,
):
    out = df.copy()
    col = f"p_stab_{feature_col}_{method}"
    if recent_folds is not None:
        col += f"_r{recent_folds}"
    out[col] = np.nan

    min_prior_folds = int(cfg.get("stability_min_prior_folds", 8))
    min_rows = int(cfg.get("stability_min_train_rows", 1000))
    min_pos = int(cfg.get("stability_min_train_positives", 25))

    diag = []
    for td in sorted(pd.to_datetime(out["date"].unique())):
        prior_all = out[
            (out["date"] < td)
            & out["y6"].notna()
            & out[feature_col].notna()
        ].copy()
        idx = out.index[
            (out["date"] == td)
            & out["y6"].notna()
            & out[feature_col].notna()
        ]
        if len(idx) == 0:
            continue

        prior_folds = int(prior_all["date"].nunique())
        if prior_folds < min_prior_folds:
            continue

        train = training_window(prior_all, recent_folds)
        train_pos = int(train["y6"].sum())
        if (
            len(train) < min_rows
            or train_pos < min_pos
            or train["y6"].nunique() < 2
        ):
            continue

        cal = fit_calibrator(
            method,
            train[feature_col].to_numpy(float),
            train["y6"].astype(int).to_numpy(),
        )
        pred = cal.predict(out.loc[idx, feature_col].to_numpy(float))
        out.loc[idx, col] = pred

        diag.append({
            "date": td,
            "policy": col,
            "prior_folds": prior_folds,
            "train_folds": int(train["date"].nunique()),
            "train_rows": int(len(train)),
            "train_positives": train_pos,
        })
    return out, col, pd.DataFrame(diag)


def prior_anchor(prior: pd.DataFrame, recent_folds: int | None = None) -> float:
    q = training_window(prior, recent_folds)
    if q.empty:
        return np.nan
    # Equal weight per historical fold prevents recent high-coverage years from
    # overwhelming the base-rate estimate simply because they have more rows.
    rates = q.groupby("date")["y6"].mean().dropna()
    return float(rates.mean()) if len(rates) else np.nan


def shift_to_mean_probability(p: np.ndarray, target_mean: float) -> np.ndarray:
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


def forward_rank_anchor(
    df: pd.DataFrame,
    method: str,
    cfg: dict,
    anchor_recent_folds: int | None,
):
    out, base_col, diag = forward_fixed(
        df, "p_fund_rank", method, cfg, recent_folds=None
    )
    suffix = "all" if anchor_recent_folds is None else f"r{anchor_recent_folds}"
    col = f"{base_col}_anchor_{suffix}"
    out[col] = np.nan

    for td in sorted(pd.to_datetime(out["date"].unique())):
        idx = out.index[(out["date"] == td) & out[base_col].notna()]
        if len(idx) == 0:
            continue
        prior = out[
            (out["date"] < td)
            & out["y6"].notna()
            & out["p_fund_raw"].notna()
        ].copy()
        anchor = prior_anchor(prior, anchor_recent_folds)
        if not np.isfinite(anchor):
            continue
        out.loc[idx, col] = shift_to_mean_probability(
            out.loc[idx, base_col].to_numpy(float), anchor
        )
    return out, col, diag


def fold_diagnostics(df: pd.DataFrame, policies: list[str]) -> pd.DataFrame:
    rows = []
    for td, g in df.groupby("date"):
        for pcol in policies:
            q = g.dropna(subset=["y6", pcol])
            if q.empty:
                continue
            y = q["y6"].astype(int).to_numpy()
            p = q[pcol].to_numpy(float)
            br = float(y.mean())
            pr = (
                float(average_precision_score(y, p))
                if len(np.unique(y)) >= 2 else np.nan
            )
            rows.append({
                "date": pd.Timestamp(td),
                "policy": pcol,
                "rows": int(len(q)),
                "positives": int(y.sum()),
                "base_rate": br,
                "mean_probability": float(np.mean(p)),
                "pr_auc": pr,
                "pr_auc_lift": float(pr / br) if br > 0 and np.isfinite(pr) else np.nan,
            })
    return pd.DataFrame(rows)


def pass_gate(m: dict, cfg: dict) -> tuple[bool, list[str]]:
    reasons = []
    if m.get("folds", 0) < int(cfg.get("fund_gate_min_folds", 8)):
        reasons.append("folds")
    if m.get("n", 0) < int(cfg.get("fund_gate_min_rows", 3000)):
        reasons.append("rows")
    if m.get("positives", 0) < int(cfg.get("fund_gate_min_positives", 100)):
        reasons.append("positives")
    if not np.isfinite(m.get("pr_auc_lift", np.nan)) or (
        m["pr_auc_lift"] < float(cfg.get("fund_gate_min_pr_auc_lift", 1.10))
    ):
        reasons.append("pr_auc_lift")
    if not np.isfinite(m.get("calibration_slope", np.nan)) or (
        m["calibration_slope"] < float(cfg.get("fund_gate_min_calibration_slope", 0.50))
    ):
        reasons.append("calibration_slope")
    return len(reasons) == 0, reasons


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    d = pd.read_parquet(args.input)
    d["date"] = pd.to_datetime(d["date"])
    d = d.dropna(subset=["p_fund_raw"]).copy()
    d = add_fold_rank(d)

    policies = []
    policy_diags = []

    specs = [
        ("p_fund_raw", "platt", None),
        ("p_fund_raw", "beta", None),
        ("p_fund_rank", "platt", None),
        ("p_fund_rank", "beta", None),
        ("p_fund_rank", "isotonic", None),
        ("p_fund_rank", "platt", 4),
        ("p_fund_rank", "beta", 4),
        ("p_fund_rank", "platt", 8),
        ("p_fund_rank", "beta", 8),
    ]

    work = d.copy()
    for feature, method, recent in specs:
        tmp, col, diag = forward_fixed(
            work, feature, method, cfg, recent_folds=recent
        )
        work[col] = tmp[col]
        policies.append(col)
        if len(diag):
            policy_diags.append(diag)

    for method in ["platt", "beta"]:
        for anchor_recent in [None, 4, 8]:
            tmp, col, diag = forward_rank_anchor(
                work, method, cfg, anchor_recent
            )
            work[col] = tmp[col]
            policies.append(col)
            if len(diag):
                policy_diags.append(diag)

    # Existing nested policy remains an explicit benchmark.
    if "p_fund_cal" in work.columns:
        policies.append("p_fund_cal")
    if "p_fund_cal_old" in work.columns:
        policies.append("p_fund_cal_old")

    results = []
    for col in policies:
        m = metrics(work, col)
        passed, reasons = pass_gate(m, cfg)
        results.append({
            "policy": col,
            **m,
            "passed": passed,
            "fail_reasons": "|".join(reasons),
        })

    table = pd.DataFrame(results)
    if len(table):
        table = table.sort_values(
            ["passed", "pr_auc_lift", "calibration_slope", "logloss"],
            ascending=[False, False, False, True],
        ).reset_index(drop=True)
    table.to_csv(outdir / "calibration_policy_comparison.csv", index=False)

    fd = fold_diagnostics(work, policies)
    fd.to_csv(outdir / "calibration_by_fold.csv", index=False)

    raw_fd = within_fold_rank_metrics(work, "p_fund_raw")
    raw_fd.to_csv(outdir / "raw_signal_by_fold.csv", index=False)

    if policy_diags:
        pd.concat(policy_diags, ignore_index=True).drop_duplicates().to_csv(
            outdir / "policy_training_diagnostics.csv", index=False
        )

    keep = ["date", "symbol", "y6", "p_fund_raw", "p_fund_rank"] + policies
    keep = [c for c in keep if c in work.columns]
    work[keep].to_parquet(
        outdir / "calibration_stability_predictions.parquet", index=False
    )

    best = table.iloc[0].to_dict() if len(table) else {}
    summary = {
        "purpose": "V9.5 calibration stability challenger",
        "leakage_policy": (
            "all calibration fits use strictly prior OOS folds; current-fold "
            "percentile ranks use current scores only and never current labels"
        ),
        "final_quality_gates_unchanged": {
            "min_folds": int(cfg.get("fund_gate_min_folds", 8)),
            "min_rows": int(cfg.get("fund_gate_min_rows", 3000)),
            "min_positives": int(cfg.get("fund_gate_min_positives", 100)),
            "min_pr_auc_lift": float(cfg.get("fund_gate_min_pr_auc_lift", 1.10)),
            "min_calibration_slope": float(cfg.get("fund_gate_min_calibration_slope", 0.50)),
        },
        "best_policy": best,
        "raw_within_fold": {
            "folds": int(raw_fd["date"].nunique()) if len(raw_fd) else 0,
            "median_pr_auc_lift": float(raw_fd["pr_auc_lift"].median()) if len(raw_fd) else None,
            "mean_pr_auc_lift": float(raw_fd["pr_auc_lift"].mean()) if len(raw_fd) else None,
            "folds_above_1x": int((raw_fd["pr_auc_lift"] > 1.0).sum()) if len(raw_fd) else 0,
            "folds_above_1_1x": int((raw_fd["pr_auc_lift"] >= 1.10).sum()) if len(raw_fd) else 0,
        },
        "production_action": (
            "do not change V9.5 production weight unless a leakage-free policy "
            "passes every unchanged gate"
        ),
    }
    json.dump(summary, open(outdir / "summary.json", "w"), indent=2, default=str)
    print(table.to_string(index=False), flush=True)
    print(json.dumps(summary, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
