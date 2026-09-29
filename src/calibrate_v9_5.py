from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import calibrate_v9_2 as base
import calibrate_v9_3_1 as v931
import fundamentals_pit as fpit


META_FEATURES = {
    "fund_quality_completeness",
    "fund_age_days",
    "fund_scope_consolidated",
    "fund_specialized_financial",
    "fund_mapping_score",
    "fund_mapped_fields",
    "fund_concept_mapping_fraction",
    "fund_synthetic_context",
}
FUND_FEATURES = [c for c in fpit.FEATURE_COLS if c not in META_FEATURES]

DIRECTION = {
    "fund_revenue_yoy": 1,
    "fund_revenue_qoq": 1,
    "fund_revenue_accel": 1,
    "fund_pat_yoy": 1,
    "fund_pat_qoq": 1,
    "fund_pat_turnaround": 1,
    "fund_operating_margin": 1,
    "fund_margin_delta_yoy": 1,
    "fund_interest_coverage": 1,
    "fund_debt_to_equity": -1,
    "fund_debt_change_yoy": -1,
    "fund_current_ratio": 1,
    "fund_ocf_to_pat": 1,
    "fund_fcf_margin": 1,
    "fund_shares_change_yoy": -1,
    "fund_promoter_change_yoy": 1,
    "fund_pledge_change_yoy": -1,
    "fund_revenue_yoy_ann": 1,
    "fund_pat_yoy_ann": 1,
    "fund_debt_change_yoy_ann": -1,
    "fund_ocf_to_pat_ann": 1,
    "fund_roe_proxy_ann": 1,
}


def add_fund_ranks(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in FUND_FEATURES:
        if c not in out.columns:
            out[c] = np.nan

    for _, idx in out.groupby("date").groups.items():
        for c in FUND_FEATURES:
            # rank(pct=True, ascending=True) gives the largest observation a
            # percentile near 1. For "lower is better" fields reverse it.
            direction = int(DIRECTION.get(c, 1))
            out.loc[idx, f"{c}_rank"] = out.loc[idx, c].rank(
                pct=True,
                method="average",
                ascending=(direction > 0),
            )
    return out


def fund_model_cols() -> list[str]:
    return [f"{c}_rank" for c in FUND_FEATURES]


def make_models(seed: int):
    elastic = Pipeline([
        ("imp", SimpleImputer(strategy="median", add_indicator=True)),
        ("sc", StandardScaler()),
        (
            "lr",
            LogisticRegression(
                penalty="elasticnet",
                solver="saga",
                l1_ratio=0.35,
                C=0.25,
                class_weight="balanced",
                max_iter=2500,
                random_state=seed,
            ),
        ),
    ])
    gbm = Pipeline([
        ("imp", SimpleImputer(strategy="median", add_indicator=True)),
        (
            "gbm",
            HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=220,
                max_leaf_nodes=15,
                min_samples_leaf=35,
                l2_regularization=2.0,
                random_state=seed + 1,
            ),
        ),
    ])
    return elastic, gbm


def usable_fund_mask(df: pd.DataFrame, cfg: dict) -> pd.Series:
    min_comp = float(cfg.get("fund_min_completeness", 0.30))
    min_core = float(cfg.get("fund_min_core_completeness", 0.34))
    max_age = float(cfg.get("fund_max_age_days", 550))

    overall = df["fund_quality_completeness"].fillna(0) >= min_comp
    core = df.get(
        "fund_core_completeness",
        pd.Series(0.0, index=df.index),
    ).fillna(0) >= min_core

    # A row is usable if it has either the richer modern feature set OR a
    # sufficiently complete legacy turnaround core. Missing optional features
    # are handled by the model's imputer; they are never filled with zero.
    mask = overall | core

    # Require at least one of the two primary turnaround measurements when
    # those columns exist, so "quality" cannot be satisfied only by metadata.
    primary = pd.Series(False, index=df.index)
    for col in ["fund_revenue_yoy", "fund_pat_yoy"]:
        if col in df.columns:
            primary |= df[col].notna()
    mask &= primary

    if "fund_age_days" in df.columns:
        mask &= df["fund_age_days"].fillna(np.inf) <= max_age

    min_map = float(cfg.get("fund_min_mapping_score", 0.55))
    min_fields = int(cfg.get("fund_min_mapped_fields", 3))
    if "fund_mapping_score" in df.columns:
        mask &= df["fund_mapping_score"].fillna(0.0) >= min_map
    if "fund_mapped_fields" in df.columns:
        mask &= df["fund_mapped_fields"].fillna(0.0) >= min_fields

    if bool(cfg.get("fund_exclude_specialized_financial", True)):
        mask &= df.get(
            "fund_specialized_financial",
            pd.Series(0.0, index=df.index),
        ).fillna(0) < 0.5

    return mask


def _predict_one(model, train, test, cols):
    tr = train.dropna(subset=["y6"]).copy()
    if len(tr) < 500 or tr["y6"].sum() < 20 or tr["y6"].nunique() < 2:
        return np.full(len(test), np.nan)
    try:
        model.fit(tr[cols], tr["y6"].astype(int))
        return model.predict_proba(test[cols])[:, 1]
    except Exception:
        return np.full(len(test), np.nan)


def walk_forward_fundamental(data: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rows = []
    dates = sorted(pd.to_datetime(data["date"].unique()))
    first_year = int(cfg.get("fund_first_oos_year", 2017))
    train_years = int(cfg.get("fund_rolling_train_years", 8))
    cols = fund_model_cols()
    seed = int(cfg["random_seed"])

    for td in dates:
        if td.year < first_year:
            continue

        cur_all = data[data["date"] == td].copy()
        test = cur_all[usable_fund_mask(cur_all, cfg)].copy()
        if test.empty:
            continue

        start = td - pd.DateOffset(years=train_years)
        train_all = data[(data["date"] < td) & (data["date"] >= start)].copy()
        train = train_all[usable_fund_mask(train_all, cfg)].copy()

        elastic, gbm = make_models(seed + td.year * 10 + td.month)
        p1 = _predict_one(elastic, train, test, cols)
        p2 = _predict_one(gbm, train, test, cols)
        arr = np.vstack([p1, p2]).T

        z = test[
            [
                "date", "symbol", "y6", "fund_quality_completeness",
                "fund_age_days", "fund_specialized_financial",
                "fund_mapping_score", "fund_mapped_fields",
                "fund_concept_mapping_fraction", "fund_synthetic_context",
            ]
        ].copy()
        z["p_fund_raw"] = np.nanmean(arr, axis=1)
        z["fund_model_dispersion"] = np.nanstd(arr, axis=1)
        z.attrs.clear()
        rows.append(z)

        print(
            f"FUND OOS {td.date()} train={len(train):,} test={len(test):,} "
            f"positives={int(test['y6'].fillna(0).sum())}",
            flush=True,
        )

    if not rows:
        return pd.DataFrame()
    # Pandas propagates attrs from parent slices. Some upstream attrs can hold
    # DataFrames, whose equality comparison is ambiguous during concat.
    clean = []
    for frame in rows:
        frame = frame.copy()
        frame.attrs.clear()
        clean.append(frame)
    out = pd.concat(clean, ignore_index=True)
    out.attrs.clear()
    return out


def calibration_metrics(df, target="y6", pcol="p_fund_cal"):
    q = df.dropna(subset=[target, pcol]).copy()
    if q.empty or q[target].nunique() < 2:
        return {}
    y = q[target].astype(int).to_numpy()
    p = np.clip(q[pcol].to_numpy(float), base.EPS, 1 - base.EPS)
    slope, intercept = base.calibration_slope_intercept(y, p)
    pr = float(average_precision_score(y, p))
    br = float(y.mean())
    return {
        "n": int(len(q)),
        "positives": int(y.sum()),
        "base_rate": br,
        "pr_auc": pr,
        "pr_auc_lift": float(pr / br) if br > 0 else np.nan,
        "brier": float(brier_score_loss(y, p)),
        "logloss": float(log_loss(y, p)),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
    }


def fundamental_gate(hist: pd.DataFrame, cfg: dict) -> dict:
    m = calibration_metrics(hist, "y6", "p_fund_cal")
    rec = {**m, "passed": False, "reasons": []}

    min_rows = int(cfg.get("fund_gate_min_rows", 3000))
    min_pos = int(cfg.get("fund_gate_min_positives", 100))
    min_folds = int(cfg.get("fund_gate_min_folds", 8))
    min_pr = float(cfg.get("fund_gate_min_pr_auc_lift", 1.10))
    min_slope = float(cfg.get("fund_gate_min_calibration_slope", 0.50))

    if rec.get("n", 0) < min_rows:
        rec["reasons"].append(f"n<{min_rows}")
    if rec.get("positives", 0) < min_pos:
        rec["reasons"].append(f"positives<{min_pos}")

    folds = int(hist.dropna(subset=["y6", "p_fund_cal"])["date"].nunique())
    rec["folds"] = folds
    if folds < min_folds:
        rec["reasons"].append(f"folds<{min_folds}")

    if not np.isfinite(rec.get("pr_auc_lift", np.nan)) or rec["pr_auc_lift"] < min_pr:
        rec["reasons"].append(f"pr_auc_lift<{min_pr}")
    if not np.isfinite(rec.get("calibration_slope", np.nan)) or rec["calibration_slope"] < min_slope:
        rec["reasons"].append(f"calibration_slope<{min_slope}")

    rec["passed"] = len(rec["reasons"]) == 0
    return rec


def risk_survivors(g: pd.DataFrame, spec: dict, cfg: dict) -> pd.DataFrame:
    k = int(cfg.get("selection_k", 10))
    q = g.dropna(
        subset=["p_cal", "p_dd30_cal", "model_dispersion"]
    ).copy()
    if len(q) < k:
        return pd.DataFrame()

    if spec.get("baseline", False):
        return q.sort_values(
            ["p_cal", "model_dispersion"], ascending=[False, True]
        ).copy()

    pool_n = min(int(spec["pool_n"]), len(q))
    pool = q.sort_values(
        ["p_cal", "model_dispersion"], ascending=[False, True]
    ).head(pool_n).copy()
    rd = float(spec["risk_drop"])
    if rd > 0:
        cutoff = float(pool["p_dd30_cal"].quantile(1.0 - rd))
        pool = pool[pool["p_dd30_cal"] <= cutoff].copy()
    return pool


def candidate_coverage(g: pd.DataFrame, spec: dict, cfg: dict) -> dict:
    surv = risk_survivors(g, spec, cfg)
    if surv.empty:
        return {"survivors": 0, "covered": 0, "coverage": 0.0}
    covered = surv.dropna(
        subset=["p_fund_cal", "fund_model_dispersion"]
    )
    return {
        "survivors": int(len(surv)),
        "covered": int(len(covered)),
        "coverage": float(len(covered) / len(surv)),
    }


def select_blended(g: pd.DataFrame, spec: dict, fund_weight: float, cfg: dict):
    k = int(cfg.get("selection_k", 10))
    wf = float(fund_weight)

    # Exact safe fallback: when fundamental weight is zero, call the proven
    # V9.3.1 selector on the full technical universe.
    if wf <= 0:
        sel = v931.select_topk(g, spec, k)
        if not sel.empty:
            sel = sel.copy()
            sel["selection_score_v95"] = sel["selection_score"]
        return sel

    surv = risk_survivors(g, spec, cfg)
    if len(surv) < k:
        return pd.DataFrame()
    pool = surv.dropna(
        subset=["p_fund_cal", "fund_model_dispersion"]
    ).copy()
    if len(pool) < k:
        return pd.DataFrame()

    ws = float(spec.get("w_safety", 0.0))
    wc = float(spec.get("w_consensus", 0.0))
    alpha_total = max(0.0, 1.0 - ws - wc)

    pool["comp100"] = pool["p_cal"].rank(pct=True, method="average")
    pool["compfund"] = pool["p_fund_cal"].rank(pct=True, method="average")
    pool["compsafety"] = pool["p_dd30_cal"].rank(
        pct=True, method="average", ascending=False
    )
    pool["compconsensus"] = pool["model_dispersion"].rank(
        pct=True, method="average", ascending=False
    )

    pool["selection_score_v95"] = (
        alpha_total
        * ((1.0 - wf) * pool["comp100"] + wf * pool["compfund"])
        + ws * pool["compsafety"]
        + wc * pool["compconsensus"]
    )
    return pool.sort_values(
        ["selection_score_v95", "p_cal", "model_dispersion"],
        ascending=[False, False, True],
    ).head(k).copy()


def fold_metrics(g, spec, w, cfg):
    k = int(cfg.get("selection_k", 10))
    all_valid = g.dropna(
        subset=["y6", "dd30_6m", "p_cal", "p_dd30_cal", "model_dispersion"]
    ).copy()
    if len(all_valid) < max(k, 20):
        return None

    cov = candidate_coverage(all_valid, spec, cfg)
    min_cov = float(cfg.get("fund_min_candidate_coverage", 0.70))
    applied = float(w) > 0 and cov["coverage"] >= min_cov and cov["covered"] >= k
    actual_w = float(w) if applied else 0.0

    sel = select_blended(all_valid, spec, actual_w, cfg)
    if sel.empty or len(sel) < k:
        actual_w = 0.0
        applied = False
        sel = select_blended(all_valid, spec, 0.0, cfg)
    if sel.empty or len(sel) < k:
        return None

    base_rate = float(all_valid["y6"].mean())
    precision = float(sel["y6"].mean())
    return {
        "precision_2x": precision,
        "lift_2x": precision / base_rate if base_rate > 0 else np.nan,
        "hit": float(precision > 0),
        "dd30_rate": float(sel["dd30_6m"].mean()),
        "base_rate_2x": base_rate,
        "candidate_coverage": cov["coverage"],
        "fund_weight_requested": float(w),
        "fund_weight_applied": actual_w,
        "fund_applied": float(applied),
    }


def metrics_by_fold(hist, spec, w, cfg):
    rows = []
    for td, g in hist.groupby("date"):
        m = fold_metrics(g, spec, w, cfg)
        if m is not None:
            m["date"] = pd.Timestamp(td)
            rows.append(m)
    return pd.DataFrame(rows)


def aggregate(hist, spec, w, cfg):
    t = metrics_by_fold(hist, spec, w, cfg)
    if t.empty:
        return None
    lift = np.clip(
        t["lift_2x"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float),
        0,
        10,
    )
    return {
        "folds": int(t["date"].nunique()),
        "mean_precision_2x": float(t["precision_2x"].mean()),
        "median_precision_2x": float(t["precision_2x"].median()),
        "mean_capped_lift_2x": float(lift.mean()) if len(lift) else np.nan,
        "hit_fold_rate": float(t["hit"].mean()),
        "hit_folds": int(t["hit"].sum()),
        "mean_dd30_rate": float(t["dd30_rate"].mean()),
        "median_dd30_rate": float(t["dd30_rate"].median()),
        "mean_candidate_coverage": float(t["candidate_coverage"].mean()),
        "median_candidate_coverage": float(t["candidate_coverage"].median()),
        "applied_fold_rate": float(t["fund_applied"].mean()),
        "applied_folds": int(t["fund_applied"].sum()),
    }


def alpha_retention(cand, baseline, cfg):
    r = float(cfg.get("fund_alpha_retention", 0.95))
    fields = [
        "mean_precision_2x",
        "mean_capped_lift_2x",
        "hit_fold_rate",
        "median_precision_2x",
    ]
    ok = True
    gates = {}
    for f in fields:
        actual = cand.get(f, np.nan)
        required = r * baseline.get(f, np.nan)
        gates[f] = {"actual": actual, "required": required}
        if not (
            np.isfinite(actual)
            and np.isfinite(required)
            and actual + 1e-12 >= required
        ):
            ok = False
    return ok, gates


def utility(a, baseline):
    if a is None or baseline is None:
        return -np.inf
    ratios = []
    for f in ["mean_precision_2x", "mean_capped_lift_2x", "hit_fold_rate"]:
        b = baseline.get(f, np.nan)
        v = a.get(f, np.nan)
        ratios.append(
            v / b if np.isfinite(v) and np.isfinite(b) and b > 0 else 0.0
        )
    dd_b = baseline.get("mean_dd30_rate", np.nan)
    dd_a = a.get("mean_dd30_rate", np.nan)
    dd_gain = (
        (dd_b - dd_a) / dd_b
        if np.isfinite(dd_b) and np.isfinite(dd_a) and dd_b > 0
        else 0.0
    )
    return float(
        0.45 * ratios[0]
        + 0.35 * ratios[1]
        + 0.20 * ratios[2]
        + 0.10 * dd_gain
    )


def optimize_weight(hist, spec, cfg):
    baseline = aggregate(hist, spec, 0.0, cfg)
    gate = fundamental_gate(hist, cfg)
    if baseline is None:
        return 0.0, pd.DataFrame(), baseline, gate

    weights = [
        float(x)
        for x in cfg.get(
            "fund_weight_grid",
            [0.0, 0.05, 0.10, 0.15, 0.20, 0.25],
        )
    ]
    base_u = utility(baseline, baseline)
    min_gain = float(cfg.get("fund_min_utility_gain", 0.02))
    best_w = 0.0
    best_u = base_u
    rows = []

    for w in weights:
        if w > 0 and not gate["passed"]:
            continue
        a = aggregate(hist, spec, w, cfg)
        if a is None:
            continue
        feasible, gates = alpha_retention(a, baseline, cfg)
        u = utility(a, baseline)
        if w == 0:
            feasible = True
        rows.append({
            "fund_weight": w,
            "feasible": bool(feasible),
            "utility": u,
            **a,
            "required_mean_precision_2x": gates.get(
                "mean_precision_2x", {}
            ).get("required", np.nan),
            "required_mean_capped_lift_2x": gates.get(
                "mean_capped_lift_2x", {}
            ).get("required", np.nan),
            "required_hit_fold_rate": gates.get(
                "hit_fold_rate", {}
            ).get("required", np.nan),
            "required_median_precision_2x": gates.get(
                "median_precision_2x", {}
            ).get("required", np.nan),
        })
        if (
            feasible
            and w > 0
            and u >= base_u + min_gain
            and u > best_u
        ):
            best_w = w
            best_u = u

    tab = pd.DataFrame(rows)
    if not tab.empty:
        tab = tab.sort_values(
            ["feasible", "utility", "mean_precision_2x"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
    return best_w, tab, baseline, gate


def spec_from_chosen(row):
    return {
        "name": row.config,
        "pool_n": int(row.pool_n),
        "risk_drop": float(row.risk_drop),
        "w_safety": float(row.w_safety),
        "w_consensus": float(row.w_consensus),
        "baseline": bool(row.fell_back_to_v92),
    }


def forward_v95(oos, chosen931, cfg):
    out = oos.copy()
    out["selected_v95"] = False
    out["selection_score_v95"] = np.nan
    out["selection_rank_v95"] = np.nan
    out["v95_fund_weight"] = 0.0
    out["v95_fund_gate_passed"] = False
    out["v95_candidate_coverage"] = np.nan

    rows = []
    chosen_map = {
        pd.Timestamp(r.date): r
        for r in chosen931.itertuples(index=False)
    }

    for td in sorted(chosen_map):
        g = out[out["date"] == td].copy()
        # No calibrated fundamental probability for this fold means exact
        # V9.4.1/V9.3.1 fallback; still record it for the comparison.
        prior = out[out["date"] < td].copy()
        spec = spec_from_chosen(chosen_map[td])
        requested_w, search, _, gate = optimize_weight(prior, spec, cfg)

        cov = candidate_coverage(g, spec, cfg)
        min_cov = float(cfg.get("fund_min_candidate_coverage", 0.70))
        actual_w = (
            requested_w
            if requested_w > 0
            and cov["coverage"] >= min_cov
            and cov["covered"] >= int(cfg.get("selection_k", 10))
            else 0.0
        )

        sel = select_blended(g, spec, actual_w, cfg)
        if sel.empty:
            actual_w = 0.0
            sel = select_blended(g, spec, 0.0, cfg)
        if sel.empty:
            continue

        for rank, idx in enumerate(sel.index.tolist(), start=1):
            out.loc[idx, "selected_v95"] = True
            score = (
                sel.loc[idx, "selection_score_v95"]
                if "selection_score_v95" in sel.columns
                else sel.loc[idx, "selection_score"]
            )
            out.loc[idx, "selection_score_v95"] = float(score)
            out.loc[idx, "selection_rank_v95"] = rank

        idx_date = out.index[out["date"] == td]
        out.loc[idx_date, "v95_fund_weight"] = actual_w
        out.loc[idx_date, "v95_fund_gate_passed"] = bool(gate["passed"])
        out.loc[idx_date, "v95_candidate_coverage"] = cov["coverage"]

        train_utility = np.nan
        if not search.empty:
            z = search[search["fund_weight"] == requested_w]
            if len(z):
                train_utility = float(z.iloc[0]["utility"])

        rows.append({
            "date": td,
            "risk_config": spec["name"],
            "requested_fund_weight": requested_w,
            "applied_fund_weight": actual_w,
            "fund_gate_passed": bool(gate["passed"]),
            "fund_gate_pr_auc_lift": gate.get("pr_auc_lift"),
            "fund_gate_calibration_slope": gate.get("calibration_slope"),
            "fund_gate_reasons": "|".join(gate.get("reasons", [])),
            "candidate_survivors": cov["survivors"],
            "candidate_fund_covered": cov["covered"],
            "candidate_coverage": cov["coverage"],
            "train_utility": train_utility,
            "prior_fund_calibrated_folds": int(
                prior.dropna(subset=["p_fund_cal"])["date"].nunique()
            ),
        })

    return out, pd.DataFrame(rows)


def comparison_by_fold(oos, cfg):
    rows = []
    k = int(cfg.get("selection_k", 10))
    dates = sorted(
        pd.to_datetime(oos.loc[oos["selected_v95"], "date"].unique())
    )

    for td in dates:
        g = oos[oos["date"] == td].copy()
        q = g.dropna(
            subset=["y6", "dd30_6m", "p_cal", "p_dd30_cal", "model_dispersion"]
        ).copy()
        if len(q) < max(k, 20):
            continue
        base_rate = float(q["y6"].mean())

        strategies = {
            "V9.5": g[g["selected_v95"]].copy(),
            "V9.4.1_V9.3.1": g[g["selected_v931"]].copy(),
        }

        for name, sel in strategies.items():
            if len(sel) != k:
                continue
            precision = float(sel["y6"].mean())
            rows.append({
                "date": td,
                "strategy": name,
                "k": k,
                "precision_2x": precision,
                "lift_2x": precision / base_rate if base_rate > 0 else np.nan,
                "hit": float(precision > 0),
                "dd30_rate": float(sel["dd30_6m"].mean()),
                "base_rate_2x": base_rate,
                "mean_p_cal": float(sel["p_cal"].mean()),
                "mean_p_dd30": float(sel["p_dd30_cal"].mean()),
                "fund_weight": float(g["v95_fund_weight"].iloc[0]),
                "candidate_coverage": float(g["v95_candidate_coverage"].iloc[0])
                if np.isfinite(g["v95_candidate_coverage"].iloc[0])
                else np.nan,
            })

    return pd.DataFrame(rows)


def summarize_strategy(comp, name):
    d = comp[comp["strategy"] == name].copy()
    if d.empty:
        return {}
    lift = np.clip(
        d["lift_2x"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float),
        0,
        10,
    )
    return {
        "folds": int(d["date"].nunique()),
        "mean_precision_2x": float(d["precision_2x"].mean()),
        "median_precision_2x": float(d["precision_2x"].median()),
        "mean_capped_lift_2x": float(lift.mean()) if len(lift) else np.nan,
        "median_lift_2x": float(
            d["lift_2x"].replace([np.inf, -np.inf], np.nan).median()
        ),
        "hit_fold_rate": float(d["hit"].mean()),
        "hit_folds": int(d["hit"].sum()),
        "mean_dd30_rate": float(d["dd30_rate"].mean()),
        "median_dd30_rate": float(d["dd30_rate"].median()),
        "mean_fund_weight": float(d["fund_weight"].mean()),
        "active_fund_folds": int((d["fund_weight"] > 0).sum()),
        "mean_candidate_coverage": float(
            d["candidate_coverage"].dropna().mean()
        ) if d["candidate_coverage"].notna().any() else np.nan,
    }


def fit_current_fundamental(data, current_market, fund_oos, best_fund, cfg, fundamentals):
    last = pd.Timestamp(current_market["date"].max())
    cur_features = fpit.build_snapshot_features(
        current_market[["date", "symbol"]],
        fundamentals,
        min_completeness=float(cfg.get("fund_min_completeness", 0.30)),
    )
    cur_features = add_fund_ranks(cur_features)
    usable = usable_fund_mask(cur_features, cfg)

    out = cur_features[
        [
            "date", "symbol", "fund_quality_completeness",
            "fund_age_days", "fund_specialized_financial",
            "fund_mapping_score", "fund_mapped_fields",
            "fund_concept_mapping_fraction", "fund_synthetic_context",
        ]
    ].copy()
    out["p_fund_raw"] = np.nan
    out["p_fund_cal"] = np.nan
    out["fund_model_dispersion"] = np.nan

    test = cur_features[usable].copy()
    if test.empty:
        return out

    start = last - pd.DateOffset(
        years=int(cfg.get("fund_rolling_train_years", 8))
    )
    train_all = data[
        (data["date"] < last)
        & (data["date"] >= start)
        & data["y6"].notna()
    ].copy()
    train = train_all[usable_fund_mask(train_all, cfg)].copy()

    elastic, gbm = make_models(int(cfg["random_seed"]) + 9500)
    cols = fund_model_cols()
    p1 = _predict_one(elastic, train, test, cols)
    p2 = _predict_one(gbm, train, test, cols)
    arr = np.vstack([p1, p2]).T
    raw = np.nanmean(arr, axis=1)
    disp = np.nanstd(arr, axis=1)

    cal = base.fit_final_calibrator(
        fund_oos,
        best_fund,
        "y6",
        "p_fund_raw",
    )
    pcal = raw if cal is None else cal.predict(raw)

    out.loc[test.index, "p_fund_raw"] = raw
    out.loc[test.index, "p_fund_cal"] = pcal
    out.loc[test.index, "fund_model_dispersion"] = disp
    return out


def snapshot_coverage(data, cfg):
    rows = []
    for td, g in data.groupby("date"):
        usable = usable_fund_mask(g, cfg)
        rows.append({
            "date": pd.Timestamp(td),
            "market_rows": int(len(g)),
            "fund_usable_rows": int(usable.sum()),
            "fund_usable_fraction": float(usable.mean()) if len(g) else 0.0,
            "mean_completeness": float(
                g["fund_quality_completeness"].fillna(0).mean()
            ),
            "specialized_financial_fraction": float(
                g["fund_specialized_financial"].fillna(0).mean()
            ),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_v9_5.json")
    ap.add_argument("--fundamentals", required=True)
    ap.add_argument("--output", default="outputs_v9_5")
    ap.add_argument("--legacy-dir", default=None)
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    fundamentals = fpit.read_fundamentals(args.fundamentals)

    base.cfg_h24 = int(cfg["label_days"]["y24"])
    daily = base.load_market(
        int(cfg["start_year"]),
        pd.Timestamp.today().year,
        args.legacy_dir,
    )
    daily = base.add_features(daily)
    snap = base.build_snapshots(daily, cfg)
    data = base.add_labels(snap, daily, cfg)

    ff = fpit.build_snapshot_features(
        data[["date", "symbol"]],
        fundamentals,
        min_completeness=float(cfg.get("fund_min_completeness", 0.30)),
    )
    data = data.merge(ff, on=["date", "symbol"], how="left")
    data = add_fund_ranks(data)
    cov = snapshot_coverage(data, cfg)
    cov.to_csv(outdir / "snapshot_fundamental_coverage.csv", index=False)

    print("Running fundamental-only walk-forward challenger...", flush=True)
    fund_oos = walk_forward_fundamental(data, cfg)
    if fund_oos.empty:
        raise RuntimeError("No fundamental OOS predictions generated")

    best_fund, fund_cal_tab = base.evaluate_calibrators(
        fund_oos,
        "y6",
        "p_fund_raw",
    )
    fund_oos = base.apply_forward_calibration(
        fund_oos,
        best_fund,
        "y6",
        "p_fund_raw",
        "p_fund_cal",
    )

    print("Running technical/risk baseline...", flush=True)
    market_oos = base.walk_forward(data, cfg)
    best100, tab100 = base.evaluate_calibrators(
        market_oos, "y6", "p_raw"
    )
    bestdd, tabdd = base.evaluate_calibrators(
        market_oos, "dd30_6m", "p_dd30_raw"
    )
    market_oos = base.apply_forward_calibration(
        market_oos, best100, "y6", "p_raw", "p_cal"
    )
    market_oos = base.apply_forward_calibration(
        market_oos,
        bestdd,
        "dd30_6m",
        "p_dd30_raw",
        "p_dd30_cal",
    )

    oos = market_oos.merge(
        fund_oos[
            [
                "date", "symbol", "p_fund_raw", "p_fund_cal",
                "fund_model_dispersion", "fund_quality_completeness",
                "fund_age_days", "fund_specialized_financial",
                "fund_mapping_score", "fund_mapped_fields",
                "fund_concept_mapping_fraction", "fund_synthetic_context",
            ]
        ],
        on=["date", "symbol"],
        how="left",
    )

    oos, chosen931, metric_cache = v931.forward_select(oos, cfg)
    print("Running leakage-free V9.5 weight selection...", flush=True)
    oos, chosen95 = forward_v95(oos, chosen931, cfg)
    comp = comparison_by_fold(oos, cfg)

    production_spec, risk_search, _ = v931.optimize_from_cache(
        metric_cache, cfg
    )
    if production_spec is None:
        production_spec = next(v931.cfg_grid(cfg))

    production_weight, weight_search, baseline, prod_gate = optimize_weight(
        oos, production_spec, cfg
    )

    # Current market + fundamental challenger.
    current_market = base.fit_current(
        data, daily, market_oos, best100, bestdd, cfg
    )
    current_fund = fit_current_fundamental(
        data, current_market, fund_oos, best_fund, cfg, fundamentals
    )
    current = current_market.merge(
        current_fund[
            [
                "date", "symbol", "p_fund_raw", "p_fund_cal",
                "fund_model_dispersion", "fund_quality_completeness",
                "fund_age_days", "fund_specialized_financial",
                "fund_mapping_score", "fund_mapped_fields",
                "fund_concept_mapping_fraction", "fund_synthetic_context",
            ]
        ],
        on=["date", "symbol"],
        how="left",
    )
    current["p_dd30_cal"] = current["p_dd30"]

    current_cov = candidate_coverage(current, production_spec, cfg)
    min_cov = float(cfg.get("fund_min_candidate_coverage", 0.70))
    applied_weight = (
        production_weight
        if production_weight > 0
        and prod_gate["passed"]
        and current_cov["coverage"] >= min_cov
        and current_cov["covered"] >= int(cfg.get("selection_k", 10))
        else 0.0
    )
    current_sel = select_blended(
        current, production_spec, applied_weight, cfg
    )
    if current_sel.empty:
        applied_weight = 0.0
        current_sel = select_blended(
            current, production_spec, 0.0, cfg
        )

    current["selected_v95"] = False
    current["selection_rank_v95"] = np.nan
    current["selection_score_v95"] = np.nan
    for rank, idx in enumerate(current_sel.index.tolist(), start=1):
        current.loc[idx, "selected_v95"] = True
        current.loc[idx, "selection_rank_v95"] = rank
        score = (
            current_sel.loc[idx, "selection_score_v95"]
            if "selection_score_v95" in current_sel.columns
            else current_sel.loc[idx, "selection_score"]
        )
        current.loc[idx, "selection_score_v95"] = float(score)

    current["production_fund_weight"] = applied_weight
    current["production_fund_gate_passed"] = bool(prod_gate["passed"])
    current["production_candidate_coverage"] = current_cov["coverage"]
    current["production_risk_config"] = production_spec["name"]
    current = current.sort_values(
        ["selected_v95", "selection_rank_v95", "p_cal"],
        ascending=[False, True, False],
        na_position="last",
    ).reset_index(drop=True)

    # Save outputs.
    oos.to_parquet(outdir / "oos_predictions.parquet", index=False)
    fund_oos.to_parquet(outdir / "fundamental_oos.parquet", index=False)
    fund_cal_tab.to_csv(
        outdir / "fund_calibrator_comparison.csv", index=False
    )
    tab100.to_csv(
        outdir / "calibrator_100_comparison.csv", index=False
    )
    tabdd.to_csv(
        outdir / "downside_calibrator_comparison.csv", index=False
    )
    chosen931.to_csv(
        outdir / "chosen_v931_config_by_fold.csv", index=False
    )
    chosen95.to_csv(
        outdir / "chosen_v95_weight_by_fold.csv", index=False
    )
    comp.to_csv(
        outdir / "selection_metrics_by_fold.csv", index=False
    )
    risk_search.to_csv(
        outdir / "risk_constraint_search.csv", index=False
    )
    weight_search.to_csv(
        outdir / "fund_weight_search.csv", index=False
    )
    current.to_csv(outdir / "current_selection.csv", index=False)
    current[current["selected_v95"]].head(50).to_csv(
        outdir / "current_top50.csv", index=False
    )

    s95 = summarize_strategy(comp, "V9.5")
    sbase = summarize_strategy(comp, "V9.4.1_V9.3.1")
    retention = {}
    for f in [
        "mean_precision_2x",
        "mean_capped_lift_2x",
        "hit_fold_rate",
        "median_precision_2x",
    ]:
        a = s95.get(f, np.nan)
        b = sbase.get(f, np.nan)
        retention[f] = (
            float(a / b)
            if np.isfinite(a) and np.isfinite(b) and b > 0
            else None
        )

    summary = {
        "model": cfg["model_name"],
        "pipeline_version": cfg["pipeline_version"],
        "data_start": str(daily["date"].min().date()),
        "data_end": str(daily["date"].max().date()),
        "fundamental_data_rows": int(len(fundamentals)),
        "fundamental_symbols": int(fundamentals["symbol"].nunique()),
        "fundamental_raw_oos_rows": int(
            fund_oos["p_fund_raw"].notna().sum()
        ),
        "fundamental_calibrated_oos_rows": int(
            fund_oos["p_fund_cal"].notna().sum()
        ),
        "best_fundamental_calibrator": best_fund,
        "production_fundamental_gate": prod_gate,
        "production_risk_config": production_spec,
        "production_requested_fundamental_weight": production_weight,
        "production_applied_fundamental_weight": applied_weight,
        "production_candidate_coverage": current_cov,
        "forward_history": {
            "V9.5": s95,
            "V9.4.1_V9.3.1": sbase,
            "V9.5_alpha_retention_ratios": retention,
        },
        "baseline_production_search": baseline,
        "policy": {
            "point_in_time": "only filings broadcast strictly before each snapshot may be used",
            "same_day_filings": "excluded conservatively",
            "scope": "prefer consolidated; standalone fallback",
            "specialized_financial_taxonomies": (
                "excluded from first generic challenger"
                if cfg.get("fund_exclude_specialized_financial", True)
                else "included"
            ),
            "selection": "fundamental P(2x) may receive 5-25% alpha weight only after signal-quality, coverage and 95% alpha-retention gates",
            "fallback": "exact full-universe V9.4.1/V9.3.1 selector",
        },
    }
    json.dump(
        summary,
        open(outdir / "summary.json", "w"),
        indent=2,
        default=str,
    )

    print(json.dumps(summary, indent=2, default=str), flush=True)
    cols = [
        "selection_rank_v95", "symbol", "close",
        "p_cal", "p_fund_cal", "p_dd30",
        "production_fund_weight", "selection_score_v95",
    ]
    cols = [c for c in cols if c in current.columns]
    print(
        current[current["selected_v95"]].head(20)[cols].to_string(
            index=False
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
