from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import calibrate_v9_2 as base
import calibrate_v9_3_1 as v931
import fundamentals_pit as fpit


FUND_FEATURES = [
    c for c in fpit.FEATURE_COLS
    if c not in {
        "fund_quality_completeness",
        "fund_age_days",
        "fund_scope_consolidated",
    }
]


def add_fund_ranks(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in FUND_FEATURES + [
        "fund_quality_completeness",
        "fund_age_days",
        "fund_scope_consolidated",
    ]:
        if c not in out.columns:
            out[c] = np.nan

    # Directional transformations make higher rank generally better.
    direction = {
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
        "fund_quality_completeness": 1,
        "fund_age_days": -1,
        "fund_scope_consolidated": 1,
    }

    for td, idx in out.groupby("date").groups.items():
        for c, d in direction.items():
            s = out.loc[idx, c]
            r = s.rank(pct=True, method="average", ascending=(d < 0))
            out.loc[idx, f"{c}_rank"] = r
    return out


def fund_model_cols() -> list[str]:
    return [
        f"{c}_rank"
        for c in FUND_FEATURES + [
            "fund_quality_completeness",
            "fund_age_days",
            "fund_scope_consolidated",
        ]
    ]


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
                max_iter=2000,
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


def _predict_one(model, train, test, cols):
    tr = train.dropna(subset=["y6"]).copy()
    if len(tr) < 500 or tr["y6"].sum() < 20 or tr["y6"].nunique() < 2:
        return np.full(len(test), np.nan)
    model.fit(tr[cols], tr["y6"].astype(int))
    return model.predict_proba(test[cols])[:, 1]


def walk_forward_fundamental(data: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rows = []
    dates = sorted(pd.to_datetime(data["date"].unique()))
    first_year = int(cfg.get("fund_first_oos_year", 2017))
    min_comp = float(cfg.get("fund_min_completeness", 0.30))
    train_years = int(cfg.get("fund_rolling_train_years", 8))
    cols = fund_model_cols()
    seed = int(cfg["random_seed"])

    for td in dates:
        if td.year < first_year:
            continue
        test = data[
            (data["date"] == td)
            & (data["fund_quality_completeness"] >= min_comp)
        ].copy()
        if test.empty:
            continue

        start = td - pd.DateOffset(years=train_years)
        train = data[
            (data["date"] < td)
            & (data["date"] >= start)
            & (data["fund_quality_completeness"] >= min_comp)
        ].copy()

        elastic, gbm = make_models(seed + td.year * 10 + td.month)
        p1 = _predict_one(elastic, train, test, cols)
        p2 = _predict_one(gbm, train, test, cols)
        arr = np.vstack([p1, p2]).T

        z = test[["date", "symbol", "y6"]].copy()
        z["p_fund_raw"] = np.nanmean(arr, axis=1)
        z["fund_model_dispersion"] = np.nanstd(arr, axis=1)
        z["fund_quality_completeness"] = test["fund_quality_completeness"].to_numpy()
        rows.append(z)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


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
    rec = {
        **m,
        "passed": False,
        "reasons": [],
    }

    checks = [
        ("n", int(cfg.get("fund_gate_min_rows", 3000)), ">="),
        ("positives", int(cfg.get("fund_gate_min_positives", 100)), ">="),
    ]
    for k, lim, _ in checks:
        if rec.get(k, 0) < lim:
            rec["reasons"].append(f"{k}<{lim}")

    folds = int(hist.dropna(subset=["y6", "p_fund_cal"])["date"].nunique())
    rec["folds"] = folds
    min_folds = int(cfg.get("fund_gate_min_folds", 8))
    if folds < min_folds:
        rec["reasons"].append(f"folds<{min_folds}")

    min_pr = float(cfg.get("fund_gate_min_pr_auc_lift", 1.10))
    if not np.isfinite(rec.get("pr_auc_lift", np.nan)) or rec["pr_auc_lift"] < min_pr:
        rec["reasons"].append(f"pr_auc_lift<{min_pr}")

    min_slope = float(cfg.get("fund_gate_min_calibration_slope", 0.50))
    if not np.isfinite(rec.get("calibration_slope", np.nan)) or rec["calibration_slope"] < min_slope:
        rec["reasons"].append(f"calibration_slope<{min_slope}")

    rec["passed"] = len(rec["reasons"]) == 0
    return rec


def select_topk(g, risk_spec, fund_weight, k):
    req = [
        "p_cal", "p_dd30_cal", "model_dispersion",
        "p_fund_cal", "fund_model_dispersion",
    ]
    q = g.dropna(subset=req).copy()
    if len(q) < k:
        return pd.DataFrame()

    if risk_spec.get("baseline", False):
        pool = q.sort_values(["p_cal", "model_dispersion"], ascending=[False, True]).head(k).copy()
        pool["selection_score_v95"] = pool["p_cal"]
        return pool

    pool_n = min(int(risk_spec["pool_n"]), len(q))
    pool = q.sort_values(["p_cal", "model_dispersion"], ascending=[False, True]).head(pool_n).copy()
    rd = float(risk_spec["risk_drop"])
    if rd > 0:
        cutoff = float(pool["p_dd30_cal"].quantile(1.0 - rd))
        pool = pool[pool["p_dd30_cal"] <= cutoff].copy()
    if len(pool) < k:
        return pd.DataFrame()

    wf = float(fund_weight)
    ws = float(risk_spec["w_safety"])
    wc = float(risk_spec["w_consensus"])
    alpha_total = max(0.0, 1.0 - ws - wc)

    pool["comp100"] = pool["p_cal"].rank(pct=True, method="average")
    pool["compfund"] = pool["p_fund_cal"].rank(pct=True, method="average")
    pool["compsafety"] = pool["p_dd30_cal"].rank(pct=True, method="average", ascending=False)
    consensus = (
        pool["model_dispersion"].rank(pct=True, method="average", ascending=False)
        + pool["fund_model_dispersion"].rank(pct=True, method="average", ascending=False)
    ) / 2.0

    pool["selection_score_v95"] = (
        alpha_total * ((1.0 - wf) * pool["comp100"] + wf * pool["compfund"])
        + ws * pool["compsafety"]
        + wc * consensus
    )
    return pool.sort_values(
        ["selection_score_v95", "p_cal", "model_dispersion"],
        ascending=[False, False, True],
    ).head(k).copy()


def fold_metrics(g, risk_spec, w, cfg):
    k = int(cfg.get("selection_k", 10))
    q = g.dropna(
        subset=["y6", "dd30_6m", "p_cal", "p_fund_cal", "p_dd30_cal", "model_dispersion"]
    ).copy()
    if len(q) < max(k, 20):
        return None
    sel = select_topk(q, risk_spec, w, k)
    if len(sel) < k:
        return None
    base_rate = float(q["y6"].mean())
    precision = float(sel["y6"].mean())
    return {
        "precision_2x": precision,
        "lift_2x": precision / base_rate if base_rate > 0 else np.nan,
        "hit": float(precision > 0),
        "dd30_rate": float(sel["dd30_6m"].mean()),
        "base_rate_2x": base_rate,
    }


def aggregate(hist, risk_spec, w, cfg):
    rows = []
    for td, g in hist.groupby("date"):
        m = fold_metrics(g, risk_spec, w, cfg)
        if m is not None:
            m["date"] = pd.Timestamp(td)
            rows.append(m)
    t = pd.DataFrame(rows)
    if t.empty:
        return None
    lift = np.clip(
        t["lift_2x"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float),
        0, 10,
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
        if not (np.isfinite(actual) and np.isfinite(required) and actual + 1e-12 >= required):
            ok = False
    return ok, gates


def utility(a, baseline):
    if a is None or baseline is None:
        return -np.inf
    ratios = []
    for f in ["mean_precision_2x", "mean_capped_lift_2x", "hit_fold_rate"]:
        b = baseline.get(f, np.nan)
        v = a.get(f, np.nan)
        ratios.append(v / b if np.isfinite(v) and np.isfinite(b) and b > 0 else 0.0)
    dd_b = baseline.get("mean_dd30_rate", np.nan)
    dd_a = a.get("mean_dd30_rate", np.nan)
    dd_gain = (dd_b - dd_a) / dd_b if np.isfinite(dd_b) and dd_b > 0 else 0.0
    return float(0.45 * ratios[0] + 0.35 * ratios[1] + 0.20 * ratios[2] + 0.10 * dd_gain)


def optimize_weight(hist, risk_spec, cfg):
    baseline = aggregate(hist, risk_spec, 0.0, cfg)
    gate = fundamental_gate(hist, cfg)
    if baseline is None:
        return 0.0, pd.DataFrame(), baseline, gate

    weights = [float(x) for x in cfg.get("fund_weight_grid", [0, .05, .10, .15, .20, .25])]
    base_u = utility(baseline, baseline)
    min_gain = float(cfg.get("fund_min_utility_gain", 0.02))
    best_w = 0.0
    best_u = base_u
    rows = []

    for w in weights:
        if w > 0 and not gate["passed"]:
            continue
        a = aggregate(hist, risk_spec, w, cfg)
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
            "required_mean_precision_2x": gates.get("mean_precision_2x", {}).get("required", np.nan),
            "required_mean_capped_lift_2x": gates.get("mean_capped_lift_2x", {}).get("required", np.nan),
            "required_hit_fold_rate": gates.get("hit_fold_rate", {}).get("required", np.nan),
            "required_median_precision_2x": gates.get("median_precision_2x", {}).get("required", np.nan),
        })
        if feasible and w > 0 and u >= base_u + min_gain and u > best_u:
            best_w = w
            best_u = u

    tab = pd.DataFrame(rows)
    if not tab.empty:
        tab = tab.sort_values(["feasible", "utility"], ascending=[False, False])
    return best_w, tab, baseline, gate


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

    base.cfg_h24 = int(cfg["label_days"]["y24"])
    daily = base.load_market(int(cfg["start_year"]), pd.Timestamp.today().year, args.legacy_dir)
    daily = base.add_features(daily)
    snap = base.build_snapshots(daily, cfg)
    data = base.add_labels(snap, daily, cfg)

    fundamentals = fpit.read_fundamentals(args.fundamentals)
    ff = fpit.build_snapshot_features(
        data[["date", "symbol"]],
        fundamentals,
        min_completeness=float(cfg.get("fund_min_completeness", 0.30)),
    )
    data = data.merge(ff, on=["date", "symbol"], how="left")
    data = add_fund_ranks(data)

    fund_oos = walk_forward_fundamental(data, cfg)
    if fund_oos.empty:
        raise RuntimeError("No fundamental OOS predictions generated")

    best_fund, fund_cal_tab = base.evaluate_calibrators(fund_oos, "y6", "p_fund_raw")
    fund_oos = base.apply_forward_calibration(
        fund_oos, best_fund, "y6", "p_fund_raw", "p_fund_cal"
    )

    market_oos = base.walk_forward(data, cfg)
    best100, _ = base.evaluate_calibrators(market_oos, "y6", "p_raw")
    bestdd, _ = base.evaluate_calibrators(market_oos, "dd30_6m", "p_dd30_raw")
    market_oos = base.apply_forward_calibration(market_oos, best100, "y6", "p_raw", "p_cal")
    market_oos = base.apply_forward_calibration(
        market_oos, bestdd, "dd30_6m", "p_dd30_raw", "p_dd30_cal"
    )

    oos = market_oos.merge(
        fund_oos[
            [
                "date", "symbol", "p_fund_raw", "p_fund_cal",
                "fund_model_dispersion", "fund_quality_completeness",
            ]
        ],
        on=["date", "symbol"],
        how="left",
    )

    oos, chosen931, metric_cache = v931.forward_select(oos, cfg)
    production_spec, _, _ = v931.optimize_from_cache(metric_cache, cfg)
    if production_spec is None:
        production_spec = next(v931.cfg_grid(cfg))

    weight, search, baseline, gate = optimize_weight(oos, production_spec, cfg)
    search.to_csv(outdir / "fund_weight_search.csv", index=False)
    fund_cal_tab.to_csv(outdir / "fund_calibrator_comparison.csv", index=False)
    oos.to_parquet(outdir / "oos_predictions.parquet", index=False)

    summary = {
        "model": cfg["model_name"],
        "pipeline_version": cfg["pipeline_version"],
        "fundamental_data_rows": int(len(fundamentals)),
        "fundamental_symbols": int(fundamentals["symbol"].nunique()),
        "fundamental_oos_rows": int(fund_oos["p_fund_cal"].notna().sum()),
        "fundamental_gate": gate,
        "best_fundamental_calibrator": best_fund,
        "production_risk_config": production_spec,
        "production_fundamental_weight": weight,
        "baseline_v931_like": baseline,
        "policy": {
            "point_in_time": "only filings broadcast strictly before each snapshot may be used",
            "scope": "prefer consolidated; standalone fallback",
            "selection": "fundamental challenger may receive 5-25% alpha weight only after signal-quality and 95% alpha-retention gates",
            "fallback": "zero fundamental weight, leaving V9.4.1/V9.3.1 ranking unchanged",
        },
    }
    json.dump(summary, open(outdir / "summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
