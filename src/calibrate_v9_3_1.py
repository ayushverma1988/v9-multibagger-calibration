from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss

import calibrate_v9_2 as base


def cfg_grid(cfg):
    pools = [int(x) for x in cfg.get("candidate_pool_sizes", [20, 30, 50, 75, 100])]
    drops = [float(x) for x in cfg.get("risk_drop_fractions", [0.0, 0.10, 0.20, 0.30, 0.40])]
    sw = [float(x) for x in cfg.get("safety_tiebreak_weights", [0.0, 0.05, 0.10, 0.15, 0.20])]
    cw = [float(x) for x in cfg.get("consensus_tiebreak_weights", [0.0, 0.05, 0.10, 0.15])]
    max_secondary = float(cfg.get("max_secondary_weight", 0.30))

    yield {
        "name": "V9.2_baseline",
        "pool_n": int(cfg.get("selection_k", 10)),
        "risk_drop": 0.0,
        "w_safety": 0.0,
        "w_consensus": 0.0,
        "baseline": True,
    }

    seen = set()
    for n in pools:
        for rd in drops:
            for ws in sw:
                for wc in cw:
                    if ws + wc > max_secondary + 1e-12:
                        continue
                    key = (n, round(rd, 6), round(ws, 6), round(wc, 6))
                    if key in seen:
                        continue
                    seen.add(key)
                    yield {
                        "name": f"N{n}_drop{rd:.2f}_s{ws:.2f}_c{wc:.2f}",
                        "pool_n": n,
                        "risk_drop": rd,
                        "w_safety": ws,
                        "w_consensus": wc,
                        "baseline": False,
                    }


def _rank_pct_high(s):
    return s.rank(pct=True, method="average", ascending=True)


def _rank_pct_low(s):
    return s.rank(pct=True, method="average", ascending=False)


def select_topk(g, spec, k):
    req = ["p_cal", "p_dd30_cal", "model_dispersion"]
    q = g.dropna(subset=req).copy()
    if len(q) < k:
        return pd.DataFrame()

    if spec.get("baseline", False):
        out = q.sort_values(["p_cal", "model_dispersion"], ascending=[False, True]).head(k).copy()
        out["selection_score"] = out["p_cal"]
        out["risk_cutoff"] = np.nan
        out["candidate_pool_n"] = min(k, len(q))
        return out

    pool_n = min(int(spec["pool_n"]), len(q))
    pool = q.sort_values(["p_cal", "model_dispersion"], ascending=[False, True]).head(pool_n).copy()
    if len(pool) < k:
        return pd.DataFrame()

    rd = float(spec["risk_drop"])
    if rd > 0:
        cutoff = float(pool["p_dd30_cal"].quantile(max(0.0, min(1.0, 1.0 - rd))))
        survivors = pool[pool["p_dd30_cal"] <= cutoff].copy()
    else:
        cutoff = float(pool["p_dd30_cal"].max())
        survivors = pool.copy()

    if len(survivors) < k:
        return pd.DataFrame()

    survivors["comp_alpha"] = _rank_pct_high(survivors["p_cal"])
    survivors["comp_safety"] = _rank_pct_low(survivors["p_dd30_cal"])
    survivors["comp_consensus"] = _rank_pct_low(survivors["model_dispersion"])

    ws = float(spec["w_safety"])
    wc = float(spec["w_consensus"])
    wa = 1.0 - ws - wc
    survivors["selection_score"] = (
        wa * survivors["comp_alpha"]
        + ws * survivors["comp_safety"]
        + wc * survivors["comp_consensus"]
    )
    survivors["risk_cutoff"] = cutoff
    survivors["candidate_pool_n"] = pool_n
    return survivors.sort_values(
        ["selection_score", "p_cal", "model_dispersion"],
        ascending=[False, False, True],
    ).head(k).copy()


def fold_metrics(g, spec, k):
    base_q = g.dropna(subset=["y6", "dd30_6m", "p_cal", "p_dd30_cal", "model_dispersion"]).copy()
    if len(base_q) < max(k, 20):
        return None

    sel = select_topk(base_q, spec, k)
    if len(sel) < k:
        return None

    universe_base = float(base_q["y6"].mean())
    precision = float(sel["y6"].mean())
    lift = precision / universe_base if universe_base > 0 else np.nan
    return {
        "n": int(len(base_q)),
        "precision_2x": precision,
        "lift_2x": lift,
        "hit": float(precision > 0),
        "dd30_rate": float(sel["dd30_6m"].mean()),
        "mean_p_cal": float(sel["p_cal"].mean()),
        "median_p_cal": float(sel["p_cal"].median()),
        "mean_p_dd30": float(sel["p_dd30_cal"].mean()),
        "base_rate_2x": universe_base,
    }


def metrics_by_fold(hist, spec, cfg):
    rows = []
    k = int(cfg.get("selection_k", 10))
    for td, g in hist.groupby("date"):
        m = fold_metrics(g, spec, k)
        if m is not None:
            m["date"] = pd.Timestamp(td)
            rows.append(m)
    return pd.DataFrame(rows)


def agg_metrics(t):
    if t.empty:
        return None

    finite_lift = t["lift_2x"].replace([np.inf, -np.inf], np.nan).dropna()
    dd = t["dd30_rate"].dropna()
    return {
        "folds": int(t["date"].nunique()),
        "mean_precision_2x": float(t["precision_2x"].mean()),
        "median_precision_2x": float(t["precision_2x"].median()),
        "mean_lift_2x": float(finite_lift.mean()) if len(finite_lift) else np.nan,
        "median_lift_2x": float(finite_lift.median()) if len(finite_lift) else np.nan,
        "hit_fold_rate": float(t["hit"].mean()),
        "hit_folds": int(t["hit"].sum()),
        "mean_dd30_rate": float(dd.mean()) if len(dd) else np.nan,
        "median_dd30_rate": float(dd.median()) if len(dd) else np.nan,
        "dd30_iqr": float(dd.quantile(0.75) - dd.quantile(0.25)) if len(dd) else np.nan,
        "mean_selected_p_cal": float(t["mean_p_cal"].mean()),
        "mean_selected_p_dd30": float(t["mean_p_dd30"].mean()),
    }


def alpha_gates(candidate, baseline, cfg):
    retain = float(cfg.get("alpha_retention", 0.90))
    if candidate is None or baseline is None:
        return False, {}

    gates = {
        "mean_precision": (
            candidate["mean_precision_2x"],
            retain * baseline["mean_precision_2x"],
        ),
        "mean_lift": (
            candidate["mean_lift_2x"],
            retain * baseline["mean_lift_2x"],
        ),
        "hit_fold_rate": (
            candidate["hit_fold_rate"],
            retain * baseline["hit_fold_rate"],
        ),
        "median_precision": (
            candidate["median_precision_2x"],
            retain * baseline["median_precision_2x"],
        ),
    }
    ok = True
    for _, (actual, required) in gates.items():
        if not (np.isfinite(actual) and np.isfinite(required) and actual + 1e-12 >= required):
            ok = False
            break
    return ok, {
        k: {"actual": float(v[0]), "required": float(v[1])}
        for k, v in gates.items()
    }


def optimize_config(hist, cfg):
    min_folds = int(cfg.get("selection_min_folds", 8))
    valid = hist.dropna(
        subset=["y6", "dd30_6m", "p_cal", "p_dd30_cal", "model_dispersion"]
    ).copy()
    if valid["date"].nunique() < min_folds:
        return None, pd.DataFrame(), None

    baseline_spec = next(cfg_grid(cfg))
    baseline_fold = metrics_by_fold(valid, baseline_spec, cfg)
    baseline_agg = agg_metrics(baseline_fold)
    if baseline_agg is None or baseline_agg["folds"] < min_folds:
        return None, pd.DataFrame(), baseline_agg

    rows = []
    best_spec = baseline_spec
    best_key = None

    for spec in cfg_grid(cfg):
        fm = metrics_by_fold(valid, spec, cfg)
        am = agg_metrics(fm)
        if am is None or am["folds"] < min_folds:
            continue

        feasible, gates = alpha_gates(am, baseline_agg, cfg)
        objective = (
            float(am["mean_dd30_rate"])
            + float(cfg.get("median_dd_weight", 0.35)) * float(am["median_dd30_rate"])
            + float(cfg.get("dd_iqr_weight", 0.15)) * float(am["dd30_iqr"])
        )
        if spec.get("baseline", False):
            feasible = True

        row = {
            **spec,
            **am,
            "feasible": bool(feasible),
            "objective": float(objective),
            "gate_mean_precision_required": gates.get("mean_precision", {}).get("required", np.nan),
            "gate_mean_lift_required": gates.get("mean_lift", {}).get("required", np.nan),
            "gate_hit_fold_rate_required": gates.get("hit_fold_rate", {}).get("required", np.nan),
            "gate_median_precision_required": gates.get("median_precision", {}).get("required", np.nan),
        }
        rows.append(row)

        if feasible and not spec.get("baseline", False):
            key = (
                objective,
                -am["mean_precision_2x"],
                -am["mean_lift_2x"],
                -am["hit_fold_rate"],
            )
            if best_key is None or key < best_key:
                best_key = key
                best_spec = spec.copy()

    table = pd.DataFrame(rows)
    if not table.empty:
        table = table.sort_values(
            ["feasible", "objective", "mean_precision_2x", "mean_lift_2x"],
            ascending=[False, True, False, False],
        ).reset_index(drop=True)

    return best_spec, table, baseline_agg


def build_metric_cache(oos, cfg):
    valid = oos.dropna(
        subset=["y6", "dd30_6m", "p_cal", "p_dd30_cal", "model_dispersion"]
    ).copy()
    specs = list(cfg_grid(cfg))
    k = int(cfg.get("selection_k", 10))
    rows = []
    for td, g in valid.groupby("date", sort=True):
        for spec in specs:
            m = fold_metrics(g, spec, k)
            if m is None:
                continue
            rows.append({
                "date": pd.Timestamp(td),
                "config": spec["name"],
                **m,
            })
    return pd.DataFrame(rows)


def optimize_from_cache(cache, cfg):
    min_folds = int(cfg.get("selection_min_folds", 8))
    if cache.empty:
        return None, pd.DataFrame(), None

    spec_map = {s["name"]: s for s in cfg_grid(cfg)}
    b = cache[cache["config"] == "V9.2_baseline"].copy()
    baseline_agg = agg_metrics(b)
    if baseline_agg is None or baseline_agg["folds"] < min_folds:
        return None, pd.DataFrame(), baseline_agg

    rows = []
    best_spec = spec_map["V9.2_baseline"]
    best_key = None

    for name, fm in cache.groupby("config", sort=False):
        am = agg_metrics(fm)
        if am is None or am["folds"] < min_folds:
            continue
        spec = spec_map[name]
        feasible, gates = alpha_gates(am, baseline_agg, cfg)
        objective = (
            float(am["mean_dd30_rate"])
            + float(cfg.get("median_dd_weight", 0.35)) * float(am["median_dd30_rate"])
            + float(cfg.get("dd_iqr_weight", 0.15)) * float(am["dd30_iqr"])
        )
        if spec.get("baseline", False):
            feasible = True

        rows.append({
            **spec,
            **am,
            "feasible": bool(feasible),
            "objective": float(objective),
            "gate_mean_precision_required": gates.get("mean_precision", {}).get("required", np.nan),
            "gate_mean_lift_required": gates.get("mean_lift", {}).get("required", np.nan),
            "gate_hit_fold_rate_required": gates.get("hit_fold_rate", {}).get("required", np.nan),
            "gate_median_precision_required": gates.get("median_precision", {}).get("required", np.nan),
        })

        if feasible and not spec.get("baseline", False):
            key = (
                objective,
                -am["mean_precision_2x"],
                -am["mean_lift_2x"],
                -am["hit_fold_rate"],
            )
            if best_key is None or key < best_key:
                best_key = key
                best_spec = spec.copy()

    table = pd.DataFrame(rows)
    if not table.empty:
        table = table.sort_values(
            ["feasible", "objective", "mean_precision_2x", "mean_lift_2x"],
            ascending=[False, True, False, False],
        ).reset_index(drop=True)
    return best_spec, table, baseline_agg


def forward_select(oos, cfg):
    out = oos.copy()
    out["selection_score"] = np.nan
    out["selection_rank"] = np.nan
    out["selected_v931"] = False
    out["chosen_config"] = ""
    out["chosen_pool_n"] = np.nan
    out["chosen_risk_drop"] = np.nan
    out["chosen_w_safety"] = np.nan
    out["chosen_w_consensus"] = np.nan
    out["chosen_train_objective"] = np.nan

    print("Caching configuration metrics by fold...", flush=True)
    metric_cache = build_metric_cache(out, cfg)
    chosen_rows = []
    dates = sorted(pd.to_datetime(out["date"].unique()))
    min_folds = int(cfg.get("selection_min_folds", 8))

    for td in dates:
        prior_cache = metric_cache[metric_cache["date"] < td].copy()
        prior_valid_folds = prior_cache.loc[
            prior_cache["config"] == "V9.2_baseline", "date"
        ].nunique()
        if prior_valid_folds < min_folds:
            continue

        spec, search, baseline = optimize_from_cache(prior_cache, cfg)
        if spec is None:
            continue

        cur_idx = out.index[out["date"] == td]
        cur = out.loc[cur_idx].copy()
        selected = select_topk(cur, spec, int(cfg.get("selection_k", 10)))
        if selected.empty:
            spec = next(cfg_grid(cfg))
            selected = select_topk(cur, spec, int(cfg.get("selection_k", 10)))
        if selected.empty:
            continue

        out.loc[selected.index, "selection_score"] = selected["selection_score"]
        rank_map = pd.Series(
            range(1, len(selected) + 1),
            index=selected.index,
            dtype=float,
        )
        out.loc[selected.index, "selection_rank"] = rank_map
        out.loc[selected.index, "selected_v931"] = True

        objective = np.nan
        if not search.empty:
            hit = search[search["name"] == spec["name"]]
            if len(hit):
                objective = float(hit.iloc[0]["objective"])

        out.loc[cur_idx, "chosen_config"] = spec["name"]
        out.loc[cur_idx, "chosen_pool_n"] = float(spec["pool_n"])
        out.loc[cur_idx, "chosen_risk_drop"] = float(spec["risk_drop"])
        out.loc[cur_idx, "chosen_w_safety"] = float(spec["w_safety"])
        out.loc[cur_idx, "chosen_w_consensus"] = float(spec["w_consensus"])
        out.loc[cur_idx, "chosen_train_objective"] = objective

        chosen_rows.append({
            "date": td,
            "config": spec["name"],
            "pool_n": int(spec["pool_n"]),
            "risk_drop": float(spec["risk_drop"]),
            "w_safety": float(spec["w_safety"]),
            "w_consensus": float(spec["w_consensus"]),
            "train_objective": objective,
            "prior_valid_folds": int(prior_valid_folds),
            "fell_back_to_v92": bool(spec.get("baseline", False)),
        })

    return out, pd.DataFrame(chosen_rows), metric_cache


def forward_comparison(oos, cfg):
    rows = []
    k = int(cfg.get("selection_k", 10))
    eval_dates = sorted(
        pd.to_datetime(oos.loc[oos["selected_v931"], "date"].unique())
    )
    baseline_spec = next(cfg_grid(cfg))

    for td in eval_dates:
        g = oos[oos["date"] == td].copy()

        s = g[g["selected_v931"]].copy()
        q = g.dropna(
            subset=["y6", "dd30_6m", "p_cal", "p_dd30_cal", "model_dispersion"]
        ).copy()
        if len(s) == k and len(q) >= max(k, 20):
            base_rate = float(q["y6"].mean())
            p = float(s["y6"].mean())
            rows.append(
                {
                    "date": td,
                    "strategy": "V9.3.1",
                    "k": k,
                    "precision_2x": p,
                    "lift_2x": p / base_rate if base_rate > 0 else np.nan,
                    "hit": float(p > 0),
                    "dd30_rate": float(s["dd30_6m"].mean()),
                    "mean_p_cal": float(s["p_cal"].mean()),
                    "mean_p_dd30": float(s["p_dd30_cal"].mean()),
                    "base_rate_2x": base_rate,
                }
            )

        bm = fold_metrics(g, baseline_spec, k)
        if bm is not None:
            rows.append(
                {
                    "date": td,
                    "strategy": "V9.2_pcal",
                    "k": k,
                    **{x: bm[x] for x in [
                        "precision_2x",
                        "lift_2x",
                        "hit",
                        "dd30_rate",
                        "mean_p_cal",
                        "mean_p_dd30",
                        "base_rate_2x",
                    ]},
                }
            )
    return pd.DataFrame(rows)


def summarize_strategy(comp, name):
    d = comp[comp["strategy"] == name].copy()
    if d.empty:
        return {}
    lift = d["lift_2x"].replace([np.inf, -np.inf], np.nan).dropna()
    return {
        "folds": int(d["date"].nunique()),
        "mean_precision_2x": float(d["precision_2x"].mean()),
        "median_precision_2x": float(d["precision_2x"].median()),
        "mean_lift_2x": float(lift.mean()) if len(lift) else np.nan,
        "median_lift_2x": float(lift.median()) if len(lift) else np.nan,
        "hit_fold_rate": float(d["hit"].mean()),
        "hit_folds": int(d["hit"].sum()),
        "mean_dd30_rate": float(d["dd30_rate"].mean()),
        "median_dd30_rate": float(d["dd30_rate"].median()),
        "mean_selected_p_cal": float(d["mean_p_cal"].mean()),
        "mean_selected_p_dd30": float(d["mean_p_dd30"].mean()),
    }


def build_current(current, production_spec, cfg):
    current = current.copy()
    current["p_dd30_cal"] = current["p_dd30"]
    selected = select_topk(current, production_spec, int(cfg.get("selection_k", 10)))

    current["selected_v931"] = False
    current["selection_score"] = np.nan
    current["selection_rank"] = np.nan
    current["production_config"] = production_spec["name"]
    current["production_pool_n"] = int(production_spec["pool_n"])
    current["production_risk_drop"] = float(production_spec["risk_drop"])
    current["production_w_safety"] = float(production_spec["w_safety"])
    current["production_w_consensus"] = float(production_spec["w_consensus"])

    if not selected.empty:
        for rank, idx in enumerate(selected.index.tolist(), start=1):
            current.loc[idx, "selected_v931"] = True
            current.loc[idx, "selection_score"] = float(selected.loc[idx, "selection_score"])
            current.loc[idx, "selection_rank"] = rank

    return current.sort_values(
        ["selected_v931", "selection_rank", "p_cal"],
        ascending=[False, True, False],
        na_position="last",
    ).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_v9_3_1.json")
    ap.add_argument("--output", default="outputs_v9_3_1")
    ap.add_argument("--legacy-dir", default=None)
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    base.cfg_h24 = int(cfg["label_days"]["y24"])
    end_year = pd.Timestamp.today().year

    print("Loading V9.2 market data...", flush=True)
    daily = base.load_market(int(cfg["start_year"]), end_year, args.legacy_dir)
    print("Computing V9.2 features...", flush=True)
    daily = base.add_features(daily)
    snap = base.build_snapshots(daily, cfg)
    data = base.add_labels(snap, daily, cfg)
    data.to_parquet(outdir / "snapshot_dataset.parquet", index=False)

    print("Running V9.2 walk-forward alpha/downside models...", flush=True)
    oos = base.walk_forward(data, cfg)
    if oos.empty:
        raise RuntimeError("No OOS predictions generated")

    best, cal_tab = base.evaluate_calibrators(oos, "y6", "p_raw")
    best_dd, cal_dd_tab = base.evaluate_calibrators(oos, "dd30_6m", "p_dd30_raw")
    oos = base.apply_forward_calibration(oos, best, "y6", "p_raw", "p_cal")
    oos = base.apply_forward_calibration(
        oos, best_dd, "dd30_6m", "p_dd30_raw", "p_dd30_cal"
    )
    oos["asymmetry"] = oos["p_cal"] / np.maximum(oos["p_dd30_cal"], 0.01)

    print("Running constrained V9.3.1 selector...", flush=True)
    oos, chosen = forward_select(oos, cfg)
    comparison = forward_comparison(oos, cfg)

    oos.to_parquet(outdir / "oos_predictions.parquet", index=False)
    cal_tab.to_csv(outdir / "calibrator_comparison.csv", index=False)
    cal_dd_tab.to_csv(outdir / "downside_calibrator_comparison.csv", index=False)
    chosen.to_csv(outdir / "chosen_config_by_fold.csv", index=False)
    comparison.to_csv(outdir / "selection_metrics_by_fold.csv", index=False)

    production_spec, search, baseline_all = optimize_from_cache(metric_cache, cfg)
    if production_spec is None:
        production_spec = next(cfg_grid(cfg))
    search.to_csv(outdir / "constraint_search.csv", index=False)

    current = base.fit_current(data, daily, oos, best, best_dd, cfg)
    current = build_current(current, production_spec, cfg)
    current.to_csv(outdir / "current_selection.csv", index=False)
    current[current["selected_v931"]].head(50).to_csv(
        outdir / "current_top50.csv", index=False
    )

    eval_oos = oos.dropna(subset=["y6", "p_cal"])
    y = eval_oos["y6"].astype(int).to_numpy()
    p = eval_oos["p_cal"].to_numpy()
    slope, intercept = base.calibration_slope_intercept(y, p)

    s931 = summarize_strategy(comparison, "V9.3.1")
    s92 = summarize_strategy(comparison, "V9.2_pcal")

    retention = {}
    for key in ["mean_precision_2x", "mean_lift_2x", "hit_fold_rate", "median_precision_2x"]:
        b = s92.get(key, np.nan)
        a = s931.get(key, np.nan)
        retention[key] = float(a / b) if np.isfinite(a) and np.isfinite(b) and b > 0 else None

    summary = {
        "model": cfg["model_name"],
        "pipeline_version": cfg["pipeline_version"],
        "data_start": str(daily["date"].min().date()),
        "data_end": str(daily["date"].max().date()),
        "best_calibrator": best,
        "best_downside_calibrator": best_dd,
        "oos_rows_calibrated": int(len(eval_oos)),
        "oos_positive_6m": int(y.sum()),
        "base_rate_6m": float(y.mean()),
        "brier": float(brier_score_loss(y, p)),
        "logloss": float(log_loss(y, np.clip(p, base.EPS, 1 - base.EPS))),
        "pr_auc": float(average_precision_score(y, p)),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        "selection_k": int(cfg.get("selection_k", 10)),
        "alpha_retention_required": float(cfg.get("alpha_retention", 0.90)),
        "selector_design": {
            "stage_1": "candidate pool ranked by V9.2 calibrated 2x probability",
            "stage_2": "exclude high p_dd30 candidates by within-pool risk percentile",
            "stage_3": "rank survivors with p_cal dominant; safety/consensus only small tie-break weights",
            "optimization": "minimize drawdown and drawdown instability subject to alpha-retention gates",
            "fallback": "V9.2 p_cal ranking when no non-baseline configuration passes gates",
        },
        "production_config": production_spec,
        "forward_history": {
            "V9.3.1": s931,
            "V9.2_pcal": s92,
            "alpha_retention_ratios": retention,
        },
        "production_search_baseline": baseline_all,
    }
    json.dump(summary, open(outdir / "summary.json", "w"), indent=2)

    print(json.dumps(summary, indent=2), flush=True)
    cols = [
        "selection_rank",
        "symbol",
        "close",
        "p_cal",
        "p_conservative",
        "p_dd30",
        "model_dispersion",
        "selection_score",
    ]
    cols = [c for c in cols if c in current.columns]
    print(current[current["selected_v931"]].head(20)[cols].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
