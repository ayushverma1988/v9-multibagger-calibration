from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import calibrate_v9_2 as base
import calibrate_v9_3_1 as v931
import calibrate_v9_4 as v94


SIGNALS = {
    "p25": ("hit25_6m", "hit25_6m_cal_raw"),
    "p50": ("hit50_6m", "hit50_6m_cal_raw"),
}


def signal_quality(hist: pd.DataFrame, signal: str, cfg: dict) -> dict:
    target, pcol = SIGNALS[signal]
    q = hist.dropna(subset=["date", target, pcol]).copy()
    folds = int(q["date"].nunique()) if len(q) else 0

    rec = {
        "signal": signal,
        "target": target,
        "probability_column": pcol,
        "n": int(len(q)),
        "positives": int(q[target].sum()) if len(q) else 0,
        "folds": folds,
        "base_rate": np.nan,
        "pr_auc": np.nan,
        "pr_auc_lift": np.nan,
        "calibration_slope": np.nan,
        "calibration_intercept": np.nan,
        "passed": False,
        "reasons": [],
    }

    min_rows = int(cfg.get("signal_quality_min_rows", 3000))
    min_pos = int(cfg.get("signal_quality_min_positives", 100))
    min_folds = int(cfg.get("signal_quality_min_folds", 8))
    min_pr_lift = float(cfg.get("signal_quality_min_pr_auc_lift", 1.05))
    min_slope = float(cfg.get("signal_quality_min_calibration_slope", 0.25))

    if rec["n"] < min_rows:
        rec["reasons"].append(f"rows<{min_rows}")
    if rec["positives"] < min_pos:
        rec["reasons"].append(f"positives<{min_pos}")
    if folds < min_folds:
        rec["reasons"].append(f"folds<{min_folds}")

    if len(q) and q[target].nunique() >= 2:
        try:
            m = v94.calibration_metrics(q, target, pcol)
            rec.update({
                "base_rate": float(m.get("base_rate", np.nan)),
                "pr_auc": float(m.get("pr_auc", np.nan)),
                "calibration_slope": float(m.get("calibration_slope", np.nan)),
                "calibration_intercept": float(m.get("calibration_intercept", np.nan)),
            })
            if np.isfinite(rec["base_rate"]) and rec["base_rate"] > 0 and np.isfinite(rec["pr_auc"]):
                rec["pr_auc_lift"] = float(rec["pr_auc"] / rec["base_rate"])
        except Exception as exc:
            rec["reasons"].append(f"metric_error:{type(exc).__name__}")
    else:
        rec["reasons"].append("single_class_or_empty")

    if not np.isfinite(rec["pr_auc_lift"]) or rec["pr_auc_lift"] + 1e-12 < min_pr_lift:
        rec["reasons"].append(f"pr_auc_lift<{min_pr_lift}")
    if not np.isfinite(rec["calibration_slope"]) or rec["calibration_slope"] + 1e-12 < min_slope:
        rec["reasons"].append(f"calibration_slope<{min_slope}")

    rec["passed"] = len(rec["reasons"]) == 0
    return rec


def gated_ladder_shares(cfg: dict, gates: dict):
    for shares in v94.ladder_shares(cfg):
        s100, s50, s25 = shares
        if s50 > 0 and not gates["p50"]["passed"]:
            continue
        if s25 > 0 and not gates["p25"]["passed"]:
            continue
        yield shares


def optimize_ladder_gated(hist: pd.DataFrame, risk_spec: dict, cfg: dict):
    min_folds = int(cfg.get("ladder_min_folds", 8))
    baseline_shares = (1.0, 0.0, 0.0)
    gates = {
        "p25": signal_quality(hist, "p25", cfg),
        "p50": signal_quality(hist, "p50", cfg),
    }

    bt = v94.ladder_metrics_by_fold(hist, risk_spec, baseline_shares, cfg)
    ba = v94.agg_ladder(bt)
    if ba is None or ba["folds"] < min_folds:
        return baseline_shares, pd.DataFrame(), ba, gates

    baseline_obj = v94.ladder_objective(ba, cfg)
    min_gain = float(cfg.get("ladder_min_objective_gain", 0.02))
    best = baseline_shares
    best_obj = baseline_obj
    rows = []

    for shares in gated_ladder_shares(cfg, gates):
        t = v94.ladder_metrics_by_fold(hist, risk_spec, shares, cfg)
        a = v94.agg_ladder(t)
        if a is None or a["folds"] < min_folds:
            continue

        feasible, alpha_gates = v94.alpha_gates_v94(a, ba, cfg)
        obj = v94.ladder_objective(a, cfg)
        is_baseline = shares == baseline_shares
        if is_baseline:
            feasible = True

        row = {
            "share_p100": shares[0],
            "share_p50": shares[1],
            "share_p25": shares[2],
            "baseline": is_baseline,
            "feasible": bool(feasible),
            "objective": float(obj),
            "p50_signal_allowed": bool(gates["p50"]["passed"]),
            "p25_signal_allowed": bool(gates["p25"]["passed"]),
            "p50_pr_auc_lift": gates["p50"]["pr_auc_lift"],
            "p25_pr_auc_lift": gates["p25"]["pr_auc_lift"],
            "p50_calibration_slope": gates["p50"]["calibration_slope"],
            "p25_calibration_slope": gates["p25"]["calibration_slope"],
            **a,
            "required_mean_precision_100": alpha_gates.get("mean_precision_100", {}).get("required", np.nan),
            "required_mean_capped_lift_100": alpha_gates.get("mean_capped_lift_100", {}).get("required", np.nan),
            "required_hit_fold_rate_100": alpha_gates.get("hit_fold_rate_100", {}).get("required", np.nan),
            "required_median_precision_100": alpha_gates.get("median_precision_100", {}).get("required", np.nan),
        }
        rows.append(row)

        if (
            feasible
            and not is_baseline
            and obj >= baseline_obj + min_gain
            and obj > best_obj
        ):
            best = shares
            best_obj = obj

    table = pd.DataFrame(rows)
    if not table.empty:
        table = table.sort_values(
            ["feasible", "objective", "mean_precision_100"],
            ascending=[False, False, False],
        ).reset_index(drop=True)

    return best, table, ba, gates


def forward_v941(oos: pd.DataFrame, chosen931: pd.DataFrame, cfg: dict):
    out = oos.copy()
    out["selected_v941"] = False
    out["selection_score_v941"] = np.nan
    out["selection_rank_v941"] = np.nan
    out["v941_share_p100"] = np.nan
    out["v941_share_p50"] = np.nan
    out["v941_share_p25"] = np.nan
    out["v941_gate_p50"] = False
    out["v941_gate_p25"] = False

    chosen_rows = []
    chosen_map = {
        pd.Timestamp(r.date): r
        for r in chosen931.itertuples(index=False)
    }

    for td in sorted(chosen_map):
        r = chosen_map[td]
        spec = {
            "name": r.config,
            "pool_n": int(r.pool_n),
            "risk_drop": float(r.risk_drop),
            "w_safety": float(r.w_safety),
            "w_consensus": float(r.w_consensus),
            "baseline": bool(r.fell_back_to_v92),
        }

        prior = out[out["date"] < td].copy()
        shares, search, _, gates = optimize_ladder_gated(prior, spec, cfg)

        cur = out[out["date"] == td].copy()
        sel = v94.select_ladder_topk(cur, spec, shares, int(cfg.get("selection_k", 10)))
        if sel.empty:
            shares = (1.0, 0.0, 0.0)
            sel = v94.select_ladder_topk(cur, spec, shares, int(cfg.get("selection_k", 10)))
        if sel.empty:
            continue

        for rank, idx in enumerate(sel.index.tolist(), start=1):
            out.loc[idx, "selected_v941"] = True
            out.loc[idx, "selection_score_v941"] = float(sel.loc[idx, "selection_score_v94"])
            out.loc[idx, "selection_rank_v941"] = rank

        idx_date = out.index[out["date"] == td]
        out.loc[idx_date, "v941_share_p100"] = shares[0]
        out.loc[idx_date, "v941_share_p50"] = shares[1]
        out.loc[idx_date, "v941_share_p25"] = shares[2]
        out.loc[idx_date, "v941_gate_p50"] = bool(gates["p50"]["passed"])
        out.loc[idx_date, "v941_gate_p25"] = bool(gates["p25"]["passed"])

        train_obj = np.nan
        if not search.empty:
            hit = search[
                (search["share_p100"] == shares[0])
                & (search["share_p50"] == shares[1])
                & (search["share_p25"] == shares[2])
            ]
            if len(hit):
                train_obj = float(hit.iloc[0]["objective"])

        chosen_rows.append({
            "date": td,
            "risk_config": spec["name"],
            "share_p100": shares[0],
            "share_p50": shares[1],
            "share_p25": shares[2],
            "ladder_changed_ranking": bool(shares != (1.0, 0.0, 0.0)),
            "prior_folds": int(prior["date"].nunique()),
            "train_objective": train_obj,
            "p50_allowed": bool(gates["p50"]["passed"]),
            "p50_pr_auc_lift": gates["p50"]["pr_auc_lift"],
            "p50_calibration_slope": gates["p50"]["calibration_slope"],
            "p50_quality_reasons": "|".join(gates["p50"]["reasons"]),
            "p25_allowed": bool(gates["p25"]["passed"]),
            "p25_pr_auc_lift": gates["p25"]["pr_auc_lift"],
            "p25_calibration_slope": gates["p25"]["calibration_slope"],
            "p25_quality_reasons": "|".join(gates["p25"]["reasons"]),
        })

    return out, pd.DataFrame(chosen_rows)


def comparison_by_fold(oos: pd.DataFrame, cfg: dict):
    rows = []
    k = int(cfg.get("selection_k", 10))
    dates = sorted(pd.to_datetime(oos.loc[oos["selected_v941"], "date"].unique()))

    for td in dates:
        g = oos[oos["date"] == td].copy()
        q = g.dropna(
            subset=[
                "y6", "hit50_6m", "hit25_6m", "dd30_6m",
                "p_cal", "p50_cal", "p25_cal", "model_dispersion",
            ]
        ).copy()
        if len(q) < max(k, 20):
            continue

        selections = {
            "V9.4.1": g[g["selected_v941"]].copy(),
            "V9.4": g[g["selected_v94"]].copy(),
            "V9.3.1": g[g["selected_v931"]].copy(),
            "V9.2_pcal": q.sort_values(
                ["p_cal", "model_dispersion"], ascending=[False, True]
            ).head(k).copy(),
        }

        for name, sel in selections.items():
            if len(sel) != k:
                continue
            rec = {"date": td, "strategy": name, "k": k}
            for label, suffix in [
                ("y6", "100"),
                ("hit50_6m", "50"),
                ("hit25_6m", "25"),
            ]:
                br = float(q[label].mean())
                pr = float(sel[label].mean())
                rec[f"precision_{suffix}"] = pr
                rec[f"lift_{suffix}"] = pr / br if br > 0 else np.nan
            rec["hit100"] = float(rec["precision_100"] > 0)
            rec["dd30_rate"] = float(sel["dd30_6m"].mean())
            rows.append(rec)

    return pd.DataFrame(rows)


def build_current_v941(current, threshold_cur, oos, threshold_info, production_spec, shares, gates, cfg):
    cur = v94.build_current_v94(
        current, threshold_cur, oos, threshold_info, production_spec, shares, cfg
    ).copy()
    cur = cur.rename(columns={
        "selected_v94": "selected_v941",
        "selection_score_v94": "selection_score_v941",
        "selection_rank_v94": "selection_rank_v941",
    })
    cur["production_p50_allowed"] = bool(gates["p50"]["passed"])
    cur["production_p25_allowed"] = bool(gates["p25"]["passed"])
    cur["production_p50_pr_auc_lift"] = gates["p50"]["pr_auc_lift"]
    cur["production_p25_pr_auc_lift"] = gates["p25"]["pr_auc_lift"]
    cur["production_p50_calibration_slope"] = gates["p50"]["calibration_slope"]
    cur["production_p25_calibration_slope"] = gates["p25"]["calibration_slope"]
    return cur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_v9_4_1.json")
    ap.add_argument("--output", default="outputs_v9_4_1")
    ap.add_argument("--legacy-dir", default=None)
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    base.cfg_h24 = int(cfg["label_days"]["y24"])
    end_year = pd.Timestamp.today().year

    print("Loading market data...", flush=True)
    daily = base.load_market(int(cfg["start_year"]), end_year, args.legacy_dir)
    daily = base.add_features(daily)
    snap = base.build_snapshots(daily, cfg)
    data = base.add_labels(snap, daily, cfg)
    data = v94.add_threshold_labels(data, daily, cfg)
    data.to_parquet(outdir / "snapshot_dataset.parquet", index=False)

    print("Running V9.2/V9.3.1 alpha and downside walk-forward...", flush=True)
    oos = base.walk_forward(data, cfg)
    if oos.empty:
        raise RuntimeError("No OOS predictions generated")

    labels = data[["date", "symbol", "hit25_6m", "hit50_6m", "threshold_mature_date"]]
    oos = oos.merge(labels, on=["date", "symbol"], how="left")

    best100, tab100 = base.evaluate_calibrators(oos, "y6", "p_raw")
    bestdd, tabdd = base.evaluate_calibrators(oos, "dd30_6m", "p_dd30_raw")
    oos = base.apply_forward_calibration(oos, best100, "y6", "p_raw", "p_cal")
    oos = base.apply_forward_calibration(
        oos, bestdd, "dd30_6m", "p_dd30_raw", "p_dd30_cal"
    )
    oos["p100_cal"] = oos["p_cal"]

    print("Training +25%/+50% probability models...", flush=True)
    thresh = v94.walk_forward_thresholds(data, oos["date"].unique(), cfg)
    oos = oos.merge(thresh, on=["date", "symbol"], how="left")
    oos, threshold_info = v94.calibrate_thresholds(oos)

    print("Running V9.3.1 constrained selector...", flush=True)
    oos, chosen931, metric_cache931 = v931.forward_select(oos, cfg)

    print("Running V9.4 benchmark selector...", flush=True)
    oos, chosen94 = v94.forward_v94(oos, chosen931, cfg)

    print("Running V9.4.1 signal-quality-gated selector...", flush=True)
    oos, chosen941 = forward_v941(oos, chosen931, cfg)
    comp = comparison_by_fold(oos, cfg)

    oos.to_parquet(outdir / "oos_predictions.parquet", index=False)
    tab100.to_csv(outdir / "calibrator_100_comparison.csv", index=False)
    tabdd.to_csv(outdir / "downside_calibrator_comparison.csv", index=False)
    for target in v94.THRESHOLDS:
        threshold_info[target]["table"].to_csv(
            outdir / f"{target}_calibrator_comparison.csv", index=False
        )
    chosen931.to_csv(outdir / "chosen_v931_config_by_fold.csv", index=False)
    chosen94.to_csv(outdir / "chosen_v94_ladder_by_fold.csv", index=False)
    chosen941.to_csv(outdir / "chosen_v941_gates_by_fold.csv", index=False)
    comp.to_csv(outdir / "selection_metrics_by_fold.csv", index=False)

    production_spec, risk_search, _ = v931.optimize_from_cache(metric_cache931, cfg)
    if production_spec is None:
        production_spec = next(v931.cfg_grid(cfg))

    shares, ladder_search, ladder_baseline, production_gates = optimize_ladder_gated(
        oos, production_spec, cfg
    )
    risk_search.to_csv(outdir / "risk_constraint_search.csv", index=False)
    ladder_search.to_csv(outdir / "gated_ladder_search.csv", index=False)

    current = base.fit_current(data, daily, oos, best100, bestdd, cfg)
    threshold_cur = v94.current_threshold_predictions(
        data, daily, oos, threshold_info, cfg
    )
    current = build_current_v941(
        current, threshold_cur, oos, threshold_info,
        production_spec, shares, production_gates, cfg
    )
    current.to_csv(outdir / "current_selection.csv", index=False)
    current[current["selected_v941"]].head(50).to_csv(
        outdir / "current_top50.csv", index=False
    )

    metrics25_coherent = v94.calibration_metrics(oos, "hit25_6m", "p25_cal")
    metrics50_coherent = v94.calibration_metrics(oos, "hit50_6m", "p50_cal")
    metrics100 = v94.calibration_metrics(oos, "y6", "p100_cal")

    summaries = {
        name: v94.summarize_comparison(comp, name)
        for name in ["V9.4.1", "V9.4", "V9.3.1", "V9.2_pcal"]
    }

    retention = {}
    for benchmark in ["V9.4", "V9.3.1"]:
        retention[benchmark] = {}
        for f in [
            "mean_precision_100",
            "mean_capped_lift_100",
            "hit_fold_rate_100",
            "median_precision_100",
        ]:
            a = summaries["V9.4.1"].get(f, np.nan)
            b = summaries[benchmark].get(f, np.nan)
            retention[benchmark][f] = (
                float(a / b)
                if np.isfinite(a) and np.isfinite(b) and b > 0
                else None
            )

    summary = {
        "model": cfg["model_name"],
        "pipeline_version": cfg["pipeline_version"],
        "data_start": str(daily["date"].min().date()),
        "data_end": str(daily["date"].max().date()),
        "signal_quality_policy": {
            "evaluation": "prior OOS folds only for every forward selection date",
            "probability_columns": "pre-coherence temporally calibrated P25/P50 probabilities",
            "min_pr_auc_lift": float(cfg["signal_quality_min_pr_auc_lift"]),
            "min_calibration_slope": float(cfg["signal_quality_min_calibration_slope"]),
            "min_rows": int(cfg["signal_quality_min_rows"]),
            "min_positives": int(cfg["signal_quality_min_positives"]),
            "min_folds": int(cfg["signal_quality_min_folds"]),
            "action_on_failure": "signal maximum ladder weight forced to zero",
        },
        "production_signal_quality": production_gates,
        "probability_ladder": {
            "definition": "sustained 3-session hit within 126 market sessions with liquidity confirmation",
            "p25_coherent": metrics25_coherent,
            "p50_coherent": metrics50_coherent,
            "p100": metrics100,
            "coherence_rule": "P25 >= P50 >= P100",
            "best_calibrator_p25": threshold_info["hit25_6m"]["best"],
            "best_calibrator_p50": threshold_info["hit50_6m"]["best"],
            "best_calibrator_p100": best100,
        },
        "production_risk_config": production_spec,
        "production_ladder_shares_within_upside_weight": {
            "p100": shares[0],
            "p50": shares[1],
            "p25": shares[2],
        },
        "ladder_alpha_retention_required": float(cfg.get("ladder_alpha_retention", 0.95)),
        "forward_history": summaries,
        "V9.4.1_alpha_retention_ratios": retention,
        "ladder_search_baseline": ladder_baseline,
        "fallback_rule": "failed quality gate -> zero signal weight; failed alpha/objective gate -> V9.3.1 P100-only upside ranking",
    }
    json.dump(summary, open(outdir / "summary.json", "w"), indent=2)

    print(json.dumps(summary, indent=2), flush=True)
    cols = [
        "selection_rank_v941", "symbol", "close",
        "p25_cal", "p50_cal", "p100_cal", "p_dd30",
        "p_conservative", "selection_score_v941",
    ]
    cols = [c for c in cols if c in current.columns]
    print(
        current[current["selected_v941"]].head(20)[cols].to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
