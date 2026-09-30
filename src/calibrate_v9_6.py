from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import calibrate_v9_3_1 as v931


REGIME_FEATURES = [
    "breadth_120",
    "market_median_ret120",
    "median_volatility_60",
    "median_drawdown_126",
    "dispersion_ret120",
    "breadth_20",
]


def build_regime_table(snapshot: pd.DataFrame) -> pd.DataFrame:
    x = snapshot.copy()
    x["date"] = pd.to_datetime(x["date"])
    rows = []
    for td, g in x.groupby("date", sort=True):
        rows.append({
            "date": pd.Timestamp(td),
            "breadth_120": float(np.nanmean(g["ret_120"].to_numpy(float) > 0)),
            "market_median_ret120": float(np.nanmedian(g["ret_120"].to_numpy(float))),
            "median_volatility_60": float(np.nanmedian(g["volatility_60"].to_numpy(float))),
            "median_drawdown_126": float(np.nanmedian(g["drawdown_126"].to_numpy(float))),
            "dispersion_ret120": float(np.nanstd(g["ret_120"].to_numpy(float))),
            "breadth_20": float(np.nanmean(g["ret_20"].to_numpy(float) > 0)),
        })
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def regime_weights(prior_regimes: pd.DataFrame, current: pd.Series, cfg: dict) -> pd.DataFrame:
    z = prior_regimes.dropna(subset=REGIME_FEATURES).copy()
    if z.empty:
        return pd.DataFrame(columns=["date", "regime_weight", "regime_distance"])

    vals = z[REGIME_FEATURES].to_numpy(float)
    cur = current[REGIME_FEATURES].to_numpy(float)

    med = np.nanmedian(vals, axis=0)
    q25 = np.nanquantile(vals, 0.25, axis=0)
    q75 = np.nanquantile(vals, 0.75, axis=0)
    scale = q75 - q25
    std = np.nanstd(vals, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, std)
    scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, 1.0)

    zv = (vals - med) / scale
    zc = (cur - med) / scale
    dist = np.sqrt(np.nanmean((zv - zc) ** 2, axis=1))

    bandwidth = float(cfg.get("regime_bandwidth", 1.25))
    floor = float(cfg.get("regime_weight_floor", 0.03))
    w = np.exp(-0.5 * (dist / max(bandwidth, 1e-6)) ** 2)
    w = np.maximum(w, floor)

    out = z[["date"]].copy()
    out["regime_weight"] = w
    out["regime_distance"] = dist
    return out


def effective_n(w: np.ndarray) -> float:
    w = np.asarray(w, float)
    if len(w) == 0 or np.sum(w) <= 0:
        return 0.0
    return float(np.sum(w) ** 2 / np.sum(w ** 2))


def selected_fold_metrics(g: pd.DataFrame, selected: pd.DataFrame, k: int, lift_cap: float):
    q = g.dropna(
        subset=["y6", "dd30_6m", "p_cal", "p_dd30_cal", "model_dispersion"]
    ).copy()
    s = selected.dropna(subset=["y6", "dd30_6m"]).copy()
    if len(q) < max(k, 20) or len(s) != k:
        return None
    base = float(q["y6"].mean())
    precision = float(s["y6"].mean())
    lift = precision / base if base > 0 else np.nan
    return {
        "precision_2x": precision,
        "lift_2x": lift,
        "capped_lift_2x": min(float(lift), lift_cap) if np.isfinite(lift) else np.nan,
        "hit": float(precision > 0),
        "dd30_rate": float(s["dd30_6m"].mean()),
        "base_rate_2x": base,
    }


def metrics_for_spec(oos: pd.DataFrame, spec: dict, dates, cfg: dict) -> pd.DataFrame:
    rows = []
    k = int(cfg.get("selection_k", 10))
    cap = float(cfg.get("regime_lift_cap", 10.0))
    for td in dates:
        g = oos[oos["date"] == td].copy()
        sel = v931.select_topk(g, spec, k)
        m = selected_fold_metrics(g, sel, k, cap)
        if m is not None:
            rows.append({"date": pd.Timestamp(td), **m})
    return pd.DataFrame(rows)


def metrics_incumbent(oos: pd.DataFrame, dates, cfg: dict) -> pd.DataFrame:
    rows = []
    k = int(cfg.get("selection_k", 10))
    cap = float(cfg.get("regime_lift_cap", 10.0))
    for td in dates:
        g = oos[oos["date"] == td].copy()
        sel = g[g["selected_v941"]].copy()
        m = selected_fold_metrics(g, sel, k, cap)
        if m is not None:
            rows.append({"date": pd.Timestamp(td), **m})
    return pd.DataFrame(rows)


def weighted_aggregate(metrics: pd.DataFrame, weights: pd.DataFrame, cfg: dict):
    if metrics.empty or weights.empty:
        return None
    m = metrics.merge(weights[["date", "regime_weight"]], on="date", how="inner")
    m = m.dropna(subset=["regime_weight", "precision_2x", "dd30_rate"])
    if m.empty:
        return None

    w = m["regime_weight"].to_numpy(float)
    eff = effective_n(w)
    min_eff = float(cfg.get("regime_min_effective_folds", 6.0))
    if eff < min_eff:
        return None

    def wav(col):
        q = m[col].to_numpy(float)
        ok = np.isfinite(q) & np.isfinite(w)
        if not ok.any() or np.sum(w[ok]) <= 0:
            return np.nan
        return float(np.average(q[ok], weights=w[ok]))

    out = {
        "folds": int(m["date"].nunique()),
        "effective_folds": eff,
        "weighted_precision_2x": wav("precision_2x"),
        "weighted_capped_lift_2x": wav("capped_lift_2x"),
        "weighted_hit_rate": wav("hit"),
        "weighted_dd30_rate": wav("dd30_rate"),
    }
    out["utility"] = (
        out["weighted_precision_2x"]
        + float(cfg.get("regime_utility_lift_weight", 0.02)) * out["weighted_capped_lift_2x"]
        + float(cfg.get("regime_utility_hit_weight", 0.05)) * out["weighted_hit_rate"]
        - float(cfg.get("regime_utility_dd_weight", 0.10)) * out["weighted_dd30_rate"]
    )
    return out


def alpha_retention_ok(candidate: dict, incumbent: dict, cfg: dict) -> bool:
    retain = float(cfg.get("regime_alpha_retention", 0.95))
    if candidate is None or incumbent is None:
        return False
    for c in [
        "weighted_precision_2x",
        "weighted_capped_lift_2x",
        "weighted_hit_rate",
    ]:
        a = candidate.get(c, np.nan)
        b = incumbent.get(c, np.nan)
        if not (np.isfinite(a) and np.isfinite(b)):
            return False
        if b > 0 and a + 1e-12 < retain * b:
            return False
    return True


def optimize_regime_config(
    oos: pd.DataFrame,
    prior_dates,
    weights: pd.DataFrame,
    cfg: dict,
):
    incumbent_fold = metrics_incumbent(oos, prior_dates, cfg)
    incumbent = weighted_aggregate(incumbent_fold, weights, cfg)
    if incumbent is None:
        return None, pd.DataFrame(), None

    min_gain = float(cfg.get("regime_min_utility_gain", 0.005))
    rows = []
    best = None
    best_utility = incumbent["utility"]

    for spec in v931.cfg_grid(cfg):
        fm = metrics_for_spec(oos, spec, prior_dates, cfg)
        a = weighted_aggregate(fm, weights, cfg)
        if a is None:
            continue
        retain = alpha_retention_ok(a, incumbent, cfg)
        gain = float(a["utility"] - incumbent["utility"])
        feasible = bool(retain and gain >= min_gain)

        rows.append({
            **spec,
            **a,
            "alpha_retention_passed": bool(retain),
            "utility_gain_vs_v941": gain,
            "feasible": feasible,
            "incumbent_utility": incumbent["utility"],
            "incumbent_precision_2x": incumbent["weighted_precision_2x"],
            "incumbent_capped_lift_2x": incumbent["weighted_capped_lift_2x"],
            "incumbent_hit_rate": incumbent["weighted_hit_rate"],
            "incumbent_dd30_rate": incumbent["weighted_dd30_rate"],
        })
        if feasible and a["utility"] > best_utility:
            best = spec.copy()
            best_utility = a["utility"]

    table = pd.DataFrame(rows)
    if not table.empty:
        table = table.sort_values(
            ["feasible", "utility_gain_vs_v941", "weighted_precision_2x"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
    return best, table, incumbent


def forward_regime(oos: pd.DataFrame, regimes: pd.DataFrame, cfg: dict):
    out = oos.copy()
    out["selected_v96"] = False
    out["selection_rank_v96"] = np.nan
    out["selection_score_v96"] = np.nan
    out["v96_regime_applied"] = False
    out["v96_config"] = "V9.4.1_fallback"
    out["v96_effective_prior_folds"] = np.nan
    out["v96_train_utility_gain"] = np.nan

    rmap = {pd.Timestamp(r.date): r for r in regimes.itertuples(index=False)}
    dates = sorted(pd.to_datetime(out.loc[out["selected_v941"], "date"].unique()))
    min_prior = int(cfg.get("regime_min_prior_folds", 8))
    chosen = []
    search_parts = []

    for td in dates:
        if td not in rmap:
            continue
        prior_dates = [d for d in dates if d < td and d in rmap]
        current_rows = out[out["date"] == td].copy()

        use_spec = None
        search = pd.DataFrame()
        incumbent = None
        w = pd.DataFrame()

        if len(prior_dates) >= min_prior:
            pr = regimes[regimes["date"].isin(prior_dates)].copy()
            current = regimes[regimes["date"] == td].iloc[0]
            w = regime_weights(pr, current, cfg)
            use_spec, search, incumbent = optimize_regime_config(
                out, prior_dates, w, cfg
            )

        if use_spec is None:
            sel = current_rows[current_rows["selected_v941"]].copy()
            applied = False
            config_name = "V9.4.1_fallback"
            gain = 0.0
        else:
            sel = v931.select_topk(
                current_rows, use_spec, int(cfg.get("selection_k", 10))
            )
            if len(sel) != int(cfg.get("selection_k", 10)):
                sel = current_rows[current_rows["selected_v941"]].copy()
                applied = False
                config_name = "V9.4.1_fallback"
                gain = 0.0
            else:
                applied = True
                config_name = use_spec["name"]
                hit = search[search["name"] == config_name]
                gain = float(hit.iloc[0]["utility_gain_vs_v941"]) if len(hit) else np.nan

        if len(sel) != int(cfg.get("selection_k", 10)):
            continue

        for rank, idx in enumerate(sel.index.tolist(), start=1):
            out.loc[idx, "selected_v96"] = True
            out.loc[idx, "selection_rank_v96"] = rank
            if "selection_score" in sel.columns:
                out.loc[idx, "selection_score_v96"] = float(sel.loc[idx, "selection_score"])
            elif "selection_score_v941" in sel.columns:
                out.loc[idx, "selection_score_v96"] = float(sel.loc[idx, "selection_score_v941"])
            else:
                out.loc[idx, "selection_score_v96"] = float(sel.loc[idx, "p_cal"])

        idx_date = out.index[out["date"] == td]
        out.loc[idx_date, "v96_regime_applied"] = applied
        out.loc[idx_date, "v96_config"] = config_name
        out.loc[idx_date, "v96_train_utility_gain"] = gain
        eff = effective_n(w["regime_weight"].to_numpy(float)) if len(w) else 0.0
        out.loc[idx_date, "v96_effective_prior_folds"] = eff

        chosen.append({
            "date": td,
            "regime_applied": applied,
            "config": config_name,
            "prior_folds": len(prior_dates),
            "effective_prior_folds": eff,
            "train_utility_gain": gain,
            **{f"regime_{c}": float(getattr(rmap[td], c)) for c in REGIME_FEATURES},
        })
        if not search.empty:
            s = search.copy()
            s["decision_date"] = td
            search_parts.append(s)

    search_all = pd.concat(search_parts, ignore_index=True) if search_parts else pd.DataFrame()
    return out, pd.DataFrame(chosen), search_all


def comparison(oos: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rows = []
    k = int(cfg.get("selection_k", 10))
    cap = float(cfg.get("regime_lift_cap", 10.0))
    dates = sorted(pd.to_datetime(oos.loc[oos["selected_v96"], "date"].unique()))

    for td in dates:
        g = oos[oos["date"] == td].copy()
        for name, mask in [
            ("V9.6", g["selected_v96"]),
            ("V9.4.1", g["selected_v941"]),
        ]:
            sel = g[mask].copy()
            m = selected_fold_metrics(g, sel, k, cap)
            if m is not None:
                rows.append({
                    "date": td,
                    "strategy": name,
                    "regime_applied": bool(g["v96_regime_applied"].any()) if name == "V9.6" else False,
                    **m,
                })
    return pd.DataFrame(rows)


def summarize(comp: pd.DataFrame, name: str, cfg: dict):
    d = comp[comp["strategy"] == name].copy()
    if d.empty:
        return {}
    lift = d["capped_lift_2x"].dropna()
    out = {
        "folds": int(d["date"].nunique()),
        "mean_precision_2x": float(d["precision_2x"].mean()),
        "median_precision_2x": float(d["precision_2x"].median()),
        "mean_capped_lift_2x": float(lift.mean()) if len(lift) else np.nan,
        "median_capped_lift_2x": float(lift.median()) if len(lift) else np.nan,
        "hit_fold_rate": float(d["hit"].mean()),
        "hit_folds": int(d["hit"].sum()),
        "mean_dd30_rate": float(d["dd30_rate"].mean()),
        "median_dd30_rate": float(d["dd30_rate"].median()),
    }
    out["utility"] = (
        out["mean_precision_2x"]
        + float(cfg.get("regime_utility_lift_weight", 0.02)) * out["mean_capped_lift_2x"]
        + float(cfg.get("regime_utility_hit_weight", 0.05)) * out["hit_fold_rate"]
        - float(cfg.get("regime_utility_dd_weight", 0.10)) * out["mean_dd30_rate"]
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oos", required=True)
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--config", default="config_v9_6.json")
    ap.add_argument("--output", default="outputs_v9_6")
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    oos = pd.read_parquet(args.oos)
    snap = pd.read_parquet(args.snapshot)
    oos["date"] = pd.to_datetime(oos["date"])

    required = {
        "date", "y6", "dd30_6m", "p_cal", "p_dd30_cal",
        "model_dispersion", "selected_v941",
    }
    missing = sorted(required - set(oos.columns))
    if missing:
        raise ValueError(f"V9.4.1 artifact missing required columns: {missing}")

    regimes = build_regime_table(snap)
    regimes.to_csv(outdir / "regime_by_fold.csv", index=False)

    out, chosen, search = forward_regime(oos, regimes, cfg)
    comp = comparison(out, cfg)

    out.to_parquet(outdir / "oos_predictions_v96.parquet", index=False)
    chosen.to_csv(outdir / "chosen_regime_config_by_fold.csv", index=False)
    search.to_csv(outdir / "regime_config_search.csv", index=False)
    comp.to_csv(outdir / "selection_metrics_by_fold.csv", index=False)

    s96 = summarize(comp, "V9.6", cfg)
    s941 = summarize(comp, "V9.4.1", cfg)

    retention = {}
    for key in ["mean_precision_2x", "mean_capped_lift_2x", "hit_fold_rate"]:
        a = s96.get(key, np.nan)
        b = s941.get(key, np.nan)
        retention[key] = float(a / b) if np.isfinite(a) and np.isfinite(b) and b > 0 else None

    active = int(chosen["regime_applied"].sum()) if len(chosen) else 0
    retain_req = float(cfg.get("regime_alpha_retention", 0.95))
    retention_ok = all(
        v is None or v + 1e-12 >= retain_req
        for v in retention.values()
    )
    utility_ok = (
        np.isfinite(s96.get("utility", np.nan))
        and np.isfinite(s941.get("utility", np.nan))
        and s96["utility"] >= s941["utility"]
    )
    dd_ok = (
        np.isfinite(s96.get("mean_dd30_rate", np.nan))
        and np.isfinite(s941.get("mean_dd30_rate", np.nan))
        and s96["mean_dd30_rate"] <= s941["mean_dd30_rate"] + float(cfg.get("regime_max_dd_increase", 0.03))
    )

    passed = bool(
        active >= int(cfg.get("regime_min_active_folds", 4))
        and retention_ok
        and utility_ok
        and dd_ok
    )

    summary = {
        "model": "V9.6-H6-Regime-Aware",
        "pipeline_version": cfg.get("pipeline_version"),
        "base_model": "V9.4.1-H6-Signal-Gated-Ladder",
        "method": "similar-regime weighted prior-fold selector optimization",
        "regime_features": REGIME_FEATURES,
        "leakage_policy": "current fold regime compared only with regimes and realized selection outcomes from strictly prior folds",
        "alpha_retention_required": retain_req,
        "active_regime_folds": active,
        "production_challenger_passed": passed,
        "V9.6": s96,
        "V9.4.1": s941,
        "alpha_retention_ratios": retention,
        "fallback": "exact V9.4.1 selection whenever regime history/configuration does not pass prior-fold gates",
        "production_status": "historical challenger only; live deployment requires this gate to pass first",
    }
    json.dump(summary, open(outdir / "summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
