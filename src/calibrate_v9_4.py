from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss

import calibrate_v9_2 as base
import calibrate_v9_3_1 as v931


THRESHOLDS = {
    "hit25_6m": 1.25,
    "hit50_6m": 1.50,
}


def add_threshold_labels(data: pd.DataFrame, daily: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    out = data.copy()
    calendar = np.array(sorted(daily["date"].drop_duplicates().to_numpy(dtype="datetime64[ns]")))
    cal_lookup = {d: i for i, d in enumerate(calendar)}
    maps = {}
    for sym, g in daily.groupby("symbol", sort=False):
        g = g.sort_values("date")
        dates = g["date"].to_numpy(dtype="datetime64[ns]")
        maps[sym] = (
            dates,
            g["adj_close"].to_numpy(float),
            g["turnover"].to_numpy(float),
            np.array([cal_lookup[d] for d in dates], dtype=int),
        )

    h = int(cfg["label_days"]["y6"])
    min_turn = float(cfg["min_avg_turnover_63d"])
    recs = []

    for row in out.itertuples(index=False):
        rec = row._asdict()
        dates, price, turn, calpos = maps[row.symbol]
        d = np.datetime64(pd.Timestamp(row.date).to_datetime64())
        i = np.searchsorted(dates, d, side="left")
        ci = np.searchsorted(calendar, d, side="left")

        mature = (
            i < len(dates)
            and ci < len(calendar)
            and dates[i] == d
            and calendar[ci] == d
            and ci + h < len(calendar)
        )
        for target in THRESHOLDS:
            rec[target] = np.nan
        rec["threshold_mature_date"] = pd.Timestamp(calendar[ci + h]) if mature else pd.NaT

        if mature:
            end_ci = ci + h
            j = np.searchsorted(calpos, end_ci, side="right")
            start = i + 1
            future_p = price[start:j]
            future_t = turn[start:j]
            future_cp = calpos[start:j]
            p0 = float(row.adj_close)

            if len(future_p) >= 3 and np.isfinite(p0) and p0 > 0:
                consecutive = (
                    (future_cp[1:-1] == future_cp[:-2] + 1)
                    & (future_cp[2:] == future_cp[:-2] + 2)
                )
                med = np.array(
                    [np.nanmedian(future_p[k:k + 3]) for k in range(len(future_p) - 2)]
                )
                avt = np.array(
                    [np.nanmean(future_t[k:k + 3]) for k in range(len(future_t) - 2)]
                )
                for target, multiple in THRESHOLDS.items():
                    hit = (
                        consecutive
                        & np.isfinite(med)
                        & (med >= multiple * p0)
                        & np.isfinite(avt)
                        & (avt >= min_turn)
                    )
                    rec[target] = float(bool(np.any(hit)))
            else:
                for target in THRESHOLDS:
                    rec[target] = 0.0

        recs.append(rec)

    return pd.DataFrame(recs)


def structural_raw_target(train: pd.DataFrame, test: pd.DataFrame, cfg: dict, target: str):
    tr = train.dropna(subset=[target]).copy()
    if len(tr) < 100 or tr[target].nunique() < 2:
        return np.full(len(test), np.nan)

    score_train = np.zeros(len(tr))
    score_test = np.zeros(len(test))
    for c, w in cfg["structural_weights"].items():
        rc = f"{c}_rank"
        if rc not in tr.columns or rc not in test.columns:
            continue
        score_train += float(w) * (tr[rc].fillna(0.5).to_numpy() - 0.5)
        score_test += float(w) * (test[rc].fillna(0.5).to_numpy() - 0.5)

    lr = LogisticRegression(C=0.2, class_weight="balanced", max_iter=1000)
    lr.fit(score_train.reshape(-1, 1), tr[target].astype(int))
    return lr.predict_proba(score_test.reshape(-1, 1))[:, 1]


def walk_forward_thresholds(data: pd.DataFrame, oos_dates, cfg: dict) -> pd.DataFrame:
    rows = []
    seed = int(cfg["random_seed"])

    for td in sorted(pd.to_datetime(oos_dates)):
        test = data[data["date"] == td].copy()
        if test.empty:
            continue

        start = td - pd.DateOffset(years=int(cfg["rolling_train_years"]))
        train = data[(data["date"] < td) & (data["date"] >= start)].copy()
        if "threshold_mature_date" in train.columns:
            immature = pd.to_datetime(train["threshold_mature_date"]) > td
            for target in THRESHOLDS:
                train.loc[immature, target] = np.nan

        z = test[["date", "symbol"]].copy()
        for n, target in enumerate(THRESHOLDS):
            tr = train.dropna(subset=[target])
            if len(tr) < 800 or tr[target].sum() < 20 or tr[target].nunique() < 2:
                z[f"{target}_raw"] = np.nan
                z[f"{target}_dispersion"] = np.nan
                continue

            elastic, gbm = base.make_models(seed + 300 + n * 17)
            p_struct = structural_raw_target(train, test, cfg, target)
            p_el = base.safe_predict_model(elastic, train, test, target)
            p_gbm = base.safe_predict_model(gbm, train, test, target)
            arr = np.vstack([p_struct, p_el, p_gbm]).T
            z[f"{target}_raw"] = np.nanmean(arr, axis=1)
            z[f"{target}_dispersion"] = np.nanstd(arr, axis=1)

        rows.append(z)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def calibrate_thresholds(oos: pd.DataFrame):
    out = oos.copy()
    info = {}

    for target in THRESHOLDS:
        raw = f"{target}_raw"
        raw_cal = f"{target}_cal_raw"
        best, table = base.evaluate_calibrators(out, target, raw)
        out = base.apply_forward_calibration(out, best, target, raw, raw_cal)
        info[target] = {"best": best, "table": table}

    # Coherent probability ladder: P(+25%) >= P(+50%) >= P(+100%).
    p100 = out["p_cal"].to_numpy(float)
    p50r = out["hit50_6m_cal_raw"].to_numpy(float)
    p25r = out["hit25_6m_cal_raw"].to_numpy(float)

    p50 = np.where(np.isfinite(p50r) & np.isfinite(p100), np.maximum(p50r, p100), np.nan)
    p25 = np.where(np.isfinite(p25r) & np.isfinite(p50), np.maximum(p25r, p50), np.nan)

    out["p50_cal"] = np.clip(p50, base.EPS, 1 - base.EPS)
    out["p25_cal"] = np.clip(p25, base.EPS, 1 - base.EPS)
    out["p100_cal"] = out["p_cal"]
    return out, info


def calibration_metrics(df: pd.DataFrame, target: str, pcol: str):
    q = df.dropna(subset=[target, pcol]).copy()
    if q.empty:
        return {}
    y = q[target].astype(int).to_numpy()
    p = np.clip(q[pcol].to_numpy(float), base.EPS, 1 - base.EPS)
    slope, intercept = base.calibration_slope_intercept(y, p)
    return {
        "n": int(len(q)),
        "positives": int(y.sum()),
        "base_rate": float(y.mean()),
        "brier": float(brier_score_loss(y, p)),
        "logloss": float(log_loss(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
    }


def current_universe(daily: pd.DataFrame, cfg: dict):
    last = pd.Timestamp(daily["date"].max())
    cur = daily[daily["date"] == last].copy()
    cur = cur[cur["isin"].notna() & cur["isin"].astype(str).str.startswith("INE")].copy()
    cur = cur[(cur["close"] >= cfg["price_min"]) & (cur["close"] <= cfg["price_max"])]
    cur = cur[
        (cur["avg_turnover_63"] >= cfg["min_avg_turnover_63d"])
        & (cur["history_days"] >= cfg["min_history_days"])
    ].copy()
    cur["breadth_120"] = float(np.nanmean(cur["ret_120"].to_numpy() > 0))
    cur["market_median_ret120"] = float(np.nanmedian(cur["ret_120"]))
    for c in base.RANKABLE:
        cur[f"{c}_rank"] = cur[c].rank(pct=True, method="average")
    return cur, last


def current_threshold_predictions(data, daily, oos, threshold_info, cfg):
    cur, last = current_universe(daily, cfg)
    start = last - pd.DateOffset(years=int(cfg["rolling_train_years"]))
    train = data[(data["date"] >= start)].copy()
    seed = int(cfg["random_seed"])

    out = cur[["date", "symbol"]].copy()
    for n, target in enumerate(THRESHOLDS):
        elastic, gbm = base.make_models(seed + 300 + n * 17)
        p_struct = structural_raw_target(train, cur, cfg, target)
        p_el = base.safe_predict_model(elastic, train, cur, target)
        p_gbm = base.safe_predict_model(gbm, train, cur, target)
        arr = np.vstack([p_struct, p_el, p_gbm]).T
        raw = np.nanmean(arr, axis=1)
        disp = np.nanstd(arr, axis=1)

        best = threshold_info[target]["best"]
        cal = base.fit_final_calibrator(oos, best, target, f"{target}_raw")
        pred = raw if cal is None else cal.predict(raw)
        out[f"{target}_raw"] = raw
        out[f"{target}_dispersion"] = disp
        out[f"{target}_cal_raw"] = pred

    return out


def ladder_shares(cfg):
    step = float(cfg.get("ladder_share_step", 0.05))
    n = int(round(1.0 / step))
    min100 = float(cfg.get("ladder_min_p100_share", 0.65))
    max50 = float(cfg.get("ladder_max_p50_share", 0.25))
    max25 = float(cfg.get("ladder_max_p25_share", 0.15))
    seen = set()

    baseline = (1.0, 0.0, 0.0)
    seen.add(baseline)
    yield baseline

    for i100 in range(n + 1):
        s100 = i100 * step
        if s100 + 1e-12 < min100:
            continue
        for i50 in range(n + 1 - i100):
            s50 = i50 * step
            s25 = 1.0 - s100 - s50
            if s25 < -1e-12:
                continue
            s25 = max(0.0, s25)
            if s50 > max50 + 1e-12 or s25 > max25 + 1e-12:
                continue
            key = (round(s100, 6), round(s50, 6), round(s25, 6))
            if key not in seen:
                seen.add(key)
                yield key


def risk_survivors(g: pd.DataFrame, spec: dict, k: int):
    req = ["p_cal", "p_dd30_cal", "model_dispersion", "p25_cal", "p50_cal"]
    q = g.dropna(subset=req).copy()
    if len(q) < k:
        return pd.DataFrame()

    if spec.get("baseline", False):
        return q.sort_values(["p_cal", "model_dispersion"], ascending=[False, True]).head(k).copy()

    pool_n = min(int(spec["pool_n"]), len(q))
    pool = q.sort_values(["p_cal", "model_dispersion"], ascending=[False, True]).head(pool_n).copy()
    rd = float(spec["risk_drop"])
    if rd > 0:
        cutoff = float(pool["p_dd30_cal"].quantile(1.0 - rd))
        pool = pool[pool["p_dd30_cal"] <= cutoff].copy()

    if len(pool) < k:
        return pd.DataFrame()
    return pool


def select_ladder_topk(g: pd.DataFrame, spec: dict, shares, k: int):
    surv = risk_survivors(g, spec, k)
    if len(surv) < k:
        return pd.DataFrame()

    if spec.get("baseline", False):
        out = surv.sort_values(["p_cal", "model_dispersion"], ascending=[False, True]).head(k).copy()
        out["selection_score_v94"] = out["p_cal"]
        return out

    s100, s50, s25 = map(float, shares)
    ws = float(spec["w_safety"])
    wc = float(spec["w_consensus"])
    alpha_total = max(0.0, 1.0 - ws - wc)

    surv = surv.copy()
    surv["comp100"] = surv["p100_cal"].rank(pct=True, method="average")
    surv["comp50"] = surv["p50_cal"].rank(pct=True, method="average")
    surv["comp25"] = surv["p25_cal"].rank(pct=True, method="average")
    surv["comp_safety"] = surv["p_dd30_cal"].rank(pct=True, method="average", ascending=False)
    surv["comp_consensus"] = surv["model_dispersion"].rank(pct=True, method="average", ascending=False)

    surv["selection_score_v94"] = (
        alpha_total
        * (
            s100 * surv["comp100"]
            + s50 * surv["comp50"]
            + s25 * surv["comp25"]
        )
        + ws * surv["comp_safety"]
        + wc * surv["comp_consensus"]
    )
    return surv.sort_values(
        ["selection_score_v94", "p100_cal", "model_dispersion"],
        ascending=[False, False, True],
    ).head(k).copy()


def ladder_fold_metrics(g: pd.DataFrame, spec: dict, shares, cfg: dict):
    k = int(cfg.get("selection_k", 10))
    q = g.dropna(
        subset=[
            "y6", "hit50_6m", "hit25_6m", "dd30_6m",
            "p100_cal", "p50_cal", "p25_cal", "p_dd30_cal", "model_dispersion",
        ]
    ).copy()
    if len(q) < max(k, 20):
        return None

    sel = select_ladder_topk(q, spec, shares, k)
    if len(sel) < k:
        return None

    rec = {"n": int(len(q))}
    for label, suffix in [("y6", "100"), ("hit50_6m", "50"), ("hit25_6m", "25")]:
        base_rate = float(q[label].mean())
        precision = float(sel[label].mean())
        lift = precision / base_rate if base_rate > 0 else np.nan
        rec[f"precision_{suffix}"] = precision
        rec[f"lift_{suffix}"] = lift
        rec[f"base_rate_{suffix}"] = base_rate

    rec["hit100"] = float(rec["precision_100"] > 0)
    rec["dd30_rate"] = float(sel["dd30_6m"].mean())
    rec["mean_p100"] = float(sel["p100_cal"].mean())
    rec["mean_p50"] = float(sel["p50_cal"].mean())
    rec["mean_p25"] = float(sel["p25_cal"].mean())
    return rec


def ladder_metrics_by_fold(hist, spec, shares, cfg):
    rows = []
    for td, g in hist.groupby("date"):
        m = ladder_fold_metrics(g, spec, shares, cfg)
        if m is not None:
            m["date"] = pd.Timestamp(td)
            rows.append(m)
    return pd.DataFrame(rows)


def _capped_lift(s):
    x = s.replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
    return np.clip(x, 0, 10) if len(x) else np.array([])


def agg_ladder(t: pd.DataFrame):
    if t.empty:
        return None

    out = {
        "folds": int(t["date"].nunique()),
        "mean_precision_100": float(t["precision_100"].mean()),
        "median_precision_100": float(t["precision_100"].median()),
        "hit_fold_rate_100": float(t["hit100"].mean()),
        "hit_folds_100": int(t["hit100"].sum()),
        "mean_precision_50": float(t["precision_50"].mean()),
        "median_precision_50": float(t["precision_50"].median()),
        "mean_precision_25": float(t["precision_25"].mean()),
        "median_precision_25": float(t["precision_25"].median()),
        "mean_dd30_rate": float(t["dd30_rate"].mean()),
        "median_dd30_rate": float(t["dd30_rate"].median()),
    }

    for suffix in ["100", "50", "25"]:
        cap = _capped_lift(t[f"lift_{suffix}"])
        out[f"mean_lift_{suffix}"] = float(np.nanmean(t[f"lift_{suffix}"].replace([np.inf, -np.inf], np.nan)))
        out[f"mean_capped_lift_{suffix}"] = float(cap.mean()) if len(cap) else np.nan
        out[f"capped_lift_iqr_{suffix}"] = (
            float(np.quantile(cap, 0.75) - np.quantile(cap, 0.25)) if len(cap) else np.nan
        )
    return out


def alpha_gates_v94(cand, baseline, cfg):
    retain = float(cfg.get("ladder_alpha_retention", 0.95))
    fields = ["mean_precision_100", "mean_capped_lift_100", "hit_fold_rate_100", "median_precision_100"]
    gates = {}
    ok = True
    for f in fields:
        actual = cand.get(f, np.nan)
        required = retain * baseline.get(f, np.nan)
        gates[f] = {"actual": actual, "required": required}
        if not (np.isfinite(actual) and np.isfinite(required) and actual + 1e-12 >= required):
            ok = False
    return ok, gates


def ladder_objective(a, cfg):
    if a is None:
        return -np.inf
    obj = (
        float(cfg.get("ladder_obj_w50", 0.55)) * a["mean_capped_lift_50"]
        + float(cfg.get("ladder_obj_w25", 0.30)) * a["mean_capped_lift_25"]
        + float(cfg.get("ladder_obj_w100", 0.15)) * a["mean_capped_lift_100"]
        - float(cfg.get("ladder_iqr_penalty50", 0.10)) * a["capped_lift_iqr_50"]
        - float(cfg.get("ladder_iqr_penalty25", 0.05)) * a["capped_lift_iqr_25"]
    )
    return float(obj)


def optimize_ladder(hist: pd.DataFrame, risk_spec: dict, cfg: dict):
    min_folds = int(cfg.get("ladder_min_folds", 8))
    baseline_shares = (1.0, 0.0, 0.0)
    bt = ladder_metrics_by_fold(hist, risk_spec, baseline_shares, cfg)
    ba = agg_ladder(bt)
    if ba is None or ba["folds"] < min_folds:
        return baseline_shares, pd.DataFrame(), ba

    baseline_obj = ladder_objective(ba, cfg)
    min_gain = float(cfg.get("ladder_min_objective_gain", 0.02))
    best = baseline_shares
    best_obj = baseline_obj
    rows = []

    for shares in ladder_shares(cfg):
        t = ladder_metrics_by_fold(hist, risk_spec, shares, cfg)
        a = agg_ladder(t)
        if a is None or a["folds"] < min_folds:
            continue
        feasible, gates = alpha_gates_v94(a, ba, cfg)
        obj = ladder_objective(a, cfg)
        is_baseline = shares == baseline_shares
        if is_baseline:
            feasible = True

        rows.append({
            "share_p100": shares[0],
            "share_p50": shares[1],
            "share_p25": shares[2],
            "baseline": is_baseline,
            "feasible": bool(feasible),
            "objective": obj,
            **a,
            "required_mean_precision_100": gates.get("mean_precision_100", {}).get("required", np.nan),
            "required_mean_capped_lift_100": gates.get("mean_capped_lift_100", {}).get("required", np.nan),
            "required_hit_fold_rate_100": gates.get("hit_fold_rate_100", {}).get("required", np.nan),
            "required_median_precision_100": gates.get("median_precision_100", {}).get("required", np.nan),
        })

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
    return best, table, ba


def forward_v94(oos: pd.DataFrame, chosen931: pd.DataFrame, cfg: dict):
    out = oos.copy()
    out["selected_v94"] = False
    out["selection_score_v94"] = np.nan
    out["selection_rank_v94"] = np.nan
    out["v94_share_p100"] = np.nan
    out["v94_share_p50"] = np.nan
    out["v94_share_p25"] = np.nan

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
        shares, search, baseline = optimize_ladder(prior, spec, cfg)

        cur = out[out["date"] == td].copy()
        sel = select_ladder_topk(cur, spec, shares, int(cfg.get("selection_k", 10)))
        if sel.empty:
            shares = (1.0, 0.0, 0.0)
            sel = select_ladder_topk(cur, spec, shares, int(cfg.get("selection_k", 10)))
        if sel.empty:
            continue

        for rank, idx in enumerate(sel.index.tolist(), start=1):
            out.loc[idx, "selected_v94"] = True
            out.loc[idx, "selection_score_v94"] = float(sel.loc[idx, "selection_score_v94"])
            out.loc[idx, "selection_rank_v94"] = rank

        idx_date = out.index[out["date"] == td]
        out.loc[idx_date, "v94_share_p100"] = shares[0]
        out.loc[idx_date, "v94_share_p50"] = shares[1]
        out.loc[idx_date, "v94_share_p25"] = shares[2]

        row = {
            "date": td,
            "risk_config": spec["name"],
            "share_p100": shares[0],
            "share_p50": shares[1],
            "share_p25": shares[2],
            "ladder_changed_ranking": bool(shares != (1.0, 0.0, 0.0)),
            "prior_folds": int(prior["date"].nunique()),
        }
        if not search.empty:
            hit = search[
                (search["share_p100"] == shares[0])
                & (search["share_p50"] == shares[1])
                & (search["share_p25"] == shares[2])
            ]
            row["train_objective"] = float(hit.iloc[0]["objective"]) if len(hit) else np.nan
        chosen_rows.append(row)

    return out, pd.DataFrame(chosen_rows)


def comparison_by_fold(oos: pd.DataFrame, cfg: dict):
    rows = []
    k = int(cfg.get("selection_k", 10))
    dates = sorted(pd.to_datetime(oos.loc[oos["selected_v94"], "date"].unique()))

    for td in dates:
        g = oos[oos["date"] == td].copy()
        q = g.dropna(
            subset=["y6", "hit50_6m", "hit25_6m", "dd30_6m", "p_cal", "p50_cal", "p25_cal"]
        ).copy()
        if len(q) < max(k, 20):
            continue

        selections = {
            "V9.4": g[g["selected_v94"]].copy(),
            "V9.3.1": g[g["selected_v931"]].copy(),
            "V9.2_pcal": q.sort_values(["p_cal", "model_dispersion"], ascending=[False, True]).head(k).copy(),
        }

        for name, sel in selections.items():
            if len(sel) != k:
                continue
            rec = {"date": td, "strategy": name, "k": k}
            for label, suffix in [("y6", "100"), ("hit50_6m", "50"), ("hit25_6m", "25")]:
                br = float(q[label].mean())
                pr = float(sel[label].mean())
                rec[f"precision_{suffix}"] = pr
                rec[f"lift_{suffix}"] = pr / br if br > 0 else np.nan
            rec["hit100"] = float(rec["precision_100"] > 0)
            rec["dd30_rate"] = float(sel["dd30_6m"].mean())
            rows.append(rec)

    return pd.DataFrame(rows)


def summarize_comparison(comp, name):
    d = comp[comp["strategy"] == name].copy()
    if d.empty:
        return {}

    out = {
        "folds": int(d["date"].nunique()),
        "mean_precision_100": float(d["precision_100"].mean()),
        "median_precision_100": float(d["precision_100"].median()),
        "hit_fold_rate_100": float(d["hit100"].mean()),
        "hit_folds_100": int(d["hit100"].sum()),
        "mean_precision_50": float(d["precision_50"].mean()),
        "median_precision_50": float(d["precision_50"].median()),
        "mean_precision_25": float(d["precision_25"].mean()),
        "median_precision_25": float(d["precision_25"].median()),
        "mean_dd30_rate": float(d["dd30_rate"].mean()),
        "median_dd30_rate": float(d["dd30_rate"].median()),
    }
    for suffix in ["100", "50", "25"]:
        cap = _capped_lift(d[f"lift_{suffix}"])
        out[f"mean_capped_lift_{suffix}"] = float(cap.mean()) if len(cap) else np.nan
        out[f"median_lift_{suffix}"] = float(
            d[f"lift_{suffix}"].replace([np.inf, -np.inf], np.nan).median()
        )
    return out


def build_current_v94(current, threshold_cur, oos, threshold_info, production_spec, shares, cfg):
    cur = current.merge(threshold_cur, on=["date", "symbol"], how="left").copy()
    cur["p_dd30_cal"] = cur["p_dd30"]
    cur["p100_cal"] = cur["p_cal"]

    p50r = cur["hit50_6m_cal_raw"].to_numpy(float)
    p25r = cur["hit25_6m_cal_raw"].to_numpy(float)
    p100 = cur["p100_cal"].to_numpy(float)
    p50 = np.where(np.isfinite(p50r), np.maximum(p50r, p100), np.nan)
    p25 = np.where(np.isfinite(p25r) & np.isfinite(p50), np.maximum(p25r, p50), np.nan)
    cur["p50_cal"] = np.clip(p50, base.EPS, 1 - base.EPS)
    cur["p25_cal"] = np.clip(p25, base.EPS, 1 - base.EPS)

    sel = select_ladder_topk(cur, production_spec, shares, int(cfg.get("selection_k", 10)))
    cur["selected_v94"] = False
    cur["selection_score_v94"] = np.nan
    cur["selection_rank_v94"] = np.nan
    for rank, idx in enumerate(sel.index.tolist(), start=1):
        cur.loc[idx, "selected_v94"] = True
        cur.loc[idx, "selection_score_v94"] = float(sel.loc[idx, "selection_score_v94"])
        cur.loc[idx, "selection_rank_v94"] = rank

    cur["production_risk_config"] = production_spec["name"]
    cur["production_share_p100"] = shares[0]
    cur["production_share_p50"] = shares[1]
    cur["production_share_p25"] = shares[2]
    return cur.sort_values(
        ["selected_v94", "selection_rank_v94", "p100_cal"],
        ascending=[False, True, False],
        na_position="last",
    ).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_v9_4.json")
    ap.add_argument("--output", default="outputs_v9_4")
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
    data = add_threshold_labels(data, daily, cfg)
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
    oos = base.apply_forward_calibration(oos, bestdd, "dd30_6m", "p_dd30_raw", "p_dd30_cal")
    oos["p100_cal"] = oos["p_cal"]

    print("Training +25%/+50% probability models...", flush=True)
    thresh = walk_forward_thresholds(data, oos["date"].unique(), cfg)
    oos = oos.merge(thresh, on=["date", "symbol"], how="left")
    oos, threshold_info = calibrate_thresholds(oos)

    print("Running V9.3.1 constrained selector...", flush=True)
    oos, chosen931 = v931.forward_select(oos, cfg)

    print("Running V9.4 upside-ladder selector...", flush=True)
    oos, chosen94 = forward_v94(oos, chosen931, cfg)
    comp = comparison_by_fold(oos, cfg)

    oos.to_parquet(outdir / "oos_predictions.parquet", index=False)
    tab100.to_csv(outdir / "calibrator_100_comparison.csv", index=False)
    tabdd.to_csv(outdir / "downside_calibrator_comparison.csv", index=False)
    for target in THRESHOLDS:
        threshold_info[target]["table"].to_csv(
            outdir / f"{target}_calibrator_comparison.csv", index=False
        )
    chosen931.to_csv(outdir / "chosen_v931_config_by_fold.csv", index=False)
    chosen94.to_csv(outdir / "chosen_v94_ladder_by_fold.csv", index=False)
    comp.to_csv(outdir / "selection_metrics_by_fold.csv", index=False)

    production_spec, risk_search, _ = v931.optimize_config(oos, cfg)
    if production_spec is None:
        production_spec = next(v931.cfg_grid(cfg))
    shares, ladder_search, ladder_baseline = optimize_ladder(oos, production_spec, cfg)
    risk_search.to_csv(outdir / "risk_constraint_search.csv", index=False)
    ladder_search.to_csv(outdir / "ladder_search.csv", index=False)

    current = base.fit_current(data, daily, oos, best100, bestdd, cfg)
    threshold_cur = current_threshold_predictions(data, daily, oos, threshold_info, cfg)
    current = build_current_v94(
        current, threshold_cur, oos, threshold_info, production_spec, shares, cfg
    )
    current.to_csv(outdir / "current_selection.csv", index=False)
    current[current["selected_v94"]].head(50).to_csv(outdir / "current_top50.csv", index=False)

    metrics25 = calibration_metrics(oos, "hit25_6m", "p25_cal")
    metrics50 = calibration_metrics(oos, "hit50_6m", "p50_cal")
    metrics100 = calibration_metrics(oos, "y6", "p100_cal")

    s94 = summarize_comparison(comp, "V9.4")
    s931 = summarize_comparison(comp, "V9.3.1")
    s92 = summarize_comparison(comp, "V9.2_pcal")

    retention_vs_931 = {}
    for f in ["mean_precision_100", "mean_capped_lift_100", "hit_fold_rate_100", "median_precision_100"]:
        a = s94.get(f, np.nan)
        b = s931.get(f, np.nan)
        retention_vs_931[f] = float(a / b) if np.isfinite(a) and np.isfinite(b) and b > 0 else None

    summary = {
        "model": cfg["model_name"],
        "pipeline_version": cfg["pipeline_version"],
        "data_start": str(daily["date"].min().date()),
        "data_end": str(daily["date"].max().date()),
        "probability_ladder": {
            "definition": "sustained 3-session hit within 126 market sessions with liquidity confirmation",
            "p25": metrics25,
            "p50": metrics50,
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
        "ladder_alpha_retention_required_vs_v931": float(cfg.get("ladder_alpha_retention", 0.95)),
        "forward_history": {
            "V9.4": s94,
            "V9.3.1": s931,
            "V9.2_pcal": s92,
            "V9.4_alpha_retention_vs_V9.3.1": retention_vs_931,
        },
        "ladder_search_baseline": ladder_baseline,
        "fallback_rule": "if intermediate-upside ladder cannot improve its prior-fold objective while preserving 2x alpha gates, keep V9.3.1 p100-only upside ranking",
    }
    json.dump(summary, open(outdir / "summary.json", "w"), indent=2)

    print(json.dumps(summary, indent=2), flush=True)
    cols = [
        "selection_rank_v94", "symbol", "close",
        "p25_cal", "p50_cal", "p100_cal", "p_dd30",
        "p_conservative", "selection_score_v94",
    ]
    cols = [c for c in cols if c in current.columns]
    print(current[current["selected_v94"]].head(20)[cols].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
