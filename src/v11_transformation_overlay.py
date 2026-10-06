from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

import calibrate_v9_2 as base
import v10_2_market_integrity as market_integrity


EVENT_FEATURES = [
    "evt_capacity_180",
    "evt_commissioning_180",
    "evt_order_180",
    "evt_new_product_180",
    "evt_future_product_180",
    "evt_customer_180",
    "evt_debt_reduction_180",
    "evt_credit_positive_180",
    "evt_earnings_positive_180",
    "evt_risk_180",
    "evt_catalyst_breadth_180",
    "evt_catalyst_recent_90",
]
OWNERSHIP_FEATURES = [
    "promoter_buy_net_180",
    "promoter_buy_count_180",
    "promoter_pct",
    "promoter_delta_qoq",
]

FUTURE_RE = re.compile(
    r"\b(electric vehicle|\bev\b|battery|energy storage|semiconductor|chip|solar|renewable|green hydrogen|hydrogen|data cent(?:er|re)|defen[cs]e|aerospace|drone|robot(?:ics)?|automation|artificial intelligence|machine learning|power electronics|electronics manufacturing|ems)\b",
    re.I,
)
CAPACITY_RE = re.compile(
    r"\b(capacity expansion|expand(?:ing|ed)? capacity|capacity addition|new plant|greenfield|brownfield|capex|capital expenditure|setting up (?:a )?(?:new )?(?:plant|facility|unit)|new manufacturing (?:plant|facility|unit))\b",
    re.I,
)
COMMISSION_RE = re.compile(
    r"\b(commission(?:ed|ing)|commercial production|commencement of (?:commercial )?production|commenced (?:commercial )?(?:production|operations)|plant (?:is )?operational|facility (?:is )?operational|production has started|operations have started|trial production)\b",
    re.I,
)
ORDER_RE = re.compile(
    r"\b(received (?:an? )?order|order win|order worth|purchase order|work order|letter of award|\bloa\b|contract awarded|awarded (?:a )?contract)\b",
    re.I,
)
PRODUCT_RE = re.compile(
    r"\b(new product|product launch|launch(?:ed|es|ing)?|introduc(?:e|ed|es|ing) (?:a )?new|commerciali[sz](?:e|ed|ation)|new solution|new platform|new technology)\b",
    re.I,
)
CUSTOMER_RE = re.compile(
    r"\b(new customer|customer win|strategic customer|long[- ]term agreement|supply agreement|vendor approval|approved vendor|customer qualification|export order|export approval)\b",
    re.I,
)
DEBT_RE = re.compile(r"\b(debt reduction|reduce(?:d|s|ing)? debt|deleverag|debt free|repay(?:ment|ing)? debt)\b", re.I)
RISK_RE = re.compile(r"\b(default|insolvency|cirp|fraud|forensic audit|auditor resignation|litigation|penalty|show cause|pledge invocation)\b", re.I)


def norm(v):
    if pd.isna(v):
        return None
    s = str(v).strip().upper()
    return None if s in {"", "NAN", "NONE", "<NA>", "NULL"} else s


def security_key(isin, symbol):
    i = norm(isin)
    return "I:" + i if i else "S:" + (norm(symbol) or "")


def availability_date(s):
    z = pd.to_datetime(s, utc=True, errors="coerce")
    try:
        return z.dt.tz_convert("Asia/Kolkata").dt.tz_localize(None).dt.normalize()
    except Exception:
        return pd.to_datetime(s, dayfirst=True, errors="coerce").dt.normalize()


def _num(s):
    return pd.to_numeric(s.astype(str).str.replace(",", "", regex=False), errors="coerce")


def add_accumulation_features(daily: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for _, g in daily.groupby("symbol", sort=False):
        g = g.sort_values("date").copy()
        p = pd.to_numeric(g["adj_close"], errors="coerce")
        v = pd.to_numeric(g["volume"], errors="coerce").fillna(0.0)
        r1 = p.pct_change()
        v20 = v.rolling(20, min_periods=12).mean()
        v60 = v.rolling(60, min_periods=30).mean()
        med60 = v.rolling(60, min_periods=30).median()
        g["rvol_20_60"] = v20 / v60.replace(0, np.nan) - 1.0
        upv = v.where(r1 > 0, 0.0).rolling(20, min_periods=12).sum()
        allv = v.rolling(20, min_periods=12).sum()
        g["up_volume_share_20"] = upv / allv.replace(0, np.nan)
        active = ((r1 > 0) & (v > med60)).astype(float)
        g["accumulation_days_20"] = active.rolling(20, min_periods=12).mean()
        modest = 1.0 - np.clip(np.abs(pd.to_numeric(g.get("ret_20"), errors="coerce")) / 0.50, 0, 1)
        g["accumulation_absorption"] = (
            np.clip(g["rvol_20_60"], 0, 3)
            * np.clip(g["up_volume_share_20"] - 0.45, 0, 0.55)
            * modest.fillna(0.0)
        )
        pieces.append(g)
    return pd.concat(pieces, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)


def prepare_events(path: str | Path) -> pd.DataFrame:
    cols = [
        "published_ts", "symbol", "isin", "headline", "details", "event_type",
        "event_direction", "event_strength",
    ]
    e = pd.read_parquet(path)
    for c in cols:
        if c not in e:
            e[c] = pd.NA
    e = e[cols].copy()
    e["avail_date"] = availability_date(e["published_ts"])
    e["key"] = [security_key(i, s) for i, s in zip(e["isin"], e["symbol"])]
    typ = e["event_type"].fillna("other").astype(str).str.lower()
    strength = pd.to_numeric(e["event_strength"], errors="coerce").fillna(0.5).clip(0, 1)
    direction = pd.to_numeric(e["event_direction"], errors="coerce").fillna(0.0).clip(-1, 1)
    txt = (e["headline"].fillna("").astype(str) + " " + e["details"].fillna("").astype(str)).str.lower()

    cap = typ.eq("capacity_expansion") | txt.str.contains(CAPACITY_RE, na=False)
    commission = txt.str.contains(COMMISSION_RE, na=False)
    order = typ.eq("order_win") | txt.str.contains(ORDER_RE, na=False)
    product = txt.str.contains(PRODUCT_RE, na=False)
    future_product = product & txt.str.contains(FUTURE_RE, na=False)
    customer = typ.eq("customer_supplier") | txt.str.contains(CUSTOMER_RE, na=False)
    debt = typ.eq("debt_reduction") | txt.str.contains(DEBT_RE, na=False)
    credit_pos = typ.eq("credit_rating") & (direction > 0)
    earnings_pos = typ.eq("earnings") & (direction > 0)
    risk = typ.isin(["dilution", "auditor_change", "litigation", "regulatory"]) | txt.str.contains(RISK_RE, na=False)

    e["_capacity"] = np.where(cap, strength, 0.0)
    e["_commissioning"] = np.where(commission, np.maximum(strength, 0.8), 0.0)
    e["_order"] = np.where(order, strength, 0.0)
    e["_new_product"] = np.where(product, strength, 0.0)
    e["_future_product"] = np.where(future_product, np.maximum(strength, 0.7), 0.0)
    e["_customer"] = np.where(customer, strength, 0.0)
    e["_debt"] = np.where(debt, strength, 0.0)
    e["_credit_pos"] = np.where(credit_pos, np.maximum(direction, 0) * strength, 0.0)
    e["_earnings_pos"] = np.where(earnings_pos, np.maximum(direction, 0) * strength, 0.0)
    e["_risk"] = np.where(risk, strength, 0.0)
    material = cap | commission | order | product | customer | debt | credit_pos | earnings_pos
    e["_material"] = material.astype(float)
    return e[e["avail_date"].notna() & e["key"].ne("S:")].sort_values(["key", "avail_date"]).reset_index(drop=True)


def _find_col(df, names):
    low = {str(c).lower(): c for c in df.columns}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    return None


def prepare_insider(path: str | Path) -> pd.DataFrame:
    x = pd.read_parquet(path).copy()
    date_col = _find_col(x, ["intimDt", "intimationDate", "anex", "date"])
    x["avail_date"] = pd.to_datetime(x[date_col], dayfirst=True, errors="coerce").dt.normalize() if date_col else pd.NaT
    ic = _find_col(x, ["isin_norm", "isin", "secIsin"])
    sc = _find_col(x, ["symbol_norm", "symbol"])
    iv = x[ic] if ic else pd.Series(index=x.index, dtype=object)
    sv = x[sc] if sc else pd.Series(index=x.index, dtype=object)
    x["key"] = [security_key(i, s) for i, s in zip(iv, sv)]

    tc = _find_col(x, ["tdpTransactionType", "transactionType", "tkdAcqm"])
    t = x[tc].fillna("").astype(str).str.lower() if tc else pd.Series("", index=x.index)
    sign = np.where(t.str.contains("sell|sale|disposal|dispose", regex=True), -1.0,
            np.where(t.str.contains("buy|purchase|acquisition|acquire", regex=True), 1.0, 0.0))

    pc = _find_col(x, ["personCategory", "category"])
    person = x[pc].fillna("").astype(str).str.lower() if pc else pd.Series("", index=x.index)
    promoter = person.str.contains("promoter", regex=False).to_numpy(float)

    mc = _find_col(x, ["acqMode", "mode"])
    mode = x[mc].fillna("").astype(str).str.lower() if mc else pd.Series("", index=x.index)
    quality = np.full(len(x), 0.5, float)
    quality[mode.str.contains("open market|market purchase", regex=True).to_numpy()] = 1.0
    quality[mode.str.contains("preferential|warrant|rights", regex=True).to_numpy()] = 0.6
    quality[mode.str.contains("inter-se|inter se|off market|transfer|gift", regex=True).to_numpy()] = 0.15
    quality[mode.str.contains("esop|employee", regex=True).to_numpy()] = 0.0

    candidates = []
    for nm in ["buyValue", "sellValue", "secVal", "securityValue", "value"]:
        c = _find_col(x, [nm])
        if c is not None:
            candidates.append(_num(x[c]).abs())
    value = pd.concat(candidates, axis=1).max(axis=1, skipna=True).fillna(0.0) if candidates else pd.Series(0.0, index=x.index)
    magnitude = np.log1p(value.to_numpy(float))
    signed = sign * promoter * quality * magnitude
    x["_promoter_signed"] = signed
    x["_promoter_buy_count"] = (signed > 0).astype(float)
    return x[x["avail_date"].notna() & x["key"].ne("S:")].sort_values(["key", "avail_date"]).reset_index(drop=True)


def prepare_shareholding(path: str | Path) -> pd.DataFrame:
    s = pd.read_parquet(path).copy()
    b = pd.to_datetime(s.get("broadcastDate", pd.Series(index=s.index, dtype=object)), dayfirst=True, errors="coerce")
    sub = pd.to_datetime(s.get("submissionDate", pd.Series(index=s.index, dtype=object)), dayfirst=True, errors="coerce")
    s["avail_date"] = b.fillna(sub).dt.normalize()
    s["key"] = [security_key(i, sy) for i, sy in zip(s.get("isin_norm", s.get("isin")), s.get("symbol_norm", s.get("symbol")))]
    pct_col = _find_col(s, ["pr_and_prgrp", "promoter_pct", "promoterHolding"])
    s["promoter_pct"] = _num(s[pct_col]).clip(0, 100) if pct_col else np.nan
    return s[s["avail_date"].notna() & s["promoter_pct"].notna()].sort_values(["key", "avail_date"]).reset_index(drop=True)


def _source_map(df, cols):
    out = {}
    if df is None or df.empty:
        return out
    for k, g in df.groupby("key", sort=False):
        gg = g.sort_values("avail_date")
        out[k] = {
            "dates": gg["avail_date"].to_numpy(dtype="datetime64[ns]"),
            **{c: gg[c].to_numpy() for c in cols},
        }
    return out


def attach_point_in_time(rows: pd.DataFrame, events: pd.DataFrame, insider: pd.DataFrame | None, shareholding: pd.DataFrame | None) -> pd.DataFrame:
    r = rows.copy()
    r["_key_v11"] = [security_key(i, s) for i, s in zip(r.get("isin"), r.get("symbol"))]
    r["_symkey_v11"] = ["S:" + (norm(s) or "") for s in r.get("symbol")]
    em = _source_map(events, [
        "_capacity", "_commissioning", "_order", "_new_product", "_future_product",
        "_customer", "_debt", "_credit_pos", "_earnings_pos", "_risk", "_material",
    ])
    im = _source_map(insider, ["_promoter_signed", "_promoter_buy_count"]) if insider is not None else {}
    sm = _source_map(shareholding, ["promoter_pct"]) if shareholding is not None else {}

    rec = []
    for z in r[["_key_v11", "_symkey_v11", "date"]].itertuples(index=False):
        key, symkey, d0 = z
        d = np.datetime64(pd.Timestamp(d0).normalize())
        vals = {c: 0.0 for c in EVENT_FEATURES}
        vals.update({c: np.nan for c in OWNERSHIP_FEATURES})
        vals["promoter_buy_net_180"] = 0.0
        vals["promoter_buy_count_180"] = 0.0

        e = em.get(key) or em.get(symkey)
        if e:
            dates = e["dates"]
            hi = np.searchsorted(dates, d, side="right")
            lo180 = np.searchsorted(dates, d - np.timedelta64(180, "D"), side="left")
            lo90 = np.searchsorted(dates, d - np.timedelta64(90, "D"), side="left")
            if hi > lo180:
                age = (d - dates[lo180:hi]).astype("timedelta64[D]").astype(float)
                w = np.exp(-np.maximum(age, 0) / 90.0)
                def ws(c):
                    return float(np.nansum(e[c][lo180:hi].astype(float) * w))
                vals["evt_capacity_180"] = ws("_capacity")
                vals["evt_commissioning_180"] = ws("_commissioning")
                vals["evt_order_180"] = ws("_order")
                vals["evt_new_product_180"] = ws("_new_product")
                vals["evt_future_product_180"] = ws("_future_product")
                vals["evt_customer_180"] = ws("_customer")
                vals["evt_debt_reduction_180"] = ws("_debt")
                vals["evt_credit_positive_180"] = ws("_credit_pos")
                vals["evt_earnings_positive_180"] = ws("_earnings_pos")
                vals["evt_risk_180"] = ws("_risk")
                kinds = ["_capacity", "_commissioning", "_order", "_new_product", "_future_product", "_customer", "_debt", "_credit_pos", "_earnings_pos"]
                vals["evt_catalyst_breadth_180"] = float(sum(np.nansum(e[c][lo180:hi].astype(float)) > 0 for c in kinds))
            if hi > lo90:
                vals["evt_catalyst_recent_90"] = float(np.nansum(e["_material"][lo90:hi].astype(float)))

        q = im.get(key) or im.get(symkey)
        if q:
            dates = q["dates"]
            hi = np.searchsorted(dates, d, side="right")
            lo = np.searchsorted(dates, d - np.timedelta64(180, "D"), side="left")
            if hi > lo:
                vals["promoter_buy_net_180"] = float(np.nansum(q["_promoter_signed"][lo:hi].astype(float)))
                vals["promoter_buy_count_180"] = float(np.nansum(q["_promoter_buy_count"][lo:hi].astype(float)))

        s = sm.get(key) or sm.get(symkey)
        if s:
            dates = s["dates"]
            hi = np.searchsorted(dates, d, side="right")
            if hi > 0:
                vals["promoter_pct"] = float(s["promoter_pct"][hi - 1])
                if hi > 1:
                    vals["promoter_delta_qoq"] = float(s["promoter_pct"][hi - 1] - s["promoter_pct"][hi - 2])
        rec.append(vals)

    add = pd.DataFrame(rec, index=r.index)
    for c in EVENT_FEATURES + OWNERSHIP_FEATURES:
        r[c] = add[c]
    return r.drop(columns=["_key_v11", "_symkey_v11"])


def positive_rank(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").fillna(0.0)
    out = pd.Series(0.0, index=s.index)
    m = x > 0
    if m.any():
        out.loc[m] = x.loc[m].rank(pct=True, method="average")
    return out


def standard_rank(s: pd.Series, neutral=0.5) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    if x.notna().sum() < 2:
        return pd.Series(neutral, index=s.index, dtype=float)
    return x.rank(pct=True, method="average").fillna(neutral)


def add_discovery_scores(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    out = []
    w = cfg["weights"]
    pconf = cfg["priced_in"]
    gate = cfg["gate"]
    for _, g0 in df.groupby("date", sort=True):
        g = g0.copy()
        catalyst_raw = (
            1.00 * g["evt_capacity_180"].fillna(0)
            + 1.25 * g["evt_commissioning_180"].fillna(0)
            + 0.90 * g["evt_order_180"].fillna(0)
            + 0.85 * g["evt_new_product_180"].fillna(0)
            + 1.00 * g["evt_future_product_180"].fillna(0)
            + 0.70 * g["evt_customer_180"].fillna(0)
            + 0.40 * g["evt_catalyst_breadth_180"].fillna(0)
            + 0.35 * g["evt_catalyst_recent_90"].fillna(0)
        )
        business_raw = (
            0.80 * g["evt_earnings_positive_180"].fillna(0)
            + 0.70 * g["evt_debt_reduction_180"].fillna(0)
            + 0.50 * g["evt_credit_positive_180"].fillna(0)
        )
        promoter_raw = np.maximum(pd.to_numeric(g["promoter_buy_net_180"], errors="coerce").fillna(0), 0)
        promoter_delta = np.maximum(pd.to_numeric(g["promoter_delta_qoq"], errors="coerce").fillna(0), 0)

        catalyst = positive_rank(catalyst_raw)
        business = positive_rank(business_raw)
        promoter = 0.75 * positive_rank(promoter_raw) + 0.25 * positive_rank(promoter_delta)
        accumulation = (
            0.35 * standard_rank(g["rvol_20_60"])
            + 0.30 * standard_rank(g["up_volume_share_20"])
            + 0.20 * standard_rank(g["accumulation_days_20"])
            + 0.15 * positive_rank(g["accumulation_absorption"])
        )

        ret120 = pd.to_numeric(g.get("ret_120"), errors="coerce").fillna(0.0)
        ret252 = pd.to_numeric(g.get("ret_252"), errors="coerce").fillna(0.0)
        p120 = np.clip((ret120 - float(pconf["ret120_soft"])) / max(float(pconf["ret120_hard"]) - float(pconf["ret120_soft"]), 1e-9), 0, 1)
        p252 = np.clip((ret252 - float(pconf["ret252_soft"])) / max(float(pconf["ret252_hard"]) - float(pconf["ret252_soft"]), 1e-9), 0, 1)
        priced = 0.55 * p120 + 0.45 * p252
        commissioning_override = g["evt_commissioning_180"].fillna(0) > 0
        priced = np.where(commissioning_override, priced * 0.65, priced)
        early = 1.0 - priced

        p100 = g["p100_cal"] if "p100_cal" in g else g.get("p_cal", pd.Series(np.nan, index=g.index))
        pdd = g["p_dd30_cal"] if "p_dd30_cal" in g else g.get("p_dd30", pd.Series(np.nan, index=g.index))
        technical = (
            0.50 * standard_rank(p100)
            + 0.20 * standard_rank(g.get("trend_consistency_60", pd.Series(np.nan, index=g.index)))
            + 0.20 * standard_rank(g.get("turnover_accel", pd.Series(np.nan, index=g.index)))
            + 0.10 * standard_rank(g.get("mom_accel", pd.Series(np.nan, index=g.index)))
        )
        risk = 0.55 * standard_rank(pdd) + 0.45 * positive_rank(g["evt_risk_180"])
        ownership_accum = 0.55 * promoter + 0.45 * accumulation

        score = (
            float(w["catalyst"]) * catalyst
            + float(w["business_inflection"]) * business
            + float(w["ownership_accumulation"]) * ownership_accum
            + float(w["early_stage"]) * early
            + float(w["technical_confirmation"]) * technical
            - float(cfg["risk_penalty"]) * risk
        )

        catalyst_gate = catalyst_raw >= float(gate["catalyst_raw_min"])
        promoter_gate = (promoter >= float(gate["promoter_score_min"])) & (accumulation >= float(gate["accumulation_score_min"]))
        technical_gate = technical >= float(gate["technical_min"])
        priced_gate = (priced <= float(gate["max_priced_in_penalty"])) | commissioning_override
        eligible = (catalyst_gate | promoter_gate) & technical_gate & priced_gate

        g["catalyst_raw"] = catalyst_raw
        g["business_inflection_raw"] = business_raw
        g["catalyst_score"] = catalyst
        g["business_inflection_score"] = business
        g["promoter_conviction_score"] = promoter
        g["accumulation_score"] = accumulation
        g["ownership_accumulation_score"] = ownership_accum
        g["technical_confirmation_score"] = technical
        g["priced_in_penalty"] = priced
        g["early_stage_score"] = early
        g["risk_score"] = risk
        g["discovery_score"] = score
        g["gate_catalyst"] = catalyst_gate
        g["gate_promoter_accumulation"] = promoter_gate
        g["gate_technical"] = technical_gate
        g["gate_not_overpriced"] = priced_gate
        g["transformation_eligible"] = eligible
        out.append(g)
    return pd.concat(out, ignore_index=True)


def evaluate(scored: pd.DataFrame, control: pd.DataFrame, k: int, min_date: str) -> tuple[pd.DataFrame, dict]:
    rows = []
    start = pd.Timestamp(min_date)
    for td, g in scored[scored["date"] >= start].groupby("date"):
        mature = g.dropna(subset=["y6"]).copy() if "y6" in g else pd.DataFrame()
        if mature.empty:
            continue
        sel = mature[mature["transformation_eligible"]].sort_values("discovery_score", ascending=False).head(k)
        if len(sel) < k:
            continue
        base_rate = float(mature["y6"].mean())
        precision = float(sel["y6"].mean())
        dd = float(sel["dd30_6m"].mean()) if "dd30_6m" in sel and sel["dd30_6m"].notna().any() else np.nan
        rows.append({
            "date": td,
            "qualified": int(mature["transformation_eligible"].sum()),
            "precision_100": precision,
            "lift_100": precision / base_rate if base_rate > 0 else np.nan,
            "dd30_rate": dd,
        })
    tab = pd.DataFrame(rows)

    ctl_rows = []
    if not control.empty and "y6" in control:
        for td, g in control[control["date"] >= start].groupby("date"):
            q = g.dropna(subset=["y6"]).copy()
            if len(q) < k:
                continue
            if "selected_v941" in q and int(q["selected_v941"].fillna(False).sum()) >= k:
                sel = q[q["selected_v941"].fillna(False)].head(k)
            else:
                pc = "p100_cal" if "p100_cal" in q else "p_cal"
                sel = q.sort_values(pc, ascending=False).head(k)
            br = float(q["y6"].mean())
            ctl_rows.append({
                "date": td,
                "control_precision_100": float(sel["y6"].mean()),
                "control_lift_100": float(sel["y6"].mean()) / br if br > 0 else np.nan,
                "control_dd30_rate": float(sel["dd30_6m"].mean()) if "dd30_6m" in sel and sel["dd30_6m"].notna().any() else np.nan,
            })
    ctl = pd.DataFrame(ctl_rows)
    merged = tab.merge(ctl, on="date", how="left") if len(tab) else tab
    summary = {
        "evaluated_folds": int(len(merged)),
        "mean_precision_100": float(merged["precision_100"].mean()) if len(merged) else None,
        "mean_lift_100": float(merged["lift_100"].mean()) if len(merged) else None,
        "hit_fold_rate_100": float((merged["precision_100"] > 0).mean()) if len(merged) else None,
        "mean_dd30_rate": float(merged["dd30_rate"].mean()) if len(merged) else None,
        "mean_control_precision_100": float(merged["control_precision_100"].mean()) if len(merged) and "control_precision_100" in merged else None,
        "mean_control_lift_100": float(merged["control_lift_100"].mean()) if len(merged) and "control_lift_100" in merged else None,
        "mean_control_dd30_rate": float(merged["control_dd30_rate"].mean()) if len(merged) and "control_dd30_rate" in merged else None,
    }
    return merged, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--baseline-oos", required=True)
    ap.add_argument("--current-universe", required=True)
    ap.add_argument("--events", required=True)
    ap.add_argument("--pit-dir", required=True)
    ap.add_argument("--legacy-dir", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    oos = pd.read_parquet(args.baseline_oos).copy()
    oos["date"] = pd.to_datetime(oos["date"])
    current = pd.read_csv(args.current_universe).copy()
    current["date"] = pd.to_datetime(current["date"])
    needed = pd.concat([
        oos[["date", "symbol", "isin"]],
        current[["date", "symbol", "isin"]],
    ], ignore_index=True).drop_duplicates(["date", "symbol"])

    print("Loading market history for transformation/accumulation features...", flush=True)
    market_integrity.install_on_base()
    end_year = int(pd.Timestamp.today().year)
    daily = base.load_market(int(cfg["start_year"]), end_year, args.legacy_dir)
    daily = market_integrity.normalize_split_bonus_volume(daily, int(cfg["start_year"]), end_year)
    daily = market_integrity.stitch_symbol_changes_same_isin(daily)
    daily = base.add_features(daily)
    daily = add_accumulation_features(daily)

    mcols = [
        "date", "symbol", "isin", "ret_20", "ret_60", "ret_120", "ret_252", "mom_accel",
        "turnover_accel", "trend_consistency_60", "rvol_20_60", "up_volume_share_20",
        "accumulation_days_20", "accumulation_absorption",
    ]
    market_rows = daily[mcols].merge(needed[["date", "symbol"]], on=["date", "symbol"], how="inner")

    print("Preparing point-in-time catalyst and ownership sources...", flush=True)
    events = prepare_events(args.events)
    pit = Path(args.pit_dir)
    insider_path = pit / "insider_trades.parquet"
    share_path = pit / "shareholding_master.parquet"
    insider = prepare_insider(insider_path) if insider_path.exists() else pd.DataFrame()
    share = prepare_shareholding(share_path) if share_path.exists() else pd.DataFrame()

    def enrich(base_rows):
        x = base_rows.merge(market_rows.drop(columns=["isin"], errors="ignore"), on=["date", "symbol"], how="left")
        x = attach_point_in_time(x, events, insider, share)
        return add_discovery_scores(x, cfg)

    print("Scoring historical V10.2 OOS universe with transformation-first overlay...", flush=True)
    hist = enrich(oos)
    hist.to_parquet(outdir / "historical_scored.parquet", index=False)

    print("Scoring current universe...", flush=True)
    cur = enrich(current)
    cur = cur.sort_values(["transformation_eligible", "discovery_score"], ascending=[False, False]).reset_index(drop=True)
    cur.to_csv(outdir / "current_universe_scored.csv", index=False)
    current_top = cur[cur["transformation_eligible"]].head(int(cfg["selection_k"])).copy()
    current_top.to_csv(outdir / "current_transformation_watchlist.csv", index=False)

    fold_metrics, metrics = evaluate(hist, oos, int(cfg["selection_k"]), cfg["evaluation_start"])
    fold_metrics.to_csv(outdir / "fold_metrics.csv", index=False)

    summary = {
        "model": cfg["model_name"],
        "objective": "discover transformation stories before they become obvious",
        "architecture": "fundamental/corporate catalyst first; promoter/volume accumulation second; technical confirmation last",
        "control": "V10.2 integrity-corrected V9.4.1",
        "rules_frozen_before_backtest": True,
        "weights": cfg["weights"],
        "risk_penalty": cfg["risk_penalty"],
        "gate": cfg["gate"],
        "priced_in": cfg["priced_in"],
        "current_date": str(pd.Timestamp(current["date"].max()).date()),
        "current_qualified": int(cur["transformation_eligible"].sum()),
        "current_selected": int(len(current_top)),
        "evaluation": metrics,
        "source_coverage": {
            "events": int(len(events)),
            "insider": int(len(insider)),
            "shareholding": int(len(share)),
        },
        "principles": [
            "capacity expansion/commissioning, order wins and new/futuristic products are discovery signals",
            "promoter open-market buying and persistent abnormal volume can independently qualify a candidate",
            "technical momentum is confirmation, not the primary discovery engine",
            "large prior rerating receives an explicit already-priced-in penalty",
            "no outcome-based feature/weight tuning is allowed in this frozen run",
        ],
    }
    json.dump(summary, open(outdir / "summary.json", "w"), indent=2, default=str)
    print(json.dumps(summary, indent=2, default=str), flush=True)

    show = [
        "date", "symbol", "close", "discovery_score", "catalyst_score", "business_inflection_score",
        "promoter_conviction_score", "accumulation_score", "early_stage_score", "technical_confirmation_score",
        "priced_in_penalty", "p100_cal", "p_dd30_cal", "transformation_eligible",
    ]
    show = [c for c in show if c in current_top]
    if len(current_top):
        print(current_top[show].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
