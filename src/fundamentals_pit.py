from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_ID = [
    "symbol",
    "period_end",
    "broadcast_ts",
    "statement_scope",
    "period_months",
]

VALUE_COLS = [
    "revenue",
    "operating_profit",
    "pbt",
    "pat",
    "finance_cost",
    "total_assets",
    "total_equity",
    "total_debt",
    "current_assets",
    "current_liabilities",
    "operating_cash_flow",
    "capex",
    "shares_outstanding",
    "promoter_pct",
    "pledged_pct",
]

FEATURE_COLS = [
    "fund_revenue_yoy",
    "fund_revenue_qoq",
    "fund_revenue_accel",
    "fund_pat_yoy",
    "fund_pat_qoq",
    "fund_pat_turnaround",
    "fund_operating_margin",
    "fund_margin_delta_yoy",
    "fund_interest_coverage",
    "fund_debt_to_equity",
    "fund_debt_change_yoy",
    "fund_current_ratio",
    "fund_ocf_to_pat",
    "fund_fcf_margin",
    "fund_shares_change_yoy",
    "fund_promoter_change_yoy",
    "fund_pledge_change_yoy",
    "fund_revenue_yoy_ann",
    "fund_pat_yoy_ann",
    "fund_debt_change_yoy_ann",
    "fund_ocf_to_pat_ann",
    "fund_roe_proxy_ann",
    "fund_quality_completeness",
    "fund_age_days",
    "fund_scope_consolidated",
    "fund_specialized_financial",
    "fund_mapping_score",
    "fund_mapped_fields",
    "fund_concept_mapping_fraction",
    "fund_synthetic_context",
]


def _safe_div(a, b, floor=1e-9):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    den = np.where(np.abs(b) >= floor, np.abs(b), np.nan)
    return a / den


def _signed_growth(cur, prev, floor=1.0):
    if not (np.isfinite(cur) and np.isfinite(prev)):
        return np.nan
    return (cur - prev) / max(abs(prev), floor)


def _ratio(a, b):
    if not (np.isfinite(a) and np.isfinite(b)) or abs(b) < 1e-9:
        return np.nan
    return a / b


def normalize_fundamentals(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    missing = [c for c in REQUIRED_ID if c not in x.columns]
    if missing:
        raise ValueError(f"Missing fundamental identity columns: {missing}")

    x["symbol"] = x["symbol"].astype(str).str.upper().str.strip()
    x["period_end"] = pd.to_datetime(x["period_end"], errors="coerce")
    x["broadcast_ts"] = pd.to_datetime(x["broadcast_ts"], errors="coerce", utc=True)
    x["statement_scope"] = (
        x["statement_scope"].astype(str).str.lower().str.strip()
        .replace({
            "consolidated": "consolidated",
            "standalone": "standalone",
            "non-consolidated": "standalone",
            "non consolidated": "standalone",
        })
    )
    x["period_months"] = pd.to_numeric(x["period_months"], errors="coerce")

    for c in VALUE_COLS:
        if c not in x.columns:
            x[c] = np.nan
        x[c] = pd.to_numeric(x[c], errors="coerce")

    if "filing_id" not in x.columns:
        x["filing_id"] = (
            x["symbol"].astype(str)
            + "|"
            + x["period_end"].astype(str)
            + "|"
            + x["broadcast_ts"].astype(str)
            + "|"
            + x["statement_scope"].astype(str)
        )
    if "source_url" not in x.columns:
        x["source_url"] = ""
    if "revision" not in x.columns:
        x["revision"] = False
    if "taxonomy_family" not in x.columns:
        x["taxonomy_family"] = "generic_or_unknown"
    x["taxonomy_family"] = x["taxonomy_family"].astype(str).str.lower().str.strip()
    if "mapping_score" not in x.columns:
        x["mapping_score"] = 0.0
    if "mapped_field_count" not in x.columns:
        x["mapped_field_count"] = 0
    if "concept_mapping_fraction" not in x.columns:
        x["concept_mapping_fraction"] = 0.0
    if "synthetic_context_used" not in x.columns:
        x["synthetic_context_used"] = False
    x["mapping_score"] = pd.to_numeric(x["mapping_score"], errors="coerce").fillna(0.0)
    x["mapped_field_count"] = pd.to_numeric(x["mapped_field_count"], errors="coerce").fillna(0)
    x["concept_mapping_fraction"] = pd.to_numeric(
        x["concept_mapping_fraction"], errors="coerce"
    ).fillna(0.0)

    x = x.dropna(subset=["symbol", "period_end", "broadcast_ts", "period_months"])
    x = x[x["period_months"].isin([3, 6, 9, 12])].copy()
    x = x.sort_values(
        ["symbol", "period_end", "statement_scope", "period_months", "broadcast_ts"]
    )

    # Keep all revisions in the raw PIT table. Snapshot selection later chooses
    # the latest filing that was actually known before the snapshot date.
    return x.reset_index(drop=True)


def read_fundamentals(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(p)
    elif p.suffix.lower() in {".csv", ".txt"}:
        df = pd.read_csv(p)
    else:
        raise ValueError(f"Unsupported fundamentals file: {p}")
    return normalize_fundamentals(df)


def _latest_known(rows: pd.DataFrame, snapshot_ts: pd.Timestamp) -> pd.DataFrame:
    q = rows[rows["broadcast_ts"] < snapshot_ts].copy()
    if q.empty:
        return q
    # Latest revision that was known at the time for each statement identity.
    q = q.sort_values("broadcast_ts").drop_duplicates(
        ["symbol", "period_end", "statement_scope", "period_months"],
        keep="last",
    )
    return q


def _choose_scope(rows: pd.DataFrame) -> str | None:
    if rows.empty:
        return None
    scopes = set(rows["statement_scope"].dropna().astype(str))
    if "consolidated" in scopes:
        return "consolidated"
    if "standalone" in scopes:
        return "standalone"
    return sorted(scopes)[0] if scopes else None


def _near_period(rows: pd.DataFrame, target: pd.Timestamp, tol_days=45):
    if rows.empty:
        return None
    d = (rows["period_end"] - target).abs().dt.days
    ok = d <= tol_days
    if not ok.any():
        return None
    return rows.loc[d[ok].idxmin()]


def _latest_period(rows: pd.DataFrame):
    if rows.empty:
        return None
    return rows.sort_values(["period_end", "broadcast_ts"]).iloc[-1]


def _feature_one(symbol_rows: pd.DataFrame, snapshot_date: pd.Timestamp) -> dict:
    # Conservative availability: same-day filings are excluded. A filing must
    # have been broadcast strictly before the snapshot date UTC midnight.
    snap_ts = pd.Timestamp(snapshot_date).tz_localize("UTC")
    known = _latest_known(symbol_rows, snap_ts)

    out = {c: np.nan for c in FEATURE_COLS}
    out["fund_quality_completeness"] = 0.0

    if known.empty:
        return out

    scope = _choose_scope(known)
    known = known[known["statement_scope"] == scope].copy()
    out["fund_scope_consolidated"] = float(scope == "consolidated")
    families = known.get("taxonomy_family", pd.Series(dtype=str)).dropna().astype(str).str.lower()
    specialized = families.str.contains("bank|insurance|nbfc", regex=True).any() if len(families) else False
    out["fund_specialized_financial"] = float(bool(specialized))

    qtr = known[known["period_months"] == 3].copy()
    ann = known[known["period_months"] == 12].copy()

    cur = _latest_period(qtr)
    if cur is not None:
        out["fund_mapping_score"] = float(cur.get("mapping_score", 0.0))
        out["fund_mapped_fields"] = float(cur.get("mapped_field_count", 0.0))
        out["fund_concept_mapping_fraction"] = float(
            cur.get("concept_mapping_fraction", 0.0)
        )
        out["fund_synthetic_context"] = float(
            bool(cur.get("synthetic_context_used", False))
        )
        pe = pd.Timestamp(cur["period_end"])
        yoy = _near_period(qtr[qtr["period_end"] < pe], pe - pd.DateOffset(years=1))
        prevq = _near_period(qtr[qtr["period_end"] < pe], pe - pd.DateOffset(months=3))
        yoy_prevq = None
        if prevq is not None:
            yoy_prevq = _near_period(
                qtr[qtr["period_end"] < pd.Timestamp(prevq["period_end"])],
                pd.Timestamp(prevq["period_end"]) - pd.DateOffset(years=1),
            )

        out["fund_age_days"] = float((pd.Timestamp(snapshot_date) - pe).days)

        if yoy is not None:
            out["fund_revenue_yoy"] = _signed_growth(cur["revenue"], yoy["revenue"])
            out["fund_pat_yoy"] = _signed_growth(cur["pat"], yoy["pat"])
            out["fund_margin_delta_yoy"] = (
                _ratio(cur["operating_profit"], cur["revenue"])
                - _ratio(yoy["operating_profit"], yoy["revenue"])
            )
            out["fund_debt_change_yoy"] = _signed_growth(
                cur["total_debt"], yoy["total_debt"], floor=1.0
            )
            out["fund_shares_change_yoy"] = _signed_growth(
                cur["shares_outstanding"], yoy["shares_outstanding"], floor=1.0
            )
            if np.isfinite(cur["promoter_pct"]) and np.isfinite(yoy["promoter_pct"]):
                out["fund_promoter_change_yoy"] = cur["promoter_pct"] - yoy["promoter_pct"]
            if np.isfinite(cur["pledged_pct"]) and np.isfinite(yoy["pledged_pct"]):
                out["fund_pledge_change_yoy"] = cur["pledged_pct"] - yoy["pledged_pct"]
            out["fund_pat_turnaround"] = float(
                np.isfinite(cur["pat"])
                and np.isfinite(yoy["pat"])
                and cur["pat"] > 0
                and yoy["pat"] <= 0
            )

        if prevq is not None:
            out["fund_revenue_qoq"] = _signed_growth(cur["revenue"], prevq["revenue"])
            out["fund_pat_qoq"] = _signed_growth(cur["pat"], prevq["pat"])

        if prevq is not None and yoy is not None and yoy_prevq is not None:
            prior_yoy_growth = _signed_growth(prevq["revenue"], yoy_prevq["revenue"])
            if np.isfinite(out["fund_revenue_yoy"]) and np.isfinite(prior_yoy_growth):
                out["fund_revenue_accel"] = out["fund_revenue_yoy"] - prior_yoy_growth

        out["fund_operating_margin"] = _ratio(cur["operating_profit"], cur["revenue"])
        out["fund_interest_coverage"] = _ratio(cur["operating_profit"], cur["finance_cost"])
        out["fund_debt_to_equity"] = _ratio(cur["total_debt"], cur["total_equity"])
        out["fund_current_ratio"] = _ratio(cur["current_assets"], cur["current_liabilities"])
        out["fund_ocf_to_pat"] = _ratio(cur["operating_cash_flow"], cur["pat"])
        fcf = (
            cur["operating_cash_flow"] - abs(cur["capex"])
            if np.isfinite(cur["operating_cash_flow"]) and np.isfinite(cur["capex"])
            else np.nan
        )
        out["fund_fcf_margin"] = _ratio(fcf, cur["revenue"])

    a = _latest_period(ann)
    if a is not None:
        ape = pd.Timestamp(a["period_end"])
        ay = _near_period(ann[ann["period_end"] < ape], ape - pd.DateOffset(years=1))
        if ay is not None:
            out["fund_revenue_yoy_ann"] = _signed_growth(a["revenue"], ay["revenue"])
            out["fund_pat_yoy_ann"] = _signed_growth(a["pat"], ay["pat"])
            out["fund_debt_change_yoy_ann"] = _signed_growth(
                a["total_debt"], ay["total_debt"], floor=1.0
            )
        out["fund_ocf_to_pat_ann"] = _ratio(a["operating_cash_flow"], a["pat"])
        out["fund_roe_proxy_ann"] = _ratio(a["pat"], a["total_equity"])

    measurable = [
        c for c in FEATURE_COLS
        if c not in {
            "fund_quality_completeness",
            "fund_scope_consolidated",
            "fund_specialized_financial",
            "fund_mapping_score",
            "fund_mapped_fields",
            "fund_concept_mapping_fraction",
            "fund_synthetic_context",
        }
    ]
    vals = np.array([out[c] for c in measurable], dtype=float)
    out["fund_quality_completeness"] = float(np.isfinite(vals).mean())
    return out


def build_snapshot_features(
    snapshots: pd.DataFrame,
    fundamentals: pd.DataFrame,
    min_completeness: float = 0.0,
) -> pd.DataFrame:
    req = {"date", "symbol"}
    if not req.issubset(snapshots.columns):
        raise ValueError("snapshots must contain date and symbol")

    f = normalize_fundamentals(fundamentals)
    grouped = {sym: g.copy() for sym, g in f.groupby("symbol", sort=False)}

    rows = []
    for r in snapshots[["date", "symbol"]].itertuples(index=False):
        sym = str(r.symbol).upper()
        feat = _feature_one(
            grouped.get(sym, f.iloc[0:0]),
            pd.Timestamp(r.date),
        )
        feat["date"] = pd.Timestamp(r.date)
        feat["symbol"] = sym
        rows.append(feat)

    out = pd.DataFrame(rows)
    if min_completeness > 0:
        for c in FEATURE_COLS:
            if c not in out.columns:
                out[c] = np.nan
        out["fund_usable"] = (
            out["fund_quality_completeness"] >= float(min_completeness)
        )
    else:
        out["fund_usable"] = True
    return out


def synthetic_self_test() -> dict:
    # Filing broadcast after the first snapshot must never leak backwards.
    raw = pd.DataFrame([
        {
            "symbol": "TEST",
            "period_end": "2024-06-30",
            "broadcast_ts": "2024-08-10T10:00:00Z",
            "statement_scope": "Consolidated",
            "period_months": 3,
            "revenue": 100,
            "operating_profit": 10,
            "pat": 5,
            "finance_cost": 2,
            "total_equity": 50,
            "total_debt": 25,
        },
        {
            "symbol": "TEST",
            "period_end": "2025-06-30",
            "broadcast_ts": "2025-08-10T10:00:00Z",
            "statement_scope": "Consolidated",
            "period_months": 3,
            "revenue": 140,
            "operating_profit": 21,
            "pat": 12,
            "finance_cost": 2,
            "total_equity": 60,
            "total_debt": 20,
        },
    ])
    snaps = pd.DataFrame([
        {"date": "2025-07-31", "symbol": "TEST"},
        {"date": "2025-08-31", "symbol": "TEST"},
    ])
    z = build_snapshot_features(snaps, raw)
    before = z.iloc[0]
    after = z.iloc[1]
    assert not np.isfinite(before["fund_revenue_yoy"])
    assert abs(after["fund_revenue_yoy"] - 0.40) < 1e-9
    assert after["fund_pat_yoy"] > 1.0
    return {
        "rows": int(len(z)),
        "no_future_filing_leak": bool(not np.isfinite(before["fund_revenue_yoy"])),
        "post_filing_revenue_yoy": float(after["fund_revenue_yoy"]),
        "post_filing_pat_yoy": float(after["fund_pat_yoy"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--fundamentals")
    ap.add_argument("--snapshots")
    ap.add_argument("--output")
    ap.add_argument("--min-completeness", type=float, default=0.25)
    args = ap.parse_args()

    if args.self_test:
        print(json.dumps(synthetic_self_test(), indent=2))
        return

    if not (args.fundamentals and args.snapshots and args.output):
        raise SystemExit("--fundamentals, --snapshots and --output are required")

    fundamentals = read_fundamentals(args.fundamentals)
    sp = Path(args.snapshots)
    snapshots = pd.read_parquet(sp) if sp.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(sp)
    out = build_snapshot_features(snapshots, fundamentals, args.min_completeness)
    op = Path(args.output)
    op.parent.mkdir(parents=True, exist_ok=True)
    if op.suffix.lower() in {".parquet", ".pq"}:
        out.to_parquet(op, index=False)
    else:
        out.to_csv(op, index=False)
    print(json.dumps({
        "rows": int(len(out)),
        "usable": int(out["fund_usable"].sum()),
        "mean_completeness": float(out["fund_quality_completeness"].mean()),
    }, indent=2))


if __name__ == "__main__":
    main()
