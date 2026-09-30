from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import calibrate_v9_2 as base
import fundamentals_pit as fpit


def assert_true(cond, msg):
    if not bool(cond):
        raise AssertionError(msg)


def synthetic_market(days=140):
    dates = pd.bdate_range("2024-01-02", periods=days)
    d = pd.DataFrame({
        "date": dates,
        "symbol": "TEST",
        "adj_close": 100.0,
        "close": 100.0,
        "turnover": 10_000_000.0,
    })
    return d


def snapshot_from_day(daily, idx):
    r = daily.iloc[idx]
    rec = {
        "date": r["date"],
        "symbol": r["symbol"],
        "adj_close": float(r["adj_close"]),
    }
    if "security_key" in daily.columns:
        rec["security_key"] = r["security_key"]
    return pd.DataFrame([rec])


def test_label_maturity():
    daily = synthetic_market(140)
    cfg = {
        "label_days": {"y6": 126, "y12": 252, "y24": 504},
        "min_avg_turnover_63d": 5_000_000,
    }
    mature = base.add_labels(snapshot_from_day(daily, 0), daily, cfg).iloc[0]
    immature = base.add_labels(snapshot_from_day(daily, 20), daily, cfg).iloc[0]

    assert_true(np.isfinite(mature["y6"]), "mature y6 must be labeled")
    assert_true(mature["y6"] == 0.0, "flat mature series must be negative")
    assert_true(pd.notna(mature["y6_mature_date"]), "mature date missing")
    assert_true(pd.isna(immature["y6"]), "immature y6 must be NaN, never 0")
    assert_true(pd.isna(immature["y6_mature_date"]), "immature mature_date must be NaT")
    return {
        "mature_label": float(mature["y6"]),
        "immature_is_nan": bool(pd.isna(immature["y6"])),
    }


def test_three_session_2x_rule():
    cfg = {
        "label_days": {"y6": 126, "y12": 252, "y24": 504},
        "min_avg_turnover_63d": 5_000_000,
    }

    spike = synthetic_market(140)
    spike.loc[20, "adj_close"] = 250.0
    z1 = base.add_labels(snapshot_from_day(spike, 0), spike, cfg).iloc[0]
    assert_true(z1["y6"] == 0.0, "one-session spike must not count as 2x")

    sustained = synthetic_market(140)
    sustained.loc[20:22, "adj_close"] = [205.0, 210.0, 208.0]
    z2 = base.add_labels(snapshot_from_day(sustained, 0), sustained, cfg).iloc[0]
    assert_true(z2["y6"] == 1.0, "three consecutive liquid 2x sessions must count")
    # The production rule uses the median of each 3-session window. Thus the\n    # first qualifying window is sessions 19-21: [100, 205, 210] has median\n    # 205 >= 2x, deliberately rejecting a one-day spike but not requiring all\n    # three closes themselves to exceed 2x.\n    assert_true(z2["days_to_2x"] == 21.0, "3-session median confirmation offset changed")

    illiquid = synthetic_market(140)
    illiquid.loc[20:22, "adj_close"] = [205.0, 210.0, 208.0]
    illiquid.loc[20:22, "turnover"] = 100_000.0
    z3 = base.add_labels(snapshot_from_day(illiquid, 0), illiquid, cfg).iloc[0]
    assert_true(z3["y6"] == 0.0, "illiquid 2x sequence must not count")
    return {
        "single_spike_rejected": True,
        "three_session_median_hit": True,
        "illiquid_hit_rejected": True,
    }


def test_split_artifact_protection():
    cfg = {
        "label_days": {"y6": 126, "y12": 252, "y24": 504},
        "min_avg_turnover_63d": 5_000_000,
    }
    daily = synthetic_market(140)
    # Simulate a raw-close discontinuity from a corporate action while the
    # split/bonus-normalized close correctly remains flat.
    daily.loc[20:25, "close"] = 220.0
    daily.loc[20:25, "adj_close"] = 100.0
    z = base.add_labels(snapshot_from_day(daily, 0), daily, cfg).iloc[0]
    assert_true(z["y6"] == 0.0, "raw close split artifact leaked into 2x label")
    return {"split_artifact_rejected": True}


def test_ticker_rename_continuity():
    dates = pd.bdate_range("2024-01-02", periods=180)
    daily = pd.DataFrame({
        "date": dates,
        "symbol": ["OLD"] * 90 + ["NEW"] * 90,
        "isin": ["INE000TEST01"] * 180,
        "close": 100.0,
        "adj_close": 100.0,
        "volume": 100_000.0,
        "turnover": 10_000_000.0,
    })
    daily = base.add_security_key(daily)
    feat = base.add_features(daily)

    first_new = feat[feat["symbol"] == "NEW"].sort_values("date").iloc[0]
    assert_true(
        int(first_new["history_days"]) == 91,
        "ticker rename reset feature history despite same known ISIN",
    )

    # A pre-rename snapshot must be able to observe a post-rename 2x event.
    daily.loc[100:102, "adj_close"] = [205.0, 210.0, 208.0]
    daily.loc[100:102, "close"] = [205.0, 210.0, 208.0]
    cfg = {
        "label_days": {"y6": 126, "y12": 252, "y24": 504},
        "min_avg_turnover_63d": 5_000_000,
    }
    snap = snapshot_from_day(daily, 40)
    z = base.add_labels(snap, daily, cfg).iloc[0]
    assert_true(
        z["y6"] == 1.0,
        "ticker rename broke future label continuity for same ISIN",
    )
    return {
        "feature_history_continues": True,
        "pre_rename_label_sees_post_rename_hit": True,
        "first_new_history_days": int(first_new["history_days"]),
    }


def fundamentals_fixture():
    rows = [
        {
            "symbol": "TEST", "period_end": "2024-06-30",
            "broadcast_ts": "2024-08-10T10:00:00Z",
            "statement_scope": "Consolidated", "period_months": 3,
            "revenue": 100.0, "operating_profit": 10.0, "pbt": 8.0,
            "pat": 5.0, "finance_cost": 2.0,
        },
        {
            "symbol": "TEST", "period_end": "2025-06-30",
            "broadcast_ts": "2025-08-10T10:00:00Z",
            "statement_scope": "Consolidated", "period_months": 3,
            "revenue": 120.0, "operating_profit": 15.0, "pbt": 12.0,
            "pat": 8.0, "finance_cost": 2.0, "revision": False,
        },
        {
            "symbol": "TEST", "period_end": "2025-06-30",
            "broadcast_ts": "2025-09-10T10:00:00Z",
            "statement_scope": "Consolidated", "period_months": 3,
            "revenue": 150.0, "operating_profit": 20.0, "pbt": 17.0,
            "pat": 12.0, "finance_cost": 2.0, "revision": True,
        },
    ]
    return pd.DataFrame(rows)


def test_fundamental_pit_and_revisions():
    raw = fundamentals_fixture()
    snaps = pd.DataFrame([
        {"date": "2025-08-10", "symbol": "TEST"},  # same-day original excluded
        {"date": "2025-08-31", "symbol": "TEST"},  # original known
        {"date": "2025-09-30", "symbol": "TEST"},  # revision known
    ])
    z = fpit.build_snapshot_features(snaps, raw)

    same_day = z.iloc[0]
    original = z.iloc[1]
    revised = z.iloc[2]

    assert_true(
        not np.isfinite(same_day["fund_revenue_yoy"]),
        "same-day filing leaked into snapshot",
    )
    assert_true(
        abs(float(original["fund_revenue_yoy"]) - 0.20) < 1e-9,
        "original PIT filing not used before revision",
    )
    assert_true(
        abs(float(revised["fund_revenue_yoy"]) - 0.50) < 1e-9,
        "latest-known revision not used after its broadcast",
    )
    return {
        "same_day_excluded": True,
        "original_yoy": float(original["fund_revenue_yoy"]),
        "revised_yoy": float(revised["fund_revenue_yoy"]),
    }


def static_source_checks():
    src = Path(__file__).with_name("calibrate_v9_2.py").read_text()
    pit = Path(__file__).with_name("fundamentals_pit.py").read_text()
    checks = {
        "labels_use_adj_close": "p0=float(row.adj_close)" in src,
        "immature_labels_nan": "if mature else np.nan" in src,
        "same_day_fundamentals_excluded": 'rows["broadcast_ts"] < snapshot_ts' in pit,
        "latest_known_revision": 'keep="last"' in pit,
    }
    for k, v in checks.items():
        assert_true(v, f"static integrity check failed: {k}")
    return checks


def main():
    results = {
        "label_maturity": test_label_maturity(),
        "three_session_2x": test_three_session_2x_rule(),
        "corporate_action": test_split_artifact_protection(),
        "ticker_rename_continuity": test_ticker_rename_continuity(),
        "fundamental_pit": test_fundamental_pit_and_revisions(),
        "static_checks": static_source_checks(),
    }
    results["passed"] = True
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
