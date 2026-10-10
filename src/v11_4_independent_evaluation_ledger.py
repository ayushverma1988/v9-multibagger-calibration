"""Immutable prospective research protocol and outcome-blind observations.

This adds evaluation discipline; it cannot supply six months of future prices
or certify the incomplete full financial/catalyst model. Old dates cannot be
registered as new blind observations after a recipe has been frozen.
"""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path

import pandas as pd
import numpy as np

HORIZON_SESSIONS = 126
ACCEPTANCE = {"minimum_fully_assessable_dates": 12, "mean_top10_jaccard": .80,
              "worst_date_p05_jaccard": .60}


def fingerprint(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def aware(value):
    t = pd.Timestamp(value)
    if pd.isna(t) or t.tzinfo is None:
        raise ValueError("Evaluation clocks must be timezone-aware")
    return t.tz_convert("UTC")


def immutable_write(path, raw):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # O_EXCL prevents reruns, concurrent jobs and accidental replacement.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)


def freeze_protocol(folder, model_path, recipe, frozen_at_utc):
    stamp = aware(frozen_at_utc)
    model = Path(model_path).read_bytes()
    if not model or not recipe.get("feature_names") or not recipe.get("source_code_sha256"):
        raise ValueError("Actual frozen weights, recipe and source fingerprints required")
    protocol = {"schema_version": 1, "scope": "FIXED_21_INPUT_RESEARCH_PROSPECTIVE_EVALUATION",
                "frozen_at_utc": stamp.isoformat(), "model_sha256": fingerprint(model),
                "recipe": recipe, "horizon_market_sessions": HORIZON_SESSIONS,
                "selection_count_per_view": 10, "views": ["GLOBAL", "EARLY", "SECOND_LEG", "EXTENDED"],
                "accessible_2x_label": {"median_adjusted_close_over_three_consecutive_sessions_ge_2x": True,
                                       "mean_three_session_turnover_INR_ge": 5000000},
                "acceptance_gates": ACCEPTANCE, "unknown_selected_outcomes_never_replaced": True,
                "six_month_hits_not_portfolio_PnL": True,
                "source_recovery_after_market_date_is_not_blind_observation": True,
                "full_financial_catalyst_model_source_qualified": False,
                "production_approved": False}
    raw = encoded(protocol)
    target = Path(folder) / "protocol.json"
    immutable_write(target, raw)
    return protocol, fingerprint(raw)


def load_protocol(path, expected_sha256):
    raw = Path(path).read_bytes()
    if fingerprint(raw) != expected_sha256:
        raise ValueError("Frozen protocol differs from independently pinned receipt")
    return json.loads(raw)


def record_observation(folder, protocol, protocol_sha256, model_path, ranked_candidates,
                       source_sha256, source_first_seen_utc, decision_close_utc,
                       next_session_open_utc, first_recorded_utc, view="GLOBAL"):
    recorded, close, next_open = map(aware, (first_recorded_utc, decision_close_utc, next_session_open_utc))
    if close < aware(protocol["frozen_at_utc"]) or recorded < close or recorded >= next_open:
        raise ValueError("Backfilled/late observation cannot qualify as prospective")
    if fingerprint(Path(model_path).read_bytes()) != protocol["model_sha256"]:
        raise ValueError("Weights changed after the independent protocol freeze")
    if fingerprint(encoded(protocol)) != protocol_sha256:
        raise ValueError("Protocol identity changed")
    if view not in protocol["views"]:
        raise ValueError("View not declared before outcomes")
    if not source_sha256 or set(source_sha256) != set(source_first_seen_utc):
        raise ValueError("All source hashes and first-retrieval clocks required")
    for key, digest in source_sha256.items():
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Invalid source fingerprint")
        if aware(source_first_seen_utc[key]) > recorded:
            raise ValueError("Source acquired after the recorded decision")
    x = ranked_candidates.copy()
    if any(str(c).lower().startswith(("y6", "y12", "y24", "outcome", "integrity_y", "observed_2x")) for c in x):
        raise ValueError("Future outcome metadata cannot enter the decision cohort")
    if not {"isin", "symbol", "rank", "probability"}.issubset(x):
        raise ValueError("Complete source security identity and frozen ranks required")
    if x["isin"].duplicated().any() or x["rank"].duplicated().any():
        raise ValueError("Duplicate selection rank or cross-listed security")
    probabilities = pd.to_numeric(x["probability"], errors="coerce")
    if not probabilities.between(0, 1).all():
        raise ValueError("Invalid prospective model probabilities")
    if len(x) < 10 or sorted(x["rank"].tolist()) != list(range(1, len(x) + 1)):
        raise ValueError("Full decision-eligible ranking and at least ten stocks required")
    x = x.sort_values("rank")
    if not x["probability"].is_monotonic_decreasing:
        raise ValueError("Ranks disagree with predeclared probability ordering")
    rows = x.head(10)[["isin", "symbol", "rank", "probability"]].to_dict("records")
    record = {"protocol_sha256": protocol_sha256, "model_sha256": protocol["model_sha256"],
              "view": view, "decision_close_utc": close.isoformat(), "next_session_open_utc": next_open.isoformat(),
              "first_recorded_utc": recorded.isoformat(), "source_sha256": source_sha256,
              "source_first_seen_utc": source_first_seen_utc, "candidate_count": len(x),
              "full_candidate_ranking_sha256": fingerprint(x.to_json(orient="records", double_precision=15).encode()),
              "selected": rows, "outcome_status": "WAITING_FOR_126_ACTUAL_MARKET_SESSIONS"}
    date = close.tz_convert("Asia/Kolkata").date().isoformat()
    path = Path(folder) / "observations" / date / (view + ".json")
    immutable_write(path, encoded(record))
    return record


def assess_accessible_event(start_adjusted_close, future_prices, official_sessions, evaluation_cutoff_utc):
    """Assess only after 126 actual exchange sessions; missing rows stay unknown.

The caller supplies official split/bonus-adjusted prices and a separately
verified market calendar. Exchange presence/adjustment proofs remain required.
"""
    cutoff = aware(evaluation_cutoff_utc)
    sessions = pd.DatetimeIndex([aware(s) for s in official_sessions])
    if sessions.has_duplicates or not sessions.is_monotonic_increasing:
        raise ValueError("Official future session calendar is not unique and ordered")
    mature = len(sessions) >= HORIZON_SESSIONS and sessions[HORIZON_SESSIONS - 1] <= cutoff
    if not mature:
        return {"status": "NOT_MATURE", "observed_2x": None, "required_sessions": HORIZON_SESSIONS}
    if not math.isfinite(start_adjusted_close) or start_adjusted_close <= 0:
        raise ValueError("Invalid adjusted decision price")
    x = future_prices.copy()
    if not {"session_close_utc", "adjusted_close", "turnover_INR", "source_verified"}.issubset(x):
        raise ValueError("Original verified price and adjustment evidence required")
    x["session_close_utc"] = x["session_close_utc"].map(aware)
    if x["session_close_utc"].duplicated().any():
        raise ValueError("Duplicate future company session")
    x = x.set_index("session_close_utc").reindex(sessions[:HORIZON_SESSIONS])
    values = x[["adjusted_close", "turnover_INR"]].apply(pd.to_numeric, errors="coerce")
    good = (x["source_verified"].eq(True) & values["adjusted_close"].gt(0) & values["turnover_INR"].ge(0)
            & pd.Series(np.isfinite(values.to_numpy()).all(axis=1), index=x.index))
    if not good.all():
        return {"status": "UNKNOWN_MISSING_OR_UNVERIFIED_SESSIONS", "observed_2x": None,
                "missing_sessions": int((~good).sum())}
    hit = (values["adjusted_close"].rolling(3).median().ge(2 * start_adjusted_close) &
           values["turnover_INR"].rolling(3).mean().ge(5000000)).any()
    return {"status": "MATURE_ASSESSABLE", "observed_2x": int(hit),
            "maturity_utc": sessions[HORIZON_SESSIONS - 1].isoformat()}
