"""Fail-closed IST scheduler gate for original NSE daily V11.4 forward research.

Admits only Mon-Fri at or after 21:00 IST, same local exchange date. This
is a session *precondition*, not confirmation the NSE actually traded:
holidays, exchange disruptions, publishing lag and dual-bhavcopy identity
are independently validated later by the original NSE market source collector.
No future labels, no stock outcomes, no trader-signal overwrite.
"""
from __future__ import annotations
import argparse
import json
from datetime import datetime, time
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")
EARLIEST_IST = time(21, 0)
KNOWN_EVENTS = ("workflow_dispatch", "schedule")


def session_gate(now: datetime, event_name: str) -> dict:
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("Time must carry an explicit timezone for NSE schedule")
    if event_name not in KNOWN_EVENTS:
        raise ValueError("Unsupported V11.4 forward research trigger")
    ist = now.astimezone(IST)
    weekday = ist.weekday()  # Monday=0 ... Sunday=6
    if weekday >= 5:
        reason = "NON_TRADING_WEEKEND_IN_ASIA_KOLKATA"
    elif ist.time() < EARLIEST_IST:
        reason = "NSE_DUAL_ARCHIVE_EOD_PUBLICATION_SAFETY_WINDOW_NOT_REACHED"
    else:
        reason = "WEEKDAY_AFTER_IST_PUBLICATION_GUARD_STILL_REQUIRES_NSE_HOLIDAY_AND_DUAL_BHAVCOPY_CHECK"
    allowed = weekday < 5 and ist.time() >= EARLIEST_IST
    return {
        "gate_version": "v11_4_IST_20261010",
        "IST_date": ist.date().isoformat(),
        "IST_clock": ist.strftime("%H:%M"),
        "event": event_name,
        "weekday_index_Mon_0": weekday,
        "clock_guard_earliest_HHMM_IST": EARLIEST_IST.strftime("%H:%M"),
        "eligible_for_source_check_only": allowed,
        "reason": reason,
        "NSE_traded_on_day_confirmed_by_gate": False,
        "exchange_holiday_calendar_confirmed_by_gate": False,
        "original_2026_10_08_frozen_stock_picks_modified": False,
        "historical_accuracy_claim": None
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--event", choices=KNOWN_EVENTS, required=True)
    p.add_argument("--asof", help="Override for regression testing only: ISO-aware datetime")
    p.add_argument("--github-output", help="GitHub Actions step-output file")
    args = p.parse_args()
    now = datetime.fromisoformat(args.asof) if args.asof else datetime.now(tz=IST)
    report = session_gate(now, args.event)
    print(json.dumps(report, indent=2), flush=True)
    if args.github_output:
        with open(args.github_output, "a", encoding="utf-8") as f:
            f.write("run_live=" + ("true" if report["eligible_for_source_check_only"] else "false") + "\n")
            f.write("market_IST_date=" + report["IST_date"] + "\n")
            f.write("session_reason=" + report["reason"] + "\n")


if __name__ == "__main__":
    main()
