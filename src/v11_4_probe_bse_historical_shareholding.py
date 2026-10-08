"""Read-only BSE LODR Reg-31 historical shareholding INDEX availability probe.

Tests a five-company sample only. It does not promote models, alter backtests,
fabricate old ownership values, or treat a quarter-end date as the publication date.
Source API: https://api.bseindia.com/BseIndiaAPI/api/Corp_Shareholding_ng/w

Older company XBRL/HTML links are NOT parsed by this diagnostic.
"""
from __future__ import annotations

import csv
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import requests

BASE = "https://api.bseindia.com/BseIndiaAPI/api/Corp_Shareholding_ng/w"
SAMPLE = (
    ("RELIANCE", "500325"),
    ("TCS", "532540"),
    ("INFY", "500209"),
    ("SBIN", "500112"),
    ("HDFCBANK", "500180"),
)
YEARS = tuple(range(2017, 2022))
OUT = Path("outputs_v11_4_bse_historical_probe")
FIELDS = (
    "symbol", "bse_scripcode", "quarter_end", "filing_available_ts",
    "is_xbrl", "has_xbrl_url", "filing_url", "company_name",
)

def parse_timestamp(value):
    if not value:
        return None
    try:
        stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        for fmt in ("%b %d %Y %I:%M%p", "%b %d %Y  %I:%M%p", "%Y-%m-%d"):
            try:
                stamp = datetime.strptime(str(value).strip(), fmt)
                break
            except ValueError:
                continue
        else:
            return None
    if stamp.tzinfo is None:
        # BSE filing fields are Indian local time; pin to IST if lacking offset.
        from datetime import timedelta
        stamp = stamp.replace(tzinfo=timezone(timedelta(hours=5, minutes=30)))
    return stamp.astimezone(timezone.utc).isoformat()

def valid_row(row):
    end = parse_timestamp(row.get("EndDate") or row.get("DisplayDT"))
    available = parse_timestamp(row.get("D") or row.get("broadcastTime"))
    if not end or not available:
        return None
    end_day = end[:10]
    year = int(end_day[:4])
    if year not in YEARS:
        return None
    # Do not use a record with missing availability or publication predating quarter close.
    if available < end:
        return None
    attachment = str(row.get("XBRLAttachment") or "").strip()
    if attachment.startswith("/XBRLFILES/"):
        filing_url = "https://www.bseindia.com" + attachment
    else:
        filing_url = ""
    return {
        "quarter_end": end_day,
        "filing_available_ts": available,
        "is_xbrl": int(str(row.get("IsXBRL", "0")) == "1"),
        "has_xbrl_url": int(bool(filing_url)),
        "filing_url": filing_url,
        "company_name": str(row.get("Company_NAme") or ""),
    }

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sess = requests.Session()
    sess.headers.update({
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": "https://www.bseindia.com/",
        "Accept": "application/json, text/plain, */*",
    })
    # BSE is reportedly Origin-header sensitive; do not add that header.
    allrows = []
    diagnostics = []
    for symbol, code in SAMPLE:
        error = ""
        data = None
        response_info = {}
        for attempt in range(1, 3):
            try:
                res = sess.get(BASE, params={"scripcode": code, "flag": "0", "indtype": ""}, timeout=25)
                response_info = {"status": res.status_code, "content_type": res.headers.get("Content-Type", ""), "bytes": len(res.content)}
                res.raise_for_status()
                if "json" not in res.headers.get("Content-Type", "").lower():
                    raise ValueError("Unexpected non-JSON response (possible provider gate)")
                data = res.json()
                if not isinstance(data, dict) or not isinstance(data.get("Table"), list):
                    raise ValueError("Response lacks expected Table list")
                break
            except (requests.RequestException, ValueError) as exc:
                error = type(exc).__name__ + ": " + str(exc)[:160]
                if attempt == 1:
                    time.sleep(2)
        found = 0
        xbrl = 0
        if data is not None:
            for raw in data["Table"]:
                if not isinstance(raw, dict):
                    continue
                clean = valid_row(raw)
                if clean is None:
                    continue
                allrows.append({"symbol": symbol, "bse_scripcode": code, **clean})
                found += 1
                xbrl += int(clean["has_xbrl_url"])
        diagnostics.append({
            "symbol": symbol, "code": code, "valid_2017_2021_filing_rows": found,
            "valid_2017_2021_xbrl_links": xbrl, "error": error if data is None else "",
            **response_info,
        })
        print(f"BSE sample {symbol}: historical rows={found}, XBRL URLs={xbrl}, error={error if data is None else 'none'}", flush=True)
        time.sleep(1)

    counts = Counter(int(r["quarter_end"][:4]) for r in allrows)
    covered = defaultdict(set)
    for r in allrows:
        covered[int(r["quarter_end"][:4])].add(r["symbol"])
    report = {
        "sample_companies": len(SAMPLE),
        "sample_names": [x[0] for x in SAMPLE],
        "source": BASE,
        "years_checked": list(YEARS),
        "historical_filings_by_year": {str(y): counts[y] for y in YEARS},
        "companies_with_filings_by_year": {str(y): len(covered[y]) for y in YEARS},
        "total_historical_filings": len(allrows),
        "total_historical_xbrl_links": sum(r["has_xbrl_url"] for r in allrows),
        "successful_indexes": sum(not d["error"] for d in diagnostics),
        "endpoint_accessible_in_runner": any(not d["error"] for d in diagnostics),
        "historical_index_coverage_observed": any(int(r["quarter_end"][:4]) < 2022 for r in allrows),
        "pit_policy": "Keep original publication/broadcast time; never backdate filings to quarter end.",
        "scope": "Five-company read-only source pilot; not complete market backfill or validation.",
        "next_gate": "Resolve symbol/ISIN-to-BSE-scrip mapping, parse historical XBRL/PDF, quantify per-fold coverage before revisiting 12/18 gate.",
        "company_diagnostics": diagnostics,
    }
    with (OUT / "historical_filing_index.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(allrows)
    (OUT / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if not report["endpoint_accessible_in_runner"]:
        sys.exit("BSE historical endpoint not accessible: inspect response diagnostics")
    if not report["historical_index_coverage_observed"]:
        sys.exit("BSE historical availability not yet demonstrated")

if __name__ == "__main__":
    main()
