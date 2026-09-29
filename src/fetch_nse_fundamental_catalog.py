from __future__ import annotations

import argparse
import json
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests


BASE = "https://www.nseindia.com"
PAGE = BASE + "/companies-listing/corporate-filings-financial-results"
LEGACY = BASE + "/api/corporates-financial-results"
INTEGRATED = BASE + "/api/integrated-filing-results"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/134.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json,text/plain,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": PAGE,
    "X-Requested-With": "XMLHttpRequest",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    r = s.get(PAGE, timeout=30)
    r.raise_for_status()
    return s


def _get_json(s: requests.Session, url: str, params: dict[str, Any], retries=4):
    last = None
    for i in range(retries):
        try:
            r = s.get(url, params=params, timeout=45)
            if r.status_code in (401, 403):
                s.get(PAGE, timeout=30)
                time.sleep(1.0 + i)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            last = exc
            time.sleep(1.0 + 1.5 * i)
    raise RuntimeError(f"NSE request failed after {retries} attempts: {url}: {last}")


def _records(obj: Any) -> list[dict]:
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if not isinstance(obj, dict):
        return []

    for key in (
        "data",
        "records",
        "rows",
        "result",
        "results",
        "tableData",
        "integratedFiling",
    ):
        v = obj.get(key)
        if isinstance(v, list):
            return [x for x in v if isinstance(x, dict)]
        if isinstance(v, dict):
            z = _records(v)
            if z:
                return z

    # Some NSE APIs return several top-level arrays.
    out = []
    for v in obj.values():
        if isinstance(v, list) and v and isinstance(v[0], dict):
            out.extend(v)
    return out


def _first(d: dict, *keys):
    low = {str(k).lower(): v for k, v in d.items()}
    for k in keys:
        if k in d and d[k] not in (None, "", "-"):
            return d[k]
        v = low.get(str(k).lower())
        if v not in (None, "", "-"):
            return v
    return None


def _parse_dt(x):
    if x in (None, "", "-"):
        return pd.NaT
    return pd.to_datetime(x, errors="coerce", dayfirst=True, utc=True)


def _norm(row: dict, source: str) -> dict:
    symbol = _first(row, "symbol", "sm_symbol", "nseSymbol")
    company = _first(row, "companyName", "company", "sm_name", "issuer")
    period_end = _first(
        row,
        "toDate",
        "periodEnded",
        "period_ended",
        "quarterEndDate",
        "re_to_dt",
    )
    broadcast = _first(
        row,
        "broadCastDate",
        "broadcastDate",
        "broadcast_date",
        "sort_date",
        "filingDate",
        "submissionDate",
    )
    xbrl = _first(row, "xbrl", "xbrlLink", "xbrl_url", "xbrlFile", "file_url")
    detail = _first(row, "resultDetailedDataLink", "resultDetailLink", "detailedDataLink")
    scope = _first(row, "consolidated", "consolidatedOrStandalone", "typeOfResult")
    audited = _first(row, "audited", "auditedUnaudited", "auditStatus")
    relating = _first(row, "relatingTo", "period", "quarter", "filing_type")
    filing_id = _first(row, "id", "filingId", "seqId", "fileName", "file_name")

    return {
        "source": source,
        "symbol": str(symbol).upper().strip() if symbol is not None else None,
        "company": company,
        "period_end": pd.to_datetime(period_end, errors="coerce", dayfirst=True),
        "broadcast_ts": _parse_dt(broadcast),
        "statement_scope": scope,
        "audited": audited,
        "relating_to": relating,
        "filing_id": filing_id,
        "xbrl_url": xbrl,
        "detail_url": detail,
        "raw_json": json.dumps(row, ensure_ascii=False, default=str),
    }


def fetch_integrated(
    s: requests.Session,
    from_date: date,
    to_date: date,
    page_size=200,
    max_pages=200,
) -> list[dict]:
    out = []
    for page in range(1, max_pages + 1):
        params = {
            "type": "Integrated Filing- Financials",
            "index": "equities",
            "from_date": from_date.strftime("%d-%m-%Y"),
            "to_date": to_date.strftime("%d-%m-%Y"),
            "page": page,
            "size": page_size,
        }
        obj = _get_json(s, INTEGRATED, params)
        rec = _records(obj)
        if not rec:
            break
        out.extend(_norm(x, "nse_integrated") for x in rec)
        if len(rec) < page_size:
            break
    return out


def fetch_legacy(
    s: requests.Session,
    from_date: date,
    to_date: date,
    periods=("Quarterly", "Annual"),
) -> list[dict]:
    out = []
    for period in periods:
        params = {
            "index": "equities",
            "period": period,
            "from_date": from_date.strftime("%d-%m-%Y"),
            "to_date": to_date.strftime("%d-%m-%Y"),
        }
        try:
            obj = _get_json(s, LEGACY, params)
        except Exception as exc:
            print(f"warning: legacy {period} fetch failed: {exc}", flush=True)
            continue
        out.extend(_norm(x, f"nse_legacy_{period.lower()}") for x in _records(obj))
    return out


def fetch_catalog(from_date: date, to_date: date) -> pd.DataFrame:
    s = _session()
    rows = []
    rows.extend(fetch_integrated(s, from_date, to_date))
    rows.extend(fetch_legacy(s, from_date, to_date))
    if not rows:
        return pd.DataFrame(
            columns=[
                "source", "symbol", "company", "period_end", "broadcast_ts",
                "statement_scope", "audited", "relating_to", "filing_id",
                "xbrl_url", "detail_url", "raw_json",
            ]
        )
    df = pd.DataFrame(rows)
    df = df.dropna(subset=["symbol", "broadcast_ts"]).copy()
    df = df.sort_values(["broadcast_ts", "symbol", "source"])
    df = df.drop_duplicates(
        ["symbol", "period_end", "broadcast_ts", "source", "xbrl_url"],
        keep="last",
    ).reset_index(drop=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--to-date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--output", required=True)
    ap.add_argument("--require-records", action="store_true")
    args = ap.parse_args()

    f = datetime.strptime(args.from_date, "%Y-%m-%d").date()
    t = datetime.strptime(args.to_date, "%Y-%m-%d").date()
    if f > t:
        raise SystemExit("from-date must be <= to-date")

    df = fetch_catalog(f, t)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() in {".parquet", ".pq"}:
        df.to_parquet(out, index=False)
    else:
        df.to_csv(out, index=False)

    stats = {
        "from_date": str(f),
        "to_date": str(t),
        "rows": int(len(df)),
        "symbols": int(df["symbol"].nunique()) if len(df) else 0,
        "with_xbrl": int(df["xbrl_url"].notna().sum()) if len(df) else 0,
        "sources": df["source"].value_counts().to_dict() if len(df) else {},
    }
    print(json.dumps(stats, indent=2))
    if args.require_records and df.empty:
        raise SystemExit("No NSE financial filing records returned")


if __name__ == "__main__":
    main()
