from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import requests

BASE = "https://www.nseindia.com"
API = BASE + "/api/corporate-announcements"

HEADERS = {
    "user-agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/129.0 Safari/537.36"
    ),
    "accept-language": "en-US,en;q=0.9",
    "accept": "application/json,text/plain,*/*",
    "referer": BASE + "/companies-listing/corporate-filings-announcements",
}

EVENT_RULES = [
    ("order_win", ["order", "contract", "letter of award", "purchase order", "work order"], 1),
    ("capacity_expansion", ["capacity", "commissioning", "new plant", "expansion", "commercial production"], 1),
    ("debt_reduction", ["debt reduction", "repayment", "prepayment", "deleveraging"], 1),
    ("credit_rating", ["credit rating", "rating upgrade", "rating downgrade", "credit watch", "outlook"], 0),
    ("promoter_activity", ["promoter", "insider trading", "insider transaction"], 0),
    ("pledge_change", ["pledge", "encumbrance"], 0),
    ("dilution", ["preferential", "warrant", "qip", "rights issue", "allotment"], -1),
    ("buyback", ["buyback", "share repurchase"], 1),
    ("management_change", ["change in management", "appointment", "resignation", "key managerial"], 0),
    ("auditor_change", ["auditor", "qualified opinion"], 0),
    ("regulatory", ["penalty", "regulatory", "show cause", "inspection", "sebi order"], -1),
    ("litigation", ["litigation", "court order", "arbitration"], -1),
    ("customer_supplier", ["customer", "supplier", "client"], 0),
    ("earnings", ["financial results", "results", "earnings", "guidance", "profit warning"], 0),
    ("corporate_action", ["split", "bonus", "merger", "demerger", "scheme of arrangement"], 0),
]


def request_session() -> requests.Session:
    s = requests.Session()
    r = s.get(BASE, headers=HEADERS, timeout=45)
    r.raise_for_status()
    return s


def fetch_month(year: int, month: int, retries: int = 5):
    start = pd.Timestamp(year=year, month=month, day=1)
    end = start + pd.offsets.MonthEnd(1)
    params = {
        "index": "equities",
        "from_date": start.strftime("%d-%m-%Y"),
        "to_date": end.strftime("%d-%m-%Y"),
    }

    last = None
    for attempt in range(1, retries + 1):
        try:
            s = request_session()
            r = s.get(API, params=params, headers=HEADERS, timeout=90)
            if r.status_code in {401, 403, 429}:
                raise RuntimeError(f"NSE HTTP {r.status_code}")
            r.raise_for_status()
            data = r.json()
            if isinstance(data, dict) and "data" in data:
                data = data["data"]
            if not isinstance(data, list):
                raise RuntimeError(f"unexpected payload type {type(data).__name__}")
            return data, {
                "from_date": params["from_date"],
                "to_date": params["to_date"],
                "status_code": r.status_code,
                "attempt": attempt,
            }
        except Exception as exc:
            last = repr(exc)
            time.sleep(min(30, 2 ** attempt))
    raise RuntimeError(f"failed after {retries} attempts: {last}")


def first(row: dict, *names):
    for n in names:
        v = row.get(n)
        if v is not None and str(v).strip() not in {"", "nan", "None"}:
            return v
    return None


def parse_ts(v):
    if v is None:
        return pd.NaT
    # NSE announcement timestamps are exchange-local (India) when no timezone
    # is supplied. Preserve that fact, then normalize to UTC.
    t = pd.to_datetime(v, errors="coerce", dayfirst=True)
    if pd.isna(t):
        return pd.NaT
    if getattr(t, "tzinfo", None) is None:
        t = t.tz_localize("Asia/Kolkata")
    return t.tz_convert("UTC")


def classify(subject: str, details: str):
    text = (str(subject) + " " + str(details)).lower()
    for event_type, needles, direction in EVENT_RULES:
        if any(n in text for n in needles):
            # Direction here is a prior semantic tag, not a learned return
            # label. Ambiguous families remain neutral.
            return event_type, float(direction), 0.5
    return "other", 0.0, 0.25


def deterministic_id(source_record_id: str, published_ts, symbol: str, subject: str):
    raw = "|".join([
        "NSE",
        str(source_record_id),
        str(published_ts),
        str(symbol),
        str(subject),
    ])
    return hashlib.sha256(raw.encode("utf-8", errors="ignore")).hexdigest()


def normalize(rows: list[dict]) -> pd.DataFrame:
    out = []
    for r in rows:
        symbol = str(first(r, "symbol", "sm_symbol") or "").strip().upper()
        isin = first(r, "sm_isin", "isin", "ISIN")
        isin = str(isin).strip().upper() if isin is not None else None
        if isin in {"", "NAN", "NONE", "NULL"}:
            isin = None

        subject = str(first(r, "desc", "subject", "SUBJECT") or "").strip()
        details = str(first(r, "attchmntText", "details", "DETAILS") or "").strip()
        published = parse_ts(first(r, "an_dt", "broadcastDateTime", "sort_date", "dt"))
        if pd.isna(published):
            continue

        recid = first(r, "seq_id", "csvName", "bflag", "orgid")
        if recid is None:
            recid = deterministic_id("", published, symbol, subject)

        attach = first(r, "attchmntFile", "attachment", "ATTACHMENT")
        attach_url = urljoin(BASE + "/", str(attach)) if attach else None

        event_type, direction, strength = classify(subject, details)
        sec_key = "ISIN:" + isin if isin else "SYM:" + symbol

        raw_text = (subject + "\n" + details).strip()
        out.append({
            "event_id": deterministic_id(str(recid), published, symbol, subject),
            "published_ts": published,
            "source": "NSE",
            "source_record_id": str(recid),
            "symbol": symbol,
            "isin": isin,
            "security_key": sec_key,
            "company_name": first(r, "sm_name", "companyName", "COMPANY NAME"),
            "industry_raw": first(r, "smIndustry", "industry"),
            "headline": subject,
            "details": details,
            "event_type": event_type,
            "event_direction": direction,
            "event_strength": strength,
            "document_url": attach_url,
            "raw_text_hash": hashlib.sha256(raw_text.encode("utf-8", errors="ignore")).hexdigest(),
            "exchange_received_ts": parse_ts(first(r, "exchdisstime", "exchangeReceivedTime")),
            "source_json": json.dumps(r, ensure_ascii=False, default=str),
        })

    d = pd.DataFrame(out)
    if d.empty:
        return d
    d = d.sort_values(["published_ts", "source_record_id", "event_id"])
    d = d.drop_duplicates(["source", "source_record_id"], keep="first")
    return d.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--month", type=int, required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows, meta = fetch_month(args.year, args.month)
    raw_path = out / "raw_announcements.json"
    raw_path.write_text(json.dumps(rows, ensure_ascii=False, default=str), encoding="utf-8")

    d = normalize(rows)
    if len(d):
        d.to_parquet(out / "events.parquet", index=False)
        d.to_csv(out / "events.csv", index=False)
    else:
        # Keep an explicit empty checkpoint rather than treating an empty month
        # as a failed/incomplete job.
        pd.DataFrame(columns=[
            "event_id","published_ts","source","source_record_id","symbol","isin",
            "security_key","headline","event_type","event_direction",
            "event_strength","document_url","raw_text_hash"
        ]).to_parquet(out / "events.parquet", index=False)

    report = {
        "year": args.year,
        "month": args.month,
        "raw_rows": int(len(rows)),
        "normalized_rows": int(len(d)),
        "symbols": int(d["symbol"].nunique()) if len(d) else 0,
        "isins": int(d["isin"].nunique()) if len(d) else 0,
        "start": str(d["published_ts"].min()) if len(d) else None,
        "end": str(d["published_ts"].max()) if len(d) else None,
        "event_type_counts": d["event_type"].value_counts(dropna=False).to_dict() if len(d) else {},
        "fetch": meta,
    }
    (out / "coverage.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
