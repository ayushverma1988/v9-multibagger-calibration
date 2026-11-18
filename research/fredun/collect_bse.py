"""Fetch BSE day-end prices and issuer original documents for the Fredun review."""
import csv
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
import gzip
import io
import json
from pathlib import Path
import re

import collect_sources as sc

BSE_START = date(2025, 10, 1)
FIRST_NSE_DAY = date(2026, 8, 17)
EXTRA_DATES = {date(2026, 11, 17), date(2026, 11, 18), FIRST_NSE_DAY}
SAMPLE = {"20260622", "20260623", "20260715", "20260716", "20260814", "20260817", "20261117", "20261118"}

def bse_session(day):
    code = day.strftime("%Y%m%d")
    name = f"bse_{code}.csv"
    url = f"https://www.bseindia.com/download/BhavCopy/Equity/BhavCopy_BSE_CM_0_0_0_{code}_F_0000.CSV"
    body = sc.cached(name, url, keep=False)
    if body is None:
        return None
    try:
        reader = csv.DictReader(io.StringIO(body.decode("utf-8-sig")), skipinitialspace=True)
        required = {"TradDt", "Src", "Sgmt", "ISIN", "FinInstrmId", "OpnPric",
                    "HghPric", "LwPric", "ClsPric", "TtlTradgVol"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Unexpected BSE field definitions")
        selected = [row for row in reader if row["ISIN"] == sc.ISIN or row["FinInstrmId"] == sc.CODE]
        if len(selected) != 1:
            raise ValueError(f"{len(selected)} candidate security rows")
        row = selected[0]
        if (row["ISIN"], row["FinInstrmId"], row["TradDt"], row["Src"], row["Sgmt"]) != (
                sc.ISIN, sc.CODE, day.isoformat(), "BSE", "CM"):
            raise ValueError("Unexpected BSE identity, source or date")
        if code in SAMPLE:
            with gzip.open(sc.OUT / (name + ".gz"), "wb") as f:
                f.write(body)
        return {"date": day.isoformat(), "file": name, "row": row}
    except (KeyError, ValueError, UnicodeError) as exc:
        print("BSE_UNAVAILABLE", code, str(exc)[:100], flush=True)
        return None

def bse_history():
    dates = [
        FIRST_NSE_DAY - timedelta(days=n)
        for n in range(1, (FIRST_NSE_DAY - BSE_START).days + 1)
    ]
    dates = sorted({d for d in dates if d.weekday() < 5} | EXTRA_DATES, reverse=True)
    # Establish access and schema on an anchor before dispatching the historical requests.
    first = bse_session(dates.pop(0))
    with ThreadPoolExecutor(max_workers=5) as pool:
        rows = [r for r in pool.map(bse_session, dates) if r is not None]
    if first:
        rows.insert(0, first)
    (sc.OUT / "bse_official_history.json").write_text(json.dumps(rows, indent=2))
    print("BSE_OFFICIAL_ROWS", len(rows), min((r["date"] for r in rows), default=None), flush=True)

def filing_originals():
    attachment = "https://www.bseindia.com/xml-data/corpfiling/"
    docs = {
        "annual_FY26.pdf": attachment+"AttachHis/3be196be-c2ec-415f-b849-4f8308943486.pdf",
        "annual_FY25.pdf": attachment+"AttachHis/5674f15d-d757-44a1-9382-ce8400f34057.pdf",
        "monitoring_20261113.pdf": attachment+"AttachLive/3c4d8d6d-b61a-4930-af42-2c1d41dcbb03.pdf",
        "deviation_20261114.pdf": attachment+"AttachLive/82b73c03-c836-4dd6-aa5f-a619e45d8e02.pdf",
        "warrant_20261114.pdf": attachment+"AttachLive/af90cf5e-165e-41c6-87fd-624eeffa215a.pdf",
        "earnings_call_ppt.pdf": attachment+"AttachHis/b31900a9-fef0-46cc-972d-14cd6d12cd89.pdf",
        "bonus_notice.html":"https://www.bseindia.com/markets/MarketInfo/DispNewNoticesCirculars.aspx?page=20260714-22",
        "research_revised.html":"https://www.bazaarwatch.com/announcement/291391/fredun-pharmaceuticals-ltd-revised-outcome-of-the-board-meeting-dated-05th-november-2026",
        "research_original.html":"https://www.bazaarwatch.com/announcement/288333/fredun-pharmaceuticals-ltd-outcome-of-the-board-meeting-dated-05th-november-2026",
        "research_icici.html":"https://www.icicidirect.com/research/equity/rapid-results/fredun-pharmaceuticals-ltd",
    }
    with ThreadPoolExecutor(max_workers=3) as pool:
        outputs = list(pool.map(lambda rec: sc.cached(*rec), docs.items()))
    for label, raw in zip(docs, outputs):
        if raw is not None:
            print("FILING", label, len(raw), raw.startswith(b"%PDF"), flush=True)
        if label == "research_revised.html" and raw is not None:
            assets = sorted(set(re.findall(rb"[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}\.pdf", raw)))
            for name in assets[:8]:
                pic = name.decode()
                sc.cached("bse_revision_candidate_" + pic, attachment+"AttachLive/"+pic)

if __name__ == "__main__":
    sc.OUT = Path("fredun_bse_capture")
    sc.OUT.mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        for _ in pool.map(lambda fn: fn(), (bse_history, filing_originals)):
            pass
    (sc.OUT / "source_receipts.json").write_text(json.dumps(sc.RECEIPTS, indent=2, sort_keys=True))
