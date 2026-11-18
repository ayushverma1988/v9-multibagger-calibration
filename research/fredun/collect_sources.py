"""Capture dated exchange and independent source records for Fredun Pharmaceuticals."""
import csv
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import threading
from urllib.parse import urlsplit
import zipfile

import requests

OUT = Path("fredun_capture")
SYMBOL, ISIN, CODE = "FREDUN", "INE194R01017", "539730"
ASOF = date(2026, 11, 18)
NSE = "https://www.nseindia.in"
BSE_API = "https://api.bseindia.com/BseIndiaAPI/api"
STOPS = set()
LOCK = threading.Lock()
TLS = threading.local()
RECEIPTS = {}

def cached(name, url, headers=None, keep=True):
    host = urlsplit(url).hostname
    with LOCK:
        if host in STOPS:
            return None
    proof = {"url": url, "observed_utc": datetime.now(timezone.utc).isoformat()}
    try:
        if not hasattr(TLS, "session"):
            TLS.session = requests.Session()
            TLS.session.max_redirects = 3
            TLS.session.headers["User-Agent"] = "Mozilla/5.0"
        with TLS.session.get(url, headers=headers, timeout=(7, 16), stream=True) as result:
            proof["status"] = result.status_code
            proof["resolved_url"] = result.url
            proof["content_type"] = result.headers.get("Content-Type")
            if result.status_code in {401, 403, 429}:
                with LOCK:
                    STOPS.add(host)
            result.raise_for_status()
            if urlsplit(result.url).hostname not in {
                host, "www.nseindia.com", "www.nseindia.in", "nsearchives.nseindia.com",
                "nsearchives.nseindia.in", "www.bseindia.com", "bseindia.com", "www.fredungroup.com",
                "fredungroup.com",
            }:
                raise ValueError("The provider redirected outside the source")
            pieces = []
            size = 0
            for chunk in result.iter_content(64 * 1024):
                size += len(chunk)
                if size > 24_000_000:
                    raise ValueError("Source exceeded request size limit")
                pieces.append(chunk)
            body = b"".join(pieces)
            proof.update(bytes=len(body), sha256=hashlib.sha256(body).hexdigest())
            if keep:
                (OUT / name).write_bytes(body)
            return body
    except (ValueError, requests.RequestException) as error:
        proof["error"] = str(error)[:180]
        return None
    finally:
        with LOCK:
            RECEIPTS[name] = proof

def nse_history():
    head = {"Referer": NSE + "/", "Accept": "application/json"}
    for lower, upper in [("18-11-2025", "18-11-2026"), ("01-04-2026", "18-11-2026")]:
        suffix = f"symbol={SYMBOL}&series=%5B%22EQ%22%5D&from={lower}&to={upper}"
        cached(f"nse_history_{lower}.json", NSE + "/api/historical/cm/equity?" + suffix, head)
    cached("nse_actions.json", NSE + "/api/corporates-corporateActions?index=equities&symbol=FREDUN&from_date=01-01-2024&to_date=18-11-2026", head)
    cached("nse_announcements.json", NSE + "/api/corporate-announcements?index=equities&symbol=FREDUN&from_date=01-04-2026&to_date=18-11-2026", head)

def provider_history():
    for venue in ("NS", "BO"):
        for provider in ("query1", "query2"):
            url = f"https://{provider}.finance.yahoo.com/v8/finance/chart/{SYMBOL}.{venue}?interval=1d&range=5y&events=div%2Csplits"
            raw = cached(f"{provider}_{venue}.json", url)
            if raw:
                try:
                    c = json.loads(raw)["chart"]["result"][0]
                    print("YAHOO", venue, provider, c["meta"].get("symbol"), len(c.get("timestamp",[])), flush=True)
                    break
                except (IndexError, KeyError, TypeError, ValueError):
                    continue
    cached("stockanalysis_nse.html", "https://stockanalysis.com/quote/nse/FREDUN/history/?p=1y")
    cached("stockanalysis_bse.html", "https://stockanalysis.com/quote/bom/539730/history/?p=1y")
    cached("screener_consolidated.html", "https://www.screener.in/company/FREDUN/consolidated/")
    cached("ft_nse_history.html","https://markets.ft.com/data/equities/tearsheet/historical?s=FREDUN%3ANSI")

def official_bar(day):
    date_id = day.strftime("%Y%m%d")
    name = f"nse_{date_id}.zip"
    link = f"https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_{date_id}_F_0000.csv.zip"
    raw = cached(name, link, keep=False)
    if not raw:
        return None
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            members = [f for f in z.namelist() if f.lower().endswith(".csv")]
            if len(members) != 1:
                raise ValueError("Unexpected file format")
            data = list(csv.DictReader(io.StringIO(z.read(members[0]).decode("utf-8-sig")), skipinitialspace=True))
        if not data or data[0]["Src"] != "NSE" or data[0]["TradDt"] != day.isoformat():
            raise ValueError("Unexpected day or venue")
        selected = [r for r in data if r["ISIN"] == ISIN or r["TckrSymb"] == SYMBOL]
        if day.isoformat() in {"2026-11-17", "2026-11-18"} or selected:
            (OUT / name).write_bytes(raw)
        return {"date": day.isoformat(), "rows": selected, "file": name, "total": len(data)}
    except (KeyError, UnicodeError, ValueError, zipfile.BadZipFile) as exc:
        print("INVALID_NSE", date_id, str(exc)[:100], flush=True)
        return None

def official_history():
    start = date(2026, 3, 2)
    dates = [ASOF - timedelta(days=n) for n in range((ASOF - start).days + 1)]
    dates = [d for d in dates if d.weekday() < 5 or d == date(2026, 11, 8)]
    with ThreadPoolExecutor(max_workers=6) as pool:
        items = [x for x in pool.map(official_bar, dates) if x is not None]
    (OUT / "nse_official_history.json").write_text(json.dumps(items, indent=2))
    has_fredun = [x for x in items if x["rows"]]
    print("OFFICIAL", len(items), "sessions:", len(has_fredun), "Fredun sessions", 
          min((x["date"] for x in has_fredun), default=None), max((x["date"] for x in has_fredun), default=None), flush=True)

def bse_filings():
    head = {"Referer": "https://www.bseindia.com/", "Origin": "https://www.bseindia.com", "Accept": "application/json"}
    docs = {}
    windows = [("20261101", "20261118"), ("20261001", "20261031"), ("20260901", "20260930"),
               ("20260801", "20260831"), ("20260701", "20260731"), ("20260601", "20260630"),
               ("20260501", "20260531"), ("20260201", "20260228")]
    for low, high in windows:
        url = f"{BSE_API}/AnnSubCategoryGetData/w?Pageno=1&strCat=-1&strPrevDate={low}&strScrip={CODE}&strSearch=P&strToDate={high}&strType=C&subcategory=-1"
        raw = cached(f"bse_ann_{high}.json", url, head)
        if raw is None:
            continue
        try:
            records = json.loads(raw).get("Table", [])
        except (AttributeError, ValueError):
            continue
        for obj in records:
            headline = str(obj.get("HEADLINE", "")) + " " + str(obj.get("NEWSSUB", ""))
            target = str(obj.get("ATTACHMENTNAME", ""))
            if re.fullmatch(r"[a-fA-F0-9-]{30,}\.pdf", target) and re.search(
                    r"financial result|outcome of board|annual report|integrated filing|press release|business update|preferential|split|bonus|acquisiti", headline, re.I):
                docs[target] = (high, headline)
    for name, (high, headline) in list(docs.items())[:32]:
        raw = cached(f"bse_{high}_{name}", "https://www.bseindia.com/xml-data/corpfiling/AttachLive/" + name)
        if raw is not None and not raw.startswith(b"%PDF"):
            print("BSE_NONPDF", name, flush=True)
    print("BSE_DOCUMENTS", len(docs), flush=True)
    for endpoint, query, label in [
        ("StockReachGraph/w", f"scripcode={CODE}&flag=1Y&fromdate=&todate=&seriesid=", "history"),
        ("CorporateAction/w", f"scripcode={CODE}", "actions"),
        ("ComHeadernew/w", f"scripcode={CODE}&seriesid=", "profile"),
    ]:
        cached(f"bse_{label}.json", f"{BSE_API}/{endpoint}?{query}", head)

def additional():
    vals = {
        "issuer.html":"https://www.fredungroup.com/",
        "quarter_filing.html":"https://financialfilings.com/filings/fredun-pharmaceuticals-limited/interim-quarterly-report/2026/75641225/",
        "quarter_correction.html":"https://financialfilings.com/filings/fredun-pharmaceuticals-limited/interim-quarterly-report/2026/75642320/",
        "annual_filing.html":"https://financialfilings.com/filings/fredun-pharmaceuticals-limited/annual-report/2026/47996437/",
        "results_release.html":"https://financialfilings.com/filings/fredun-pharmaceuticals-limited/earnings-release/2026/75760644/",
        "bse_board_filing.pdf":"https://www.bseindia.com/xml-data/corpfiling/AttachLive/d8af6f66-5ff5-4d04-8ed9-cfec5ea3ed18.pdf",
    }
    for label, url in vals.items():
        cached(label, url)

if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=5) as pool:
        for f in pool.map(lambda fn: fn(), (nse_history, provider_history, official_history, bse_filings, additional)):
            pass
    (OUT / "source_receipts.json").write_text(json.dumps(RECEIPTS, indent=2, sort_keys=True))
    pass_count = sum("sha256" in x for x in RECEIPTS.values())
    print("COLLECTION_RECEIPT", pass_count, "of", len(RECEIPTS), "responses captured", flush=True)
