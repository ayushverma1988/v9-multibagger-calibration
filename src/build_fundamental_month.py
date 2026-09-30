from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import tarfile
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import fetch_nse_fundamental_catalog as capi
import legacy_financial_normalizer as legacy
import xbrl_normalizer as xn


DOWNLOAD_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/134 Safari/537.36",
    "Accept": "application/xml,text/xml,text/html,application/xhtml+xml,application/zip,*/*",
    "Referer": "https://www.nseindia.com/",
}


def _real_xbrl(v) -> bool:
    s = str(v or "").strip()
    return (
        s not in {"", "-", "nan", "None"}
        and not s.endswith("/-")
        and "/xbrl/-" not in s.lower()
    )


def _detail_link(raw_json: str):
    try:
        d = json.loads(raw_json)
    except Exception:
        return None
    for k in (
        "resultDetailedDataLink",
        "resultDetailDataLink",
        "resultDetailedLink",
        "financialResultLink",
    ):
        v = d.get(k)
        if v and str(v).strip() not in {"-", "nan", "None"}:
            return str(v).strip()
    return None


def _resolve(v: str) -> str:
    v = str(v).strip()
    if v.startswith("//"):
        return "https:" + v
    if v.startswith("http://") or v.startswith("https://"):
        return v
    return capi.BASE.rstrip("/") + "/" + v.lstrip("/")


def _fetch(url: str, timeout=25, retries=2):
    last = None
    for i in range(retries):
        try:
            r = requests.get(url, headers=DOWNLOAD_HEADERS, timeout=timeout)
            r.raise_for_status()
            if not r.content:
                raise ValueError("empty response")
            return r.content, r.headers.get("content-type", "")
        except Exception as exc:
            last = exc
    raise RuntimeError(f"{url}: {last}")


def _xml_payloads(blob: bytes, ctype: str):
    if blob[:2] == b"PK" or "zip" in str(ctype).lower():
        with zipfile.ZipFile(io.BytesIO(blob)) as zf:
            for name in zf.namelist():
                if name.lower().endswith((".xml", ".xbrl", ".xhtml", ".html", ".htm")):
                    yield name, zf.read(name)
    else:
        yield "instance.xml", blob


def _safe_file(symbol: str, url: str, suffix: str):
    h = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
    s = re.sub(r"[^A-Za-z0-9_.-]", "_", str(symbol))[:30]
    return f"{s}_{h}{suffix}"


def prepare_catalog(year: int, month: int) -> pd.DataFrame:
    start = pd.Timestamp(year=year, month=month, day=1)
    end = start + pd.offsets.MonthEnd(1)
    d = capi.fetch_catalog(start.date(), end.date())
    if d.empty:
        return d

    d = d.copy()
    d["detail_url"] = d["raw_json"].map(_detail_link)
    d["fetch_mode"] = np.where(
        d["xbrl_url"].map(_real_xbrl),
        "xbrl",
        np.where(d["detail_url"].notna(), "legacy_html", "none"),
    )
    d["fetch_url"] = np.where(
        d["fetch_mode"] == "xbrl",
        d["xbrl_url"],
        d["detail_url"],
    )
    d["fetch_url"] = d["fetch_url"].map(
        lambda x: _resolve(x) if pd.notna(x) and str(x) not in {"", "None", "nan"} else None
    )

    # The legacy endpoint may surface the same filing through both Quarterly
    # and Annual queries. Fetch a concrete archive URL only once.
    d = d.sort_values(["broadcast_ts", "symbol", "source"]).drop_duplicates(
        ["symbol", "broadcast_ts", "fetch_url"],
        keep="last",
    )
    return d.reset_index(drop=True)


def _process_one(rd: dict, raw_dir: Path, mapping: dict):
    mode = rd["fetch_mode"]
    url = rd.get("fetch_url")
    if mode == "none" or not url:
        raise RuntimeError("no usable archive URL")

    blob, ctype = _fetch(url)
    suffix = ".html" if mode == "legacy_html" else ".bin"
    raw_name = _safe_file(rd.get("symbol"), url, suffix)
    (raw_dir / raw_name).write_bytes(blob)

    if mode == "legacy_html":
        row = legacy.parse_legacy_result_html(
            blob,
            symbol=rd.get("symbol"),
            broadcast_ts=rd.get("broadcast_ts"),
            source_url=url,
            source=rd.get("source"),
            fallback_period_end=rd.get("period_end"),
            fallback_scope=rd.get("statement_scope"),
            filing_id=rd.get("filing_id"),
        )
        row["archive_member"] = raw_name
        row["raw_archive_file"] = f"raw/{raw_name}"
        return [row]

    meta = {
        "symbol": rd.get("symbol"),
        "broadcast_ts": rd.get("broadcast_ts"),
        "statement_scope": rd.get("statement_scope"),
        "filing_id": rd.get("filing_id"),
        "source_url": url,
        "xbrl_url": url,
        "source": rd.get("source"),
    }
    rows = []
    for member, payload in _xml_payloads(blob, ctype):
        z = xn.normalize_xbrl(payload, meta, mapping)
        for row in z:
            row["archive_member"] = member
            row["raw_archive_file"] = f"raw/{raw_name}"
        rows.extend(z)
    if not rows:
        raise RuntimeError("XBRL normalized to zero rows")
    return rows


def coverage(catalog, norm, errors):
    fields = [
        "revenue", "operating_profit", "pbt", "pat", "finance_cost",
        "total_assets", "total_equity", "total_debt",
        "current_assets", "current_liabilities",
        "operating_cash_flow", "capex", "shares_outstanding",
    ]
    return {
        "catalog_rows": int(len(catalog)),
        "unique_fetch_urls": int(catalog["fetch_url"].nunique()) if len(catalog) else 0,
        "xbrl_rows": int((catalog["fetch_mode"] == "xbrl").sum()) if len(catalog) else 0,
        "legacy_html_rows": int((catalog["fetch_mode"] == "legacy_html").sum()) if len(catalog) else 0,
        "unusable_rows": int((catalog["fetch_mode"] == "none").sum()) if len(catalog) else 0,
        "normalized_rows": int(len(norm)),
        "normalized_symbols": int(norm["symbol"].nunique()) if len(norm) else 0,
        "errors": int(len(errors)),
        "success_fraction": float(
            (len(catalog) - len(errors)) / len(catalog)
        ) if len(catalog) else 0.0,
        "taxonomy_families": norm["taxonomy_family"].value_counts(dropna=False).to_dict() if len(norm) else {},
        "field_coverage": {
            c: float(norm[c].notna().mean()) if len(norm) and c in norm else 0.0
            for c in fields
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--month", type=int, required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--mapping", default="config/xbrl_concept_map_v9_5.json")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    outdir = Path(args.output_dir)
    raw_dir = outdir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    mapping = xn.load_map(args.mapping)

    cat = prepare_catalog(args.year, args.month)
    cat.to_parquet(outdir / "catalog.parquet", index=False)
    print(
        json.dumps({
            "year": args.year,
            "month": args.month,
            "catalog_rows": int(len(cat)),
            "modes": cat["fetch_mode"].value_counts().to_dict() if len(cat) else {},
        }, indent=2),
        flush=True,
    )

    rows = []
    errs = []

    work = [
        r._asdict()
        for r in cat.itertuples(index=False)
        if r.fetch_mode != "none"
    ]
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        fut = {
            ex.submit(_process_one, rd, raw_dir, mapping): rd
            for rd in work
        }
        for n, f in enumerate(as_completed(fut), start=1):
            rd = fut[f]
            try:
                rows.extend(f.result())
            except Exception as exc:
                errs.append({
                    "symbol": rd.get("symbol"),
                    "broadcast_ts": rd.get("broadcast_ts"),
                    "fetch_mode": rd.get("fetch_mode"),
                    "fetch_url": rd.get("fetch_url"),
                    "error": repr(exc),
                })
            if n % 100 == 0:
                print(
                    f"processed={n}/{len(work)} normalized={len(rows)} errors={len(errs)}",
                    flush=True,
                )

    norm = pd.DataFrame(rows)
    if len(norm):
        norm["period_end"] = pd.to_datetime(norm["period_end"], errors="coerce")
        norm["broadcast_ts"] = pd.to_datetime(norm["broadcast_ts"], errors="coerce", utc=True)
        norm = norm.dropna(
            subset=["symbol", "period_end", "broadcast_ts", "period_months"]
        ).copy()
        norm = norm.sort_values(
            ["symbol", "period_end", "period_months", "broadcast_ts", "mapping_score"]
        ).drop_duplicates(
            ["symbol", "period_end", "period_months", "broadcast_ts", "statement_scope"],
            keep="last",
        )
    err = pd.DataFrame(errs)

    norm.to_parquet(outdir / "fundamentals_normalized.parquet", index=False)
    err.to_csv(outdir / "errors.csv", index=False)
    rep = coverage(cat, norm, err)
    json.dump(rep, open(outdir / "coverage.json", "w"), indent=2, default=str)

    # Store raw files as a single compressed checkpoint for the month.
    with tarfile.open(outdir / "raw_archive.tar.gz", "w:gz") as tf:
        for p in raw_dir.glob("*"):
            tf.add(p, arcname=p.name)

    print(json.dumps(rep, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
