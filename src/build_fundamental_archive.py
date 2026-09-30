from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import time
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.parse import urljoin, urlparse

import numpy as np
import pandas as pd
import requests

import fetch_nse_fundamental_catalog as catalog_api
import xbrl_normalizer as xn
import legacy_financial_normalizer as legacy


def month_windows(start: date, end: date):
    cur = start.replace(day=1)
    while cur <= end:
        if cur.month == 12:
            nxt = date(cur.year + 1, 1, 1)
        else:
            nxt = date(cur.year, cur.month + 1, 1)
        a = max(start, cur)
        b = min(end, nxt - timedelta(days=1))
        yield a, b
        cur = nxt


def fetch_catalog_archive(start: date, end: date, sleep_s=0.25) -> pd.DataFrame:
    parts = []
    for a, b in month_windows(start, end):
        print(f"catalog {a} -> {b}", flush=True)
        try:
            z = catalog_api.fetch_catalog(a, b)
        except Exception as exc:
            print(f"warning: catalog window failed {a} {b}: {exc}", flush=True)
            z = pd.DataFrame()
        if len(z):
            parts.append(z)
        time.sleep(sleep_s)

    if not parts:
        return pd.DataFrame()
    d = pd.concat(parts, ignore_index=True)
    d = d.sort_values(["broadcast_ts", "symbol", "source"])
    d = d.drop_duplicates(
        ["symbol", "period_end", "broadcast_ts", "source", "xbrl_url"],
        keep="last",
    ).reset_index(drop=True)
    return d


def _download_session():
    s = requests.Session()
    s.headers.update({
        **catalog_api.HEADERS,
        "Accept": "application/xml,text/xml,application/xhtml+xml,text/html,application/zip,*/*",
    })
    try:
        s.get(catalog_api.PAGE, timeout=30)
    except Exception:
        pass
    return s


def resolve_url(u: str) -> str:
    u = str(u).strip()
    if u.startswith("//"):
        return "https:" + u
    if u.startswith("http://") or u.startswith("https://"):
        return u
    return urljoin(catalog_api.BASE + "/", u.lstrip("/"))


def safe_stem(row: pd.Series) -> str:
    raw = "|".join([
        str(row.get("symbol", "")),
        str(row.get("period_end", "")),
        str(row.get("broadcast_ts", "")),
        str(row.get("statement_scope", "")),
        str(row.get("filing_id", "")),
        str(row.get("xbrl_url", "")),
    ])
    h = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:14]
    sym = re.sub(r"[^A-Za-z0-9_.-]", "_", str(row.get("symbol", "UNK")))[:30]
    return f"{sym}_{h}"


def download_bytes(s: requests.Session, url: str, retries=4) -> tuple[bytes, str]:
    last = None
    for i in range(retries):
        try:
            r = s.get(url, timeout=60, allow_redirects=True)
            if r.status_code in (401, 403):
                try:
                    s.get(catalog_api.PAGE, timeout=30)
                except Exception:
                    pass
                time.sleep(1.0 + i)
                continue
            r.raise_for_status()
            if not r.content:
                raise ValueError("empty response")
            return r.content, r.headers.get("content-type", "")
        except Exception as exc:
            last = exc
            time.sleep(1.0 + 1.5 * i)
    raise RuntimeError(f"download failed: {url}: {last}")


def extract_xml_payloads(blob: bytes, content_type: str):
    if blob[:2] == b"PK" or "zip" in content_type.lower():
        with zipfile.ZipFile(io.BytesIO(blob)) as zf:
            for name in zf.namelist():
                low = name.lower()
                if low.endswith((".xml", ".xbrl", ".xhtml", ".html", ".htm")):
                    yield name, zf.read(name)
        return
    yield "instance.xml", blob


def _usable_link(v) -> bool:
    if v is None:
        return False
    s = str(v).strip()
    if not s or s.lower() in {"nan", "none", "-"}:
        return False
    if s.rstrip("/").endswith("/-") or s.endswith("/-"):
        return False
    return True


def normalize_archive(catalog: pd.DataFrame, outdir: Path, mapping_path: str | Path):
    raw_dir = outdir / "raw_xbrl"
    raw_dir.mkdir(parents=True, exist_ok=True)
    mapping = xn.load_map(mapping_path)
    s = _download_session()

    rows = []
    errors = []
    stats = {"xbrl": 0, "legacy_html": 0, "skipped_no_source": 0}

    for n, r in enumerate(catalog.itertuples(index=False), start=1):
        rd = r._asdict()
        xurl = rd.get("xbrl_url")
        detail = rd.get("detail_url")
        use_xbrl = _usable_link(xurl)
        use_legacy = (not use_xbrl) and _usable_link(detail)

        if not use_xbrl and not use_legacy:
            stats["skipped_no_source"] += 1
            if n % 500 == 0:
                print(
                    f"processed={n}/{len(catalog)} xbrl={stats['xbrl']} "
                    f"legacy={stats['legacy_html']} normalized={len(rows)} "
                    f"errors={len(errors)} skipped={stats['skipped_no_source']}",
                    flush=True,
                )
            continue

        source_url = resolve_url(xurl if use_xbrl else detail)
        stem = safe_stem(pd.Series(rd))
        try:
            blob, ctype = download_bytes(s, source_url)
            raw_suffix = ".bin" if use_xbrl else ".html"
            raw_path = raw_dir / f"{stem}{raw_suffix}"
            raw_path.write_bytes(blob)

            if use_xbrl:
                stats["xbrl"] += 1
                meta = {
                    "symbol": rd.get("symbol"),
                    "period_end": rd.get("period_end"),
                    "broadcast_ts": rd.get("broadcast_ts"),
                    "statement_scope": rd.get("statement_scope"),
                    "filing_id": rd.get("filing_id"),
                    "source_url": source_url,
                    "xbrl_url": source_url,
                    "source": rd.get("source"),
                }

                found = 0
                for member, payload in extract_xml_payloads(blob, ctype):
                    try:
                        z = xn.normalize_xbrl(payload, meta, mapping)
                        for row in z:
                            row["archive_member"] = member
                            row["raw_archive_file"] = str(raw_path.relative_to(outdir))
                        rows.extend(z)
                        found += len(z)
                    except Exception as exc:
                        errors.append({
                            "symbol": rd.get("symbol"),
                            "broadcast_ts": rd.get("broadcast_ts"),
                            "source_url": source_url,
                            "stage": "normalize_xbrl_member",
                            "member": member,
                            "error": repr(exc),
                        })
                if found == 0:
                    errors.append({
                        "symbol": rd.get("symbol"),
                        "broadcast_ts": rd.get("broadcast_ts"),
                        "source_url": source_url,
                        "stage": "normalize_xbrl_file",
                        "member": "",
                        "error": "no normalized duration rows",
                    })

            else:
                stats["legacy_html"] += 1
                try:
                    row = legacy.parse_legacy_result_html(
                        blob,
                        symbol=rd.get("symbol"),
                        broadcast_ts=rd.get("broadcast_ts"),
                        source_url=source_url,
                        source=rd.get("source"),
                        fallback_period_end=rd.get("period_end"),
                        fallback_scope=rd.get("statement_scope") or "",
                        filing_id=rd.get("filing_id"),
                    )
                    if (
                        row
                        and pd.notna(row.get("period_end"))
                        and int(row.get("mapped_field_count", 0)) > 0
                    ):
                        row["archive_member"] = "legacy_result.html"
                        row["raw_archive_file"] = str(raw_path.relative_to(outdir))
                        rows.append(row)
                    else:
                        errors.append({
                            "symbol": rd.get("symbol"),
                            "broadcast_ts": rd.get("broadcast_ts"),
                            "source_url": source_url,
                            "stage": "normalize_legacy_html",
                            "member": "",
                            "error": "legacy page produced no mapped financial fields",
                        })
                except Exception as exc:
                    errors.append({
                        "symbol": rd.get("symbol"),
                        "broadcast_ts": rd.get("broadcast_ts"),
                        "source_url": source_url,
                        "stage": "normalize_legacy_html",
                        "member": "",
                        "error": repr(exc),
                    })

        except Exception as exc:
            errors.append({
                "symbol": rd.get("symbol"),
                "broadcast_ts": rd.get("broadcast_ts"),
                "source_url": source_url,
                "stage": "download_xbrl" if use_xbrl else "download_legacy_html",
                "member": "",
                "error": repr(exc),
            })

        if n % 100 == 0:
            print(
                f"processed={n}/{len(catalog)} xbrl={stats['xbrl']} "
                f"legacy={stats['legacy_html']} normalized={len(rows)} "
                f"errors={len(errors)} skipped={stats['skipped_no_source']}",
                flush=True,
            )

    norm = pd.DataFrame(rows)
    if len(norm):
        norm["period_end"] = pd.to_datetime(norm["period_end"], errors="coerce")
        norm["broadcast_ts"] = pd.to_datetime(
            norm["broadcast_ts"], errors="coerce", utc=True
        )
        norm["statement_scope"] = (
            norm["statement_scope"].astype(str).str.lower()
            .replace({
                "consolidated": "consolidated",
                "standalone": "standalone",
                "non-consolidated": "standalone",
                "non consolidated": "standalone",
            })
        )
        for col in [
            "revenue", "operating_profit", "pbt", "pat", "finance_cost",
            "total_assets", "total_equity", "total_debt",
            "current_assets", "current_liabilities", "operating_cash_flow",
            "capex", "shares_outstanding", "promoter_pct", "pledged_pct",
        ]:
            if col not in norm.columns:
                norm[col] = np.nan
        norm = norm.dropna(subset=["symbol", "period_end", "broadcast_ts", "period_months"])
        norm = norm.sort_values(
            ["symbol", "period_end", "period_months", "broadcast_ts", "mapping_score"]
        )
        norm = norm.drop_duplicates(
            ["symbol", "period_end", "period_months", "broadcast_ts", "statement_scope"],
            keep="last",
        ).reset_index(drop=True)

    err = pd.DataFrame(errors)
    return norm, err, stats

def coverage_report(catalog: pd.DataFrame, norm: pd.DataFrame, errors: pd.DataFrame, stats: dict):
    fields = [
        "revenue", "operating_profit", "pbt", "pat", "finance_cost",
        "total_assets", "total_equity", "total_debt",
        "current_assets", "current_liabilities", "operating_cash_flow",
        "capex", "shares_outstanding", "promoter_pct", "pledged_pct",
    ]
    xbrl_valid = (
        catalog["xbrl_url"].map(_usable_link).sum()
        if len(catalog) and "xbrl_url" in catalog else 0
    )
    detail_valid = (
        catalog["detail_url"].map(_usable_link).sum()
        if len(catalog) and "detail_url" in catalog else 0
    )
    rep = {
        "catalog_rows": int(len(catalog)),
        "catalog_symbols": int(catalog["symbol"].nunique()) if len(catalog) else 0,
        "catalog_valid_xbrl_links": int(xbrl_valid),
        "catalog_legacy_detail_links": int(detail_valid),
        "downloaded_xbrl_files": int(stats.get("xbrl", 0)),
        "downloaded_legacy_html_files": int(stats.get("legacy_html", 0)),
        "skipped_no_source": int(stats.get("skipped_no_source", 0)),
        "normalized_rows": int(len(norm)),
        "normalized_symbols": int(norm["symbol"].nunique()) if len(norm) else 0,
        "errors": int(len(errors)),
        "taxonomy_families": (
            norm["taxonomy_family"].value_counts(dropna=False).to_dict()
            if len(norm) and "taxonomy_family" in norm else {}
        ),
        "period_months": (
            norm["period_months"].value_counts(dropna=False).sort_index().to_dict()
            if len(norm) and "period_months" in norm else {}
        ),
        "field_coverage": {},
    }
    if len(norm):
        for col in fields:
            rep["field_coverage"][col] = (
                float(norm[col].notna().mean()) if col in norm else 0.0
            )
        rep["mapping_score_mean"] = float(norm["mapping_score"].mean())
        rep["mapped_field_count_mean"] = float(norm["mapped_field_count"].mean())
        by_year = []
        years = sorted(norm["broadcast_ts"].dt.year.dropna().unique())
        for y in years:
            z = norm[norm["broadcast_ts"].dt.year == y]
            by_year.append({
                "year": int(y),
                "rows": int(len(z)),
                "symbols": int(z["symbol"].nunique()),
                "revenue_coverage": float(z["revenue"].notna().mean()),
                "pat_coverage": float(z["pat"].notna().mean()),
                "balance_sheet_coverage": float(
                    z[["total_assets", "total_equity", "total_debt"]]
                    .notna().mean().mean()
                ),
            })
        rep["by_year"] = by_year
    return rep

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-date", required=True)
    ap.add_argument("--to-date", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--mapping", default="config/xbrl_concept_map_v9_5.json")
    args = ap.parse_args()

    start = datetime.strptime(args.from_date, "%Y-%m-%d").date()
    end = datetime.strptime(args.to_date, "%Y-%m-%d").date()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    cat = fetch_catalog_archive(start, end)
    cat.to_parquet(outdir / "catalog.parquet", index=False)

    if cat.empty:
        rep = coverage_report(cat, pd.DataFrame(), pd.DataFrame(), {"xbrl":0,"legacy_html":0,"skipped_no_source":0})
        json.dump(rep, open(outdir / "coverage.json", "w"), indent=2)
        print(json.dumps(rep, indent=2))
        return

    norm, errors, stats = normalize_archive(cat, outdir, args.mapping)
    norm.to_parquet(outdir / "fundamentals_normalized.parquet", index=False)
    errors.to_csv(outdir / "errors.csv", index=False)

    rep = coverage_report(cat, norm, errors, stats)
    json.dump(rep, open(outdir / "coverage.json", "w"), indent=2)
    print(json.dumps(rep, indent=2, default=str))


if __name__ == "__main__":
    main()
