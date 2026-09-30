from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

import build_fundamental_archive as yearly
import build_fundamental_month as monthly
import legacy_financial_normalizer as legacy
import xbrl_normalizer as xn
from renormalize_saved_archives import finalize, coverage_report


def find_one(root: Path, pattern: str):
    xs = list(root.rglob(pattern))
    return xs[0] if xs else None


def tar_index(path: Path):
    tf = tarfile.open(path, "r:*")
    members = {Path(m.name).name: m for m in tf.getmembers() if m.isfile()}
    return tf, members


def tar_read(tf, members, name: str):
    m = members.get(Path(name).name)
    if m is None:
        return None
    f = tf.extractfile(m)
    return f.read() if f is not None else None


def usable(v) -> bool:
    if v is None:
        return False
    s = str(v).strip()
    return (
        bool(s)
        and s.lower() not in {"nan", "none", "-"}
        and not s.rstrip("/").endswith("/-")
    )


def normalize_blob(blob: bytes, mode: str, rd: dict, mapping: dict, source_url: str):
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

    if mode == "legacy_html":
        return [legacy.parse_legacy_result_html(
            blob,
            symbol=rd.get("symbol"),
            broadcast_ts=rd.get("broadcast_ts"),
            source_url=source_url,
            source=rd.get("source"),
            fallback_period_end=rd.get("period_end"),
            fallback_scope=rd.get("statement_scope") or "",
            filing_id=rd.get("filing_id"),
        )]

    out = []
    for member, payload in yearly.extract_xml_payloads(blob, ""):
        z = xn.normalize_xbrl(payload, meta, mapping)
        for row in z:
            row["archive_member"] = member
        out.extend(z)
    return out


def process_monthly_source(catalog_root: Path, raw_root: Path, year: int, month: int, mapping: dict):
    cp = find_one(catalog_root, "catalog.parquet")
    tp = find_one(raw_root, "*.tar.gz")
    if cp is None or tp is None:
        raise RuntimeError("monthly catalog/raw artifact missing")

    cat = pd.read_parquet(cp)
    rows, errors = [], []
    tf, members = tar_index(tp)
    try:
        for r in cat.itertuples(index=False):
            rd = r._asdict()
            mode = str(rd.get("fetch_mode", "none"))
            url = rd.get("fetch_url")
            if mode == "none" or not usable(url):
                continue
            suffix = ".html" if mode == "legacy_html" else ".bin"
            raw_name = monthly._safe_file(rd.get("symbol"), str(url), suffix)
            blob = tar_read(tf, members, raw_name)
            if blob is None:
                errors.append({
                    "symbol": rd.get("symbol"), "stage": "raw_lookup",
                    "raw_name": raw_name, "error": "raw file not found",
                })
                continue
            try:
                z = normalize_blob(blob, mode, rd, mapping, str(url))
                for q in z:
                    q["archive_year"] = year
                    q["archive_month"] = month
                    q["raw_archive_file"] = raw_name
                rows.extend(z)
            except Exception as exc:
                errors.append({
                    "symbol": rd.get("symbol"), "stage": "normalize",
                    "raw_name": raw_name, "error": repr(exc),
                })
    finally:
        tf.close()
    return cat, rows, errors


def process_yearly_source(catalog_root: Path, raw_root: Path, year: int, month: int, mapping: dict):
    cp = find_one(catalog_root, "catalog.parquet")
    tp = find_one(raw_root, "*.tar.gz")
    if cp is None or tp is None:
        raise RuntimeError("yearly catalog/raw artifact missing")

    cat = pd.read_parquet(cp)
    cat["broadcast_ts"] = pd.to_datetime(cat["broadcast_ts"], errors="coerce", utc=True)
    cat = cat[
        (cat["broadcast_ts"].dt.year == year)
        & (cat["broadcast_ts"].dt.month == month)
    ].copy()

    rows, errors = [], []
    tf, members = tar_index(tp)
    try:
        for r in cat.itertuples(index=False):
            rd = r._asdict()
            xurl = rd.get("xbrl_url")
            detail = rd.get("detail_url")
            use_xbrl = usable(xurl)
            use_legacy = (not use_xbrl) and usable(detail)
            if not use_xbrl and not use_legacy:
                continue

            mode = "xbrl" if use_xbrl else "legacy_html"
            source_url = yearly.resolve_url(xurl if use_xbrl else detail)
            raw_name = yearly.safe_stem(pd.Series(rd)) + (".bin" if use_xbrl else ".html")
            blob = tar_read(tf, members, raw_name)
            if blob is None:
                errors.append({
                    "symbol": rd.get("symbol"), "stage": "raw_lookup",
                    "raw_name": raw_name, "error": "raw file not found",
                })
                continue

            try:
                z = normalize_blob(blob, mode, rd, mapping, source_url)
                for q in z:
                    q["archive_year"] = year
                    q["archive_month"] = month
                    q["raw_archive_file"] = raw_name
                rows.extend(z)
            except Exception as exc:
                errors.append({
                    "symbol": rd.get("symbol"), "stage": "normalize",
                    "raw_name": raw_name, "error": repr(exc),
                })
    finally:
        tf.close()
    return cat, rows, errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--month", type=int, required=True)
    ap.add_argument("--source-kind", choices=["monthly", "yearly"], required=True)
    ap.add_argument("--catalog-root", required=True)
    ap.add_argument("--raw-root", required=True)
    ap.add_argument("--mapping", default="config/xbrl_concept_map_v9_5.json")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    mapping = xn.load_map(args.mapping)

    if args.source_kind == "monthly":
        cat, rows, errors = process_monthly_source(
            Path(args.catalog_root), Path(args.raw_root),
            args.year, args.month, mapping,
        )
    else:
        cat, rows, errors = process_yearly_source(
            Path(args.catalog_root), Path(args.raw_root),
            args.year, args.month, mapping,
        )

    d = finalize(rows)
    err = pd.DataFrame(errors)
    d.to_parquet(out / "fundamentals_renormalized.parquet", index=False)
    err.to_csv(out / "renormalize_errors.csv", index=False)

    rep = coverage_report(d, errors, len(cat))
    rep.update({
        "year": args.year,
        "month": args.month,
        "source_kind": args.source_kind,
        "catalog_rows_for_month": int(len(cat)),
    })
    json.dump(rep, open(out / "coverage.json", "w"), indent=2, default=str)

    print(json.dumps({
        "year": args.year,
        "month": args.month,
        "source_kind": args.source_kind,
        "catalog_rows": int(len(cat)),
        "normalized_rows": int(len(d)),
        "symbols": int(d["symbol"].nunique()) if len(d) else 0,
        "errors": int(len(errors)),
        "field_coverage": rep.get("field_coverage", {}),
    }, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
