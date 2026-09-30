from __future__ import annotations

import argparse
import json
import re
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

import build_fundamental_archive as yearly
import build_fundamental_month as monthly
import legacy_financial_normalizer as legacy
import xbrl_normalizer as xn


FIELDS = [
    "revenue", "operating_profit", "pbt", "pat", "finance_cost",
    "total_assets", "total_equity", "total_debt",
    "current_assets", "current_liabilities",
    "operating_cash_flow", "capex", "shares_outstanding",
    "promoter_pct", "pledged_pct",
]


def _artifact_key(path: Path):
    return path.name


def _find_one(root: Path, pattern: str):
    xs = list(root.rglob(pattern))
    return xs[0] if xs else None


def _tar_index(path: Path):
    tf = tarfile.open(path, "r:*")
    members = {
        Path(m.name).name: m
        for m in tf.getmembers()
        if m.isfile()
    }
    return tf, members


def _tar_read(tf, members, name):
    m = members.get(Path(name).name)
    if m is None:
        return None
    f = tf.extractfile(m)
    return f.read() if f is not None else None


def _normalize_blob(blob, mode, rd, mapping):
    url = (
        rd.get("fetch_url")
        or rd.get("xbrl_url")
        or rd.get("source_url")
        or ""
    )
    source = rd.get("source")
    meta = {
        "symbol": rd.get("symbol"),
        "period_end": rd.get("period_end"),
        "broadcast_ts": rd.get("broadcast_ts"),
        "statement_scope": rd.get("statement_scope"),
        "filing_id": rd.get("filing_id"),
        "source_url": url,
        "xbrl_url": url,
        "source": source,
    }

    if mode == "legacy_html":
        return [
            legacy.parse_legacy_result_html(
                blob,
                symbol=rd.get("symbol"),
                broadcast_ts=rd.get("broadcast_ts"),
                source_url=url,
                source=source,
                fallback_period_end=rd.get("period_end"),
                fallback_scope=rd.get("statement_scope"),
                filing_id=rd.get("filing_id"),
            )
        ]

    out = []
    for member, payload in yearly.extract_xml_payloads(blob, ""):
        rows = xn.normalize_xbrl(payload, meta, mapping)
        for row in rows:
            row["archive_member"] = member
        out.extend(rows)
    return out


def _monthly_pairs(catalog_root: Path, raw_root: Path):
    cats = {}
    for p in catalog_root.iterdir():
        if not p.is_dir():
            continue
        m = re.fullmatch(r"v9-5-month-(20\d{2})-(\d{1,2})", p.name)
        if m:
            cats[(int(m.group(1)), int(m.group(2)))] = p

    raws = {}
    for p in raw_root.iterdir():
        if not p.is_dir():
            continue
        m = re.fullmatch(r"v9-5-month-raw-(20\d{2})-(\d{1,2})", p.name)
        if m:
            raws[(int(m.group(1)), int(m.group(2)))] = p

    for key in sorted(cats):
        if key in raws:
            yield key, cats[key], raws[key]


def renormalize_monthlies(catalog_root, raw_root, mapping):
    rows = []
    errors = []
    files = 0

    for (year, month), cdir, rdir in _monthly_pairs(catalog_root, raw_root):
        cp = _find_one(cdir, "catalog.parquet")
        tp = _find_one(rdir, "*.tar.gz")
        if cp is None or tp is None:
            errors.append({
                "year": year, "month": month, "stage": "pair",
                "error": "missing catalog or raw tar",
            })
            continue

        cat = pd.read_parquet(cp)
        tf, members = _tar_index(tp)
        try:
            for r in cat.itertuples(index=False):
                rd = r._asdict()
                mode = str(rd.get("fetch_mode", "none"))
                url = rd.get("fetch_url")
                if mode == "none" or not url or str(url) in {"nan", "None", ""}:
                    continue
                suffix = ".html" if mode == "legacy_html" else ".bin"
                raw_name = monthly._safe_file(rd.get("symbol"), str(url), suffix)
                blob = _tar_read(tf, members, raw_name)
                if blob is None:
                    errors.append({
                        "year": year, "month": month, "symbol": rd.get("symbol"),
                        "stage": "raw_lookup", "raw_name": raw_name,
                        "error": "raw file not found",
                    })
                    continue
                try:
                    z = _normalize_blob(blob, mode, rd, mapping)
                    for q in z:
                        q["archive_year"] = year
                        q["archive_month"] = month
                        q["raw_archive_file"] = raw_name
                    rows.extend(z)
                    files += 1
                except Exception as exc:
                    errors.append({
                        "year": year, "month": month, "symbol": rd.get("symbol"),
                        "stage": "normalize", "raw_name": raw_name,
                        "error": repr(exc),
                    })
        finally:
            tf.close()

        print(
            f"monthly {year}-{month:02d}: total_rows={len(rows)} errors={len(errors)}",
            flush=True,
        )

    return rows, errors, files


def _year_pairs(catalog_root: Path, raw_root: Path):
    cats = {}
    for p in catalog_root.iterdir():
        if not p.is_dir():
            continue
        m = re.fullmatch(r"v9-5-normalized-(20\d{2})", p.name)
        if m:
            cats[int(m.group(1))] = p

    raws = {}
    for p in raw_root.iterdir():
        if not p.is_dir():
            continue
        m = re.fullmatch(r"v9-5-raw-xbrl-(20\d{2})", p.name)
        if m:
            raws[int(m.group(1))] = p

    for year in sorted(cats):
        if year in raws:
            yield year, cats[year], raws[year]


def renormalize_yearlies(catalog_root, raw_root, mapping):
    rows = []
    errors = []
    files = 0

    for year, cdir, rdir in _year_pairs(catalog_root, raw_root):
        cp = _find_one(cdir, "catalog.parquet")
        tp = _find_one(rdir, "*.tar.gz")
        if cp is None or tp is None:
            errors.append({
                "year": year, "stage": "pair",
                "error": "missing catalog or raw tar",
            })
            continue

        cat = pd.read_parquet(cp)
        tf, members = _tar_index(tp)
        try:
            for r in cat.itertuples(index=False):
                rd = r._asdict()
                xurl = rd.get("xbrl_url")
                if not xurl or str(xurl).lower() in {"nan", "none", "-", ""}:
                    continue
                raw_name = yearly.safe_stem(pd.Series(rd)) + ".bin"
                blob = _tar_read(tf, members, raw_name)
                if blob is None:
                    errors.append({
                        "year": year, "symbol": rd.get("symbol"),
                        "stage": "raw_lookup", "raw_name": raw_name,
                        "error": "raw file not found",
                    })
                    continue
                try:
                    z = _normalize_blob(blob, "xbrl", rd, mapping)
                    for q in z:
                        q["archive_year"] = year
                        q["archive_month"] = np.nan
                        q["raw_archive_file"] = raw_name
                    rows.extend(z)
                    files += 1
                except Exception as exc:
                    errors.append({
                        "year": year, "symbol": rd.get("symbol"),
                        "stage": "normalize", "raw_name": raw_name,
                        "error": repr(exc),
                    })
        finally:
            tf.close()

        print(
            f"yearly {year}: total_rows={len(rows)} errors={len(errors)}",
            flush=True,
        )

    return rows, errors, files


def finalize(rows):
    d = pd.DataFrame(rows)
    if d.empty:
        return d

    d["period_end"] = pd.to_datetime(d["period_end"], errors="coerce")
    d["broadcast_ts"] = pd.to_datetime(d["broadcast_ts"], errors="coerce", utc=True)
    for c in FIELDS:
        if c not in d.columns:
            d[c] = np.nan
        d[c] = pd.to_numeric(d[c], errors="coerce")

    d = d.dropna(subset=["symbol", "period_end", "broadcast_ts", "period_months"])
    d = d[d["period_months"].isin([3, 6, 9, 12])]
    d = d.sort_values(
        [
            "symbol", "period_end", "period_months", "broadcast_ts",
            "statement_scope", "mapped_field_count", "mapping_score",
        ],
        ascending=[True, True, True, True, True, False, False],
    ).drop_duplicates(
        ["symbol", "period_end", "period_months", "broadcast_ts", "statement_scope"],
        keep="first",
    )
    return d.reset_index(drop=True)


def coverage_report(d, errors, files):
    rep = {
        "normalized_source_files": int(files),
        "normalized_rows": int(len(d)),
        "symbols": int(d["symbol"].nunique()) if len(d) else 0,
        "errors": int(len(errors)),
        "broadcast_start": str(d["broadcast_ts"].min()) if len(d) else None,
        "broadcast_end": str(d["broadcast_ts"].max()) if len(d) else None,
        "taxonomy_families": (
            d["taxonomy_family"].value_counts(dropna=False).to_dict()
            if len(d) else {}
        ),
        "field_coverage": {
            c: float(d[c].notna().mean()) if len(d) else 0.0
            for c in FIELDS
        },
        "mapping_score_mean": float(d["mapping_score"].mean()) if len(d) else 0.0,
        "mapped_field_count_mean": float(d["mapped_field_count"].mean()) if len(d) else 0.0,
    }

    by_year = []
    if len(d):
        for year, z in d.groupby(d["broadcast_ts"].dt.year):
            by_year.append({
                "year": int(year),
                "rows": int(len(z)),
                "symbols": int(z["symbol"].nunique()),
                "families": "|".join(
                    f"{k}:{v}" for k, v in z["taxonomy_family"].value_counts().items()
                ),
                "revenue": float(z["revenue"].notna().mean()),
                "operating_profit": float(z["operating_profit"].notna().mean()),
                "pbt": float(z["pbt"].notna().mean()),
                "pat": float(z["pat"].notna().mean()),
                "finance_cost": float(z["finance_cost"].notna().mean()),
                "assets": float(z["total_assets"].notna().mean()),
                "equity": float(z["total_equity"].notna().mean()),
                "debt": float(z["total_debt"].notna().mean()),
                "ocf": float(z["operating_cash_flow"].notna().mean()),
                "capex": float(z["capex"].notna().mean()),
                "shares": float(z["shares_outstanding"].notna().mean()),
                "mean_fields": float(z["mapped_field_count"].mean()),
                "mean_mapping_score": float(z["mapping_score"].mean()),
            })
    rep["by_year"] = by_year
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--monthly-catalog-root", required=True)
    ap.add_argument("--monthly-raw-root", required=True)
    ap.add_argument("--yearly-catalog-root", required=True)
    ap.add_argument("--yearly-raw-root", required=True)
    ap.add_argument("--mapping", default="config/xbrl_concept_map_v9_5.json")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    mapping = xn.load_map(args.mapping)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    mrows, merr, mfiles = renormalize_monthlies(
        Path(args.monthly_catalog_root),
        Path(args.monthly_raw_root),
        mapping,
    )
    yrows, yerr, yfiles = renormalize_yearlies(
        Path(args.yearly_catalog_root),
        Path(args.yearly_raw_root),
        mapping,
    )

    d = finalize(mrows + yrows)
    errors = merr + yerr

    d.to_parquet(out / "fundamentals_renormalized.parquet", index=False)
    pd.DataFrame(errors).to_csv(out / "renormalize_errors.csv", index=False)

    rep = coverage_report(d, errors, mfiles + yfiles)
    json.dump(rep, open(out / "coverage.json", "w"), indent=2, default=str)
    pd.DataFrame(rep["by_year"]).to_csv(out / "coverage_by_year.csv", index=False)
    print(json.dumps(rep, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
