"""Independent PIT ownership backfill feasibility study.

Uses a community NSE-XBRL derived file ONLY in this isolated diagnostic.
Do not promote to production without NSE-filing cross-check and license attribution.
Upstream: https://github.com/aditya-jha/nse-historical-membership
Data attribution: Aditya Jha (2026), CC BY 4.0, LICENSE-DATA.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import pandas as pd

SOURCE = Path("upstream_nse/shareholding_history/data")
OUT = Path("outputs_v11_4_nse_ownership_probe")
OUT.mkdir(parents=True, exist_ok=True)

def main():
    flat_path = SOURCE / "parsed/_flat.csv"
    assert flat_path.is_file(), f"Missing open-source historical CSV: {flat_path}"
    flat = pd.read_csv(flat_path, low_memory=False)
    required = {"ticker", "period", "source_filename", "promoter_pct", "public_pct"}
    assert required.issubset(flat.columns), f"Missing columns: {required - set(flat.columns)}"
    flat["ticker"] = flat["ticker"].astype(str).str.upper().str.strip()
    flat["period"] = flat["period"].astype(str).str.strip()
    flat["source_filename"] = flat["source_filename"].astype(str).str.strip()
    flat["promoter_pct"] = pd.to_numeric(flat["promoter_pct"], errors="coerce")
    flat["public_pct"] = pd.to_numeric(flat["public_pct"], errors="coerce")

    metadata = []
    manifests = list((SOURCE / "filings_index").glob("*.json"))
    for file in manifests:
        try:
            j = json.loads(file.read_text())
        except (OSError, ValueError):
            continue
        ticker = str(j.get("ticker") or file.stem).upper().strip()
        for record in j.get("filings", []):
            url = str(record.get("xbrl") or "").strip()
            filename = url.rsplit("/", 1)[-1]
            if not filename.endswith(".xml"):
                continue
            metadata.append({
                "ticker": ticker,
                "source_filename": filename,
                "broadcast_date": record.get("broadcast_date"),
                "submission_date": record.get("submission_date"),
                "quarter_end": record.get("date"),
                "xbrl_url": url,
                "revised": record.get("revised"),
            })
    index = pd.DataFrame(metadata)
    assert not index.empty, "No indexed NSE XBRL filings found"
    index = index.drop_duplicates(["ticker", "source_filename"])
    records = flat.merge(index, on=["ticker", "source_filename"], how="left", validate="many_to_one")
    broadcast = pd.to_datetime(records["broadcast_date"], format="%d-%b-%Y %H:%M:%S", errors="coerce")
    records["available_at_utc"] = broadcast.dt.tz_localize("Asia/Kolkata", ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")
    records["period_end"] = pd.to_datetime(records["quarter_end"], format="%d-%b-%Y", errors="coerce", utc=True)
    records["numeric_ok"] = (
        records["promoter_pct"].between(0, 100)
        & records["public_pct"].between(0, 100)
        & ((records["promoter_pct"] + records["public_pct"] - 100).abs() <= 5)
    )
    records["source_ok"] = (
        records["available_at_utc"].notna()
        & records["period_end"].notna()
        & (records["available_at_utc"] >= records["period_end"])
        & records["numeric_ok"]
    )
    valid = records[records["source_ok"]].copy()
    valid = valid.sort_values("available_at_utc").drop_duplicates(["ticker", "source_filename"])

    # Save reproducible candidate facts for a separate, source-verified PIT validation.
    # No quarter-end substitution for missing publication timestamp is permitted.
    export_columns = [
        "ticker", "period", "period_end", "available_at_utc",
        "promoter_pct", "fii_pct", "dii_pct", "public_pct", "noninst_pct",
        "source_filename", "xbrl_url", "quarter_end", "submission_date",
        "broadcast_date", "revised",
    ]
    export_columns = [c for c in export_columns if c in valid.columns]
    valid[export_columns].to_parquet(
        OUT / "ownership_historical_pit_candidates.parquet",
        index=False, compression="zstd",
    )
    (OUT / "SOURCE_ATTRIBUTION.txt").write_text(
        "Derived from Aditya Jha (2026), NSE Historical Membership/Shareholding History,\n"
        "https://github.com/aditya-jha/nse-historical-membership\n"
        "Original derived dataset: CC BY 4.0, subject to upstream LICENSE-DATA.\n"
        "The underlying regulatory XBRL documents belong to their publishers.\n"
        "This export is unverified and is not approved for live investment use.\n",
        encoding="utf-8",
    )

    snapfiles = list(Path("control_seed").rglob("snapshot_dataset.parquet"))
    assert snapfiles, "Missing accepted control snapshot"
    snap = pd.read_parquet(snapfiles[0], columns=["date", "symbol"])
    snap["date"] = pd.to_datetime(snap["date"], errors="coerce").dt.normalize()
    snap["symbol"] = snap["symbol"].astype(str).str.upper().str.strip()
    folds = pd.read_csv("fold_dates.csv")
    assert "date" in folds, "Fold dates file missing date column"

    foldrows = []
    for d in pd.to_datetime(folds["date"], errors="coerce").dropna().sort_values().unique():
        fold = pd.Timestamp(d).normalize()
        cut = fold.tz_localize("Asia/Kolkata") + pd.Timedelta(hours=23, minutes=59, seconds=59)
        cut = cut.tz_convert("UTC")
        universe = set(snap.loc[snap["date"].eq(fold), "symbol"])
        cand = valid[
            valid["ticker"].isin(universe)
            & (valid["available_at_utc"] <= cut)
            & (valid["available_at_utc"] >= cut - pd.Timedelta(days=500))
        ]
        covered = cand["ticker"].nunique()
        foldrows.append({
            "fold_date": fold.strftime("%Y-%m-%d"),
            "universe_n": len(universe),
            "nse_xbrl_pit_covered_symbols": covered,
            "nse_xbrl_pit_coverage": covered / max(len(universe), 1),
            "quarterly_ownership_gate_25pct": covered >= 0.25 * len(universe) and len(universe) > 0,
        })
    fold_df = pd.DataFrame(foldrows)
    fold_df.to_csv(OUT / "ownership_candidate_coverage_by_fold.csv", index=False)
    by_year = (valid.groupby(valid["period_end"].dt.year)["ticker"].nunique().rename("symbols_with_valid_xbrl")
        .reset_index().rename(columns={"period_end": "period_year"}))
    by_year.to_csv(OUT / "historic_xbrl_symbols_by_year.csv", index=False)

    summary = {
        "scope": "INDEPENDENT_FEASIBILITY_ONLY; no production input or gate changed",
        "source": "https://github.com/aditya-jha/nse-historical-membership",
        "license": "CC BY 4.0 (attribution required); upstream NSE XBRL facts, unverified third-party parsing",
        "source_sha256": hashlib.sha256(flat_path.read_bytes()).hexdigest(),
        "flat_rows": len(flat),
        "indexed_xml_files": len(index),
        "upstream_manifest_files": len(manifests),
        "joined_index_records": int(records["broadcast_date"].notna().sum()),
        "valid_timestamp_and_percent_rows": len(valid),
        "valid_symbols_total": int(valid["ticker"].nunique()),
        "folds_total": len(fold_df),
        "folds_reaching_25pct_ownership_coverage": int(fold_df["quarterly_ownership_gate_25pct"].sum()),
        "folds_2017_2021_reaching_gate": int(fold_df.loc[fold_df["fold_date"] < "2022-01-01", "quarterly_ownership_gate_25pct"].sum()),
        "max_coverage_2017_2021": float(fold_df.loc[fold_df["fold_date"] < "2022-01-01", "nse_xbrl_pit_coverage"].max()),
        "note": "Independent potential ownership coverage only; whole V11.4 12/18 validation still requires all official evidence and proper PIT news demand.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str), flush=True)
    print(fold_df.to_string(index=False), flush=True)

if __name__ == "__main__":
    main()
