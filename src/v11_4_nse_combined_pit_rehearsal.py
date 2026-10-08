"""Build a read-only historical ownership staging file for the 18-fold PIT audit.

NEVER write into accepted V10/V11 production artifacts. Third-party records are
provisional until full source identity and fact QC; this stages COVERAGE ONLY.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd

FIELDS = ["symbol_norm", "date", "broadcastDate", "isin_norm", "origin",
          "promoter_pct", "public_pct", "filing_url"]

def official_frame(path):
    x = pd.read_parquet(path)
    for need in ("symbol_norm", "broadcastDate", "date"):
        if need not in x:
            raise ValueError("Official PIT archive lacks " + need)
    out = pd.DataFrame({
        "symbol_norm": x["symbol_norm"].astype(str).str.upper().str.strip(),
        "date": pd.to_datetime(x["date"], utc=True, errors="coerce"),
        "broadcastDate": pd.to_datetime(x["broadcastDate"], utc=True, errors="coerce"),
        "isin_norm": x["isin_norm"].astype(str) if "isin_norm" in x else "",
        "origin": "official_nse_archive",
        "promoter_pct": float("nan"),
        "public_pct": float("nan"),
        "filing_url": x["xbrl"].astype(str) if "xbrl" in x else "",
    })
    return out[out["date"].notna() & out["broadcastDate"].notna()]

def candidate_frame(path):
    x = pd.read_parquet(path)
    needed = ["ticker", "period_end", "available_at_utc", "promoter_pct", "public_pct", "xbrl_url"]
    for need in needed:
        if need not in x:
            raise ValueError("Community candidate archive lacks " + need)
    period = pd.to_datetime(x["period_end"], utc=True, errors="coerce")
    available = pd.to_datetime(x["available_at_utc"], utc=True, errors="coerce")
    promoter = pd.to_numeric(x["promoter_pct"], errors="coerce")
    public = pd.to_numeric(x["public_pct"], errors="coerce")
    out = pd.DataFrame({
        "symbol_norm": x["ticker"].astype(str).str.upper().str.strip(),
        "date": period,
        "broadcastDate": available,
        "isin_norm": "",
        "origin": "third_party_nse_xbrl_provisional",
        "promoter_pct": promoter,
        "public_pct": public,
        "filing_url": x["xbrl_url"].astype(str),
    })
    # Do not replace available official quarterly records from 2022 onwards.
    # Old-quarter revisions first published in 2022 remain as-of their broadcast date.
    cutoff = pd.Timestamp("2022-01-01", tz="UTC")
    keep = (
        out["date"].notna() &
        (out["date"] < cutoff) &
        out["broadcastDate"].notna() &
        (out["broadcastDate"] >= out["date"]) &
        out["promoter_pct"].between(0, 100) &
        out["public_pct"].between(0, 100) &
        ((out["promoter_pct"] + out["public_pct"] - 100).abs() <= 5) &
        out["filing_url"].str.match(r"^https://nsearchives\.nseindia\.com/corporate/xbrl/.*\.xml$")
    )
    return out.loc[keep]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--official", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--verification-summary", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    qa = json.loads(Path(a.verification_summary).read_text())
    if not qa.get("source_sample_pass") or qa.get("verified_sample_n", 0) < 8:
        raise SystemExit("Original-document sample QA has not passed. Staging blocked.")
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    official = official_frame(a.official)
    third = candidate_frame(a.candidate)
    all_rows = pd.concat([official, third], ignore_index=True)[FIELDS]
    all_rows = all_rows.drop_duplicates(["symbol_norm", "date", "broadcastDate", "origin"])
    all_rows.to_parquet(out / "ownership_merged_REHEARSAL_ONLY.parquet", index=False, compression="zstd")
    summary = {
        "scope": "COVERAGE_REHEARSAL_ONLY",
        "production_promotion_allowed": False,
        "official_rows": len(official),
        "historical_third_party_rows": len(third),
        "combined_rows": len(all_rows),
        "third_party_year_min": str(third["date"].min()) if len(third) else None,
        "third_party_year_max": str(third["date"].max()) if len(third) else None,
        "source_sample_verified": qa["verified_sample_n"],
        "source_sample_years": qa["verified_years"],
        "cautions": ["Full source-by-source validation incomplete",
                     "Ticker-to-ISIN identity stitching incomplete for legacy symbols",
                     "Annual 5-7 year fundamentals and historical external demand not yet rebuilt",
                     "This is not a prediction-performance backtest"],
    }
    (out / "staging_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)

if __name__ == "__main__":
    main()
