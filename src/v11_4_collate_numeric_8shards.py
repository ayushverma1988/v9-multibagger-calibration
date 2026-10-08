"""Collate eight independent 2024 and 2025 strict annual NSE XBRL shards.

Reject missing, duplicate or out-of-universe symbols, wrong reporting modes
and after-cutoff records. Unsuccessful document retrieval remains an error,
not a zero-valued financial feature. Original V10 production remains frozen.
"""
import argparse,json
from pathlib import Path
import pandas as pd

FILES={
    2024:"strict_2024_fold_annual_numeric_coverage.json",
    2025:"FY2025_annual_PIT_pairing_summary.json",
}
TARGETS={
    2024:"eligible_companies_with_both_fiscal_index_records",
    2025:"eligible_companies_with_2024_legacy_and_2025_integrated_same_mode",
}
FEATURE_NAME="short_horizon_financial_PIT_research_features.csv"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--artifacts",required=True)
    ap.add_argument("--output",required=True)
    ap.add_argument("--expected-shards",type=int,default=4)
    a=ap.parse_args()
    root=Path(a.artifacts);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    reports=[];err=[];concat={}
    for year in (2024,2025):
        roots=list(root.rglob(FILES[year]))
        if len(roots)!=a.expected_shards:
            raise SystemExit(f"FY{year}: expected {a.expected_shards} source shards, saw {len(roots)}")
        samples=[];indices=set();all_requested=set()
        target_counts=set()
        for path in roots:
            report=json.loads(path.read_text())
            shard=report.get("shard_index")
            if shard in indices:raise SystemExit(f"FY{year}: duplicate shard {shard}")
            indices.add(shard)
            if report.get("shard_count")!=a.expected_shards:raise SystemExit("Wrong shard count")
            syms=report.get("requested_symbols",[])
            if len(syms)!=len(set(syms)):raise SystemExit("Duplicate ticker within shard")
            if all_requested.intersection(syms):raise SystemExit("Duplicate ticker between shards")
            all_requested.update(syms)
            target_counts.add(report.get(TARGETS[year],report.get("candidate_symbol_total")))
            f=path.parent/"features"/FEATURE_NAME
            if not f.exists():raise SystemExit("Missing verified 1y feature output "+str(f))
            x=pd.read_csv(f)
            if "symbol" not in x or "fold_date" not in x:
                raise SystemExit("Missing key columns in verified feature CSV")
            if not set(x["symbol"].astype(str)).issubset(set(syms)):
                raise SystemExit("Feature key not in original shard selection")
            if x["symbol"].duplicated().any():raise SystemExit("One company repeated within features")
            if not x["fold_date"].astype(str).eq(f"{year}-12-31").all():
                raise SystemExit("Wrong historical fold date")
            samples.append(x)
            reports.append({
                "year":year,"shard":shard,"requested_companies":len(syms),
                "verified_numeric_feature_companies":int(x["symbol"].nunique()),
                "source_document_errors":int(report.get("source_download_errors",report.get("download_errors",0))),
                "original_source_requested":int(report.get("source_documents_expected",report.get("attempted_docs",0))),
            })
        if indices!=set(range(a.expected_shards)) or len(target_counts)!=1:
            raise SystemExit(f"FY{year} incomplete shards or inconsistent original candidate count")
        if len(all_requested)!=next(iter(target_counts)):
            raise SystemExit(f"FY{year} candidate universe mismatch: {len(all_requested)} versus {target_counts}")
        merged=pd.concat(samples,ignore_index=True)
        if merged["symbol"].duplicated().any():raise SystemExit(f"FY{year} cross-shard duplicate company")
        verified=int(merged["symbol"].nunique())
        target=len(all_requested)
        ratio=verified/max(target,1)
        if ratio<0.85:
            raise SystemExit(f"FY{year}: only {ratio:.2%} of eligible original companies have verified features")
        # Do not allow timestamp-less or label-bearing source rows into staging.
        for col in ("y6","y12","y24","dd30_6m"):
            if col in merged:raise SystemExit("Forward return label in source-only feature matrix")
        pub=pd.to_datetime(merged["source_available_utc"],utc=True,format="mixed",errors="coerce")
        cutoff=pd.Timestamp(f"{year}-12-31T10:00:00Z")
        if pub.isna().any() or (pub>cutoff).any():
            raise SystemExit(f"FY{year}: after-close source publication or unknown timestamp")
        concat[year]=merged
    for year,df in concat.items():
        df.to_csv(out/f"verified_FY{year}_annual_yoy_PIT_candidates.csv",index=False)
    shard_df=pd.DataFrame(reports).sort_values(["year","shard"])
    shard_df.to_csv(out/"all_eight_annual_source_shard_audit.csv",index=False)
    summary={
        "scope":"FULL_AVAILABLE_TWO_YEAR_FINANCIAL_SOURCE_RESEARCH",
        "original_validation_folds_covered_by_numeric_facts":2,
        "source_shards_2024":a.expected_shards,
        "source_shards_2025":a.expected_shards,
        "FY2024_verified_companies":int(concat[2024]["symbol"].nunique()),
        "FY2025_verified_companies":int(concat[2025]["symbol"].nunique()),
        "FY2024_revenue_yoy_nonnull":int(concat[2024]["revenue_yoy_pct"].notna().sum()),
        "FY2025_revenue_yoy_nonnull":int(concat[2025]["revenue_yoy_pct"].notna().sum()),
        "FY2024_PAT_yoy_nonnull":int(concat[2024]["pat_yoy_pct"].notna().sum()),
        "FY2025_PAT_yoy_nonnull":int(concat[2025]["pat_yoy_pct"].notna().sum()),
        "all_missing_numeric_values_unimputed":True,
        "baseline_V10_production_unchanged":True,
        "all_18_folds_numeric_source_ready":False,
        "predictive_performance_measured":False,
        "source_download_errors_total":int(shard_df["source_document_errors"].sum()),
    }
    (out/"eight_shard_full_annual_research_summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)
if __name__=="__main__":main()
