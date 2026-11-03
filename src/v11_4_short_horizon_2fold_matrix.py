"""Construct frozen-universe PIT short-horizon annual feature MATRIX, research only.

Uncovered stocks keep NaN numerical features; never interpret missing
earnings history as zero growth. Historical model outcomes and labels are not
read, fitted or modified here.
"""
import argparse,json
from pathlib import Path
import pandas as pd

def fold_close(v):
    return (pd.Timestamp(v).normalize().tz_localize("Asia/Kolkata")
            +pd.Timedelta(hours=15,minutes=30)).tz_convert("UTC")

FEATURES=["revenue_yoy_pct","pat_yoy_pct","pat_margin_latest_pct","pat_margin_delta_pp",
          "profit_turnaround","earlier_profit_negative","latest_fiscal_age_days"]

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--snapshot",required=True)
    p.add_argument("--features-2024",required=True)
    p.add_argument("--features-2025",required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    snap=pd.read_parquet(a.snapshot,columns=["date","symbol"])
    snap["date"]=pd.to_datetime(snap["date"],errors="coerce").dt.normalize()
    snap["symbol"]=snap["symbol"].astype(str).str.upper().str.strip()
    rows=[];checks=[]
    for fold,source in [("2024-12-31",a.features_2024),("2025-12-31",a.features_2025)]:
        z=pd.read_csv(source)
        req={"fold_date","symbol","reporting_mode","source_available_utc","qa_status"}|set(FEATURES)
        if not req.issubset(z):raise SystemExit("Missing input PIT fields: "+str(req-set(z)))
        if not z["fold_date"].astype(str).eq(fold).all():raise SystemExit("Fold contamination")
        if not z["qa_status"].eq("CANDIDATE_SHORT_HORIZON_ONLY").all():
            raise SystemExit("Unapproved numeric fact source encountered")
        z["symbol"]=z["symbol"].astype(str).str.upper().str.strip()
        z["available"]=pd.to_datetime(z["source_available_utc"],utc=True,format="mixed",errors="coerce")
        cutoff=fold_close(fold)
        if z["available"].isna().any() or (z["available"]>cutoff).any():
            raise SystemExit("Future or missing historical financial publication timestamps")
        current=snap[snap["date"].eq(pd.Timestamp(fold))][["symbol"]].drop_duplicates()
        if (~z["symbol"].isin(set(current["symbol"]))).any():
            raise SystemExit("Source candidate outside original historical stock universe")
        z=z.sort_values(["symbol","reporting_mode"],
                        ascending=[True,True]).drop_duplicates("symbol",keep="first")
        clean=z[["symbol","reporting_mode","source_available_utc"]+FEATURES]
        joined=current.merge(clean,on="symbol",how="left",validate="1:1")
        joined["date"]=fold
        joined["has_verified_annual_yoy_source"]=joined["source_available_utc"].notna()
        joined["original_fold_asof_utc"]=cutoff.isoformat()
        rows.append(joined)
        checks.append({
            "fold":fold,"original_universe_symbols":len(current),
            "matched_annual_yoy_companies":int(joined["has_verified_annual_yoy_source"].sum()),
            "coverage":float(joined["has_verified_annual_yoy_source"].mean()),
            "nonnull_revenue_yoy":int(joined["revenue_yoy_pct"].notna().sum()),
            "nonnull_PAT_yoy":int(joined["pat_yoy_pct"].notna().sum()),
            "no_unmatched_companies_dropped":len(joined)==len(current),
        })
    matrix=pd.concat(rows,ignore_index=True)
    if matrix.duplicated(["date","symbol"]).any():raise SystemExit("Duplicate as-of rows")
    if matrix["revenue_yoy_pct"].notna().sum()<20:
        raise SystemExit("Insufficient verified 1y source coverage for research matrix")
    matrix.to_parquet(out/"v11_4_research_2024_2025_annual_yoy_PIT.parquet",index=False,compression="zstd")
    checks_df=pd.DataFrame(checks)
    checks_df.to_csv(out/"research_feature_coverage_by_fold.csv",index=False)
    summary={
        "scope":"FROZEN_UNIVERSE_FEATURE_MATRIX_2_FOLDS_ONLY",
        "folds":2,"rows":len(matrix),"frozen_universe_no_survivorship_substitution":True,
        "source_revenue_values_are_XBRL_verified_only":True,
        "all_missing_numerics_remain_NaN":True,
        "historical_label_leakage_possible":False,
        "old_5y_7y_missingness_untouched":True,
        "final_18fold_model_backtest_done":False,
        "v10_production_changed":False,
        "coverage":checks,
    }
    (out/"short_horizon_2fold_matrix_summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)

if __name__=="__main__":main()
