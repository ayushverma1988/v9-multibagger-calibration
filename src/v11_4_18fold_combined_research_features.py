"""Frozen V11.4 research feature matrix: 18-fold NSE event counts + verified 2fold annual trends.

No labels and no future outcome features are transferred into the research
feature matrix. Financials absent outside 2024/25 remain NaN, NEVER zero.
No selection rules, weights, model fitting, or production modifications.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd

PRICE_FEATURES=("ret_20","ret_60","ret_120","ret_252","mom_accel",
   "vol_accel","turnover_accel","off_high_252","above_low_252",
   "trend_consistency_60","volatility_60","avg_turnover_63","integrity_feature_clean")
ANNUAL_FEATURES=("revenue_yoy_pct","pat_yoy_pct","pat_margin_latest_pct",
   "pat_margin_delta_pp","profit_turnaround","earlier_profit_negative",
   "latest_fiscal_age_days")
FORBIDDEN=("y6","y12","y24","dd30_6m","days_to_2x","hit25_6m","hit50_6m")

def check_source_cutoff(df,timefield,datefield="date"):
    cutoff=pd.to_datetime(df[datefield],errors="raise").dt.tz_localize("Asia/Kolkata") +pd.Timedelta(hours=15,minutes=30)
    clock=pd.to_datetime(df[timefield],utc=True,format="mixed",errors="coerce")
    valid=clock.notna()&(clock<=cutoff.dt.tz_convert("UTC"))
    return valid

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--snapshot",required=True)
    p.add_argument("--catalysts",required=True)
    p.add_argument("--financials",required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    base=pd.read_parquet(a.snapshot,columns=["date","symbol",*PRICE_FEATURES])
    base["date"]=pd.to_datetime(base["date"],errors="coerce").dt.strftime("%Y-%m-%d")
    base["symbol"]=base["symbol"].astype(str).str.upper().str.strip()
    if base.duplicated(["date","symbol"]).any():raise SystemExit("Duplicate frozen market snapshot rows")
    cat=pd.read_parquet(a.catalysts)
    cat["date"]=cat["date"].astype(str)
    cat["symbol"]=cat["symbol"].astype(str).str.upper().str.strip()
    if cat.duplicated(["date","symbol"]).any():raise SystemExit("Duplicate event feature rows")
    if not check_source_cutoff(cat,"historical_asof_utc").all():
        raise SystemExit("Some event snapshots violate historical NSE cutoff")
    event_cols=[s for s in cat if s.startswith("nse_")]
    if not event_cols:raise SystemExit("No official NSE historical event features found")
    fin=pd.read_parquet(a.financials)
    fin["date"]=fin["date"].astype(str)
    fin["symbol"]=fin["symbol"].astype(str).str.upper().str.strip()
    if fin.duplicated(["date","symbol"]).any():raise SystemExit("Duplicate original PIT financial company rows")
    if not set(ANNUAL_FEATURES).issubset(fin):raise SystemExit("Missing annual financial feature columns")
    available=fin[fin["has_verified_annual_yoy_source"]].copy()
    if len(available) and not check_source_cutoff(available,"source_available_utc").all():
        raise SystemExit("Financial filings recorded after original fold")
    fin_cols=list(ANNUAL_FEATURES)+["has_verified_annual_yoy_source","source_available_utc","reporting_mode"]
    final=(cat[["date","symbol","historical_asof_utc",*event_cols]]
           .merge(base,on=["date","symbol"],how="left",validate="1:1")
           .merge(fin[["date","symbol",*fin_cols]],on=["date","symbol"],how="left",validate="1:1"))
    if final.duplicated(["date","symbol"]).any():raise SystemExit("Nonunique training research keys")
    if len(final)!=len(cat):raise SystemExit("Catalyst universe rows dropped")
    if any(t in final.columns for t in FORBIDDEN):raise SystemExit("Forward outcome leaked into feature matrix")
    if final["integrity_feature_clean"].isna().any():raise SystemExit("Original V10 feature eligibility missing")
    if final["has_verified_annual_yoy_source"].notna().sum()!=len(fin):
        raise SystemExit("Original 2-fold finance join did not preserve row coverage")
    # In the other 16 folds, all annual numeric fields are completely missing, not zero.
    old=final[~final["date"].isin(["2024-12-31","2025-12-31"])]
    if old[list(ANNUAL_FEATURES)].notna().any().any():
        raise SystemExit("Unvalidated pre-2024 annual numeric data emerged")
    final.to_parquet(out/"v11_4_18fold_source_PIT_FEATURES_RESEARCH.parquet",index=False,compression="zstd")
    fold=(final.groupby("date").agg(
        original_stock_rows=("symbol","size"),
        annual_source_coverage=("has_verified_annual_yoy_source","count"),
        revenue_yoy_coverage=("revenue_yoy_pct","count"),
        catalyst_any_180d=("nse_catalyst_total_180d",lambda s:int((s>0).sum())),
        eligible_original_price_features=("integrity_feature_clean",lambda s:int(s.fillna(False).sum()))
    ).reset_index())
    fold.to_csv(out/"18fold_source_feature_coverage.csv",index=False)
    summary={
        "scope":"READ_ONLY_HISTORICAL_PIT_FEATURE_MATRIX_NO_OUTCOMES",
        "folds":int(final["date"].nunique()),
        "historical_rows":len(final),
        "annual_numeric_folds":sorted(final.loc[final["revenue_yoy_pct"].notna(),"date"].unique().tolist()),
        "historical_NSE_catalyst_columns":len(event_cols),
        "historical_NSE_lookbacks":[90,180,365],
        "numeric_annual_features_available_in_all_18_folds":False,
        "forward_target_label_columns_in_matrix":False,
        "missing_annual_values_imputed_to_zero":False,
        "V10_production_modified":False,
        "predictive_accuracy_measured":False
    }
    (out/"18fold_research_feature_summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)
    print(fold.tail(4).to_string(index=False),flush=True)
    if summary["folds"]!=18 or len(final)<18000:raise SystemExit("Frozen 18-fold research matrix incomplete")
if __name__=="__main__":main()
