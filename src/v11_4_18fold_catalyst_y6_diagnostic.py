"""Frozen 18-fold catalyst-vs-future-2x association diagnostic (NOT a trained predictor).

This script deliberately keeps future labels OUT of the original PIT matrix.
It joins labels solely in read-only evaluation and reports ALL predefined
catalyst categories, avoiding best-of-many cherry picking. Association is not
causation, a portfolio, a new model or out-of-sample improvement over V10.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd

TYPES=("capacity_expansion","order_win","regulatory","promoter_activity",
       "corporate_action","buyback","dilution","earnings")
WINDOWS=(90,180,365)
EVAL_ASOF=pd.Timestamp("2026-10-08")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--features",required=True)
    p.add_argument("--frozen-labels",required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args();out=Path(a.output);out.mkdir(exist_ok=True,parents=True)
    feat=pd.read_parquet(a.features)
    feat["date"]=pd.to_datetime(feat["date"],errors="coerce").dt.normalize()
    keys=["date","symbol"]
    if any(c in feat for c in ("y6","y12","y24","dd30_6m")):
        raise SystemExit("Forward labels present in source-only training feature matrix")
    need={"nse_"+name+"_180d" for name in TYPES}
    if not need.issubset(feat):
        raise SystemExit("Missing predeclared 180-day NSE metadata event categories")
    labels=pd.read_parquet(a.frozen_labels,columns=[
        "date","symbol","y6","y6_mature_date","integrity_y6_clean",
        "integrity_feature_clean","dd30_6m","ret_120"])
    labels["date"]=pd.to_datetime(labels["date"],errors="coerce").dt.normalize()
    labels["y6_mature_date"]=pd.to_datetime(labels["y6_mature_date"],errors="coerce").dt.normalize()
    if labels.duplicated(keys).any() or feat.duplicated(keys).any():
        raise SystemExit("Duplicate company-date identities; refuse biased association")
    combined=feat.merge(labels,on=keys,how="left",validate="1:1",suffixes=("","_frozen"))
    ok=(combined["y6"].isin([0,1])&
        combined["y6_mature_date"].notna()&
        (combined["y6_mature_date"]<=EVAL_ASOF)&
        combined["integrity_y6_clean"].eq(True)&
        combined["integrity_feature_clean_frozen"].eq(True)&
        combined["integrity_feature_clean"].eq(True))
    clean=combined[ok].copy()
    baseline=clean.groupby("date").agg(
        eligible_rows=("symbol","size"),historical_y6_hits=("y6","sum"),
        y6_rate=("y6","mean"),dd30_rate=("dd30_6m","mean")).reset_index()
    baseline.to_csv(out/"frozen_y6_base_rate_by_fold.csv",index=False)
    rows=[];folds=[]
    for typ in TYPES:
        for window in WINDOWS:
            col=f"nse_{typ}_{window}d"
            if col not in clean:raise SystemExit("Missing "+col)
            exposed=clean[col].gt(0)
            n=int(exposed.sum())
            hits=int(clean.loc[exposed,"y6"].sum())
            allmean=float(clean["y6"].mean())
            rate=hits/n if n else None
            rows.append({
                "event_type":typ,"lookback_days":window,
                "eligible_stock_folds":len(clean),
                "exposed_stock_folds":n,"y6_double_hits":hits,
                "exposed_y6_rate":rate,"all_market_y6_rate":allmean,
                "unadjusted_rate_ratio":rate/allmean if n and allmean>0 else None,
                "exposed_dd30_rate":float(clean.loc[exposed,"dd30_6m"].mean()) if n else None,
                "nonexposed_y6_rate":float(clean.loc[~exposed,"y6"].mean()) if (~exposed).any() else None,
                "confounding_not_controlled":True,
                "predictor_trained":False,
            })
            for fold,subset in clean.groupby("date"):
                x=subset[subset[col].gt(0)]
                folds.append({"date":str(fold.date()),"event_type":typ,
                              "lookback_days":window,
                              "eligible_stock_folds":len(subset),"exposed":len(x),
                              "exposed_y6_hits":int(x["y6"].sum()),
                              "exposed_y6_rate":float(x["y6"].mean()) if len(x) else None})
    pd.DataFrame(rows).to_csv(out/"all_prespecified_catalyst_types_descriptive_y6.csv",index=False)
    pd.DataFrame(folds).to_csv(out/"by_fold_catalyst_descriptive_y6.csv",index=False)
    foldcount=clean["date"].nunique()
    summary={
        "scope":"UNADJUSTED_READ_ONLY_OUTCOME_ASSOCIATION_NOT_MODEL_BACKTEST",
        "frozen_fold_dates_in_source":int(feat["date"].nunique()),
        "folds_with_matured_clean_y6":int(foldcount),
        "eligible_stock_dates":len(clean),
        "universe_stock_dates_before_label_exclusion":len(feat),
        "prespecified_event_classes":len(TYPES),
        "prespecified_lookbacks":list(WINDOWS),
        "all_market_y6_base_rate":float(clean["y6"].mean()) if len(clean) else None,
        "frozen_V10_selection_comparison_completed":False,
        "does_not_validate_incremental_alpha":True,
        "no_label_leak_into_source_feature_file":True,
        "V10_production_untouched":True,
        "method":"Event-presence group outcome frequency only, not adjusted causal inference.",
    }
    (out/"catalyst_y6_diagnostic_summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)
    print(pd.DataFrame(rows)[["event_type","lookback_days","exposed_stock_folds","y6_double_hits","unadjusted_rate_ratio"]].to_string(index=False),flush=True)
    if foldcount<12:raise SystemExit("Fewer than 12 eligible mature y6 folds")
    if len(clean)<10000:raise SystemExit("Insufficient historical clean market y6 rows")
if __name__=="__main__":main()
