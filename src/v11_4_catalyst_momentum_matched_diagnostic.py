"""Exploratory, predefined NSE catalyst associations after fold/momentum/liquidity matching.

NO training or model selection. Uses only 18 original frozen folds and
matured/clean y6 outcomes, and reports every event type + horizon. Within
each fold, exposure is compared to nonexposure in the same historical
20%-rank momentum and liquidity strata (minimum 3 unexposed controls).
Still observational/confounded, not V10 incremental-alfa validation.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd
import numpy as np

TYPES=("capacity_expansion","order_win","regulatory","promoter_activity",
       "corporate_action","buyback","dilution","earnings")
WINDOWS=(90,180,365)
EVAL_ASOF=pd.Timestamp("2026-10-08")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--features",required=True)
    p.add_argument("--frozen-snapshot",required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    f=pd.read_parquet(a.features)
    labels=pd.read_parquet(a.frozen_snapshot,columns=[
        "date","symbol","y6","y6_mature_date","integrity_y6_clean",
        "integrity_feature_clean"])
    f["date"]=pd.to_datetime(f["date"],errors="coerce").dt.normalize()
    labels["date"]=pd.to_datetime(labels["date"],errors="coerce").dt.normalize()
    labels["y6_mature_date"]=pd.to_datetime(labels["y6_mature_date"],errors="coerce")
    if f.duplicated(["date","symbol"]).any() or labels.duplicated(["date","symbol"]).any():
        raise SystemExit("Duplicate frozen fold company keys")
    if any(c in f for c in ("y6","y12","y24","dd30_6m")):
        raise SystemExit("Future outcome label accidentally inside PIT model feature matrix")
    merged=f.merge(labels,on=["date","symbol"],how="left",suffixes=("","_label"),validate="1:1")
    m=merged[(merged["y6"].isin([0,1]))&
        (merged["y6_mature_date"]<=EVAL_ASOF)&
        merged["integrity_y6_clean"].eq(True)&
        merged["integrity_feature_clean"].eq(True)&
        merged["integrity_feature_clean_label"].eq(True)].copy()
    for c in ("ret_120","avg_turnover_63"):
        m[c]=pd.to_numeric(m[c],errors="coerce").replace([np.inf,-np.inf],np.nan)
        m=m[m[c].notna()]
    m["momentum_bin"]=np.minimum(4,np.floor((m.groupby("date")["ret_120"].rank(pct=True)-1e-9)*5)).astype(int)
    m["liquidity_bin"]=np.minimum(4,np.floor((m.groupby("date")["avg_turnover_63"].rank(pct=True)-1e-9)*5)).astype(int)
    m["stratum"]=m["date"].dt.strftime("%Y-%m-%d")+"|"+m["momentum_bin"].astype(str)+"|"+m["liquidity_bin"].astype(str)
    records=[];perfold=[]
    for typ in TYPES:
        for window in WINDOWS:
            signal=f"nse_{typ}_{window}d"
            if signal not in m:raise SystemExit("Missing predefined catalyst class: "+signal)
            m["exposed"]=m[signal].gt(0)
            pooled=m.groupby(["stratum","exposed"],observed=True)["y6"].agg(["size","sum"]).unstack("exposed",fill_value=0)
            if not (("size",True) in pooled.columns and ("size",False) in pooled.columns):
                raise SystemExit("Insufficient exposure/control sample")
            valid=(pooled["size",True]>0)&(pooled["size",False]>=3)
            matched=pooled.loc[valid]
            n_exposed=int(matched["size",True].sum())
            hits_exposed=float(matched["sum",True].sum())
            expected_controls=float((matched["size",True]*
                 matched["sum",False]/matched["size",False]).sum())
            ratio=(hits_exposed/expected_controls) if expected_controls>0 else None
            records.append({
                "event_type":typ,"lookback_days":window,
                "matched_exposed_stock_folds":n_exposed,
                "all_exposed_stock_folds":int(m["exposed"].sum()),
                "matched_exposure_fraction":n_exposed/max(int(m["exposed"].sum()),1),
                "observed_matched_y6_rate":hits_exposed/max(n_exposed,1),
                "expected_matched_control_y6_rate":expected_controls/max(n_exposed,1),
                "within_fold_momentum_liquidity_matched_rate_ratio":ratio,
                "matched_strata":len(matched),
                "causal_interpretation_valid":False,
                "model_training_performed":False,
            })
            strata_df=matched.copy()
            strata_df["date"]=pd.to_datetime(strata_df.index.str[:10])
            for date,grp in strata_df.groupby("date"):
                exposed=int(grp["size",True].sum())
                observed=float(grp["sum",True].sum())
                expected=float((grp["size",True]*grp["sum",False]/grp["size",False]).sum())
                perfold.append({
                    "fold_date":str(date.date()),"event_type":typ,"lookback_days":window,
                    "exposed_matched":exposed,"observed_y6_hits":observed,
                    "expected_control_y6_hits":expected,
                    "withinfold_rate_ratio":observed/expected if expected>0 else None,
                })
    pd.DataFrame(records).to_csv(out/"catalyst_all_classes_matched_y6_exploratory.csv",index=False)
    pd.DataFrame(perfold).to_csv(out/"catalyst_matched_y6_by_fold.csv",index=False)
    summary={
        "scope":"UNTRAINED_OBSERVATIONAL_MOMENTUM_LIQUIDITY_MATCHED_EXPLORATION",
        "frozen_mature_clean_y6_stockfolds":len(m),
        "folds":int(m["date"].nunique()),
        "preset_catalyst_event_classes":len(TYPES),
        "preset_lookback_windows":list(WINDOWS),
        "matched_on":"same fold, within-fold quintile(ret_120), within-fold quintile(avg_turnover_63)",
        "control_min_per_stratum":3,
        "all_24_comparisons_reported":True,
        "multiplicity_not_corrected":True,
        "V10_rank_selection_comparison_completed":False,
        "true_model_accuracy_improvement_demonstrated":False,
        "training_scores_unmodified":True
    }
    (out/"catalyst_matched_association_summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)
    print(pd.DataFrame(records)[["event_type","lookback_days","matched_exposed_stock_folds","matched_exposure_fraction","within_fold_momentum_liquidity_matched_rate_ratio"]].to_string(index=False),flush=True)
    if summary["folds"]!=18 or len(m)<10000:raise SystemExit("Not enough original historical folds")
if __name__=="__main__":main()
