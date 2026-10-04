from __future__ import annotations

import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

import calibrate_v9_2 as base
import calibrate_v9_3_1 as v931
import calibrate_v9_4 as v94
import calibrate_v9_4_1 as v941
import calibrate_v9_4_1_integrity as prod
import v10_2_market_integrity as market_integrity
import v10_4_features as f104


def coverage(df):
    out={}
    for c in f104.FEATURES_V10_4:
        if c in df:
            x=pd.to_numeric(df[c],errors="coerce")
            out[c]={
                "non_null":int(x.notna().sum()),
                "coverage":float(x.notna().mean()),
                "non_zero":int((x.fillna(0).abs()>1e-12).sum()),
            }
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",default="config_v9_4_1.json")
    ap.add_argument("--output",default="outputs_v10_4_challenger")
    ap.add_argument("--legacy-dir",default=None)
    ap.add_argument("--events",required=True)
    ap.add_argument("--pit-dir",required=True)
    args=ap.parse_args()

    cfg=json.load(open(args.config))
    outdir=Path(args.output); outdir.mkdir(parents=True,exist_ok=True)
    pit=Path(args.pit_dir)
    base.cfg_h24=int(cfg["label_days"]["y24"])
    end_year=pd.Timestamp.today().year

    print("Preparing frozen V10.4 point-in-time sources...",flush=True)
    ev=f104.prepare_events(args.events)
    shp=f104.prepare_shareholding(pit/"shareholding_master.parquet") if (pit/"shareholding_master.parquet").exists() else pd.DataFrame()
    ins=f104.prepare_insider(pit/"insider_trades.parquet") if (pit/"insider_trades.parquet").exists() else pd.DataFrame()
    fin=f104.prepare_financial(pit/"financial_results_metadata.parquet") if (pit/"financial_results_metadata.parquet").exists() else pd.DataFrame()
    print("sources events",len(ev),"shareholding",len(shp),"insider",len(ins),"financial",len(fin),flush=True)

    print("Loading integrity-corrected market...",flush=True)
    market_integrity.install_on_base()
    daily=base.load_market(int(cfg["start_year"]),end_year,args.legacy_dir)
    daily=market_integrity.normalize_split_bonus_volume(daily,int(cfg["start_year"]),end_year)
    daily=market_integrity.stitch_symbol_changes_same_isin(daily)
    daily=base.add_features(daily)
    daily=prod.mark_integrity(daily,args.events)

    print("Building clean snapshots + V10.4 features...",flush=True)
    snap=prod.build_clean_snapshots(daily,cfg)
    snap=f104.attach_features(snap,ev,shp,ins,fin)
    feature_cov=coverage(snap)

    # Freeze the model matrix extension before any walk-forward outcomes are
    # evaluated. Structural score weights/ranks remain exactly V10.2.
    original_features=list(base.MODEL_FEATURES)
    base.MODEL_FEATURES=original_features+[c for c in f104.FEATURES_V10_4 if c not in original_features]

    data=base.add_labels(snap,daily,cfg)
    data=v94.add_threshold_labels(data,daily,cfg)
    data=prod.censor_integrity_labels(data)
    data.to_parquet(outdir/"snapshot_dataset.parquet",index=False)

    print("Running 18-fold frozen-logic V10.4 challenger...",flush=True)
    oos=base.walk_forward(data,cfg)
    if oos.empty:raise RuntimeError("No OOS predictions generated")
    labels=data[["date","symbol","hit25_6m","hit50_6m","threshold_mature_date"]]
    oos=oos.merge(labels,on=["date","symbol"],how="left")

    best100,tab100=base.evaluate_calibrators(oos,"y6","p_raw")
    bestdd,tabdd=base.evaluate_calibrators(oos,"dd30_6m","p_dd30_raw")
    oos=base.apply_forward_calibration(oos,best100,"y6","p_raw","p_cal")
    oos=base.apply_forward_calibration(oos,bestdd,"dd30_6m","p_dd30_raw","p_dd30_cal")
    oos["p100_cal"]=oos["p_cal"]

    thresh=v94.walk_forward_thresholds(data,oos["date"].unique(),cfg)
    oos=oos.merge(thresh,on=["date","symbol"],how="left")
    oos,threshold_info=v94.calibrate_thresholds(oos)

    oos,chosen931,metric_cache931=v931.forward_select(oos,cfg)
    oos,chosen94=v94.forward_v94(oos,chosen931,cfg)
    oos,chosen941=v941.forward_v941(oos,chosen931,cfg)
    comp=v941.comparison_by_fold(oos,cfg)

    oos.to_parquet(outdir/"oos_predictions.parquet",index=False)
    comp.to_csv(outdir/"selection_metrics_by_fold.csv",index=False)
    tab100.to_csv(outdir/"calibrator_100_comparison.csv",index=False)
    tabdd.to_csv(outdir/"downside_calibrator_comparison.csv",index=False)
    chosen931.to_csv(outdir/"chosen_v931_config_by_fold.csv",index=False)
    chosen94.to_csv(outdir/"chosen_v94_ladder_by_fold.csv",index=False)
    chosen941.to_csv(outdir/"chosen_v941_gates_by_fold.csv",index=False)

    production_spec,risk_search,_=v931.optimize_from_cache(metric_cache931,cfg)
    if production_spec is None:production_spec=next(v931.cfg_grid(cfg))
    shares,ladder_search,ladder_baseline,production_gates=v941.optimize_ladder_gated(oos,production_spec,cfg)
    risk_search.to_csv(outdir/"risk_constraint_search.csv",index=False)
    ladder_search.to_csv(outdir/"gated_ladder_search.csv",index=False)

    print("Scoring current market with V10.4 features...",flush=True)
    daily_current=daily.copy()
    daily_current.loc[~daily_current["integrity_feature_clean"],"avg_turnover_63"]=0.0
    daily_current=f104.attach_current_to_daily(daily_current,ev,shp,ins,fin)
    current=base.fit_current(data,daily_current,oos,best100,bestdd,cfg)
    threshold_cur=v94.current_threshold_predictions(data,daily_current,oos,threshold_info,cfg)
    current=v941.build_current_v941(current,threshold_cur,oos,threshold_info,production_spec,shares,production_gates,cfg)
    current.to_csv(outdir/"current_selection.csv",index=False)
    current[current["selected_v941"]].head(50).to_csv(outdir/"current_top50.csv",index=False)

    summaries={name:v94.summarize_comparison(comp,name) for name in ["V9.4.1","V9.4","V9.3.1","V9.2_pcal"]}
    p100=v94.calibration_metrics(oos,"y6","p100_cal")
    p50=v94.calibration_metrics(oos,"hit50_6m","p50_cal")
    p25=v94.calibration_metrics(oos,"hit25_6m","p25_cal")

    summary={
        "model":"V10.4 fundamental-event-ownership challenger",
        "control_model":"V10.2 integrity-corrected V9.4.1",
        "base_pipeline_version":cfg["pipeline_version"],
        "data_start":str(daily["date"].min().date()),
        "data_end":str(daily["date"].max().date()),
        "new_features":list(f104.FEATURES_V10_4),
        "feature_coverage":feature_cov,
        "point_in_time_sources":{
            "event_rows":int(len(ev)),"shareholding_rows":int(len(shp)),
            "insider_rows":int(len(ins)),"financial_metadata_rows":int(len(fin)),
        },
        "model_changes":{
            "structural_score":"unchanged",
            "selector":"unchanged",
            "labels":"unchanged",
            "integrity_policy":"unchanged from accepted V10.2",
            "model_feature_count_control":len(original_features),
            "model_feature_count_candidate":len(base.MODEL_FEATURES),
        },
        "blocking_jumps":int(daily["_blocking_jump"].sum()),
        "snapshots":int(len(data)),
        "forward_history":summaries,
        "probability_ladder":{"p25":p25,"p50":p50,"p100":p100,"best_calibrator_p100":best100},
        "production_signal_quality":production_gates,
        "production_risk_config":production_spec,
        "production_ladder_shares":{"p100":shares[0],"p50":shares[1],"p25":shares[2]},
        "rule_freeze":"published_v10_4_feature_freeze/summary.json",
    }
    json.dump(summary,open(outdir/"summary.json","w"),indent=2,default=str)
    print(json.dumps(summary,indent=2,default=str))


if __name__=="__main__":
    main()
