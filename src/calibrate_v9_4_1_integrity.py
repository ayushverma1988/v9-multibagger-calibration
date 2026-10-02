from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd

import calibrate_v9_2 as base
import calibrate_v9_3_1 as v931
import calibrate_v9_4 as v94
import calibrate_v9_4_1 as v941
import v10_2_selected_window_integrity as integ


BLOCKING_CLASSES={"adjustment_factor_discontinuity","corporate_action_or_restructuring"}

import v10_2_market_integrity as market_integrity


def mark_integrity(daily: pd.DataFrame, events_path: str) -> pd.DataFrame:
    x=daily.sort_values(["symbol","date"]).copy()
    x["_adj_ret_i"]=x.groupby("symbol")["adj_close"].pct_change()
    x["_raw_ret_i"]=x.groupby("symbol")["close"].pct_change()
    x["_blocking_jump"]=False
    x["_blocking_type"]=""

    # Structured merger/demerger events are blocking boundaries regardless of
    # the observed one-day return. They change the economic identity/value of
    # the security and are not representable by a simple split factor.
    structured=market_integrity.structured_identity_events(
        int(x["date"].dt.year.min()),int(x["date"].dt.year.max())
    )
    identity_rows=[]
    if len(structured):
        for r in structured.itertuples(index=False):
            typ=str(getattr(r,"type","")).lower()
            if typ not in {"merger","demerger"}:
                continue
            isin=integ.norm_str(getattr(r,"isin",None))
            sym=integ.norm_str(getattr(r,"resolved_symbol",None) or getattr(r,"symbol",None))
            ex=pd.Timestamp(getattr(r,"ex_date"))
            q=x[x["isin"].map(integ.norm_str)==isin].copy() if isin else pd.DataFrame()
            if q.empty and sym:
                q=x[x["symbol"].map(integ.norm_str)==sym].copy()
            if q.empty:
                identity_rows.append({
                    "event_type":typ,"event_date":str(ex.date()),"isin":isin,
                    "resolved_symbol":sym,"mapped":False,"mapped_date":None,
                })
                continue
            q=q.sort_values("date")
            dates=q["date"].to_numpy(dtype="datetime64[ns]")
            pos=int(np.searchsorted(dates,np.datetime64(ex),side="left"))
            mapped_idx=None
            if pos<len(q):
                mapped_idx=q.index[pos]
            else:
                last_idx=q.index[-1]
                last_date=pd.Timestamp(q.loc[last_idx,"date"])
                if 0 <= (ex-last_date).days <= 30:
                    mapped_idx=last_idx
            if mapped_idx is not None:
                x.at[mapped_idx,"_blocking_jump"]=True
                x.at[mapped_idx,"_blocking_type"]=typ
                identity_rows.append({
                    "event_type":typ,"event_date":str(ex.date()),"isin":isin,
                    "resolved_symbol":sym,"mapped":True,
                    "mapped_date":str(pd.Timestamp(x.at[mapped_idx,"date"]).date()),
                })
            else:
                identity_rows.append({
                    "event_type":typ,"event_date":str(ex.date()),"isin":isin,
                    "resolved_symbol":sym,"mapped":False,"mapped_date":None,
                })
    x.attrs["structured_identity_event_audit"]=identity_rows

    ca=integ.prepare_events(events_path)

    cand=x[np.isfinite(x["_adj_ret_i"]) & (x["_adj_ret_i"].abs()>=.50)].copy()
    for idx,r in cand.iterrows():
        if bool(x.at[idx,"_blocking_jump"]) and str(x.at[idx,"_blocking_type"]) in {"merger","demerger"}:
            continue
        ar=float(r["_adj_ret_i"])
        rr=float(r["_raw_ret_i"]) if np.isfinite(r["_raw_ret_i"]) else np.nan

        # Pure adjustment-factor discontinuity: normalized price jumps but raw
        # traded close does not. This is observable from prices alone.
        if np.isfinite(rr) and abs(ar-rr)>=.25 and abs(rr)<.30:
            x.at[idx,"_blocking_jump"]=True
            x.at[idx,"_blocking_type"]="adjustment_factor_discontinuity"
            continue

        # For large raw traded-price jumps, require point-in-time corporate-
        # action evidence. Use a narrower operational window than the diagnostic
        # audit to avoid unrelated events months earlier.
        if np.isfinite(rr) and abs(rr)>=.50:
            jd=pd.Timestamp(r["date"]).normalize()
            lo,hi=jd-pd.Timedelta(days=120),jd+pd.Timedelta(days=14)
            q=ca[(ca["event_date"]>=lo)&(ca["event_date"]<=hi)].copy()
            isin=integ.norm_str(r.get("isin",None)); sym=integ.norm_str(r["symbol"])
            if isin:
                qi=q[q["isin_norm"]==isin]
                q=qi if len(qi) else q[q["symbol_norm"]==sym]
            else:
                q=q[q["symbol_norm"]==sym]
            if len(q):
                x.at[idx,"_blocking_jump"]=True
                x.at[idx,"_blocking_type"]="corporate_action_or_restructuring"

    # Feature cleanliness is strictly backward-looking and label-free.
    clean=pd.Series(True,index=x.index,dtype=bool)
    for _,inds0 in x.groupby("symbol",sort=False).groups.items():
        inds=list(inds0)
        b=x.loc[inds,"_blocking_jump"].astype(int)
        dirty=b.rolling(253,min_periods=1).max().astype(bool)
        clean.loc[inds]=~dirty.to_numpy()
    x["integrity_feature_clean"]=clean
    return x


def build_clean_snapshots(daily: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    # Reuse the frozen snapshot/ranking code, but make dirty rows ineligible
    # before universe breadth/ranks are calculated.
    d=daily.copy()
    d.loc[~d["integrity_feature_clean"],"avg_turnover_63"]=0.0
    s=base.build_snapshots(d,cfg)

    # Add forward cleanliness flags using trading-day offsets to the next
    # confirmed blocking jump. These flags do not use outcomes.
    groups={sym:g.sort_values("date").reset_index(drop=True) for sym,g in daily.groupby("symbol",sort=False)}
    rec=[]
    for r in s.itertuples(index=False):
        g=groups.get(r.symbol)
        if g is None or g.empty:
            vals={"integrity_y6_clean":False,"integrity_y12_clean":False,"integrity_y24_clean":False}
        else:
            dates=g["date"].to_numpy(dtype="datetime64[ns]")
            pos=int(np.searchsorted(dates,np.datetime64(pd.Timestamp(r.date)),side="right")-1)
            jidx=np.flatnonzero(g["_blocking_jump"].to_numpy(dtype=bool))
            future=jidx[jidx>pos]-pos
            vals={
                "integrity_y6_clean":bool(not np.any((future>=1)&(future<=126))),
                "integrity_y12_clean":bool(not np.any((future>=1)&(future<=252))),
                "integrity_y24_clean":bool(not np.any((future>=1)&(future<=504))),
            }
        rec.append(vals)
    flags=pd.DataFrame(rec,index=s.index)
    for c in flags.columns:s[c]=flags[c].astype(bool)
    return s


def censor_integrity_labels(data: pd.DataFrame) -> pd.DataFrame:
    d=data.copy()
    if "integrity_y6_clean" in d:
        bad=~d["integrity_y6_clean"].fillna(False)
        for c in ["y6","dd30_6m","hit25_6m","hit50_6m"]:
            if c in d:d.loc[bad,c]=np.nan
        if "days_to_2x" in d:
            # days_to_2x is also consumed by the survival model; if the 24m
            # path is dirty it is censored below rather than here.
            pass
    if "integrity_y12_clean" in d and "y12" in d:
        d.loc[~d["integrity_y12_clean"].fillna(False),"y12"]=np.nan
    if "integrity_y24_clean" in d:
        bad=~d["integrity_y24_clean"].fillna(False)
        if "y24" in d:d.loc[bad,"y24"]=np.nan
        if "days_to_2x" in d:d.loc[bad,"days_to_2x"]=np.nan
    return d


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",default="config_v9_4_1.json")
    ap.add_argument("--output",default="outputs_v9_4_1_integrity")
    ap.add_argument("--legacy-dir",default=None)
    ap.add_argument("--events",required=True)
    args=ap.parse_args()

    cfg=json.load(open(args.config))
    outdir=Path(args.output); outdir.mkdir(parents=True,exist_ok=True)
    base.cfg_h24=int(cfg["label_days"]["y24"])
    end_year=pd.Timestamp.today().year

    print("Loading market and marking integrity discontinuities...",flush=True)
    market_integrity.install_on_base()
    daily=base.load_market(int(cfg["start_year"]),end_year,args.legacy_dir)
    pd.DataFrame(market_integrity.LAST_ACTION_RESOLUTION_AUDIT).to_csv(
        outdir/"split_bonus_symbol_resolution_audit.csv",index=False
    )
    daily=market_integrity.normalize_split_bonus_volume(
        daily,int(cfg["start_year"]),end_year
    )
    pd.DataFrame(market_integrity.LAST_VOLUME_NORMALIZATION_AUDIT).to_csv(
        outdir/"split_bonus_volume_normalization_audit.csv",index=False
    )
    daily=market_integrity.stitch_symbol_changes_same_isin(daily)
    pd.DataFrame(market_integrity.LAST_SYMBOL_STITCH_AUDIT).to_csv(
        outdir/"symbol_name_change_stitch_audit.csv",index=False
    )
    daily=base.add_features(daily)
    daily=mark_integrity(daily,args.events)
    pd.DataFrame(daily.attrs.get("structured_identity_event_audit",[])).to_csv(
        outdir/"merger_demerger_event_audit.csv",index=False
    )

    # Hard integrity assertions for the two forensic cases that exposed the bug.
    # These are data-correctness checks only; they do not use model outcomes.
    tips=daily[(daily["isin"].map(integ.norm_str)=="INE716B01029") & (daily["date"]==pd.Timestamp("2023-04-21"))]
    if len(tips):
        tr=float(tips.iloc[0]["_adj_ret_i"])
        if abs(tr) >= 0.50:
            raise RuntimeError(f"TIPS/TIPSMUSIC split normalization unresolved: adjusted return={tr:.6f}")
    mirza=daily[(daily["symbol"]=="MIRZAINT") & (daily["date"]==pd.Timestamp("2023-03-29"))]
    if len(mirza) and not bool(mirza.iloc[0]["_blocking_jump"]):
        raise RuntimeError("MIRZAINT demerger discontinuity was not marked blocking")

    jump_audit=daily[daily["_blocking_jump"]][
        ["date","symbol","isin","close","adj_close","_adj_ret_i","_raw_ret_i","_blocking_type"]
    ].copy()
    jump_audit.to_csv(outdir/"blocking_jumps.csv",index=False)

    print("Building integrity-clean snapshots/labels...",flush=True)
    snap=build_clean_snapshots(daily,cfg)
    data=base.add_labels(snap,daily,cfg)
    data=v94.add_threshold_labels(data,daily,cfg)
    data=censor_integrity_labels(data)
    data.to_parquet(outdir/"snapshot_dataset.parquet",index=False)

    print("Running frozen V9.4.1 modeling on integrity-clean data...",flush=True)
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
    tab100.to_csv(outdir/"calibrator_100_comparison.csv",index=False)
    tabdd.to_csv(outdir/"downside_calibrator_comparison.csv",index=False)
    for target in v94.THRESHOLDS:
        threshold_info[target]["table"].to_csv(outdir/f"{target}_calibrator_comparison.csv",index=False)
    chosen931.to_csv(outdir/"chosen_v931_config_by_fold.csv",index=False)
    chosen94.to_csv(outdir/"chosen_v94_ladder_by_fold.csv",index=False)
    chosen941.to_csv(outdir/"chosen_v941_gates_by_fold.csv",index=False)
    comp.to_csv(outdir/"selection_metrics_by_fold.csv",index=False)

    production_spec,risk_search,_=v931.optimize_from_cache(metric_cache931,cfg)
    if production_spec is None:production_spec=next(v931.cfg_grid(cfg))
    shares,ladder_search,ladder_baseline,production_gates=v941.optimize_ladder_gated(oos,production_spec,cfg)
    risk_search.to_csv(outdir/"risk_constraint_search.csv",index=False)
    ladder_search.to_csv(outdir/"gated_ladder_search.csv",index=False)

    # Make dirty current rows ineligible before current-universe ranks.
    daily_current=daily.copy()
    daily_current.loc[~daily_current["integrity_feature_clean"],"avg_turnover_63"]=0.0
    current=base.fit_current(data,daily_current,oos,best100,bestdd,cfg)
    threshold_cur=v94.current_threshold_predictions(data,daily_current,oos,threshold_info,cfg)
    current=v941.build_current_v941(
        current,threshold_cur,oos,threshold_info,production_spec,shares,production_gates,cfg
    )
    current.to_csv(outdir/"current_selection.csv",index=False)
    current[current["selected_v941"]].head(50).to_csv(outdir/"current_top50.csv",index=False)

    summaries={name:v94.summarize_comparison(comp,name) for name in ["V9.4.1","V9.4","V9.3.1","V9.2_pcal"]}
    metrics25=v94.calibration_metrics(oos,"hit25_6m","p25_cal")
    metrics50=v94.calibration_metrics(oos,"hit50_6m","p50_cal")
    metrics100=v94.calibration_metrics(oos,"y6","p100_cal")
    summary={
        "model":"V9.4.1 integrity-corrected challenger",
        "base_pipeline_version":cfg["pipeline_version"],
        "data_start":str(daily["date"].min().date()),
        "data_end":str(daily["date"].max().date()),
        "integrity_policy":{
            "feature_lag_max_trading_days":252,
            "feature_price_rows_covered":253,
            "primary_label_censor_trading_days":126,
            "secondary_label_censor_trading_days":{"y12":252,"y24":504},
            "blocking_classes":sorted(BLOCKING_CLASSES),
            "corporate_action_match_window_days":{"before":120,"after":14},
            "uses_outcomes_for_integrity_decision":False,
            "split_bonus_identity_resolution":"action ISIN -> symbol active on ex-date via symbol_history/nse.parquet; fallback to action symbol",
            "dividends_in_adjusted_close":False,
            "demergers_mergers_scaled":False,
            "merger_demerger_policy":"structural reset at mapped effective trading boundary; feature history excluded for 252 lags and forward labels censored when crossing event",
            "name_symbol_change_policy":"same ISIN -> stitch identifier history with no price reset; unresolved/new ISIN -> no synthetic stitching",
            "split_bonus_volume_policy":"pre-event share volume normalized by inverse price factor; monetary turnover unchanged",
        },
        "blocking_jumps":int(daily["_blocking_jump"].sum()),
        "snapshots_after_integrity_gate":int(len(data)),
        "y6_censored_rows":int((~data["integrity_y6_clean"]).sum()),
        "y12_censored_rows":int((~data["integrity_y12_clean"]).sum()),
        "y24_censored_rows":int((~data["integrity_y24_clean"]).sum()),
        "forward_history":summaries,
        "probability_ladder":{
            "p25_coherent":metrics25,
            "p50_coherent":metrics50,
            "p100":metrics100,
            "best_calibrator_p25":threshold_info["hit25_6m"]["best"],
            "best_calibrator_p50":threshold_info["hit50_6m"]["best"],
            "best_calibrator_p100":best100,
        },
        "production_signal_quality":production_gates,
        "production_risk_config":production_spec,
        "production_ladder_shares_within_upside_weight":{"p100":shares[0],"p50":shares[1],"p25":shares[2]},
        "fallback_rule":"This run is an integrity challenger only; original V9.4.1 remains frozen until comparison and selected-window audit pass.",
    }
    json.dump(summary,open(outdir/"summary.json","w"),indent=2,default=str)
    print(json.dumps(summary,indent=2,default=str))


if __name__=="__main__":
    main()
