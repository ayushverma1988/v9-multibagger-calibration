from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd

import calibrate_v9_2 as base
import calibrate_v9_4 as v94
import v10_3_bse_current as bsemod


def find_file(root,name):
    hits=list(Path(root).rglob(name))
    if not hits:
        raise FileNotFoundError(f"{name} not found under {root}")
    return hits[0]


def censor_structural_forward(data,daily):
    d=data.copy()
    groups={s:g.sort_values("date").reset_index(drop=True) for s,g in daily.groupby("symbol",sort=False)}
    clean=[]
    for r in d.itertuples(index=False):
        g=groups.get(r.symbol)
        ok=True
        if g is None or g.empty:
            ok=False
        else:
            dates=g["date"].to_numpy(dtype="datetime64[ns]")
            pos=int(np.searchsorted(dates,np.datetime64(pd.Timestamp(r.date)),side="right")-1)
            events=np.flatnonzero(g["_blocking_event"].to_numpy(dtype=bool))
            future=events[events>pos]-pos
            ok=not np.any((future>=1)&(future<=126))
        clean.append(bool(ok))
    d["integrity_y6_clean"]=clean
    bad=~d["integrity_y6_clean"]
    for c in ["y6","dd30_6m"]:
        if c in d:d.loc[bad,c]=np.nan
    return d


def calibrated_from_prior(prior,pcol,target,raw):
    method,_=base.choose_calibrator_temporal(prior,target,pcol)
    if method is None:
        return np.full(len(raw),np.nan),None
    if method=="none":
        return np.asarray(raw,float),method
    q=prior.dropna(subset=[target,pcol])
    cal=base._fit_cal(method,q[pcol].to_numpy(),q[target].astype(int).to_numpy())
    return cal.predict(np.asarray(raw,float)),method


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--accepted-dir",required=True)
    ap.add_argument("--output",required=True)
    ap.add_argument("--config",default="config_v9_4_1.json")
    args=ap.parse_args()
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    cfg=json.load(open(args.config))
    base.cfg_h24=int(cfg["label_days"]["y24"])

    nse_data=pd.read_parquet(find_file(args.accepted_dir,"snapshot_dataset.parquet"))
    oos=pd.read_parquet(find_file(args.accepted_dir,"oos_predictions.parquet"))
    model_summary=json.load(open(find_file(args.accepted_dir,"summary.json")))
    spec=dict(model_summary["production_risk_config"])
    shares=(1.0,0.0,0.0)

    bse=bsemod.load_bse_history(2024,2026)
    actions=bsemod.resolve_actions(bsemod.load_bse_actions(2024,2026),bsemod.load_bse_symbol_history())
    bse,_=bsemod.apply_bse_split_bonus(bse,actions)
    bse,_=bsemod.stitch_same_isin(bse)
    bse,_=bsemod.mark_bse_structural_integrity(bse,actions)
    bse=base.add_features(bse)

    # Remove feature-contaminated snapshot dates before ranks are calculated.
    bf=bse.copy()
    bf.loc[~bf["integrity_feature_clean"],"avg_turnover_63"]=0.0
    snap=base.build_snapshots(bf,cfg)
    lab=base.add_labels(snap,bse,cfg)
    lab=censor_structural_forward(lab,bse)

    rows=[]; selections=[]
    seed=int(cfg["random_seed"])
    for td in sorted(pd.to_datetime(lab["date"].unique())):
        test=lab[(lab["date"]==td)&lab["y6"].notna()].copy()
        if len(test)<20:
            continue
        start=td-pd.DateOffset(years=int(cfg["rolling_train_years"]))
        train=nse_data[(nse_data["date"]<td)&(nse_data["date"]>=start)].copy()
        for target,mcol in [("y6","y6_mature_date"),("y12","y12_mature_date"),("y24","y24_mature_date"),("dd30_6m","dd30_6m_mature_date")]:
            if mcol in train.columns:
                train.loc[pd.to_datetime(train[mcol])>td,target]=np.nan
        train=train[train["y6"].notna()].copy()
        if len(train)<800 or train["y6"].sum()<20:
            continue

        elastic,gbm=base.make_models(seed)
        ps=base.structural_raw(train,test,cfg)
        pe=base.safe_predict_model(elastic,train,test,"y6")
        pg=base.safe_predict_model(gbm,train,test,"y6")
        ph=base.survival_raw(train,test,seed)
        arr=np.vstack([ps,pe,pg,ph]).T
        raw=np.nanmean(arr,axis=1)

        dd1,dd2=base.make_models(seed+101)
        pdd1=base.safe_predict_model(dd1,train,test,"dd30_6m")
        pdd2=base.safe_predict_model(dd2,train,test,"dd30_6m")
        ddarr=np.vstack([pdd1,pdd2]).T
        ddraw=np.nanmean(ddarr,axis=1)

        prior=oos[oos["date"]<td].copy()
        pcal,pmethod=calibrated_from_prior(prior,"p_raw","y6",raw)
        pdd,dmethod=calibrated_from_prior(prior,"p_dd30_raw","dd30_6m",ddraw)

        q=test[["date","symbol","isin","close","y6","dd30_6m"]].copy()
        q["p_cal"]=pcal
        q["p100_cal"]=pcal
        q["p50_cal"]=pcal
        q["p25_cal"]=pcal
        q["p_dd30_cal"]=pdd
        q["model_dispersion"]=np.nanstd(arr,axis=1)
        sel=v94.select_ladder_topk(q,spec,shares,int(cfg.get("selection_k",10)))
        if len(sel)!=int(cfg.get("selection_k",10)):
            continue

        base_rate=float(q["y6"].mean())
        precision=float(sel["y6"].mean())
        lift=precision/base_rate if base_rate>0 else np.nan
        dd=float(sel["dd30_6m"].mean())
        rows.append({
            "date":str(pd.Timestamp(td).date()),
            "eligible":int(len(q)),
            "base_rate_2x":base_rate,
            "precision_2x":precision,
            "lift_2x":lift,
            "hit_fold":float(precision>0),
            "dd30_rate":dd,
            "p100_calibrator":pmethod,
            "dd_calibrator":dmethod,
        })
        z=sel[["date","symbol","isin","close","y6","dd30_6m","p100_cal","p_dd30_cal","model_dispersion"]].copy()
        z["fold"]=str(pd.Timestamp(td).date())
        selections.append(z)

    metrics=pd.DataFrame(rows)
    metrics.to_csv(out/"bse_transfer_metrics_by_fold.csv",index=False)
    if selections:
        pd.concat(selections,ignore_index=True).to_csv(out/"bse_transfer_selections.csv",index=False)

    capped=np.clip(metrics["lift_2x"].dropna().to_numpy(float),0,10) if len(metrics) else np.array([])
    summary={
        "stage":"V10.3 limited BSE transfer diagnostic",
        "diagnostic_only":True,
        "model_or_selector_tuning_performed":False,
        "history_limit":"BSE source begins in 2024, so this cannot replace the 18-fold NSE validation",
        "folds":int(len(metrics)),
        "selected_rows":int(len(metrics)*int(cfg.get("selection_k",10))),
        "mean_precision_2x":float(metrics["precision_2x"].mean()) if len(metrics) else None,
        "median_precision_2x":float(metrics["precision_2x"].median()) if len(metrics) else None,
        "hit_fold_rate_2x":float(metrics["hit_fold"].mean()) if len(metrics) else None,
        "mean_capped_lift_2x":float(capped.mean()) if len(capped) else None,
        "mean_dd30":float(metrics["dd30_rate"].mean()) if len(metrics) else None,
        "fold_dates":metrics["date"].tolist() if len(metrics) else [],
        "interpretation":"Use only as cross-exchange transfer evidence. Do not retune V10.2 based on these few BSE folds.",
    }
    json.dump(summary,open(out/"summary.json","w"),indent=2,default=str)
    print(json.dumps(summary,indent=2,default=str))
    if len(metrics): print(metrics.to_string(index=False),flush=True)


if __name__=="__main__":
    main()
