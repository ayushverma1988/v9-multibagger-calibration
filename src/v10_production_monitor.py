from __future__ import annotations

import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

import calibrate_v9_2 as base
import calibrate_v9_4_1_integrity as prod
import v10_2_market_integrity as market_integrity

EPS=1e-6


def _read_existing(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    x=pd.read_csv(path)
    if "prediction_date" in x:
        x["prediction_date"]=pd.to_datetime(x["prediction_date"])
    return x


def append_batch(ledger: pd.DataFrame, top: pd.DataFrame, model_cfg: dict) -> pd.DataFrame:
    top=top.copy()
    top["date"]=pd.to_datetime(top["date"])
    if top["date"].nunique()!=1 or len(top)!=10:
        raise RuntimeError("Production batch must contain exactly one date and 10 rows")
    pred_date=pd.Timestamp(top["date"].iloc[0]).normalize()

    rows=[]
    for r in top.itertuples(index=False):
        rows.append({
            "prediction_date":pred_date,
            "symbol":str(r.symbol),
            "isin":str(r.isin) if pd.notna(r.isin) else "",
            "rank":int(r.production_rank),
            "close":float(r.close),
            "adj_close":float(r.adj_close) if hasattr(r,"adj_close") and pd.notna(r.adj_close) else float(r.close),
            "target_2x":float(r.target_2x),
            "p100_cal":float(r.p100_cal if hasattr(r,"p100_cal") else r.p_cal),
            "p_dd30_cal":float(r.p_dd30_cal if hasattr(r,"p_dd30_cal") and pd.notna(r.p_dd30_cal) else r.p_dd30),
            "selection_score":float(r.production_selection_score) if hasattr(r,"production_selection_score") and pd.notna(r.production_selection_score) else np.nan,
            "avg_turnover_63":float(r.avg_turnover_63),
            "model_version":model_cfg["production_model"],
            "selector":model_cfg["frozen_selector"]["name"],
            "outcome_status":"pending",
            "mature_date":pd.NaT,
            "hit2x_6m":np.nan,
            "dd30_6m":np.nan,
            "max_return_6m":np.nan,
            "min_return_6m":np.nan,
            "integrity_censored":False,
        })
    new=pd.DataFrame(rows)
    if ledger.empty:
        out=new
    else:
        out=pd.concat([ledger,new],ignore_index=True,sort=False)
    out["prediction_date"]=pd.to_datetime(out["prediction_date"])
    out=out.sort_values(["prediction_date","rank","symbol"])
    out=out.drop_duplicates(["prediction_date","symbol"],keep="last").reset_index(drop=True)
    return out


def _identity_group(daily: pd.DataFrame, isin: str, symbol: str) -> pd.DataFrame:
    isin=str(isin or "").strip().upper()
    symbol=str(symbol or "").strip().upper()
    if isin and isin not in {"NAN","NONE","<NA>"} and "isin" in daily:
        q=daily[daily["isin"].astype(str).str.upper().eq(isin)].copy()
        if len(q):
            return q.sort_values("date")
    return daily[daily["symbol"].astype(str).str.upper().eq(symbol)].copy().sort_values("date")


def evaluate_pending(ledger: pd.DataFrame, cfg: dict, events: str, legacy_dir: str|None, data_end: pd.Timestamp) -> pd.DataFrame:
    if ledger.empty:
        return ledger
    pending=ledger[ledger["outcome_status"].fillna("pending").eq("pending")].copy()
    if pending.empty:
        return ledger

    # Avoid a full market reload until at least one prediction is old enough
    # to plausibly be 126 trading sessions mature.
    oldest=pd.to_datetime(pending["prediction_date"]).min()
    if pd.isna(oldest) or (data_end-oldest).days < 150:
        return ledger

    base.cfg_h24=504
    market_integrity.install_on_base()
    daily=base.load_market(2003,int(data_end.year),legacy_dir)
    daily=market_integrity.normalize_split_bonus_volume(daily,2003,int(data_end.year))
    daily=market_integrity.stitch_symbol_changes_same_isin(daily)
    daily=prod.mark_integrity(daily,events)
    daily["date"]=pd.to_datetime(daily["date"])
    calendar=np.array(sorted(daily["date"].drop_duplicates().to_numpy(dtype="datetime64[ns]")))
    cal_lookup={d:i for i,d in enumerate(calendar)}
    h=int(cfg["monitoring"]["horizon_sessions"])
    min_turn=5_000_000.0

    out=ledger.copy()
    for idx,row in out[out["outcome_status"].fillna("pending").eq("pending")].iterrows():
        pdte=pd.Timestamp(row["prediction_date"]).normalize()
        d=np.datetime64(pdte)
        ci=cal_lookup.get(d)
        if ci is None or ci+h>=len(calendar):
            continue
        mature=pd.Timestamp(calendar[ci+h])
        g=_identity_group(daily,row.get("isin",""),row["symbol"])
        if g.empty:
            continue
        dates=g["date"].to_numpy(dtype="datetime64[ns]")
        calpos=np.array([cal_lookup.get(x,-1) for x in dates],dtype=int)
        valid=calpos>=0
        g=g.loc[valid].copy()
        dates=dates[valid]; calpos=calpos[valid]
        start=np.searchsorted(calpos,ci,side="right")
        stop=np.searchsorted(calpos,ci+h,side="right")
        fut=g.iloc[start:stop].copy()
        if fut.empty:
            out.at[idx,"outcome_status"]="matured"
            out.at[idx,"mature_date"]=mature
            out.at[idx,"hit2x_6m"]=0.0
            out.at[idx,"dd30_6m"]=1.0
            out.at[idx,"max_return_6m"]=-1.0
            out.at[idx,"min_return_6m"]=-1.0
            continue

        # Structural/adjustment discontinuities censor prospective evaluation
        # exactly as production integrity policy does.
        if "_blocking_jump" in fut and bool(fut["_blocking_jump"].fillna(False).any()):
            out.at[idx,"outcome_status"]="censored_integrity"
            out.at[idx,"mature_date"]=mature
            out.at[idx,"integrity_censored"]=True
            continue

        p0=float(row["adj_close"])
        pp=fut["adj_close"].to_numpy(float)
        tt=fut["turnover"].to_numpy(float)
        cp=np.array([cal_lookup[np.datetime64(x)] for x in fut["date"]],dtype=int)

        hit2x=False
        if len(pp)>=3:
            consecutive=(cp[1:-1]==cp[:-2]+1)&(cp[2:]==cp[:-2]+2)
            med=np.array([np.nanmedian(pp[k:k+3]) for k in range(len(pp)-2)])
            avt=np.array([np.nanmean(tt[k:k+3]) for k in range(len(tt)-2)])
            hit2x=bool(np.any(consecutive & np.isfinite(med) & (med>=2*p0) & np.isfinite(avt) & (avt>=min_turn)))

        disappeared=int(cp[-1]) < ci+h
        dd30=bool((len(pp)>0 and np.nanmin(pp)<=0.7*p0) or disappeared)
        out.at[idx,"outcome_status"]="matured"
        out.at[idx,"mature_date"]=mature
        out.at[idx,"hit2x_6m"]=float(hit2x)
        out.at[idx,"dd30_6m"]=float(dd30)
        out.at[idx,"max_return_6m"]=float(np.nanmax(pp)/p0-1)
        out.at[idx,"min_return_6m"]=float(np.nanmin(pp)/p0-1)

    return out


def calibration_slope(y,p):
    y=np.asarray(y,int); p=np.clip(np.asarray(p,float),EPS,1-EPS)
    if len(y)<20 or len(np.unique(y))<2 or y.sum()<5:
        return np.nan
    m=LogisticRegression(C=1e6,solver="lbfgs",max_iter=2000)
    m.fit(logit(p).reshape(-1,1),y)
    return float(m.coef_[0,0])


def drift_report(ledger: pd.DataFrame, cfg: dict):
    m=ledger[ledger["outcome_status"].eq("matured")].copy()
    if m.empty:
        return {
            "status":"insufficient_prospective_history",
            "mature_predictions":0,"mature_batches":0,"warnings":[],
            "automatic_retuning":False,
        }, pd.DataFrame()

    batches=[]
    for d,g in m.groupby("prediction_date"):
        if len(g)==0: continue
        batches.append({
            "prediction_date":pd.Timestamp(d),
            "n":int(len(g)),
            "precision_2x":float(g["hit2x_6m"].mean()),
            "hit_batch":float(g["hit2x_6m"].sum()>0),
            "dd30_rate":float(g["dd30_6m"].mean()),
            "mean_p100":float(g["p100_cal"].mean()),
        })
    b=pd.DataFrame(batches)
    n=int(len(m)); nb=int(len(b))
    base0=cfg["accepted_baseline"]; mon=cfg["monitoring"]
    report={
        "status":"insufficient_prospective_history",
        "mature_predictions":n,
        "mature_batches":nb,
        "warnings":[],
        "automatic_retuning":False,
        "mean_precision_2x":float(b["precision_2x"].mean()) if len(b) else None,
        "hit_batch_rate":float(b["hit_batch"].mean()) if len(b) else None,
        "mean_dd30_rate":float(b["dd30_rate"].mean()) if len(b) else None,
    }
    if n < int(mon["min_mature_predictions_for_drift"]) or nb < int(mon["min_mature_batches_for_drift"]):
        return report,b

    y=m["hit2x_6m"].astype(int).to_numpy()
    p=np.clip(m["p100_cal"].astype(float).to_numpy(),EPS,1-EPS)
    br=float(brier_score_loss(y,p))
    slope=calibration_slope(y,p)
    report["p100_brier"]=br
    report["p100_calibration_slope"]=slope

    if report["mean_precision_2x"] < float(base0["mean_precision_100"])*float(mon["precision_retention_warning"]):
        report["warnings"].append("prospective_precision_below_70pct_of_baseline")
    if report["hit_batch_rate"] < float(base0["hit_fold_rate_100"])*float(mon["hit_rate_retention_warning"]):
        report["warnings"].append("prospective_hit_rate_below_75pct_of_baseline")
    if report["mean_dd30_rate"] > float(base0["mean_dd30_rate"])+float(mon["dd30_absolute_increase_warning"]):
        report["warnings"].append("prospective_dd30_more_than_10pp_above_baseline")
    if br > float(base0["p100_brier"])*float(mon["brier_multiple_warning"]):
        report["warnings"].append("p100_brier_more_than_1_25x_baseline")
    if np.isfinite(slope) and (slope<float(mon["calibration_slope_min_warning"]) or slope>float(mon["calibration_slope_max_warning"])):
        report["warnings"].append("p100_calibration_slope_outside_0_40_to_1_60")

    report["status"]="warning" if report["warnings"] else "healthy"
    return report,b


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--refresh-dir",required=True)
    ap.add_argument("--monitor-config",default="config_v10_production_monitor.json")
    ap.add_argument("--existing-ledger")
    ap.add_argument("--events")
    ap.add_argument("--legacy-dir")
    ap.add_argument("--output",default="outputs_v10_production_monitor")
    args=ap.parse_args()

    cfg=json.load(open(args.monitor_config))
    rdir=Path(args.refresh_dir); outdir=Path(args.output); outdir.mkdir(parents=True,exist_ok=True)
    manifest=json.load(open(rdir/"production_manifest.json"))
    top=pd.read_csv(rdir/"production_top10.csv")
    data_end=pd.Timestamp(manifest["data_end"]).normalize()

    age=(pd.Timestamp.utcnow().tz_localize(None).normalize()-data_end).days
    if age>int(cfg["monitoring"]["stale_data_days_fail"]):
        raise RuntimeError(f"Market data stale by {age} calendar days; fail-closed threshold exceeded")

    ledger=_read_existing(Path(args.existing_ledger)) if args.existing_ledger else pd.DataFrame()
    ledger=append_batch(ledger,top,cfg)
    if args.events and args.legacy_dir:
        ledger=evaluate_pending(ledger,cfg,args.events,args.legacy_dir,data_end)

    report,batches=drift_report(ledger,cfg)
    report.update({
        "production_model":cfg["production_model"],
        "latest_prediction_date":str(pd.Timestamp(top["date"].iloc[0]).date()),
        "market_data_age_days":int(age),
        "ledger_rows":int(len(ledger)),
        "ledger_batches":int(ledger["prediction_date"].nunique()),
        "model_rules_frozen":True,
        "p50_p25_production_weight":0.0,
    })

    ledger.to_csv(outdir/"prediction_ledger.csv",index=False)
    top.to_csv(outdir/"latest_top10.csv",index=False)
    if len(batches): batches.to_csv(outdir/"prospective_performance_by_batch.csv",index=False)
    else: pd.DataFrame(columns=["prediction_date","n","precision_2x","hit_batch","dd30_rate","mean_p100"]).to_csv(outdir/"prospective_performance_by_batch.csv",index=False)
    json.dump(report,open(outdir/"drift_report.json","w"),indent=2,default=str)
    json.dump({
        "status":"production_refresh_complete",
        "data_end":manifest["data_end"],
        "selected_rows":10,
        "drift_status":report["status"],
        "warnings":report["warnings"],
        "automatic_retuning":False,
    },open(outdir/"latest_status.json","w"),indent=2,default=str)
    print(json.dumps(report,indent=2,default=str))


if __name__=="__main__":
    main()
