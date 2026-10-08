"""Freeze V11.4 standalone *research* model for genuinely prospective scoring.

This is NOT a V10 comparison and does NOT turn retrospective tuning into
confirmatory backtesting. 18 originally frozen stock dates, original equity
OHLC/turnover and NSE catalyst PIT features, original matured integrity-clean
six-month labels are historical TRAINING INPUTS only. The older pipeline's
ranking, scores, probabilities and selections are never used.

After fitting, predictions can be made only on newly sourced stock-date rows
with independently verified 15:30 IST NSE event coverage and sufficiently
fresh market-close data. Do NOT put arbitrary archive 2025 data through this
interface and pretend it is an October 2026 shortlist.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from datetime import datetime,timezone
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import v11_4_standalone_train_walkforward as core

VERSION="V11.4-standalone-research-frozen-20261008"
ASOF="2026-10-08"
MIN_HISTORY_FOLDS=4
SHOCK_LO=pd.Timestamp("2020-02-20")
SHOCK_HI=pd.Timestamp("2020-06-30")
MIN_CANDIDATE_FRESHNESS_DAYS=4
FLOAT_FEATURES=list(core.MODEL_FEATURES)
CONTEMPORARY_REQUIRED={
    "date","symbol","close","avg_turnover_63","integrity_feature_clean",
    "historical_asof_utc",*FLOAT_FEATURES
}
LABELS={"y6","y12","y24","dd30_6m","y6_mature_date","integrity_y6_clean",
        "p_cal","p_raw","p100_cal","selected_v941","selected_v94"}
ACCEPTANCE={"min_test_folds":12,"mean_top10_jaccard":.80,
            "worst_fold_p05_top10_jaccard":.60}
CAL_TARGET=core.MIN_CALIBRATION_POSITIVES
PRIOR_STRENGTH=core.SPARSE_CAL_PRIOR_STRENGTH

def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):h.update(chunk)
    return h.hexdigest()

def strip_pandemic_matured_outcomes(x,enabled=True):
    if not enabled:return pd.Series(True,index=x.index)
    t=pd.to_datetime(x["y6_mature_date"],utc=True,errors="coerce").dt.tz_convert(None)
    start=pd.to_datetime(x["date"]).dt.normalize()
    # Original selection -> observed y6 maturity crossed initial COVID shock.
    return ~((start<=SHOCK_HI)&t.ge(SHOCK_LO))

def fit_calibration_params(cal_prediction,cal_y,train_y):
    q=np.clip(np.asarray(cal_prediction,dtype=float),1e-5,1-1e-5)
    y=np.asarray(cal_y,dtype=int)
    prior=np.asarray(train_y,dtype=int)
    if len(y)<50 or len(prior)<core.MIN_BASE_TRAIN_ROWS:
        raise ValueError("Insufficient mature calibration/training labels")
    logits=np.log(q/(1-q))
    if core.calibration_mode(y)=="platt":
        lr=LogisticRegression(C=core.CALIBRATION_C,solver="lbfgs",
                              class_weight=None,max_iter=400,random_state=31)
        lr.fit(logits.reshape(-1,1),y)
        if float(lr.coef_[0,0])>0:
            return {"method":"platt_positive_slope","slope":float(lr.coef_[0,0]),
                    "intercept":float(lr.intercept_[0])}
    source_rate=float(prior.mean())
    goal=(float(y.sum())+PRIOR_STRENGTH*source_rate)/(len(y)+PRIOR_STRENGTH)
    goal=float(np.clip(goal,1e-5,1-1e-5))
    lo,hi=-20.,20.
    for _ in range(55):
        mid=(lo+hi)/2.
        preds=1/(1+np.exp(-np.clip(logits+mid,-35,35)))
        if float(preds.mean())<goal:lo=mid
        else:hi=mid
    return {"method":"monotone_EB_after_negative_Platt_slope" if core.calibration_mode(y)=="platt"
            else "sparse_empirical_bayes_intercept","slope":1.,
            "intercept":(lo+hi)/2.,"shrunk_calibration_event_rate":goal}

def apply_frozen_calibration(raw,parameters):
    p=np.clip(np.asarray(raw,dtype=float),1e-5,1-1e-5)
    x=np.log(p/(1-p))
    v=parameters["slope"]*x+parameters["intercept"]
    return 1/(1+np.exp(-np.clip(v,-35,35)))

def validate_prospective_candidates(data,asof,meta):
    """Every candidate is timestamped and actual source completeness proven."""
    if LABELS.intersection(data):
        raise ValueError("Future outcome or legacy model score in prospective features")
    missing=CONTEMPORARY_REQUIRED-set(data)
    if missing:raise ValueError(f"Missing independent PIT feature slots: {sorted(missing)}")
    if len(data)<10:raise ValueError("Insufficient actual liquid company candidates")
    if data[["date","symbol"]].duplicated().any():
        raise ValueError("Duplicate historical company-date in prospective features")
    if not meta.get("original_NSE_event_catalog_verified") or not meta.get("market_close_source_verified"):
        raise ValueError("Cannot score an incomplete/unverified price or original NSE event feed")
    source_date=pd.Timestamp(asof).normalize()
    market=pd.to_datetime(data["date"],errors="coerce").dt.normalize()
    if market.isna().any() or market.nunique()!=1:
        raise ValueError("Must score one actual complete NSE closing date")
    last=market.iloc[0]
    if not pd.Timestamp("2025-12-31")<last<=source_date:
        raise ValueError("Candidate stock date invalid or from reused older backtest")
    if (source_date-last).days>MIN_CANDIDATE_FRESHNESS_DAYS:
        raise ValueError("Stale market close; never display as current")
    cutoff=core.fold_close(last)
    actual=pd.to_datetime(data["historical_asof_utc"],utc=True,errors="coerce",format="mixed")
    if actual.isna().any() or (actual!=cutoff).any():
        raise ValueError("Original NSE event-feature snapshot time is not actual market close")
    ev_complete=pd.to_datetime(meta.get("NSE_event_coverage_through_utc"),utc=True,errors="coerce")
    prices_complete=pd.to_datetime(meta.get("market_data_coverage_through_utc"),utc=True,errors="coerce")
    if pd.isna(ev_complete) or ev_complete<cutoff or pd.isna(prices_complete) or prices_complete<cutoff:
        raise ValueError("Independent event and market source coverage not proven by close")
    # Exchange classifications are required measured counts; a missing event
    # value must not silently become no announcement.
    if data[list(core.CATALYST_PREFIXES)].isna().any().any():
        raise ValueError("Missing original NSE catalyst count wrongly masquerading as zero")
    if data[list(core.PRICE_COLUMNS)].notna().mean().min()<.70:
        raise ValueError("Missing historic market feature coverage")
    return cutoff

def score_prospective(data,package,asof,meta):
    cutoff=validate_prospective_candidates(data,asof,meta)
    current=data.copy()
    elig=core.eligible_asof(current)
    curr=core.safe_featureize(current.loc[elig])
    if len(curr)<core.TOP_K:raise ValueError("Not enough integrity-clean market candidates")
    raw=package["model"].predict_proba(curr[FLOAT_FEATURES])[:,1]
    curr["p6_double_calibrated"]=apply_frozen_calibration(raw,package["calibration"])
    ranked=curr.sort_values(["p6_double_calibrated","symbol"],
                             ascending=[False,True]).head(core.TOP_K).copy()
    # Preserve an honest trading-date label. Do not imply today's session if
    # the close was from the preceding market session.
    output=ranked[["date","symbol","close","p6_double_calibrated"]].copy()
    output.insert(0,"rank",range(1,len(output)+1))
    output["model_id"]=VERSION
    output["source_cutoff_utc"]=cutoff.isoformat()
    output["six_month_outcome_verified"]=False
    output["research_only_not_investment_advice"]=True
    return output

def freeze(features,snapshot,out,asof=ASOF,exclude_covid=True):
    target=Path(out);target.mkdir(parents=True,exist_ok=True)
    inp=pd.read_parquet(features)
    if LABELS.intersection(inp):
        raise SystemExit("Source PIT feature matrix already contaminated by outcomes or old scores")
    if not set(FLOAT_FEATURES).issubset(inp):
        raise SystemExit("Missing exact 22 preserved V11.4 features")
    y=pd.read_parquet(snapshot,columns=["date","symbol","close",
         "y6","y6_mature_date","integrity_y6_clean"])
    for z in (inp,y):
        z["date"]=pd.to_datetime(z["date"],errors="coerce").dt.normalize()
        z["symbol"]=z["symbol"].astype(str).str.upper().str.strip()
        if z.duplicated(["date","symbol"]).any():raise SystemExit("Source has duplicate stock-date")
    if len(inp)!=18569 or inp["date"].nunique()!=18:
        raise SystemExit("Frozen 18-fold primary training history changed")
    allrows=inp.merge(y,on=["date","symbol"],how="left",validate="1:1")
    if len(allrows)!=len(inp):raise SystemExit("Company dropped from original training source")
    clock=pd.to_datetime(allrows["historical_asof_utc"],utc=True,format="mixed",errors="coerce")
    expected=allrows["date"].map(core.fold_close)
    if clock.isna().any() or (clock!=expected).any():
        raise SystemExit("Exchange announcement backtest PIT source close mismatch")
    cutoff=core.fold_close(asof)
    keep=core.keep_train(allrows,pd.Timestamp(asof))
    keep&=strip_pandemic_matured_outcomes(allrows,exclude_covid)
    trainable=core.safe_featureize(allrows.loc[keep].copy())
    past=sorted(trainable["date"].unique())
    if len(past)<MIN_HISTORY_FOLDS+1:
        raise SystemExit("Insufficient matured historical training and independent calibration folds")
    caldate=past[-1]
    base=trainable[trainable["date"]<caldate].copy()
    cal=trainable[trainable["date"]==caldate].copy()
    if (base["date"].nunique()<MIN_HISTORY_FOLDS or len(base)<core.MIN_BASE_TRAIN_ROWS or
        int(base["y6"].sum())<core.MIN_BASE_POSITIVES or len(cal)<50 or
        cal["y6"].nunique()!=2):
        raise SystemExit("Frozen minimum train/calibration sample coverage failed")
    for part in (base,cal):
        maturation=pd.to_datetime(part["y6_mature_date"],utc=True,errors="coerce",format="mixed")
        if not (maturation<cutoff).all():
            raise SystemExit("A six-month forward training/calibration outcome is not mature")
    model=core.make_model()
    model.fit(base[FLOAT_FEATURES],base["y6"].astype(int))
    c=model.predict_proba(cal[FLOAT_FEATURES])[:,1]
    calibration=fit_calibration_params(c,cal["y6"],base["y6"])
    check1=core.calibrate_logit(c,c,cal["y6"],base["y6"])
    check2=apply_frozen_calibration(c,calibration)
    if not np.allclose(check1,check2,rtol=1e-7,atol=1e-7):
        raise SystemExit("Frozen calibration diverges from research implementation")
    package={"model":model,"calibration":calibration,"features":FLOAT_FEATURES,
             "model_version":VERSION,"asof":asof}
    dest=target/"v11_4_frozen_research_model.joblib"
    joblib.dump(package,dest,compress=3)
    restored=joblib.load(dest)
    if not np.array_equal(apply_frozen_calibration(
        restored["model"].predict_proba(cal[FLOAT_FEATURES])[:,1],restored["calibration"]),check2):
        raise SystemExit("Serialized research model not reproducible")
    manifest={
        "scope":"FROZEN_STANDALONE_V11_4_RESEARCH_CANDIDATE_NOT_APPROVED",
        "model_id":VERSION,"freeze_asof_IST":asof,
        "trained_before_or_at":cutoff.isoformat(),
        "feature_count":len(FLOAT_FEATURES),
        "feature_names":FLOAT_FEATURES,
        "technical_and_catalysts_only_no_old_model_selection":True,
        "retrospective_design_informed_by_2019_failures":True,
        "genuinely_unseen_future_holdout_completed":False,
        "training_observation_count":len(base),"training_folds":int(base["date"].nunique()),
        "training_winners":int(base["y6"].sum()),"training_latest_fold":str(base["date"].max().date()),
        "separate_calibration_fold":str(pd.Timestamp(caldate).date()),
        "calibration_observation_count":len(cal),
        "calibration_winners":int(cal["y6"].sum()),
        "calibration_method":calibration["method"],
        "original_market_source_rows":len(inp),
        "covid_crash_crossing_labels_excluded_from_training":bool(exclude_covid),
        "original_full_history_stress_results_must_remain_visible":True,
        "historical_2024_2025_only_fundamentals_excluded_from_model":True,
        "prospective_scoring_requires_fresh_verified_NSE_prices_and_events":True,
        "future_scoring_snapshot_available_in_this_job":False,
        "production_approved":False,
        "acceptance_gates_unmodified":ACCEPTANCE,
        "source_SHA256":{"PIT_market_exchange_features":sha256(features),
                         "matured_integrity_market_labels":sha256(snapshot),
                         "frozen_model_joblib":sha256(dest),
                         "frozen_inference_source":sha256(__file__)},
        "model_source_shard_size":dest.stat().st_size
    }
    (target/"v11_4_frozen_research_manifest.json").write_text(json.dumps(manifest,indent=2))
    (target/"prospective_selection_registry_EMPTY.csv").write_text(
         "scoring_asof_IST,market_date,symbol,rank,p2x_calibrated,model_id,"
         "market_source_sha256,NSE_event_source_sha256,freeze_manifest_sha256,"
         "first_recorded_at_utc,outcome_6m_maturity_utc,observed_2x\n")
    # The empty ledger is intentional. Backfilling already-scored historical
    # selections would invalidate the prospective validation claim.
    print(json.dumps(manifest,indent=2),flush=True)
    return manifest

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--features",required=True)
    p.add_argument("--snapshot",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--asof",default=ASOF)
    p.add_argument("--include-covid-training",action="store_true")
    a=p.parse_args()
    if a.asof!=ASOF:raise SystemExit("Frozen epoch is fixed to 2026-10-08; cannot retroactively move it")
    freeze(a.features,a.snapshot,a.output,a.asof,exclude_covid=not a.include_covid_training)
if __name__=="__main__":main()
