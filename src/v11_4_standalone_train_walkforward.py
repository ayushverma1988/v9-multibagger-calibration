"""V11.4 standalone six-month 2x stock discovery model.

NOT a comparison with any older model: only NSE historical technical fields,
NSE announcements available at each original 15:30 IST close, and the original
integrity-checked six-month labels for training/evaluation.

Strict chronological protocol:
* each test fold completely excluded from training and calibration
* training labels must mature before the held-out selection moment
* a separate latest-PREVIOUS matured fold fits Platt probability calibration
* Top 10 chosen solely from contemporary PIT features (not outcomes)
* no V10 probability, V10 rankings, V10 selections, external-data backfill,
  or inaccessible 5y/7y fundamentals are allowed as predictors.
"""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss,average_precision_score,roc_auc_score

PRICE_COLUMNS=("ret_20","ret_60","ret_120","ret_252","mom_accel",
               "vol_accel","turnover_accel","off_high_252","above_low_252",
               "trend_consistency_60","volatility_60")
CATALYST_PREFIXES=(
    "nse_order_win_90d","nse_order_win_180d",
    "nse_capacity_expansion_90d","nse_capacity_expansion_180d",
    "nse_regulatory_90d_positive","nse_regulatory_180d_positive",
    "nse_promoter_activity_90d_positive",
    "nse_corporate_action_90d","nse_dilution_90d",
    "nse_earnings_90d_positive","nse_buyback_90d",
)
MODEL_FEATURES=(*PRICE_COLUMNS,*CATALYST_PREFIXES)
PRICE_MIN=20.0
PRICE_MAX=2000.0
TOP_K=10
REGULARIZATION_C=0.03
CALIBRATION_C=0.1
MIN_BASE_TRAIN_FOLDS=2
MIN_BASE_TRAIN_ROWS=1500
MIN_BASE_POSITIVES=45
MIN_CALIBRATION_POSITIVES=12
MIN_RESEARCH_TEST_FOLDS=12
STABILITY_SEEDS=(17,41,83)
# Research v11.4b: fixed bagged logistic members average out idiosyncratic
# fold-stratum training subsample variation, without tuning to test labels.
BAGGING_SEEDS=(101,211,307,419)
BAGGING_RETENTION=0.90
# Fixed 95% stratified retention per training fold; this measures
# robustness to plausible omissions in old observations, not re-tuning.
STABILITY_RETENTION=0.95

def fold_close(d):
    return (pd.Timestamp(d).normalize().tz_localize("Asia/Kolkata")
            +pd.Timedelta(hours=15,minutes=30)).tz_convert("UTC")

def keep_train(pit_data, heldout):
    heldout=pd.Timestamp(heldout)
    maturity=pd.to_datetime(pit_data["y6_mature_date"],utc=True,errors="coerce",format="mixed")
    return (pit_data["date"].lt(heldout) &
            maturity.notna() & maturity.lt(fold_close(heldout)) &
            pit_data["y6"].isin([0,1]) &
            pit_data["integrity_y6_clean"].eq(True) &
            pit_data["integrity_feature_clean"].eq(True) &
            pit_data["close"].between(PRICE_MIN,PRICE_MAX,inclusive="both") &
            pd.to_numeric(pit_data["avg_turnover_63"],errors="coerce").gt(0))

def eligible_asof(group):
    return (group["integrity_feature_clean"].eq(True)&
            group["close"].between(PRICE_MIN,PRICE_MAX,inclusive="both")&
            pd.to_numeric(group["avg_turnover_63"],errors="coerce").gt(0))

def safe_featureize(frame):
    x=frame.copy()
    for c in MODEL_FEATURES:
        x[c]=pd.to_numeric(x[c],errors="coerce").replace([np.inf,-np.inf],np.nan)
    for c in CATALYST_PREFIXES:
        x[c]=np.log1p(x[c].clip(lower=0,upper=50))
    return x

def make_model():
    return Pipeline([
        ("imputer",SimpleImputer(strategy="median")),
        ("scale",StandardScaler()),
        ("lr",LogisticRegression(
            penalty="l2",C=REGULARIZATION_C,max_iter=1300,
            class_weight=None,solver="lbfgs",random_state=31))
    ])

def calibrate_logit(raw,calibration_values,calibration_y):
    # A separate prior matured fold calibrates each held-out prediction.
    # This is NOT retrospective isotonic fitting on the test fold.
    q=np.clip(np.asarray(calibration_values,dtype=float),1e-5,1-1e-5)
    z=np.log(q/(1-q)).reshape(-1,1)
    lr=LogisticRegression(C=CALIBRATION_C,solver="lbfgs",
                          class_weight=None,max_iter=400,random_state=31)
    lr.fit(z,np.asarray(calibration_y,dtype=int))
    t=np.clip(np.asarray(raw,dtype=float),1e-5,1-1e-5)
    return lr.predict_proba(np.log(t/(1-t)).reshape(-1,1))[:,1]

def top10_similarity(a,b):
    aa=set(a);bb=set(b)
    return len(aa&bb)/len(aa|bb) if aa or bb else 1.0

def train_once(train,cal,test):
    # Bagging/ensemble operates exclusively on labels available before td.
    # A full-data estimator anchors the ensemble while four fixed 90%
    # stratified perturbations reduce one-fold coefficient instability.
    cal_preds=[];test_preds=[]
    for seed in (None,*BAGGING_SEEDS):
        local=train if seed is None else perturbation_training(
            train,seed,retention=BAGGING_RETENTION)
        model=make_model()
        model.fit(local[list(MODEL_FEATURES)],local["y6"].astype(int))
        cal_preds.append(model.predict_proba(cal[list(MODEL_FEATURES)])[:,1])
        test_preds.append(model.predict_proba(test[list(MODEL_FEATURES)])[:,1])
    cal_raw=np.mean(np.stack(cal_preds,axis=0),axis=0)
    test_raw=np.mean(np.stack(test_preds,axis=0),axis=0)
    return calibrate_logit(test_raw,cal_raw,cal["y6"])

def perturbation_training(train,seed,retention=STABILITY_RETENTION):
    # Preserve each original fold and both classes. Randomly retain 95% of
    # rows within each (fold, label) stratum; no new historical observations.
    sample=[]
    for (_, _),group in train.groupby(["date","y6"],sort=True):
        size=max(1,math.ceil(len(group)*retention))
        sample.append(group.sample(n=size,replace=False,random_state=seed))
    return pd.concat(sample,ignore_index=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--features",required=True)
    p.add_argument("--snapshot",required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    x=pd.read_parquet(a.features)
    if any(t in x for t in ("y6","y12","y24","dd30_6m","y6_mature_date")):
        raise SystemExit("Forbidden future target inside PIT predictor feature matrix")
    if not set(MODEL_FEATURES).issubset(x):
        raise SystemExit("Missing predefined original PIT market/catalyst predictors: "+str(set(MODEL_FEATURES)-set(x)))
    if "p_cal" in x or "selected_v941" in x:
        raise SystemExit("Standalone V11.4 must not consume legacy model predictions")
    snap=pd.read_parquet(a.snapshot,columns=[
        "date","symbol","close","avg_turnover_63","y6","y6_mature_date",
        "integrity_y6_clean","dd30_6m"])
    for z in (x,snap):
        z["date"]=pd.to_datetime(z["date"],errors="coerce").dt.normalize()
        z["symbol"]=z["symbol"].astype(str).str.upper().str.strip()
        if z.duplicated(["date","symbol"]).any():raise SystemExit("Duplicate historical company/date")
    if x["date"].nunique()!=18 or len(x)!=18569:
        raise SystemExit("Frozen 18-fold market source universe changed")
    joined=x.merge(snap,on=["date","symbol"],how="left",validate="1:1",suffixes=("","_snapshot"))
    if joined["close"].isna().any():
        raise SystemExit("Missing current-session close for a historical company")
    cutoff=joined["date"].map(fold_close)
    pub=pd.to_datetime(joined["historical_asof_utc"],utc=True,errors="coerce",format="mixed")
    if pub.isna().any() or (pub!=cutoff).any():
        raise SystemExit("NSE historical announcements not aligned with 15:30 fold cutoff")
    for c in ("revenue_yoy_pct","pat_yoy_pct","pat_margin_latest_pct",
              "pat_margin_delta_pp","latest_fiscal_age_days"):
        if c not in joined:raise SystemExit("Missing annual research provenance slot "+c)
    # 5y/7y numerical data only available for 2 year-end folds, so annual
    # financials remain reserved until >=12 PIT folds have audit-grade data.
    fy_coverage=joined.loc[joined["revenue_yoy_pct"].notna(),"date"].nunique()
    if fy_coverage>=12:raise SystemExit("Annual model feature activation needs separate independently specified validation")
    training=safe_featureize(joined)
    fold_records=[];picks=[];quality=[];stable=[]
    dates=sorted(training["date"].unique())
    for td in dates:
        trainable=training.loc[keep_train(training,td)].copy()
        past=sorted(trainable["date"].unique())
        if len(past)<MIN_BASE_TRAIN_FOLDS+1:
            fold_records.append({"date":str(pd.Timestamp(td).date()),"status":"not_enough_prior_mature_folds",
                                 "training_fold_count":len(past)})
            continue
        cal_fold=past[-1]
        base=trainable[trainable["date"]<cal_fold].copy()
        cal=trainable[trainable["date"]==cal_fold].copy()
        current=training[(training["date"]==td)&eligible_asof(training)].copy()
        if (base["date"].nunique()<MIN_BASE_TRAIN_FOLDS or len(base)<MIN_BASE_TRAIN_ROWS or
            int(base["y6"].sum())<MIN_BASE_POSITIVES or
            cal["y6"].nunique()<2 or int(cal["y6"].sum())<MIN_CALIBRATION_POSITIVES):
            fold_records.append({"date":str(pd.Timestamp(td).date()),"status":"insufficient_prior_training_or_calibration",
                                 "training_fold_count":int(base["date"].nunique()),
                                 "training_rows":len(base),"calibration_positive":int(cal["y6"].sum())})
            continue
        if len(current)<TOP_K:raise SystemExit("Too few stocks from independent original market/PIT universe")
        if pd.to_datetime(base["y6_mature_date"],utc=True,errors="coerce").max()>=fold_close(td):
            raise SystemExit("Six-month training outcome matured AFTER selection cutoff")
        if pd.to_datetime(cal["y6_mature_date"],utc=True,errors="coerce").max()>=fold_close(td):
            raise SystemExit("Calibration outcome matured AFTER selection cutoff")
        cal_values=cal["y6"].astype(int).to_numpy()
        pvalues=train_once(base,cal,current)
        current["v11_4_p2x_calibrated"]=pvalues
        chosen=current.sort_values(
            ["v11_4_p2x_calibrated","symbol"],ascending=[False,True]).head(TOP_K)
        if chosen["symbol"].duplicated().any():raise SystemExit("Top10 duplicate")
        for rank,r in enumerate(chosen.itertuples(index=False),1):
            picks.append({
                "date":str(pd.Timestamp(td).date()),"rank":rank,"symbol":r.symbol,
                "close":float(r.close),"p2x_calibrated":float(r.v11_4_p2x_calibrated),
                "research_standalone_V11_4":True,
                # Outcomes never included in saved selection file.
            })
        labelm=pd.to_datetime(current["y6_mature_date"],utc=True,errors="coerce")
        outcome_ok=current["y6"].isin([0,1])&current["integrity_y6_clean"].eq(True)&(
            labelm<=pd.Timestamp("2026-10-08T23:59:59Z"))
        assessed=current[outcome_ok]
        chosen_good=chosen[chosen.index.isin(assessed.index)]
        fold={
            "date":str(pd.Timestamp(td).date()),"status":"tested",
            "training_folds":int(base["date"].nunique()),"training_rows":len(base),
            "calibration_fold":str(pd.Timestamp(cal_fold).date()),
            "calibration_rows":len(cal),
            "candidates":len(current),"selected":len(chosen),
            "outcome_matured_eligible":len(assessed),
            "matured_selections":len(chosen_good),
            "six_month_double_hits":int(chosen_good["y6"].sum()),
            "six_month_double_precision":float(chosen_good["y6"].mean()) if len(chosen_good) else None,
            "six_month_universe_base_rate":float(assessed["y6"].mean()) if len(assessed) else None,
            "six_month_capped_lift":min(float(chosen_good["y6"].mean()/assessed["y6"].mean()),10)
                if len(chosen_good) and assessed["y6"].mean()>0 else None,
            "six_month_selected_dd30_rate":float(chosen_good["dd30_6m"].mean()) if len(chosen_good) else None,
            "six_month_universe_dd30_rate":float(assessed["dd30_6m"].mean()) if len(assessed) else None,
        }
        fold_records.append(fold)
        if len(assessed)>50 and assessed["y6"].nunique()==2:
            quality.append({
                "date":fold["date"],"evaluation_n":len(assessed),
                "brier":float(brier_score_loss(assessed["y6"],assessed["v11_4_p2x_calibrated"])),
                "average_precision":float(average_precision_score(
                    assessed["y6"],assessed["v11_4_p2x_calibrated"])),
                "roc_auc":float(roc_auc_score(assessed["y6"],assessed["v11_4_p2x_calibrated"])),
            })
        for seed in STABILITY_SEEDS:
            resampled=perturbation_training(base,seed)
            if resampled["y6"].nunique()!=2:raise SystemExit("Bootstrap removed all positive labels")
            alt=train_once(resampled,cal,current)
            altselected=current.assign(_p=alt).sort_values(
                ["_p","symbol"],ascending=[False,True]).head(TOP_K)
            stable.append({
                "date":fold["date"],"seed":seed,"training_sample_fraction":STABILITY_RETENTION,
                "top10_jaccard":top10_similarity(chosen["symbol"],altselected["symbol"]),
            })
        print("FOLD",json.dumps({k:fold[k] for k in ("date","training_folds","six_month_double_hits","six_month_double_precision")}),flush=True)
    frame=pd.DataFrame(fold_records)
    frame.to_csv(out/"standalone_v11_4_pit_walkforward_by_fold.csv",index=False)
    pd.DataFrame(picks).to_csv(out/"standalone_v11_4_historical_top10_research.csv",index=False)
    pd.DataFrame(quality).to_csv(out/"standalone_v11_4_probability_quality.csv",index=False)
    pd.DataFrame(stable).to_csv(out/"standalone_v11_4_selector_perturbation_stability.csv",index=False)
    tested=frame[frame["status"].eq("tested")].copy()
    full=tested[tested["matured_selections"].eq(TOP_K)]
    cand=int(full["matured_selections"].sum()) if len(full) else 0
    hits=int(full["six_month_double_hits"].sum()) if len(full) else 0
    stab=pd.DataFrame(stable)
    meanstab=float(stab["top10_jaccard"].mean()) if len(stab) else None
    foldp05=stab.groupby("date")["top10_jaccard"].quantile(.05) if len(stab) else pd.Series(dtype=float)
    worstp05=float(foldp05.min()) if len(foldp05) else None
    summary={
        "model":"V11.4 standalone NSE technical + exchange-filing catalyst core",
        "not_a_legacy_model_comparison":True,
        "original_historical_stock_folds":18,
        "training_feature_count":len(MODEL_FEATURES),
        "finance_numeric_folds_verified":int(fy_coverage),
        "finance_numeric_features_deliberately_dormant_until_12_folds":True,
        "training_algorithm":"fixed 5-member bagged L2 logistic regression with 1 previously matured fold Platt calibration",
        "bagging_member_count":1+len(BAGGING_SEEDS),
        "bagging_train_retention":BAGGING_RETENTION,
        "bagging_seeds":list(BAGGING_SEEDS),
        "fixed_C":REGULARIZATION_C,
        "previous_matured_label_only":True,
        "training_or_calibration_current_or_future_labels":False,
        "historical_test_folds":int(len(tested)),
        "fully_integrity_clean_10stock_test_folds":int(len(full)),
        "fully_comparable_oos_stock_picks":cand,
        "confirmed_2x_hits_in_clean_folds":hits,
        "six_month_oos_precision":hits/cand if cand else None,
        "six_month_oos_average_top10_precision":float(full["six_month_double_precision"].mean()) if len(full) else None,
        "six_month_mean_capped_lift":float(full["six_month_capped_lift"].mean()) if len(full) else None,
        "six_month_mean_selected_dd30_rate":float(full["six_month_selected_dd30_rate"].mean()) if len(full) else None,
        "mean_same_fold_retraining_top10_jaccard":meanstab,
        "worst_fold_p05_same_fold_retraining_jaccard":worstp05,
        "mean_top10_jaccard_gate":.80,
        "worst_fold_p05_jaccard_gate":.60,
        "selection_stability_gate_pass":bool(meanstab is not None and worstp05 is not None
                                             and meanstab>=.80 and worstp05>=.60),
        "minimum_12_folds_gate_pass":bool(len(full)>=MIN_RESEARCH_TEST_FOLDS),
        "production_approved":False,
        "model_requires_further_validation":True,
        "historical_demand_adapter_ready":False,
        "no_older_model_scoring_or_selection_used":True,
    }
    (out/"standalone_v11_4_model_summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)
    if len(full)<MIN_RESEARCH_TEST_FOLDS:
        raise SystemExit(f"Minimum 12 historically valid folds NOT met (found {len(full)}); retain diagnostic artifacts")
if __name__=="__main__":main()
