"""Frozen-recipe V11.4 fiscal + NSE momentum + corporate catalyst research model.

The 2025 June/December outcomes have already been examined by earlier *other*
finance-only experiments, so evaluations are retrospective REPLICATION, NOT new
independently blinded prospective evidence. No use of them to pick predictors,
weights, regularisation or thresholds. No changes to frozen October 2026 picks.

Fit: original 2022-Dec and 2023-Jun financial-qualified cohorts; every
training six-month label must be mature before 2024-Jun *calibration decision*.
Calibrate: original 2024-Jun cohort, label mature before 2025-Jun decision.
Research evaluation: 2025-Jun and 2025-Dec, labels mature by 2026-Oct-09.
2022-Jun coverage 69.83%, predeclared BELOW 70% gate, excluded from training.
Do not misinterpret correlated half-year panels as independent trials.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
from v11_4_eightfold_3FY_market_catalyst_source_matrix import (
    FOLDS,EXPECT_UNIVERSE,PRICE,EVENTS,FIN_FEATS,FORBIDDEN,local_1530)

TRAIN_DATES=("2022-12-30","2023-06-30")
CAL_DATE="2024-06-28"
RESEARCH_TEST_DATES=("2025-06-30","2025-12-31")
INVENTORY_ONLY_DATES=("2022-06-30","2023-12-29","2024-12-31")
EVALUATION_CUTOFF_UTC=pd.Timestamp("2026-10-09T00:00:00Z")
PRICE_FEATS=tuple(x for x in PRICE if x not in ("integrity_feature_clean","avg_turnover_63"))
FEATURES=tuple(FIN_FEATS)+PRICE_FEATS+("log_mean_63turnover_INR",)+tuple(EVENTS)
assert len(FEATURES)==len(set(FEATURES))
MIN_FIN_COVERAGE=70.
MIN_TRAIN_ROWS=1200
MIN_CAL_ROWS=700
MIN_TEST_ROWS=700
MIN_TRAIN_POS=12
MIN_CAL_POS=12
C_REG=.03
SEED=31

def safe_cutoff(date):
    return local_1530(date).tz_convert("UTC")

def check_and_join(source,labels):
    x=source.copy();label=labels.copy()
    if FORBIDDEN.intersection(x.columns):
        raise ValueError("Never load future stock outcomes into PIT feature-only matrix")
    needed={"date","symbol","historical_asof_utc","has_strict_3FY_original_fiscal_source",
            "integrity_feature_clean",*FIN_FEATS,*PRICE,*EVENTS}
    if not needed.issubset(x):
        raise ValueError("Combined 3FY original NSE financial-market-catalyst source incomplete")
    expected={"date","symbol","close","avg_turnover_63","integrity_y6_clean",
              "y6","y6_mature_date"}
    if not expected.issubset(label):
        raise ValueError("Original frozen six-month label dataset missing audit fields")
    x["date"]=pd.to_datetime(x["date"],errors="raise").dt.strftime("%Y-%m-%d")
    label["date"]=pd.to_datetime(label["date"],errors="raise").dt.strftime("%Y-%m-%d")
    for z in (x,label):
        z["symbol"]=z["symbol"].astype(str).str.upper().str.strip()
        if z.duplicated(["date","symbol"]).any():
            raise ValueError("Historical original security keys not unique")
    if len(x)!=sum(EXPECT_UNIVERSE.values()) or set(x["date"])!=set(FOLDS):
        raise ValueError("Entire original predeclared 8fold source universe changed")
    if x["historical_asof_utc"].isna().any():
        raise ValueError("Historical NSE catalyst timestamp unknown")
    published=pd.to_datetime(x["historical_asof_utc"],utc=True,errors="coerce",format="mixed")
    decision=(pd.to_datetime(x["date"]).dt.tz_localize("Asia/Kolkata")+
              pd.Timedelta(hours=15,minutes=30)).dt.tz_convert("UTC")
    if published.isna().any() or (published>decision).any():
        raise ValueError("Future NSE catalyst features published after historical decision")
    join=x.merge(label[[*sorted(expected)]],on=["date","symbol"],how="left",
                 validate="1:1",indicator=True)
    if (join["_merge"]!="both").any():
        raise ValueError("Some original historical source stocks have no frozen label identity")
    for d in FOLDS:
        fold=join.loc[join["date"].eq(d)]
        if len(fold)!=EXPECT_UNIVERSE[d]:
            raise ValueError("Historical fold row universe drift")
        known=int(fold["has_strict_3FY_original_fiscal_source"].eq(True).sum())
        if known!=FOLDS[d][1]:
            raise ValueError("Historical original NSE source-verified company count drift")
    if join["avg_turnover_63_x"].isna().any():
        raise ValueError("Historical PIT feature turnover missing")
    original=join["avg_turnover_63_x"].astype(float)
    if not np.allclose(original,join["avg_turnover_63_y"].astype(float),
                       rtol=1e-7,atol=1e-5,equal_nan=True):
        raise ValueError("Original immutable turnover feature and outcome-source market disagree")
    join=join.rename(columns={"avg_turnover_63_x":"avg_turnover_63"})
    join=join.drop(columns=["avg_turnover_63_y","_merge"])
    # Newer source-only sidecars also carry the original closing price.
    # Reconcile both observations before restoring the canonical field.
    if "close_x" in join:
        original_close=pd.to_numeric(join["close_x"],errors="coerce")
        label_close=pd.to_numeric(join["close_y"],errors="coerce")
        if (original_close.isna().any() or label_close.isna().any() or
            not np.allclose(original_close,label_close,rtol=1e-7,atol=1e-5)):
            raise ValueError("Original immutable close feature and outcome-source market disagree")
        join=join.rename(columns={"close_x":"close"}).drop(columns=["close_y"])
    return join

def eligible_at(frame,date,maturity_cutoff=None,require_mature_labels=True):
    if date not in FOLDS:raise ValueError("Unregistered research fold date")
    subset=frame.loc[frame["date"].eq(date)].copy()
    verified=subset["has_strict_3FY_original_fiscal_source"].eq(True)
    source_fraction=100*verified.sum()/len(subset)
    if date not in INVENTORY_ONLY_DATES and source_fraction<MIN_FIN_COVERAGE:
        raise ValueError(f"{date}: below 70 percent verified original 3FY financial threshold")
    if date=="2022-06-30":
        raise ValueError("2022 June is predeclared below 70% original financial coverage")
    # Test candidates must be selected before looking at future label
    # availability or integrity. Otherwise a delisting/corporate-action
    # exception can disappear from the Top 10 using hindsight.
    valid=(verified&subset["integrity_feature_clean"].eq(True)&
           subset["close"].between(20,2000)&subset["avg_turnover_63"].gt(0))
    if require_mature_labels:
        if maturity_cutoff is None:raise ValueError("Training requires an explicit maturity cutoff")
        mature=pd.to_datetime(subset["y6_mature_date"],utc=True,errors="coerce",format="mixed")
        valid&=(subset["integrity_y6_clean"].eq(True)&subset["y6"].isin([0,1])&
                mature.notna()&(mature<maturity_cutoff))
    # Missing source stock remains in audit but cannot become synthetic financial features.
    take=subset.loc[valid].copy()
    take["log_mean_63turnover_INR"]=np.log1p(take["avg_turnover_63"].astype(float))
    for key in FEATURES:
        take[key]=pd.to_numeric(take[key],errors="coerce").replace([np.inf,-np.inf],np.nan)
        if key in FIN_FEATS:
            take[key]=take[key].clip(-1,5)
        elif key in EVENTS:
            take[key]=take[key].clip(0,25)
        elif key!="log_mean_63turnover_INR":
            take[key]=take[key].clip(-3,8)
    if take[list(FIN_FEATS)].notna().sum(axis=1).eq(0).any():
        raise ValueError("Source-unverified company passed original annual completeness guard")
    if len(take)<MIN_TEST_ROWS and date in (*RESEARCH_TEST_DATES,CAL_DATE):
        raise ValueError("Too few strict-source financial and clean original six-month stocks")
    return take

def fit_and_calibrate(train,cal):
    if len(train)<MIN_TRAIN_ROWS or len(cal)<MIN_CAL_ROWS:
        raise ValueError("Insufficient PIT training and calibration securities")
    if train["y6"].sum()<MIN_TRAIN_POS or cal["y6"].sum()<MIN_CAL_POS:
        raise ValueError("Too few real six-month doublers for sound training/calibration")
    pipeline=Pipeline([
      ("imputer",SimpleImputer(strategy="median")),
      ("scale",StandardScaler()),
      ("model",LogisticRegression(C=C_REG,solver="lbfgs",max_iter=1600,
                                  random_state=SEED))])
    pipeline.fit(train[list(FEATURES)].astype(float),train["y6"].astype(int))
    calraw=np.clip(pipeline.predict_proba(cal[list(FEATURES)].astype(float))[:,1],
                   1e-5,1-1e-5)
    platt=LogisticRegression(C=.1,max_iter=800,random_state=SEED)
    platt.fit(logit(calraw).reshape(-1,1),cal["y6"].astype(int))
    if platt.coef_[0,0]<=0:
        return pipeline,None,"nonpositive_platt_slope_fallback_monotone_intercept"
    return pipeline,platt,"fixed_positive_slope_platt"

def monotone_prior_shrunk_intercept(train,calraw):
    prior=train["y6"].astype(int).mean()
    # Use known calibration prevalence, NEVER external test fold labels.
    label=calraw[1]
    calprob=calraw[0]
    target=(float(label.sum())+200*prior)/(len(label)+200)
    lo,hi=-20.,20.
    for _ in range(60):
        middle=(lo+hi)/2
        if expit(logit(calprob)+middle).mean()<target:lo=middle
        else:hi=middle
    return (lo+hi)/2

def wilson_interval(wins,n,z=1.96):
    if n<=0:return (None,None)
    p=wins/n;den=1+z*z/n
    mid=(p+z*z/(2*n))/den
    half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [float(max(0,mid-half)),float(min(1,mid+half))]

def research_train_score(source,labels):
    allrows=check_and_join(source,labels)
    train=pd.concat([eligible_at(allrows,d,safe_cutoff(CAL_DATE)) for d in TRAIN_DATES],
                    ignore_index=True)
    cal=eligible_at(allrows,CAL_DATE,safe_cutoff(RESEARCH_TEST_DATES[0]))
    model,platt,calmode=fit_and_calibrate(train,cal)
    offset=None
    if platt is None:
        r=np.clip(model.predict_proba(cal[list(FEATURES)].astype(float))[:,1],1e-5,1-1e-5)
        offset=monotone_prior_shrunk_intercept(train,(r,cal["y6"].astype(int).to_numpy()))
    score_out=[];test_summaries=[]
    for fold in RESEARCH_TEST_DATES:
        data=eligible_at(allrows,fold,require_mature_labels=False)
        raw=np.clip(model.predict_proba(data[list(FEATURES)].astype(float))[:,1],1e-5,1-1e-5)
        p=(platt.predict_proba(logit(raw).reshape(-1,1))[:,1] if platt is not None
           else expit(logit(raw)+offset))
        if not np.isfinite(p).all() or not ((p>=0)&(p<=1)).all():
            raise ValueError("Research combined model invalid calibrated probabilities")
        frame=data[["date","symbol","close","avg_turnover_63","y6",
                    "y6_mature_date","integrity_y6_clean",*FEATURES]].copy()
        frame["exploratory_p6_double_combined_research_only"]=p
        frame=frame.sort_values(["exploratory_p6_double_combined_research_only","symbol"],
                                ascending=[False,True]).reset_index(drop=True)
        frame.insert(0,"research_rank",np.arange(1,len(frame)+1))
        from v11_4_validation_metrics import evaluate_ranked
        metric=evaluate_ranked(frame,"exploratory_p6_double_combined_research_only",
                               EVALUATION_CUTOFF_UTC,float(cal["y6"].mean()))
        metric.update({"fold":fold,"financial_sources_before_original_decision":True,
            "available_stock_universe_original":len(allrows.loc[allrows["date"].eq(fold)]),
            "original_threeFY_source_verified":FOLDS[fold][1],
            "previously_seen_2025_dates_not_newly_independent_blinded_prospective_test":True})
        test_summaries.append(metric);score_out.append(frame)
    report={
      "scope":"V11_4_RESEARCH_FROZEN_RECIPE_3FY_MOMENTUM_NSE_CATALYST_COMBINED_2X_HISTORIC_REPLICATION",
      "source_only_market_financial_event_matrix_rows":len(allrows),
      "strict_original_3FY_financial_company_date_rows":int(allrows["has_strict_3FY_original_fiscal_source"].sum()),
      "actual_train_dates":list(TRAIN_DATES),
      "original_train_label_mature_prior_to_calibration_decision":CAL_DATE,
      "train_rows":len(train),"train_actual_2x_6m":int(train["y6"].sum()),
      "calibration_decision_date":CAL_DATE,
      "calibration_label_mature_prior_to_first_test_decision":RESEARCH_TEST_DATES[0],
      "calibration_rows":len(cal),"calibration_actual_2x_6m":int(cal["y6"].sum()),
      "fixed_model":"median_imputation_standard_scaled_L2_logistic_C_0.03",
      "fixed_feature_count":len(FEATURES),
      "financial_features":list(FIN_FEATS),
      "market_features":list(PRICE_FEATS)+["log_mean_63turnover_INR"],
      "verified_NSE_event_features":list(EVENTS),
      "RSI14_over80_UNVERIFIED_OMITTED":True,
      "actual_calibration_method":calmode,
      "2022June_69_83pct_financial_source_coverage_excluded_as_predeclared":True,
      "source_mutually_correlated_sixmonth_overlapping_dates_not_independent":True,
      "not_a_new_blind_independent_test_2025_labels_seen_in_earlier_finance_only_tests":True,
      "no_previous_2025_test_labels_used_for_hyperparameter_selection":True,
      "original_October_2026_frozen_V11_4_selections_unchanged":True,
      "six_month_outcomes_only_through_2026_10_09":True,
      "no_real_money_trading_or_production_approval":True,
      "independent_clean_12fold_scientific_promotion_gate_satisfied":False,
      "per_historic_fold_research_replication":test_summaries}
    return pd.concat(score_out,ignore_index=True),report

def run(args):
    data=pd.read_parquet(args.features)
    labels=pd.read_parquet(args.original_labels,columns=[
        "date","symbol","close","avg_turnover_63","integrity_y6_clean","y6","y6_mature_date"])
    ranked,report=research_train_score(data,labels)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    ranked.to_parquet(out/"combined_2025_historical_research_scores_and_actual_outcomes_PRIVATE.parquet",
                      compression="zstd",index=False)
    (out/"combined_three_fiscal_year_market_NSE_catalyst_scientific_REPLICATION_summary.json").write_text(
        json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)
    return report
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--features",required=True);p.add_argument("--original-labels",required=True)
    p.add_argument("--out",required=True)
    run(p.parse_args())
if __name__=="__main__":main()
