"""Source-corrected 3FY, specific NSE catalyst, market and RSI70 research.

Recipe specified from source availability, not from selecting a successful
2025 backtest. Those outcomes were already examined: this is retrospective
research, with no claim of an unseen holdout or production validation.
Generic promoter filings and total announcement counts are not buying or
material catalysts. Exact original 5/7-year and quarterly screens remain
separate UNKNOWN assessments until their actual inputs are recovered.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss
from v11_4_threeFY_market_catalyst_chronological_research import (
    check_and_join,eligible_at,TRAIN_DATES,CAL_DATE,RESEARCH_TEST_DATES,
    EVALUATION_CUTOFF_UTC,safe_cutoff,PRICE_FEATS,FIN_FEATS,wilson_interval,
    MIN_TRAIN_ROWS,MIN_TRAIN_POS)
from v11_4_threeFY_RSI70_promoter_source_ready import STRICT_STATUS
from v11_4_standalone_train_walkforward import (
    make_model,REGULARIZATION_C,STABILITY_SEEDS,perturbation_training,
    top10_similarity)
from v11_4_forward_research_release import fit_calibration_params,apply_frozen_calibration

SPECIFIC_EVENTS=("nse_capacity_expansion_180d_positive",
    "nse_order_win_180d_positive","nse_regulatory_180d_positive",
    "nse_earnings_180d_positive","nse_corporate_action_180d_positive",
    "nse_dilution_180d")
FEATURES=(*FIN_FEATS,*PRICE_FEATS,"log_mean_63turnover_INR",
          *SPECIFIC_EVENTS,"rsi14_wilder_fraction","rsi14_gt70_indicator")
VERSION="V11.4-systematic-3FY-specificNSE-RSI70-research-20261009"

def prepare(source,labels):
    needed={"rsi14_wilder_source","source_verification",
            "rsi14_strict_gt70_source_verified"}
    if not needed.issubset(source):raise ValueError("Source-only RSI70 sidecar missing")
    value=pd.to_numeric(source["rsi14_wilder_source"],errors="coerce")
    verified=source["source_verification"].eq(STRICT_STATUS)
    if (verified & ~value.between(0,100)).any():raise ValueError("Invalid source RSI")
    if (~verified & value.notna()).any():raise ValueError("Unverified RSI must remain UNKNOWN")
    flags=source["rsi14_strict_gt70_source_verified"].astype("boolean")
    if (flags.loc[verified].isna().any() or
        not flags.loc[verified].eq(value.loc[verified].gt(70)).all() or
        flags.loc[~verified].notna().any()):
        raise ValueError("Strict >70 RSI flags disagree with source")
    joined=check_and_join(source,labels)
    joined["rsi14_wilder_fraction"]=pd.to_numeric(
        joined["rsi14_wilder_source"],errors="coerce")/100
    joined["rsi14_gt70_indicator"]=joined["rsi14_strict_gt70_source_verified"].astype("Float64")
    return joined

def cohort(allrows,date,cutoff):
    # The old selector's guards establish original financial coverage, clean
    # prices and label maturity. RSI-unverified stocks are then explicitly
    # excluded, never assigned neutral RSI or a fabricated negative flag.
    base=eligible_at(allrows,date,cutoff)
    q=base.loc[base["source_verification"].eq(STRICT_STATUS)].copy()
    for key in FEATURES:
        q[key]=pd.to_numeric(q[key],errors="coerce").replace([np.inf,-np.inf],np.nan)
        if key in FIN_FEATS:q[key]=q[key].clip(-1,5)
        elif key in SPECIFIC_EVENTS:q[key]=np.log1p(q[key].clip(0,25))
        elif key in PRICE_FEATS:q[key]=q[key].clip(-3,8)
    if q[["rsi14_wilder_fraction","rsi14_gt70_indicator"]].isna().any().any():
        raise ValueError("Unknown RSI entered complete research cohort")
    return q

def fit_once(train,cal,test):
    # Same fixed L2 C as original core; no parameter search.
    model=make_model()
    model.fit(train[list(FEATURES)].astype(float),train["y6"].astype(int))
    cp=model.predict_proba(cal[list(FEATURES)].astype(float))[:,1]
    parameters=fit_calibration_params(cp,cal["y6"],train["y6"],min_training_rows=MIN_TRAIN_ROWS)
    raw=model.predict_proba(test[list(FEATURES)].astype(float))[:,1]
    return apply_frozen_calibration(raw,parameters),parameters

def run(source,labels,out):
    allrows=prepare(source,labels)
    train=pd.concat([cohort(allrows,d,safe_cutoff(CAL_DATE))
                     for d in TRAIN_DATES],ignore_index=True)
    cal=cohort(allrows,CAL_DATE,safe_cutoff(RESEARCH_TEST_DATES[0]))
    if (len(train)<MIN_TRAIN_ROWS or train["y6"].nunique()!=2 or
        train["y6"].sum()<MIN_TRAIN_POS or len(cal)<50 or cal["y6"].nunique()!=2):
        raise ValueError("Insufficient matured real-event train/calibration history")
    output=Path(out);output.mkdir(parents=True,exist_ok=True)
    records=[];summaries=[];stability=[]
    for date in RESEARCH_TEST_DATES:
        test=cohort(allrows,date,EVALUATION_CUTOFF_UTC)
        if len(test)<10:raise ValueError("Cannot form an original 10-stock research cohort")
        probability,parameters=fit_once(train,cal,test)
        if not np.isfinite(probability).all() or not ((probability>0)&(probability<1)).all():
            raise ValueError("Invalid calibrated model output")
        q=test[["date","symbol","close","y6",*FEATURES]].copy()
        q["research_probability_6m_2x"]=probability
        q=q.sort_values(["research_probability_6m_2x","symbol"],ascending=[False,True])
        q.insert(0,"rank",range(1,len(q)+1));records.append(q)
        y=q["y6"].astype(int);wins=int(q.head(10)["y6"].sum());rate=float(y.mean())
        summaries.append({"date":date,"source_and_label_clean_cohort":len(q),
            "actual_six_month_doublers":int(y.sum()),"cohort_base_rate":rate,
            "top10_doublers":wins,"top10_precision":wins/10,
            "top10_precision_Wilson95":wilson_interval(wins,10),
            "top10_lift":wins/10/rate if rate>0 else None,
            "average_precision":float(average_precision_score(y,q["research_probability_6m_2x"])),
            "roc_auc":float(roc_auc_score(y,q["research_probability_6m_2x"])) if y.nunique()==2 else None,
            "brier":float(brier_score_loss(y,q["research_probability_6m_2x"])),
            "calibration_method":parameters["method"],
            "already_examined_outcomes_retrospective_only":True})
        for seed in STABILITY_SEEDS:
            alternative,_=fit_once(perturbation_training(train,seed),cal,test)
            alt=test.assign(_probability=alternative).sort_values(
                ["_probability","symbol"],ascending=[False,True]).head(10)
            stability.append({"date":date,"seed":seed,
                "jaccard":top10_similarity(q.head(10)["symbol"],alt["symbol"])})
    sf=pd.DataFrame(stability)
    report={"model_id":VERSION,"scope":"RETROSPECTIVE_RESEARCH_SOURCE_CORRECTION",
        "predeclared_feature_names":list(FEATURES),"feature_count":len(FEATURES),
        "fixed_C":REGULARIZATION_C,"parameter_search_performed":False,
        "train_dates":list(TRAIN_DATES),"train_rows":len(train),
        "train_six_month_doublers":int(train["y6"].sum()),
        "calibration_date":CAL_DATE,"calibration_rows":len(cal),
        "calibration_six_month_doublers":int(cal["y6"].sum()),
        "training_labels_known_before_calibration_decision":True,
        "RSI_threshold_strictly_gt":70,"RSI_is_independent_family_not_mandatory_intersection":True,
        "RSI_unknown_not_imputed":True,"generic_promoter_filings_not_buying_predictor":True,
        "generic_announcement_total_not_material_catalyst_predictor":True,
        "three_fiscal_years_support_two_year_CAGR_only":True,
        "first_three_exact_screens_not_fabricated_or_replaced":True,
        "historical_daily_RSI_all_session_official_crosscheck_complete":False,
        "GDELT_and_Google_historical_PIT_features_available":False,
        "six_month_target_only_12m24m_horizon_model_not_fitted":True,
        "already_examined_test_dates_not_new_blinded_evidence":True,
        "required_independent_test_folds":12,"new_independent_test_folds":0,
        "production_approved":False,"original_Oct8_observation_preserved":True,
        "mean_top10_jaccard":float(sf["jaccard"].mean()),
        "worst_fold_p05_jaccard":float(sf.groupby("date")["jaccard"].quantile(.05).min()),
        "folds":summaries}
    pd.concat(records,ignore_index=True).to_parquet(
        output/"systematic_3FY_RSI70_research_rankings_PRIVATE.parquet",index=False)
    sf.to_csv(output/"systematic_3FY_RSI70_stability.csv",index=False)
    (output/"systematic_3FY_RSI70_research_summary.json").write_text(json.dumps(report,indent=2))
    return report
