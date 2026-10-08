"""Chronological, embargoed incremental catalyst model vs exact frozen V10 selection.

This is an experimental WALK-FORWARD CHALLENGER, not the complete V11.4
production model. No future-label leakage, no same-fold training, no tuning
of hyperparameters based on the test folds; all feature slots and L2
regularization are fixed. Only original V10 OOS p_cal/p_dd30_cal and NSE
filing disclosures published by each historical fold's 15:30 IST cutoff.

Train: only labeled stock-folds whose complete six-month maturity date is
BEFORE the held-out selection date. Test: fixed 10 stocks per fold matched
against EXACT original selected_v941 baseline. Outcome integrity and maturity
are used in EVALUATION ONLY, never candidate eligibility/ranking.
"""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

FROZEN_FEATURES=("logit_v10_p100","p_dd30_cal","ret_120",
                 "turnover_accel","volatility_60","order_win_90d_binary",
                 "capacity_expansion_90d_binary","promoter_positive_90d_binary",
                 "corporate_action_90d_binary","dilution_90d_binary")
NO_EVAL_BEFORE=pd.Timestamp("2019-06-28")
TODAY=pd.Timestamp("2026-10-08")
FIXED_LOGISTIC_C=0.005
MIN_PRIOR_CLEAN_LABELS=1800
MIN_PRIOR_FOLDS=3
TOP_K=10

def fold_close(date):
    return (pd.Timestamp(date).normalize().tz_localize("Asia/Kolkata")
           +pd.Timedelta(hours=15,minutes=30)).tz_convert("UTC")

def featureize(x):
    df=x.copy()
    p=pd.to_numeric(df["p_cal"],errors="coerce").clip(1e-5,1-1e-5)
    df["logit_v10_p100"]=np.log(p/(1-p))
    for n,name in (
        ("nse_order_win_90d","order_win_90d_binary"),
        ("nse_capacity_expansion_90d","capacity_expansion_90d_binary"),
        ("nse_promoter_activity_90d_positive","promoter_positive_90d_binary"),
        ("nse_corporate_action_90d","corporate_action_90d_binary"),
        ("nse_dilution_90d","dilution_90d_binary")):
        df[name]=df[n].gt(0).astype(float)
    for c in FROZEN_FEATURES:
        df[c]=pd.to_numeric(df[c],errors="coerce").replace([np.inf,-np.inf],np.nan)
    return df

def ready_training(data,heldout):
    cutoff=fold_close(heldout).tz_localize(None)
    maturity=pd.to_datetime(data["y6_mature_date"],errors="coerce")
    # Crucial: prior-label maturity is historical, not today's maturity.
    return data[(data["date"]<heldout)&
         maturity.notna()&(maturity<cutoff)&
         data["y6"].isin([0.0,1.0])&
         data["integrity_y6_clean"].eq(True)&
         data["integrity_feature_clean"].eq(True)&
         data["p_cal"].notna()]

def is_matured_clean(selected):
    maturity=pd.to_datetime(selected["y6_mature_date"],errors="coerce")
    return selected["y6"].isin([0,1])&maturity.notna()&(maturity<=TODAY)&selected["integrity_y6_clean"].eq(True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--oos",required=True)
    p.add_argument("--snapshot",required=True)
    p.add_argument("--features",required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    oos=pd.read_parquet(a.oos,columns=["date","symbol","p_cal","p_dd30_cal",
        "selected_v941","selection_rank_v941","model_dispersion","y6","dd30_6m"])
    labels=pd.read_parquet(a.snapshot,columns=[
        "date","symbol","y6_mature_date","integrity_y6_clean"])
    f=pd.read_parquet(a.features)
    wanted={"date","symbol","integrity_feature_clean","ret_120","turnover_accel","volatility_60",
            "historical_asof_utc","nse_order_win_90d","nse_capacity_expansion_90d",
            "nse_promoter_activity_90d_positive","nse_corporate_action_90d","nse_dilution_90d"}
    if not wanted.issubset(f):raise SystemExit("Missing official PIT predictor slots "+str(wanted-set(f)))
    for x in (oos,labels,f):
        x["date"]=pd.to_datetime(x["date"],errors="coerce").dt.normalize()
        x["symbol"]=x["symbol"].astype(str).str.upper().str.strip()
        if x.duplicated(["date","symbol"]).any():raise SystemExit("Nonunique historical company and date")
    prohibited={"y6","y12","y24","dd30_6m","y6_mature_date"}
    if prohibited.intersection(f):raise SystemExit("Future outcome leaked into predictor research source")
    data=oos.merge(f[list(wanted)],on=["date","symbol"],how="inner",validate="1:1")
    data=data.merge(labels,on=["date","symbol"],how="left",validate="1:1")
    if len(data)!=len(f) or data["date"].nunique()!=18:raise SystemExit("Lost historical prediction rows")
    source=pd.to_datetime(data["historical_asof_utc"],utc=True,errors="coerce",format="mixed")
    cutoff_expected=data["date"].map(fold_close)
    if source.isna().any() or (source!=cutoff_expected).any():
        raise SystemExit("Historical source availability is after the fold close")
    data=featureize(data)
    if data["selected_v941"].sum()!=180:raise SystemExit("Frozen V10 original Top10 altered")
    result=[];picks=[];fold_metrics=[]
    for td,hold in data.groupby("date",sort=True):
        if td<NO_EVAL_BEFORE:continue
        train=ready_training(data,td)
        if len(train)<MIN_PRIOR_CLEAN_LABELS or train["date"].nunique()<MIN_PRIOR_FOLDS:
            fold_metrics.append({"date":str(td.date()),"status":"insufficient_prior_matured_training",
                "train_rows":len(train),"train_folds":int(train["date"].nunique())})
            continue
        # No target-label, outcome-integrity, or future-company information is
        # consulted in constructing today's candidate eligible pool.
        eligible=hold[hold["integrity_feature_clean"].eq(True)&hold["p_cal"].notna()].copy()
        if len(eligible)<TOP_K:raise SystemExit("Too few independent current stock candidates")
        baseline=hold[hold["selected_v941"].eq(True)].sort_values(
            ["selection_rank_v941","symbol"]).copy()
        if len(baseline)!=TOP_K:raise SystemExit("Frozen V10 historical Top10 missing")
        model=Pipeline([
            ("imputer",SimpleImputer(strategy="median",add_indicator=False)),
            ("scaler",StandardScaler()),
            ("logistic",LogisticRegression(C=FIXED_LOGISTIC_C,penalty="l2",
                                        class_weight=None,max_iter=1000,random_state=42))
        ])
        fitx=train[list(FROZEN_FEATURES)]
        y=train["y6"].astype(int)
        if y.nunique()!=2 or int(y.sum())<40:
            fold_metrics.append({"date":str(td.date()),"status":"insufficient_prior_positives",
                "train_rows":len(train),"train_folds":int(train["date"].nunique())})
            continue
        model.fit(fitx,y)
        eligible["shadow_p6"]=model.predict_proba(eligible[list(FROZEN_FEATURES)])[:,1]
        chosen=eligible.sort_values(
            ["shadow_p6","model_dispersion","symbol"],ascending=[False,True,True]).head(TOP_K)
        if len(chosen)!=len(baseline):raise SystemExit("K mismatch")
        report={"date":str(td.date()),"status":"predicted",
            "train_rows":len(train),"train_folds":int(train["date"].nunique()),
            "train_latest_mature_date":str(pd.to_datetime(train["y6_mature_date"]).max().date()),
            "train_positive":int(y.sum()),
            "train_prior_to_test":bool((train["date"]<td).all()),
            "candidate_n":len(chosen),"frozen_V10_n":len(baseline),
            "shadow_overlap_with_V10":len(set(chosen["symbol"])&set(baseline["symbol"])),
        }
        valid=True
        for name,picks_df in (("shadow",chosen),("V10",baseline)):
            mask=is_matured_clean(picks_df)
            if not mask.all():valid=False
            report[f"{name}_clean_n"]=int(mask.sum())
            report[f"{name}_y6_hits"]=int(picks_df.loc[mask,"y6"].sum())
            report[f"{name}_y6_precision"]=float(picks_df.loc[mask,"y6"].mean()) if mask.any() else None
            report[f"{name}_dd30_rate"]=float(picks_df.loc[mask,"dd30_6m"].mean()) if mask.any() else None
            for rank,row in enumerate(picks_df.itertuples(index=False),start=1):
                picks.append({"date":str(td.date()),"strategy":name,"rank":rank,"symbol":row.symbol,
                    "y6":row.y6,"dd30_6m":row.dd30_6m,
                    "score":getattr(row,"shadow_p6",None) if name=="shadow" else row.p_cal,
                    "source":"exact_original_V10_OOS_plus_contemporaneous_NSE_disclosures"})
        report["fully_clean_comparable"]=bool(valid)
        fold_metrics.append(report)
    table=pd.DataFrame(fold_metrics)
    table.to_csv(out/"chronological_shadow_vs_exact_V10_by_fold.csv",index=False)
    pd.DataFrame(picks).to_csv(out/"shadow_vs_exact_V10_selected_stocks.csv",index=False)
    comp=table[table["status"].eq("predicted")&table["fully_clean_comparable"].eq(True)]
    k=int(comp["candidate_n"].sum()) if len(comp) else 0
    cy=int(comp["shadow_y6_hits"].sum()) if len(comp) else 0
    by=int(comp["V10_y6_hits"].sum()) if len(comp) else 0
    summary={
        "scope":"EXPERIMENTAL_MATURED_PRIOR_FOLD_CATALYST_SHADOW_VS_EXACT_V10",
        "frozen_original_selection_control":"V10.2 selected_v941 exactly, 10 per historical fold",
        "model":"L2 Logistic C=0.005 on 10 predeclared PIT features",
        "evaluation_folds_tested":int(table["status"].eq("predicted").sum()),
        "fully_matured_integrity_clean_comparable_folds":int(len(comp)),
        "comparable_selected_stocks_each_strategy":k,
        "shadow_6mo_double_hits":cy,
        "frozen_V10_6mo_double_hits":by,
        "shadow_6mo_precision":cy/k if k else None,
        "frozen_V10_6mo_precision":by/k if k else None,
        "shadow_V10_precision_ratio":cy/by if by else None,
        "shadow_6mo_dd30_rate":float(comp["shadow_dd30_rate"].mean()) if len(comp) else None,
        "frozen_V10_6mo_dd30_rate":float(comp["V10_dd30_rate"].mean()) if len(comp) else None,
        "no_concurrent_or_future_training_fold":True,
        "every_train_y6_label_matured_before_test":True,
        "no_hidden_hyperparameter_grid_search":True,
        "prior_descriptive_catalyst_study_exists":True,
        "independent_untouched_confirmatory_holdout":False,
        "complete_V11_4_including_historical_external_demand":False,
        "V10_production_modified":False,
        "promotion":False,
    }
    (out/"chronological_shadow_vs_frozen_V10_summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)
    print(table[["date","status","train_folds","train_rows","shadow_y6_hits","V10_y6_hits","fully_clean_comparable"]].to_string(index=False),flush=True)
    if len(comp)<12:raise SystemExit("Fewer than 12 complete clean chronologically tested folds; do not claim fair V10 comparison")
if __name__=="__main__":main()
