"""One-time, predeclared independent June-only chronological FY3 exploratory ML.

At June 2024 score-calibration time only June 2023 labels were mature.
At June 2025 test decision only June 2024 calibration labels were mature.
TEST is solely June 2025: no old December 2025 heldout labels, forecasts
or scores loaded. The model/regularization/preprocessing are the existing
3FY baseline; only more appropriate independent June chronology and genuine
calibration event count change. Test has few doublers: never deploy.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
from v11_4_threeFY_chronological_research_ML import PREDICTORS,calibrated_for_test
from v11_4_strict_annual_numeric_features import fold_close
YEARS={
 2023:("2023-06-30",(2021,2022,2023)),
 2024:("2024-06-28",(2022,2023,2024)),
 2025:("2025-06-30",(2023,2024,2025))
}
EVAL_UTC=pd.Timestamp("2026-10-09T00:00:00Z")
MIN_CAL_EVENTS=12
MIN_HOLD_EVENTS_FOR_SIGNIFICANCE=12
def make_features(frame,year,known_by):
 day,(a,b,c)=YEARS[year]
 x=frame.copy()
 x["date"]=pd.to_datetime(x["date"],errors="raise").dt.normalize()
 x["symbol"]=x["symbol"].astype(str).str.upper().str.strip()
 if not x["date"].eq(pd.Timestamp(day)).all() or len(x)<1000 or x["symbol"].duplicated().any():
  raise ValueError("Historical June fold identity not immutable")
 keys={f"FY{y}_{v}_INR" for y in (a,b,c) for v in ("revenue","PAT")}
 fields={"verified_3FY_before_actual_June_fold_close","source_eligible_mature_six_month_research_label",
 "close","avg_turnover_63","y6","y6_mature_date","integrity_y6_clean"}
 if not keys.union(fields).issubset(x):raise ValueError("June source/label/finance columns incomplete")
 pub=x["verified_3FY_before_actual_June_fold_close"].eq(True)
 # Original source-vintage clock has already been checked before saving.
 y6mature=pd.to_datetime(x["y6_mature_date"],utc=True,errors="coerce",format="mixed")
 valid=(pub&x["integrity_y6_clean"].eq(True)&x["y6"].isin([0,1])&
        x["close"].between(20,2000)&x["avg_turnover_63"].gt(0)&
        y6mature.notna()&y6mature.lt(known_by))
 if not valid.eq(x["source_eligible_mature_six_month_research_label"].eq(True)).all() and year==2025:
  raise ValueError("2025 exploratory test label maturity audit unexpectedly changed")
 eligible=x.loc[valid].copy()
 if len(eligible)<600:raise ValueError("Too few original eligible annual-fiscal trading identities")
 for yy in (a,b,c):
  for fact in ("revenue","PAT"):
   col=f"FY{yy}_{fact}_INR"
   eligible[col]=pd.to_numeric(eligible[col],errors="coerce")
 if eligible[list(keys)].isna().any().any():raise ValueError("June fiscal source has null numeric fact")
 pa=eligible[f"FY{a}_PAT_INR"];pb=eligible[f"FY{b}_PAT_INR"];pc=eligible[f"FY{c}_PAT_INR"]
 ra=eligible[f"FY{a}_revenue_INR"];rb=eligible[f"FY{b}_revenue_INR"];rc=eligible[f"FY{c}_revenue_INR"]
 if not (ra.gt(0)&rb.gt(0)&rc.gt(0)).all():raise ValueError("Invalid original 3FY sales baseline")
 eligible["sales_2yr_cagr"]=np.sqrt(rc/ra)-1
 eligible["sales_previous_yoy"]=rb/ra-1
 eligible["sales_latest_yoy"]=rc/rb-1
 eligible["sales_growth_acceleration"]=eligible["sales_latest_yoy"]-eligible["sales_previous_yoy"]
 eligible["profit_2yr_cagr"]=np.where((pa>0)&(pc>0),np.sqrt((pc/pa).clip(lower=0))-1,np.nan)
 eligible["profit_latest_yoy"]=np.where(pb>0,pc/pb-1,np.nan)
 eligible["profit_latest_margin"]=pc/rc
 eligible["profit_margin_delta"]=pc/rc-pb/rb
 eligible["profit_positive_all3"]=((pa>0)&(pb>0)&(pc>0)).astype(int)
 for name in PREDICTORS[:-1]:
  eligible[name]=pd.to_numeric(eligible[name],errors="coerce").replace([np.inf,-np.inf],np.nan).clip(-1,5)
 eligible["log_63session_turnover"]=np.log1p(pd.to_numeric(eligible["avg_turnover_63"],errors="coerce").clip(lower=0))
 return eligible
def june_threeFY_single_holdout(june23,june24,june25):
 train=make_features(june23,2023,fold_close("2024-06-28"))
 cal=make_features(june24,2024,fold_close("2025-06-30"))
 hold=make_features(june25,2025,EVAL_UTC)
 # Each June is exactly 12 months apart, with six-month labels matured before
 # the next June. This avoids overlapping training/calibration outcome windows.
 for y,x in ((2023,train),(2024,cal),(2025,hold)):
  if x["date"].nunique()!=1 or not x["date"].eq(pd.Timestamp(YEARS[y][0])).all():
   raise ValueError("Unchanged June chronological source and prediction dates required")
 positives={str(y):int(z["y6"].sum()) for y,z in ((2023,train),(2024,cal),(2025,hold))}
 if positives["2023"]<MIN_CAL_EVENTS or positives["2024"]<MIN_CAL_EVENTS:
  raise ValueError("Calibration or training has too few real original matured 2x events")
 if positives["2025"]<1:raise ValueError("No evaluable held-out twoX positives")
 # Same frozen 10-feature L2 logistic and its existing prior-shrunk positive
 # slope calibration; NEVER compare/tune using already seen Dec2025 holdout.
 model,p,calmethod=calibrated_for_test(train,cal,hold)
 if not np.isfinite(p).all() or not ((p>=0)&(p<=1)).all():
  raise ValueError("Out-of-sample research probability numerically invalid")
 scored=hold[["date","symbol","y6","close","avg_turnover_63",*PREDICTORS]].copy()
 scored["exploratory_p6_double_RESEARCH_NOT_PRODUCTION"]=p
 scored=scored.sort_values(["exploratory_p6_double_RESEARCH_NOT_PRODUCTION","symbol"],
  ascending=[False,True]).reset_index(drop=True)
 scored.insert(0,"research_rank",np.arange(1,len(scored)+1))
 yy=hold["y6"].astype(int).to_numpy();wins=int(scored.head(10)["y6"].sum())
 meta={
  "scope":"V11_4_EXPLORATORY_JUNE_ONLY_3FY_SEPARATE_CHRONOLOGICAL_HOLDOUT_NOT_PRODUCTION",
  "train_original_date":"2023-06-30",
  "calibrate_original_date":"2024-06-28",
  "unseen_original_test_date":"2025-06-30",
  "train_source_matched_matured_labeled_stocks":len(train),
  "calibration_source_matched_matured_labeled_stocks":len(cal),
  "test_source_matched_matured_labeled_stocks":len(hold),
  "train_actual_2x_6m_event_count":positives["2023"],
  "calibration_actual_2x_6m_event_count":positives["2024"],
  "test_actual_2x_6m_event_count":positives["2025"],
  "calibration_event_count_guard_at_least_12_passed":True,
  "test_holdout_minimum_12_events_guard_passed":bool(positives["2025"]>=MIN_HOLD_EVENTS_FOR_SIGNIFICANCE),
  "test_market_2x_6m_event_rate":float(yy.mean()),
  "test_top10_actual_doublers":wins,
  "test_top10_precision":wins/10,
  "test_top10_lift_over_base":float((wins/10)/yy.mean()) if yy.mean()>0 else None,
  "test_ROC_AUC":float(roc_auc_score(yy,p)) if len(np.unique(yy))==2 else None,
  "test_average_precision":float(average_precision_score(yy,p)),
  "test_Brier":float(brier_score_loss(yy,p)),
  "calibration_method":calmethod,
  "regularization_or_parameters_tuned_on_prior_Dec2025_holdout":False,
  "original_2025_Dec_test_labels_scores_NEVER_READ":True,
  "original_2026_first_sealed_V11_4_forward_predictions_unchanged":True,
  "only_one_new_separate_June2025_holdout_fold":True,
  "scientific_status":"INSUFFICIENT_HISTORIC_OUT_OF_TIME_FOLDS_AND_TEST_POSITIVE_EVENTS",
  "training_or_trading_production_approved":False}
 return scored,meta
def main():
 p=argparse.ArgumentParser()
 for y in YEARS:p.add_argument(f"--june-{y}",required=True)
 p.add_argument("--out",required=True);a=p.parse_args()
 inputs=[pd.read_parquet(getattr(a,f"june_{y}")) for y in YEARS]
 scored,result=june_threeFY_single_holdout(*inputs)
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 scored.to_csv(out/"private_original_2025_June_threeFY_holdout_ranked_stocks_and_outcomes.csv",index=False)
 (out/"original_threeFY_June_2023fit_2024cal_2025_holdout_scientific_summary.json").write_text(json.dumps(result,indent=2))
 print(json.dumps(result,indent=2),flush=True)
if __name__=="__main__":main()
