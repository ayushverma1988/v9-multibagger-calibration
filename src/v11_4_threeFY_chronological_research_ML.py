"""V11.4 independent 3-fiscal-year short-history chronological research ML pilot.

Uses only original NSE as-of 2023, 2024, 2025 year-end annual financials,
market close/volume known at each decision, and six-month 2x labels ONLY
when independently matured. Train 2023, calibrate 2024, out-of-time test
2025. No V10 baseline, no rewriting frozen October 2026 Top 10.
This is a SINGLE held-out annual fold; NOT broad backtest confirmation.
"""
from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss
from v11_4_strict_annual_numeric_features import fold_close

BASE={
 2023:{"date":"2023-12-29","years":(2021,2022,2023)},
 2024:{"date":"2024-12-31","years":(2022,2023,2024)},
 2025:{"date":"2025-12-31","years":(2023,2024,2025)}
}
PREDICTORS=("sales_2yr_cagr","sales_previous_yoy","sales_latest_yoy",
 "sales_growth_acceleration","profit_2yr_cagr","profit_latest_yoy",
 "profit_latest_margin","profit_margin_delta","profit_positive_all3",
 "log_63session_turnover")
TEST_EVAL_UTC=pd.Timestamp("2026-10-09T00:00:00Z")
MIN_TRAIN=600
MIN_CAL=600
MIN_HOLDOUT=600
MIN_POS=5

def original_financial_features(source,market,year):
 cfg=BASE[year]
 x=source.copy()
 if x["symbol"].duplicated().any():raise ValueError("Duplicate original same-fiscal stock")
 yrs=cfg["years"];earliest,mid,latest=yrs
 needed={"symbol"}|{f"FY{n}_{v}_INR" for n in yrs for v in ("revenue","PAT")}
 if not needed.issubset(x):raise ValueError("Historical original source lacks 3FY fiscal facts")
 v=pd.DataFrame({"symbol":x["symbol"].astype(str).str.upper().str.strip()})
 for n in yrs:
  for metric in ("revenue","PAT"):
   v[f"{n}_{metric}"]=pd.to_numeric(x[f"FY{n}_{metric}_INR"],errors="coerce")
 if v.filter(regex=r"_revenue$").isna().any().any() or v.filter(regex=r"_PAT$").isna().any().any():
  raise ValueError("Verified historical 3FY source contains incomplete original facts")
 a=v[f"{earliest}_revenue"];b=v[f"{mid}_revenue"];c=v[f"{latest}_revenue"]
 pa=v[f"{earliest}_PAT"];pb=v[f"{mid}_PAT"];pc=v[f"{latest}_PAT"]
 if not (a.gt(0)&b.gt(0)&c.gt(0)).all():raise ValueError("Unusable original annual sales source")
 v["sales_2yr_cagr"]=np.sqrt(c/a)-1
 v["sales_previous_yoy"]=b/a-1
 v["sales_latest_yoy"]=c/b-1
 v["sales_growth_acceleration"]=v["sales_latest_yoy"]-v["sales_previous_yoy"]
 v["profit_2yr_cagr"]=np.where((pa>0)&(pc>0),np.sqrt((pc/pa).clip(lower=0))-1,np.nan)
 v["profit_latest_yoy"]=np.where(pb>0,pc/pb-1,np.nan)
 v["profit_latest_margin"]=pc/c
 v["profit_margin_delta"]=pc/c-pb/b
 v["profit_positive_all3"]=((pa>0)&(pb>0)&(pc>0)).astype(int)
 # Fixed extreme-ratio winsorization, specified before held-out stock labels.
 for key in PREDICTORS[:-1]:
  v[key]=pd.to_numeric(v[key],errors="coerce").replace([np.inf,-np.inf],np.nan).clip(-1.,5.)
 frame=market[market["date"].eq(pd.Timestamp(cfg["date"]))].copy()
 frame["symbol"]=frame["symbol"].astype(str).str.upper().str.strip()
 if frame["symbol"].duplicated().any():raise ValueError("Ambiguous original close-date stock")
 if len(frame)<1000:raise ValueError("Original stock universe unexpectedly small")
 # no new stocks enter via the original annual files
 joined=frame.merge(v[["symbol",*PREDICTORS[:-1]]],on="symbol",how="left",validate="1:1",indicator=True)
 joined["annual_threeFY_complete"]=joined["_merge"].eq("both")
 joined["log_63session_turnover"]=np.log1p(pd.to_numeric(joined["avg_turnover_63"],errors="coerce").clip(lower=0))
 joined["annual_source_date_IST"]=cfg["date"]
 return joined

def eligibility(x,known_by):
 mature=pd.to_datetime(x["y6_mature_date"],utc=True,format="mixed",errors="coerce")
 ok=(x["annual_threeFY_complete"]&
     x["integrity_y6_clean"].eq(True)&x["y6"].isin([0,1])&
     x["close"].between(20,2000)&
     x["avg_turnover_63"].gt(0)&
     mature.notna()&(mature<known_by))
 return x.loc[ok].copy()

def calibrated_for_test(train,cal,test):
 from scipy.special import expit,logit
 model=Pipeline([("imputer",SimpleImputer(strategy="median")),
                 ("scale",StandardScaler()),
                 ("lr",LogisticRegression(C=.03,solver="lbfgs",max_iter=1400,random_state=31))])
 x=train[list(PREDICTORS)].astype(float)
 model.fit(x,train["y6"].astype(int))
 calraw=np.clip(model.predict_proba(cal[list(PREDICTORS)].astype(float))[:,1],1e-5,1-1e-5)
 testraw=np.clip(model.predict_proba(test[list(PREDICTORS)].astype(float))[:,1],1e-5,1-1e-5)
 label=cal["y6"].astype(int).to_numpy()
 if label.sum()>=12 and (len(label)-label.sum())>=12:
  platt=LogisticRegression(C=.1,max_iter=500,random_state=31)
  platt.fit(logit(calraw).reshape(-1,1),label)
  if platt.coef_[0,0]>0:
   return model,platt.predict_proba(logit(testraw).reshape(-1,1))[:,1],"positive_slope_platt"
 # Do not invert score ordering or estimate a high-variance slope from thin data.
 earlier=train["y6"].astype(int).mean()
 target=(float(label.sum())+200*earlier)/(len(label)+200)
 target=np.clip(target,1e-5,1-1e-5)
 lo,hi=-20.,20.
 for _ in range(55):
  mid=(lo+hi)/2
  if expit(logit(calraw)+mid).mean()<target:lo=mid
  else:hi=mid
 return model,expit(logit(testraw)+(lo+hi)/2),"monotone_prior_shrunk_intercept"

def historical_holdout(fin23,fin24,fin25,snapshot):
 snap=snapshot.copy()
 snap["date"]=pd.to_datetime(snap["date"],errors="coerce").dt.normalize()
 snap["symbol"]=snap["symbol"].astype(str).str.upper().str.strip()
 for needed in ("y6","y6_mature_date","integrity_y6_clean","close","avg_turnover_63"):
  if needed not in snap:raise ValueError("Missing original historical stock/label source "+needed)
 m={2023:original_financial_features(fin23,snap,2023),
    2024:original_financial_features(fin24,snap,2024),
    2025:original_financial_features(fin25,snap,2025)}
 train=eligibility(m[2023],fold_close("2024-12-31"))
 cal=eligibility(m[2024],fold_close("2025-12-31"))
 hold=eligibility(m[2025],TEST_EVAL_UTC)
 if len(train)<MIN_TRAIN or len(cal)<MIN_CAL or len(hold)<MIN_HOLDOUT:
  raise ValueError(f"Insufficient 3FY mature labeled original train/cal/test: {len(train)}/{len(cal)}/{len(hold)}")
 if any(int(x["y6"].sum())<MIN_POS for x in (train,cal,hold)):
  raise ValueError("Too few heldout or calibration 6-month doubles to evaluate model")
 if max(train["date"])>=min(cal["date"]) or max(cal["date"])>=min(hold["date"]):
  raise ValueError("Chronological dates mixed")
 model,p,calmode=calibrated_for_test(train,cal,hold)
 if not np.isfinite(p).all() or not ((p>=0)&(p<=1)).all():
  raise ValueError("Calibrated 3FY pilot has invalid estimates")
 scored=hold[["date","symbol","close","avg_turnover_63","y6",*PREDICTORS]].copy()
 scored["exploratory_p6_double_model_probability"]=p
 scored=scored.sort_values(["exploratory_p6_double_model_probability","symbol"],
                           ascending=[False,True]).reset_index(drop=True)
 scored.insert(0,"rank",np.arange(1,len(scored)+1))
 ten=scored.head(10)
 yy=hold["y6"].astype(int).to_numpy()
 cutoff_p=np.asarray(p)
 result={
  "scope":"INDEPENDENT_THREE_FISCAL_YEAR_HISTORICAL_SINGLE_FOLD_ML_HOLDOUT_NOT_CURRENT_PREDICTIONS",
  "train_year":"2023-12-29","calibration_year":"2024-12-31","unseen_test_year":"2025-12-31",
  "training_rows":len(train),"calibration_rows":len(cal),"test_rows":len(hold),
  "train_positive_twoX_sixmonth":int(train["y6"].sum()),
  "calibration_positive_twoX_sixmonth":int(cal["y6"].sum()),
  "test_positive_twoX_sixmonth":int(hold["y6"].sum()),
  "test_baseline_event_frequency":float(yy.mean()),
  "test_top10_precision":float(ten["y6"].mean()),
  "test_top10_success_count":int(ten["y6"].sum()),
  "test_top10_lift_over_test_prevalence":float(ten["y6"].mean()/yy.mean()) if yy.mean()>0 else None,
  "test_average_precision":float(average_precision_score(yy,cutoff_p)),
  "test_roc_auc":float(roc_auc_score(yy,cutoff_p)) if len(np.unique(yy))==2 else None,
  "test_brier_score":float(brier_score_loss(yy,cutoff_p)),
  "source_three_FY_fiscal_years_exactly_3":True,
  "calibration_method":calmode,"no_V10_baseline":True,
  "stock_test_date_2025_12_is_RETROSPECTIVE_not_original_2026_forward":True,
  "only_one_out_of_time_holdout_fold":True,
  "trading_commissions_slippage_circuit_limits_not_backtested":True,
  "original_oct2026_V11_4_forward_predictions_unchanged":True,
  "model_training_production_approved":False,
  "test_company_symbol_selections_PRIVATE":True}
 return scored,result

def run(fin23,fin24,fin25,snapshot,out):
 scored,summary=historical_holdout(fin23,fin24,fin25,snapshot)
 out=Path(out);out.mkdir(parents=True,exist_ok=True)
 scored.to_csv(out/"asof_2025_12_holdout_predictions_and_mature_outcomes_PRIVATE.csv",index=False)
 (out/"threeFY_chronological_single_test_fold_summary.json").write_text(json.dumps(summary,indent=2))
 print(json.dumps(summary,indent=2),flush=True)
 return summary
def main():
 a=argparse.ArgumentParser()
 for y in (2023,2024,2025):a.add_argument(f"--financial-{y}",required=True)
 a.add_argument("--snapshot",required=True);a.add_argument("--out",required=True)
 opt=a.parse_args()
 original=pd.read_parquet(opt.snapshot,columns=[
  "date","symbol","close","avg_turnover_63","y6","y6_mature_date","integrity_y6_clean"])
 run(pd.read_csv(getattr(opt,"financial_2023")),pd.read_csv(getattr(opt,"financial_2024")),
     pd.read_csv(getattr(opt,"financial_2025")),original,opt.out)
if __name__=="__main__":main()
