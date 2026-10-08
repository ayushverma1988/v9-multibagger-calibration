"""Explain disjoint standalone 2020/2021 Top10: raw vs Platt ordering.

Diagnostic only, no model selection and no fitting with held-out labels.
Requires exact already archived V11.4 18-fold PIT features and outcome maturity.
"""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.linear_model import LogisticRegression
from v11_4_standalone_train_walkforward import (
 keep_train,eligible_asof,safe_featureize,make_model,perturbation_training,
 MODEL_FEATURES,STABILITY_SEEDS,CALIBRATION_C,calibration_mode,calibrate_logit,top10_similarity
)

DATES=("2020-12-31","2021-06-30","2023-06-30")
def main():
 p=argparse.ArgumentParser();p.add_argument("--features",required=True);p.add_argument("--snapshot",required=True);p.add_argument("--output",required=True);a=p.parse_args()
 out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 f=pd.read_parquet(a.features)
 snap=pd.read_parquet(a.snapshot,columns=["date","symbol","close","avg_turnover_63","y6","y6_mature_date","integrity_y6_clean"])
 for z in (f,snap):
  z["date"]=pd.to_datetime(z["date"]).dt.normalize()
  z["symbol"]=z["symbol"].astype(str).str.upper().str.strip()
 z=safe_featureize(f.merge(snap,on=["date","symbol"],how="left",validate="1:1",suffixes=("","_snapshot")))
 allrows=[]
 for date in DATES:
  day=pd.Timestamp(date)
  trainable=z[keep_train(z,day)]
  prior=sorted(trainable["date"].unique())
  cal=trainable[trainable["date"]==prior[-1]]
  base=trainable[trainable["date"]<prior[-1]]
  test=z[z["date"].eq(day)&eligible_asof(z)]
  baseline_raw=[];baseline_cal=[]
  for seed in (None,*STABILITY_SEEDS):
   sample=base if seed is None else perturbation_training(base,seed)
   fitted=make_model()
   fitted.fit(sample[list(MODEL_FEATURES)],sample["y6"].astype(int))
   pv=fitted.predict_proba(cal[list(MODEL_FEATURES)])[:,1]
   raw=fitted.predict_proba(test[list(MODEL_FEATURES)])[:,1]
   if calibration_mode(cal["y6"])=="platt":
    c=LogisticRegression(C=CALIBRATION_C,solver="lbfgs",max_iter=400,random_state=31)
    cp=np.clip(pv,1e-5,1-1e-5)
    c.fit(np.log(cp/(1-cp)).reshape(-1,1),cal["y6"].astype(int))
    slope=float(c.coef_[0][0])
   else:slope=None
   calibrated=calibrate_logit(raw,pv,cal["y6"],sample["y6"])
   rawsel=test.assign(p=raw).sort_values(["p","symbol"],ascending=[False,True]).head(10)["symbol"].tolist()
   calsel=test.assign(p=calibrated).sort_values(["p","symbol"],ascending=[False,True]).head(10)["symbol"].tolist()
   if seed is None:baseline_raw=rawsel;baseline_cal=calsel
   allrows.append({
    "date":date,"seed":"unperturbed" if seed is None else seed,
    "prior_cal_fold":str(pd.Timestamp(prior[-1]).date()),
    "calibration_winner_count":int(cal["y6"].sum()),
    "fitted_platt_slope":slope,
    "raw_top10_overlap_with_unperturbed":top10_similarity(rawsel,baseline_raw),
    "calibrated_top10_overlap_with_unperturbed":top10_similarity(calsel,baseline_cal),
    "raw_and_calibrated_top10_overlap":top10_similarity(rawsel,calsel),
    "candidate_10_raw":"|".join(rawsel),
    "candidate_10_calibrated":"|".join(calsel),
   })
 df=pd.DataFrame(allrows)
 df.to_csv(out/"ranking_inversion_PIT_2020_2023.csv",index=False)
 print(df.drop(columns=["candidate_10_raw","candidate_10_calibrated"]).to_string(index=False),flush=True)
 print(json.dumps({"source":"PIT-only standalone V11.4 original folds",
      "rows":len(df),
      "negative_slope_cases":int((df["fitted_platt_slope"]<0).sum()),
      "production_changed":False},indent=2),flush=True)
if __name__=="__main__":main()
