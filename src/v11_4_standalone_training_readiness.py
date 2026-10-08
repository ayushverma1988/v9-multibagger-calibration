"""Inspect why historical V11.4 standalone folds are trainable/not trainable.

Research audit only; tests chronological 6m maturity and no future labels.
Prints the *feasible* separate calibration block in each historical fold.
No model fitting, no hyperparameter tuning, no benchmark comparisons.
"""
import argparse,json
from pathlib import Path
import pandas as pd
from v11_4_standalone_train_walkforward import keep_train,eligible_asof,fold_close

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--features",required=True)
 p.add_argument("--snapshot",required=True)
 p.add_argument("--output",required=True)
 a=p.parse_args()
 out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 x=pd.read_parquet(a.features,columns=["date","symbol","integrity_feature_clean"])
 y=pd.read_parquet(a.snapshot,columns=[
   "date","symbol","close","avg_turnover_63","y6",
   "y6_mature_date","integrity_y6_clean"])
 for f in (x,y):
  f["date"]=pd.to_datetime(f["date"]).dt.normalize()
  f["symbol"]=f["symbol"].astype(str).str.upper().str.strip()
 z=x.merge(y,on=["date","symbol"],validate="1:1",suffixes=("","_source"))
 report=[]
 for d in sorted(z["date"].unique()):
  train=z[keep_train(z,d)]
  groups=train.groupby("date")["y6"].agg(rows="count",positives="sum").reset_index()
  cal_latest=groups.iloc[-1] if len(groups) else None
  base=groups.iloc[:-1] if len(groups) else groups
  last2=groups.iloc[-2:] if len(groups)>=2 else groups
  before2=groups.iloc[:-2] if len(groups)>=2 else groups.iloc[:0]
  current=z[z["date"].eq(d)&eligible_asof(z)]
  exact=fold_close(d)
  row={
    "fold_date":str(pd.Timestamp(d).date()),"asof_utc":exact.isoformat(),
    "prior_mature_train_folds":len(groups),"prior_mature_train_rows":len(train),
    "prior_mature_train_doubles":int(train["y6"].sum()),
    "one_latest_calibration_fold":str(cal_latest["date"].date()) if cal_latest is not None else None,
    "one_latest_calibration_rows":int(cal_latest["rows"]) if cal_latest is not None else 0,
    "one_latest_calibration_doubles":int(cal_latest["positives"]) if cal_latest is not None else 0,
    "one_latest_model_folds":len(base),
    "one_latest_model_rows":int(base["rows"].sum()),"one_latest_model_doubles":int(base["positives"].sum()),
    "last_two_calibration_rows":int(last2["rows"].sum()),
    "last_two_calibration_doubles":int(last2["positives"].sum()),
    "last_two_base_folds":len(before2),
    "last_two_base_rows":int(before2["rows"].sum()),
    "last_two_base_doubles":int(before2["positives"].sum()),
    "current_eligible_stocks":len(current),
  }
  row["original_training_ready"]=(row["one_latest_model_folds"]>=2
    and row["one_latest_model_rows"]>=1500
    and row["one_latest_model_doubles"]>=45
    and row["one_latest_calibration_doubles"]>=12)
  report.append(row)
 result=pd.DataFrame(report)
 result.to_csv(out/"standalone_v11_4_fold_training_readiness.csv",index=False)
 summary={
  "scope":"STANDALONE_PIT_TRAINING_DIAGNOSTIC_ONLY",
  "folds":len(result),
  "original_design_ready_folds":int(result["original_training_ready"].sum()),
  "last_fold":str(result.iloc[-1]["fold_date"]),
  "no_model_fit":True,"no_benchmark_model_comparison":True,
  "no_maturity_gate_changes":True,
 }
 (out/"standalone_v11_4_training_readiness_summary.json").write_text(json.dumps(summary,indent=2))
 print(result.to_string(index=False),flush=True)
 print(json.dumps(summary,indent=2),flush=True)
 if len(result)!=18:raise SystemExit("Did not find exactly 18 original historical folds")
if __name__=="__main__":main()
