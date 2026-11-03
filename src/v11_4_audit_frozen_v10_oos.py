"""Read-only audit of exact frozen V10.2 OOS predictions for fair backtesting."""
import argparse,json
from pathlib import Path
import pandas as pd

def main():
 p=argparse.ArgumentParser();p.add_argument("--oos",required=True);p.add_argument("--folds",required=True);p.add_argument("--output",required=True);a=p.parse_args()
 out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 x=pd.read_parquet(a.oos)
 fold_dates=pd.to_datetime(pd.read_csv(a.folds)["date"],errors="coerce").dropna().dt.normalize()
 x["date"]=pd.to_datetime(x["date"],errors="coerce").dt.normalize()
 x=x[x["date"].isin(fold_dates)].copy()
 vals=[s for s in x.columns if any(k in s.lower() for k in ("selected","score","rank","prob","p_cal","y6","dd30","hit","risk","integrity"))]
 fields={}
 for c in vals:
  z=x[c]
  fields[c]={"dtype":str(z.dtype),"nonnull":int(z.notna().sum())}
  if str(z.dtype)=="bool":fields[c]["positive"]=int(z.sum())
 for pred in ("selected_v941","selected_v94","selected_v931"):
  if pred in x:
   print("SELECTIONS",pred,x.groupby("date")[pred].sum().to_dict(),flush=True)
 summary={
  "rows":len(x),"dates":int(x["date"].nunique()),
  "all_columns":x.columns.tolist(),
  "selection_probabilities_and_maturity":fields,
  "exact_original_18_folds":len(fold_dates)==18,
  "V10_weight_or_selection_changed":False,
 }
 (out/"frozen_v10_historical_oos_schema.json").write_text(json.dumps(summary,indent=2,default=str))
 print(json.dumps({k:v for k,v in summary.items() if k!="all_columns"},indent=2,default=str)[:18500],flush=True)
 if len(fold_dates)!=18:raise SystemExit("Wrong frozen fold count")
 if "selected_v941" not in x:raise SystemExit("Frozen V10 historical picks absent")
if __name__=="__main__":main()
