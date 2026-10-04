from __future__ import annotations
import argparse, glob, json
from pathlib import Path
import numpy as np, pandas as pd

def find_one(root,name):
    hits=glob.glob(str(Path(root)/"**"/name),recursive=True)
    if not hits: raise FileNotFoundError(name)
    return hits[0]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--baseline-dir",required=True)
    ap.add_argument("--v104-dir",required=True)
    ap.add_argument("--output",default="outputs_v10_5b_risk_coverage_audit")
    a=ap.parse_args()
    out=Path(a.output); out.mkdir(parents=True,exist_ok=True)

    b=pd.read_parquet(find_one(a.baseline_dir,"oos_predictions.parquet"))
    v=pd.read_parquet(find_one(a.v104_dir,"oos_predictions.parquet"))
    for d in (b,v): d["date"]=pd.to_datetime(d["date"])
    cols=["date","symbol"]
    avail=[c for c in ["p_dd30_cal","p_dd30_raw","dd_model_dispersion","p100_cal","y6","dd30_6m"] if c in v.columns]
    m=b.merge(v[cols+avail],on=cols,how="left",suffixes=("_base","_v104"),indicator=True)

    riskcol="p_dd30_cal_v104" if "p_dd30_cal_v104" in m else "p_dd30_cal"
    rawcol="p_dd30_raw_v104" if "p_dd30_raw_v104" in m else "p_dd30_raw"
    m["risk_missing"]=m[riskcol].isna()
    m["raw_available"]=m[rawcol].notna() if rawcol in m else False
    m["join_missing"]=m["_merge"].ne("both")

    bydate=m.groupby("date").agg(
        rows=("symbol","size"),
        risk_missing=("risk_missing","sum"),
        raw_available=("raw_available","sum"),
        join_missing=("join_missing","sum"),
    ).reset_index()
    bydate["coverage"]=1-bydate["risk_missing"]/bydate["rows"]

    miss=m[m["risk_missing"]].copy()
    summary={
        "rows":int(len(m)),
        "v104_rows":int(len(v)),
        "risk_non_null":int(m[riskcol].notna().sum()),
        "risk_coverage":float(m[riskcol].notna().mean()),
        "missing_rows":int(m["risk_missing"].sum()),
        "missing_with_raw_available":int((m["risk_missing"]&m["raw_available"]).sum()),
        "missing_due_join":int(m["join_missing"].sum()),
        "missing_dates":int(m.loc[m["risk_missing"],"date"].nunique()),
        "date_coverage":{str(r.date.date()):float(r.coverage) for r in bydate.itertuples()},
    }
    json.dump(summary,open(out/"summary.json","w"),indent=2,default=str)
    bydate.to_csv(out/"coverage_by_date.csv",index=False)
    miss.to_csv(out/"missing_rows.csv",index=False)
    print(json.dumps(summary,indent=2,default=str))

if __name__=="__main__": main()
