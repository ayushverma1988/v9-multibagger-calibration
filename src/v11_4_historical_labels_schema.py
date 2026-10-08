"""Inspect frozen V10 historical market/label schema without running predictions."""
import argparse,json
from pathlib import Path
import pandas as pd

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--snapshot",required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    df=pd.read_parquet(a.snapshot)
    listing=[]
    for name in df.columns:
        ser=df[name]
        item={"column":name,"dtype":str(ser.dtype),
              "nonnull":int(ser.notna().sum()),"unique":int(ser.nunique())}
        if any(k in name.lower() for k in ("return","6m","12m","doubl","label","split","adjust","date","symbol","price","target","volume","sector")):
            item["sample"]=str(ser.dropna().iloc[0])[:160] if ser.notna().any() else ""
        listing.append(item)
    outdata={"source":"Original frozen V10 historical snapshot dataset",
             "rows":len(df),"columns":listing,
             "date_range":(str(df["date"].min()),str(df["date"].max())) if "date" in df else [],
             "training_started":False}
    (out/"frozen_snapshot_schema.json").write_text(json.dumps(outdata,indent=2))
    print(json.dumps(outdata,indent=2)[:18000])
if __name__=="__main__":main()
