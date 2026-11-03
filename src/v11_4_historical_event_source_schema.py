import argparse,json
from pathlib import Path
import pandas as pd
def main():
 p=argparse.ArgumentParser();p.add_argument("--events",required=True);p.add_argument("--out",required=True);a=p.parse_args()
 x=pd.read_parquet(a.events)
 info={"count":len(x),"columns":[]}
 for col in x.columns:
  s=x[col];entry={"name":col,"dtype":str(s.dtype),"nonnull":int(s.notna().sum())}
  if s.notna().any():entry["sample"]=str(s.dropna().iloc[0])[:200]
  if any(z in col.lower() for z in ("event","type","source","kind","category")):
   try:entry["top_values"]=s.astype(str).value_counts().head(12).to_dict()
   except Exception:pass
  info["columns"].append(entry)
 folder=Path(a.out);folder.mkdir(parents=True,exist_ok=True)
 (folder/"event_schema.json").write_text(json.dumps(info,indent=2))
 print(json.dumps(info,indent=2)[:21000])
if __name__=="__main__":main()
