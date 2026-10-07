from __future__ import annotations
import argparse,json,os
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq

DATE_HINTS=[
 "date","published_ts","broadCastDate","broadcast_Date","filingDate","transaction_date",
 "quarter_end","as_of_date","snapshot_date","toDate","re_create_dt"
]
REQUIRED={
 "control_labels":{"y6","y12","y24"},
 "control_market":{"symbol","ret_120"},
 "event_core":{"symbol","published_ts"},
 "event_text_any":{"headline","details","document_url","category","event_type"},
 "financial_catalog_any":{"xbrl","broadCastDate","broadcast_Date","filingDate","toDate"},
}

def parquet_info(p:Path):
    pf=pq.ParquetFile(p)
    cols=pf.schema_arrow.names
    info={"path":str(p),"format":"parquet","rows":int(pf.metadata.num_rows),"columns":cols}
    date_ranges={}
    symbol_count=None
    read_cols=[c for c in cols if c in DATE_HINTS]
    if "symbol" in cols: read_cols.append("symbol")
    if read_cols:
        try:
            df=pd.read_parquet(p,columns=list(dict.fromkeys(read_cols)))
            if "symbol" in df: symbol_count=int(df["symbol"].astype(str).nunique())
            for c in [z for z in read_cols if z!="symbol"]:
                s=pd.to_datetime(df[c],utc=True,errors="coerce")
                if s.notna().any(): date_ranges[c]={"min":str(s.min()),"max":str(s.max()),"non_null":int(s.notna().sum())}
        except Exception as e:
            info["range_error"]=repr(e)
    info["date_ranges"]=date_ranges
    if symbol_count is not None:info["symbols"]=symbol_count
    return info

def csv_info(p:Path):
    try:
        head=pd.read_csv(p,nrows=5)
        cols=list(head.columns)
    except Exception as e:
        return {"path":str(p),"format":"csv","error":repr(e),"columns":[]}
    rows=None; symbol_count=None; date_ranges={}
    try:
        use=[c for c in cols if c in DATE_HINTS or c=="symbol"]
        if use:
            df=pd.read_csv(p,usecols=use,low_memory=False)
            rows=len(df)
            if "symbol" in df:symbol_count=int(df["symbol"].astype(str).nunique())
            for c in [z for z in use if z!="symbol"]:
                s=pd.to_datetime(df[c],utc=True,errors="coerce")
                if s.notna().any():date_ranges[c]={"min":str(s.min()),"max":str(s.max()),"non_null":int(s.notna().sum())}
        else:
            with open(p,"rb") as f: rows=max(sum(1 for _ in f)-1,0)
    except Exception as e:
        pass
    out={"path":str(p),"format":"csv","columns":cols,"date_ranges":date_ranges}
    if rows is not None:out["rows"]=int(rows)
    if symbol_count is not None:out["symbols"]=symbol_count
    return out

def inspect_dir(root):
    out=[]
    for p in sorted(Path(root).rglob("*")):
        if not p.is_file():continue
        if p.suffix.lower()==".parquet":out.append(parquet_info(p))
        elif p.suffix.lower()==".csv":out.append(csv_info(p))
        elif p.suffix.lower()==".json":
            try:
                j=json.load(open(p))
                out.append({"path":str(p),"format":"json","top_keys":list(j)[:50] if isinstance(j,dict) else []})
            except Exception as e:
                out.append({"path":str(p),"format":"json","error":repr(e)})
    return out

def has_all(cols,req):return req.issubset(set(cols))
def any_col(cols,req):return bool(set(cols)&set(req))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--events",required=True)
    ap.add_argument("--pit",required=True)
    ap.add_argument("--control",required=True)
    ap.add_argument("--output",required=True)
    a=ap.parse_args()
    inventory=[]
    for label,root in [("events",a.events),("pit",a.pit),("control",a.control)]:
        for z in inspect_dir(root):
            z["artifact_group"]=label;inventory.append(z)
    tables=[z for z in inventory if z.get("format") in {"parquet","csv"} and z.get("columns")]
    control_label_files=[z["path"] for z in tables if has_all(z["columns"],REQUIRED["control_labels"])]
    event_files=[z["path"] for z in tables if has_all(z["columns"],REQUIRED["event_core"]) and any_col(z["columns"],REQUIRED["event_text_any"])]
    financial_files=[z["path"] for z in tables if any_col(z["columns"],REQUIRED["financial_catalog_any"]) and any_col(z["columns"],{"symbol","isin","companyName"})]
    insider_files=[z["path"] for z in tables if any_col(z["columns"],{"transactionType","transaction_date","acqMode","tdpTransactionType","secAcq"})]
    shareholding_files=[z["path"] for z in tables if any_col(z["columns"],{"promoterHolding","promoter_pct","promoterAndPromoterGroup","promoter"}) and any_col(z["columns"],{"date","snapshot_date","quarter_end","broadCastDate"})]
    capabilities={
      "control_labels_present":bool(control_label_files),
      "canonical_event_text_present":bool(event_files),
      "financial_catalog_present":bool(financial_files),
      "insider_history_present":bool(insider_files),
      "shareholding_history_present":bool(shareholding_files),
      "control_label_files":control_label_files,
      "event_files":event_files[:20],
      "financial_catalog_files":financial_files[:20],
      "insider_files":insider_files[:20],
      "shareholding_files":shareholding_files[:20],
    }
    gaps=[]
    if not capabilities["control_labels_present"]:gaps.append("No table with y6/y12/y24 labels found.")
    if not capabilities["canonical_event_text_present"]:gaps.append("No canonical event table with timestamp + text fields found.")
    if not capabilities["financial_catalog_present"]:gaps.append("No historical financial filing/XBRL catalog detected.")
    gaps += [
      "Historical GDELT/Google-News/official-demand features are not assumed present; build them point-in-time per fold before final validation.",
      "Current V11.4 typed magnitude values must not be copied backward; historical order/capex/capacity must be reconstructed only from documents available by each fold.",
      "Historical financial facts must be filtered by filing/broadcast availability timestamp, not quarter end alone."
    ]
    ready_core=capabilities["control_labels_present"] and capabilities["canonical_event_text_present"] and capabilities["financial_catalog_present"]
    summary={"stage":"V11.4 PIT readiness","core_archive_ready":bool(ready_core),"capabilities":capabilities,"gaps":gaps,"inventory_count":len(inventory)}
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    json.dump(summary,open(out,"w"),indent=2,default=str)
    pd.DataFrame([{
      "artifact_group":z.get("artifact_group"),"path":z.get("path"),"format":z.get("format"),
      "rows":z.get("rows"),"symbols":z.get("symbols"),"columns":"|".join(z.get("columns",[])),
      "date_ranges":json.dumps(z.get("date_ranges",{}))
    } for z in inventory]).to_csv(out.parent/"inventory.csv",index=False)
    print(json.dumps(summary,indent=2))

if __name__=="__main__":main()
