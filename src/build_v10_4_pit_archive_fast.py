from __future__ import annotations

import argparse, json, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from huggingface_hub import hf_hub_download

import build_v10_4_pit_archive as slow

_tls=threading.local()

def tls_session():
    s=getattr(_tls,"s",None)
    if s is None:
        s=slow.session()
        s.headers["Referer"]="https://www.nseindia.com/companies-listing/corporate-filings-shareholding-pattern"
        _tls.s=s
    return s

def fetch_shp(sym):
    s=tls_session()
    url="https://www.nseindia.com/api/corporate-share-holdings-master?index=equities&symbol="+requests.utils.quote(sym)
    try:
        rows=slow.get_json(s,url,retries=3)
        if not isinstance(rows,list): rows=[]
        for z in rows:z["_query_symbol"]=sym
        time.sleep(.04)
        return sym,rows,None
    except Exception as e:
        return sym,[],repr(e)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--snapshot-file",required=True)
    ap.add_argument("--start-year",type=int,default=2016)
    ap.add_argument("--end-year",type=int,default=pd.Timestamp.today().year)
    ap.add_argument("--workers",type=int,default=6)
    ap.add_argument("--output",default="outputs_v10_4_pit_archive_fast")
    args=ap.parse_args()
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)

    snap=pd.read_parquet(args.snapshot_file,columns=["symbol","isin"])
    symbols=sorted(set(snap["symbol"].dropna().astype(str).str.upper().str.strip()))
    print("snapshot symbols",len(symbols),flush=True)

    hp=hf_hub_download(slow.REPO,"symbol_history/nse.parquet",repo_type="dataset")
    hist=pd.read_parquet(hp)
    hist["valid_from"]=pd.to_datetime(hist["valid_from"],errors="coerce")
    hist["valid_to"]=pd.to_datetime(hist["valid_to"],errors="coerce")
    hist["isin_norm"]=hist["isin"].map(slow.norm)
    hist["symbol_norm"]=hist["symbol"].map(slow.norm)

    s=slow.session()
    fin=[]; insider=[]
    for y in range(args.start_year,args.end_year+1):
        fu=f"https://www.nseindia.com/api/corporates-financial-results?index=equities&period=Quarterly&from_date=01-01-{y}&to_date=31-12-{y}"
        try:
            rows=slow.get_json(s,fu)
            if isinstance(rows,list):
                for z in rows:z["_query_year"]=y
                fin.extend(rows)
        except Exception as e: print("financial",y,"ERROR",repr(e),flush=True)

        iu=f"https://www.nseindia.com/api/corporates-pit?index=equities&from_date=01-01-{y}&to_date=31-12-{y}"
        try:
            z=slow.get_json(s,iu)
            rows=z.get("data",[]) if isinstance(z,dict) else []
            for q in rows:q["_query_year"]=y
            insider.extend(rows)
        except Exception as e: print("insider",y,"ERROR",repr(e),flush=True)
        print("year",y,"fin",len(fin),"pit",len(insider),flush=True)

    fdf=pd.DataFrame(fin)
    if len(fdf):
        for c in ["broadCastDate","filingDate","fromDate","toDate"]:
            if c in fdf:fdf[c]=slow.parse_dt(fdf[c])
        fdf["symbol_norm"]=fdf.get("symbol",pd.Series(index=fdf.index,dtype=object)).map(slow.norm)
        fdf["isin_norm"]=fdf.get("isin",pd.Series(index=fdf.index,dtype=object)).map(slow.norm)
        sub=[c for c in ["seqNumber","symbol_norm","toDate","consolidated"] if c in fdf.columns]
        fdf=fdf.sort_values(["broadCastDate","symbol_norm"]).drop_duplicates(subset=sub,keep="last")
        fdf.to_parquet(out/"financial_results_metadata.parquet",index=False)

    idf=pd.DataFrame(insider)
    if len(idf):
        for c in ["anex","tdpTransactionDate","intimDt","broadcastDt","broadcastDate"]:
            if c in idf:idf[c]=slow.parse_dt(idf[c])
        idf["symbol_norm"]=idf.get("symbol",pd.Series(index=idf.index,dtype=object)).map(slow.norm)
        idf["isin_norm"]=idf.get("isin",pd.Series(index=idf.index,dtype=object)).map(slow.norm)
        idf.to_parquet(out/"insider_trades.parquet",index=False)

    shp=[]; errors=[]
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs={ex.submit(fetch_shp,sym):sym for sym in symbols}
        for i,fut in enumerate(as_completed(futs),1):
            sym,rows,err=fut.result()
            shp.extend(rows)
            if err:errors.append({"symbol":sym,"error":err})
            if i%100==0:
                print("shareholding",i,"/",len(symbols),"rows",len(shp),"errors",len(errors),flush=True)

    sdf=pd.DataFrame(shp)
    if len(sdf):
        for c in ["broadcastDate","submissionDate","date","revisionDate","revisedDate","systemDate"]:
            if c in sdf:sdf[c]=slow.parse_dt(sdf[c])
        sdf["symbol_norm"]=sdf.get("symbol",sdf.get("_query_symbol",pd.Series(index=sdf.index,dtype=object))).map(slow.norm)
        sdf["isin_norm"]=sdf.get("isin",pd.Series(index=sdf.index,dtype=object)).map(slow.norm)
        hmap={s:g.sort_values("valid_from") for s,g in hist.groupby("symbol_norm")}
        for idx,r in sdf[sdf["isin_norm"].isna()].iterrows():
            sym=slow.norm(r.get("symbol_norm")); d=pd.Timestamp(r.get("date")) if pd.notna(r.get("date")) else pd.NaT
            g=hmap.get(sym)
            if g is None or pd.isna(d):continue
            q=g[(g["valid_from"]<=d)&((g["valid_to"].isna())|(g["valid_to"]>=d))]
            if len(q):sdf.at[idx,"isin_norm"]=q.iloc[-1]["isin_norm"]
        sub=[c for c in ["recordId","symbol_norm","date"] if c in sdf.columns]
        sdf=sdf.sort_values(["broadcastDate","symbol_norm"]).drop_duplicates(subset=sub,keep="last")
        sdf.to_parquet(out/"shareholding_master.parquet",index=False)

    pd.DataFrame(errors,columns=["symbol","error"]).to_csv(out/"shareholding_fetch_errors.csv",index=False)
    summary={
        "stage":"V10.4 fast point-in-time archive",
        "snapshot_symbols_queried":len(symbols),
        "workers":args.workers,
        "financial_rows":int(len(fdf)),
        "insider_rows":int(len(idf)),
        "insider_columns":list(idf.columns) if len(idf) else [],
        "shareholding_rows":int(len(sdf)),
        "shareholding_resolved_isin_pct":float(sdf["isin_norm"].notna().mean()) if len(sdf) else 0,
        "shareholding_errors":len(errors),
        "shareholding_date_min":str(sdf["broadcastDate"].min()) if len(sdf) and "broadcastDate" in sdf else None,
        "shareholding_date_max":str(sdf["broadcastDate"].max()) if len(sdf) and "broadcastDate" in sdf else None,
        "universe_rule":"Only symbols present in accepted V10.2 integrity-clean snapshot dataset; no outcome-based filtering.",
    }
    json.dump(summary,open(out/"summary.json","w"),indent=2,default=str)
    print(json.dumps(summary,indent=2,default=str))

if __name__=="__main__":
    main()
