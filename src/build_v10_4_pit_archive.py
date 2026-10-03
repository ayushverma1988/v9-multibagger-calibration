from __future__ import annotations

import argparse, json, time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from huggingface_hub import hf_hub_download

REPO="tejhq/indian-markets"

def norm(v):
    if pd.isna(v): return None
    s=str(v).strip().upper()
    return None if s in {"","NAN","NONE","<NA>","NULL"} else s

def session():
    s=requests.Session()
    s.headers.update({
        "User-Agent":"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/140 Safari/537.36",
        "Accept":"application/json,text/plain,*/*",
        "Accept-Language":"en-US,en;q=0.9",
        "Referer":"https://www.nseindia.com/companies-listing/corporate-filings-financial-results",
    })
    for u in [
        "https://www.nseindia.com/companies-listing/corporate-filings-financial-results",
        "https://www.nseindia.com/companies-listing/corporate-filings-shareholding-pattern",
        "https://www.nseindia.com/companies-listing/corporate-filings-insider-trading",
    ]:
        try:s.get(u,timeout=20)
        except Exception:pass
    return s

def get_json(s:requests.Session,url:str,retries:int=4)->Any:
    last=None
    for k in range(retries):
        try:
            r=s.get(url,timeout=60)
            if r.status_code==200:
                return r.json()
            last=RuntimeError(f"HTTP {r.status_code}: {r.text[:120]}")
            if r.status_code in (401,403):
                try:s.get("https://www.nseindia.com/companies-listing/corporate-filings-financial-results",timeout=20)
                except Exception:pass
        except Exception as e:last=e
        time.sleep(1.0+1.5*k)
    raise RuntimeError(f"GET failed {url}: {last!r}")

def parse_dt(x):
    return pd.to_datetime(x,dayfirst=True,errors="coerce")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--start-year",type=int,default=2016)
    ap.add_argument("--end-year",type=int,default=pd.Timestamp.today().year)
    ap.add_argument("--output",default="outputs_v10_4_pit_archive")
    args=ap.parse_args()
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)

    hp=hf_hub_download(REPO,"symbol_history/nse.parquet",repo_type="dataset")
    hist=pd.read_parquet(hp)
    hist["valid_from"]=pd.to_datetime(hist["valid_from"],errors="coerce")
    hist["valid_to"]=pd.to_datetime(hist["valid_to"],errors="coerce")
    hist["isin_norm"]=hist["isin"].map(norm)
    hist["symbol_norm"]=hist["symbol"].map(norm)
    cutoff=pd.Timestamp(f"{args.start_year}-01-01")
    active=hist[(hist["valid_to"].isna()) | (hist["valid_to"]>=cutoff)].copy()
    symbols=sorted(set(active["symbol_norm"].dropna()))

    s=session()
    fin=[]
    insider=[]
    for y in range(args.start_year,args.end_year+1):
        f="https://www.nseindia.com/api/corporates-financial-results?index=equities&period=Quarterly&from_date=01-01-%d&to_date=31-12-%d"%(y,y)
        try:
            rows=get_json(s,f)
            if isinstance(rows,list):
                for z in rows:z["_query_year"]=y
                fin.extend(rows)
        except Exception as e:
            print("financial",y,"ERROR",repr(e),flush=True)

        u="https://www.nseindia.com/api/corporates-pit?index=equities&from_date=01-01-%d&to_date=31-12-%d"%(y,y)
        try:
            z=get_json(s,u)
            rows=z.get("data",[]) if isinstance(z,dict) else []
            for q in rows:q["_query_year"]=y
            insider.extend(rows)
        except Exception as e:
            print("insider",y,"ERROR",repr(e),flush=True)
        print("year",y,"fin_total",len(fin),"pit_total",len(insider),flush=True)

    fdf=pd.DataFrame(fin)
    if len(fdf):
        for c in ["broadCastDate","filingDate","fromDate","toDate"]:
            if c in fdf:fdf[c]=parse_dt(fdf[c])
        fdf["symbol_norm"]=fdf.get("symbol",pd.Series(index=fdf.index,dtype=object)).map(norm)
        fdf["isin_norm"]=fdf.get("isin",pd.Series(index=fdf.index,dtype=object)).map(norm)
        fdf=fdf.sort_values(["broadCastDate","symbol_norm"]).drop_duplicates(
            subset=[c for c in ["seqNumber","symbol_norm","toDate","consolidated"] if c in fdf.columns],
            keep="last"
        )
        fdf.to_parquet(out/"financial_results_metadata.parquet",index=False)

    idf=pd.DataFrame(insider)
    if len(idf):
        for c in ["anex","tdpTransactionDate","intimDt","broadcastDt","broadcastDate"]:
            if c in idf:idf[c]=parse_dt(idf[c])
        idf["symbol_norm"]=idf.get("symbol",pd.Series(index=idf.index,dtype=object)).map(norm)
        idf["isin_norm"]=idf.get("isin",pd.Series(index=idf.index,dtype=object)).map(norm)
        idf.to_parquet(out/"insider_trades.parquet",index=False)

    # Shareholding master is per symbol. Query every symbol that was active at
    # any point since the start year; records are later resolved to ISIN by the
    # point-in-time symbol history. A failed symbol yields missing features,
    # never a synthetic value.
    shp=[]; errors=[]
    s.headers["Referer"]="https://www.nseindia.com/companies-listing/corporate-filings-shareholding-pattern"
    for i,sym in enumerate(symbols,1):
        url="https://www.nseindia.com/api/corporate-share-holdings-master?index=equities&symbol="+requests.utils.quote(sym)
        try:
            rows=get_json(s,url,retries=3)
            if isinstance(rows,list):
                for z in rows:
                    z["_query_symbol"]=sym
                    shp.append(z)
        except Exception as e:
            errors.append({"symbol":sym,"error":repr(e)})
        if i%100==0:
            print("shareholding",i,"/",len(symbols),"rows",len(shp),"errors",len(errors),flush=True)
        time.sleep(0.025)

    sdf=pd.DataFrame(shp)
    if len(sdf):
        for c in ["broadcastDate","submissionDate","date","revisionDate","revisedDate","systemDate"]:
            if c in sdf:sdf[c]=parse_dt(sdf[c])
        sdf["symbol_norm"]=sdf.get("symbol",sdf.get("_query_symbol",pd.Series(index=sdf.index,dtype=object))).map(norm)
        sdf["isin_norm"]=sdf.get("isin",pd.Series(index=sdf.index,dtype=object)).map(norm)

        # Resolve missing ISIN using symbol history active on report date.
        hmap={s:g.sort_values("valid_from") for s,g in active.groupby("symbol_norm")}
        miss=sdf["isin_norm"].isna()
        for idx,r in sdf[miss].iterrows():
            sym=norm(r.get("symbol_norm")); d=pd.Timestamp(r.get("date")) if pd.notna(r.get("date")) else pd.NaT
            g=hmap.get(sym)
            if g is None or pd.isna(d):continue
            q=g[(g["valid_from"]<=d)&((g["valid_to"].isna())|(g["valid_to"]>=d))]
            if len(q):sdf.at[idx,"isin_norm"]=q.iloc[-1]["isin_norm"]
        sdf=sdf.sort_values(["broadcastDate","symbol_norm"]).drop_duplicates(
            subset=[c for c in ["recordId","symbol_norm","date"] if c in sdf.columns],
            keep="last"
        )
        sdf.to_parquet(out/"shareholding_master.parquet",index=False)

    pd.DataFrame(errors).to_csv(out/"shareholding_fetch_errors.csv",index=False)

    summary={
        "stage":"V10.4 point-in-time source archive",
        "start_year":args.start_year,"end_year":args.end_year,
        "symbols_queried_shareholding":len(symbols),
        "financial_rows":int(len(fdf)),
        "financial_date_min":str(fdf["broadCastDate"].min()) if len(fdf) and "broadCastDate" in fdf else None,
        "financial_date_max":str(fdf["broadCastDate"].max()) if len(fdf) and "broadCastDate" in fdf else None,
        "insider_rows":int(len(idf)),
        "insider_columns":list(idf.columns) if len(idf) else [],
        "shareholding_rows":int(len(sdf)),
        "shareholding_resolved_isin_pct":float(sdf["isin_norm"].notna().mean()) if len(sdf) else 0,
        "shareholding_errors":len(errors),
        "shareholding_date_min":str(sdf["broadcastDate"].min()) if len(sdf) and "broadcastDate" in sdf else None,
        "shareholding_date_max":str(sdf["broadcastDate"].max()) if len(sdf) and "broadcastDate" in sdf else None,
        "point_in_time_rule":"availability timestamp is broadcast/announcement time; period/report dates are never used as information-availability dates",
    }
    json.dump(summary,open(out/"summary.json","w"),indent=2,default=str)
    print(json.dumps(summary,indent=2,default=str))

if __name__=="__main__":
    main()
