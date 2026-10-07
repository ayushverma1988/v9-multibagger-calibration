from __future__ import annotations

import argparse
import io
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

HOME="https://www.nseindia.com/"
DAILY_API="https://www.nseindia.com/api/daily-reports"
HEADERS={
    "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/134 Safari/537.36",
    "Accept":"application/json,text/plain,*/*",
    "Accept-Language":"en-US,en;q=0.9",
    "Referer":"https://www.nseindia.com/all-reports",
}

OUT_COLS=[
    "symbol","series","is_sme","company_pe","industry_pe_reference","pe_lt_industry_pe",
    "market_cap_crore","free_float_market_cap_crore","industry_basic","industry_sector",
    "valuation_source","pe_report_date","mcap_source"
]

class Client:
    def __init__(self):
        self.s=requests.Session(); self.s.headers.update(HEADERS)
        try:self.s.get(HOME,timeout=20)
        except:pass

    def get(self,url,**kwargs):
        last=None
        for i in range(4):
            try:
                r=self.s.get(url,timeout=45,**kwargs)
                if r.status_code in (401,403):
                    try:self.s.get(HOME,timeout=20)
                    except:pass
                    r=self.s.get(url,timeout=45,**kwargs)
                r.raise_for_status()
                time.sleep(.15)
                return r
            except Exception as e:
                last=e;time.sleep(min(8,1.5*(i+1)))
        raise RuntimeError(f"GET failed {url}: {last!r}")

    def daily_reports(self):
        return self.get(DAILY_API,params={"key":"CM"}).json()

    def download_report(self,item):
        path=str(item.get("filePath") or "")
        name=str(item.get("fileActlName") or "")
        if not name:
            raise RuntimeError(f"missing fileActlName for {item}")
        url=path+name if path.startswith("http") else "https://nsearchives.nseindia.com/"+path.lstrip("/") + name
        return self.get(url).content

def norm(s):
    return re.sub(r"[^a-z0-9]+"," ",str(s or "").lower()).strip()

def pick_report(meta,keywords,asof=None):
    candidates=[]
    for bucket in ["CurrentDay","PreviousDay","FutureDay"]:
        for x in meta.get(bucket,[]) if isinstance(meta,dict) else []:
            label=norm(x.get("displayName"))
            fname=norm(x.get("fileActlName"))
            if not all(k in (label+" "+fname) for k in keywords):
                continue
            td=pd.to_datetime(x.get("tradingDate"),errors="coerce")
            if asof is not None and pd.notna(td) and td.normalize()>asof.normalize():
                continue
            candidates.append((td if pd.notna(td) else pd.Timestamp.min,x))
    if not candidates:return None
    candidates.sort(key=lambda z:z[0],reverse=True)
    return candidates[0][1]

def read_csv_bytes(b):
    for enc in ["utf-8-sig","utf-8","latin1"]:
        try:return pd.read_csv(io.BytesIO(b),encoding=enc)
        except:pass
    raise RuntimeError("unable to parse report CSV")

def first_col(df,names):
    cmap={norm(c):c for c in df.columns}
    for n in names:
        if norm(n) in cmap:return cmap[norm(n)]
    for c in df.columns:
        nc=norm(c)
        for n in names:
            if norm(n) in nc:return c
    return None

def num_series(s):
    return pd.to_numeric(s.astype(str).str.replace(",","",regex=False).str.replace("₹","",regex=False),errors="coerce")

def latest_industry_map(path,asof,all_symbols):
    if not path:return pd.DataFrame(columns=["symbol","industry"])
    p=Path(path)
    if not p.exists():return pd.DataFrame(columns=["symbol","industry"])
    d=pd.read_parquet(p)
    sym="symbol_norm" if "symbol_norm" in d.columns else "symbol"
    if sym not in d.columns or "industry" not in d.columns:
        return pd.DataFrame(columns=["symbol","industry"])
    d["symbol"]=d[sym].astype(str).str.upper().str.strip()
    if "broadCastDate" in d.columns:
        d["_ts"]=pd.to_datetime(d["broadCastDate"],utc=True,errors="coerce")
        cutoff=pd.Timestamp(asof).tz_localize("UTC") if pd.Timestamp(asof).tzinfo is None else pd.Timestamp(asof).tz_convert("UTC")
        d=d[(d["_ts"].isna())|(d["_ts"]<=cutoff)]
    d=d[d["symbol"].isin(all_symbols)].copy()
    sort_cols=["symbol"]
    if "_ts" in d.columns:sort_cols.append("_ts")
    d=d.sort_values(sort_cols).drop_duplicates("symbol",keep="last")
    return d[["symbol","industry"]].rename(columns={"industry":"industry_basic"})

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--symbols-file",required=True)
    ap.add_argument("--symbol-column",default="symbol")
    ap.add_argument("--output",required=True)
    ap.add_argument("--max-symbols",type=int,default=220)
    ap.add_argument("--asof",default="")
    ap.add_argument("--financial-metadata",default="")
    ap.add_argument("--quarterly",default="")
    args=ap.parse_args()

    asof=pd.Timestamp(args.asof).normalize() if args.asof else pd.Timestamp.today().normalize()
    src=pd.read_csv(args.symbols_file)
    syms=list(dict.fromkeys(src[args.symbol_column].dropna().astype(str).str.upper().str.strip()))[:args.max_symbols]
    cl=Client();errors=[];report_diag={}
    pe_all=pd.DataFrame(columns=["symbol","company_pe"])
    mcap_all=pd.DataFrame(columns=["symbol","market_cap_crore","series"])

    try:
        meta=cl.daily_reports()
        report_diag["daily_report_buckets"]={k:len(meta.get(k,[])) for k in ["CurrentDay","PreviousDay","FutureDay"] if isinstance(meta,dict)}

        pe_item=pick_report(meta,["pe","ratio"],asof)
        if pe_item is None:
            pe_item=pick_report(meta,["pe"],asof)
        if pe_item is None:
            raise RuntimeError("NSE Daily Reports did not expose PE Ratio file")
        pedf=read_csv_bytes(cl.download_report(pe_item))
        sc=first_col(pedf,["Symbol"])
        pc=first_col(pedf,["Symbol P/E","Symbol PE","P/E"])
        if sc is None or pc is None:
            raise RuntimeError(f"PE report columns not recognized: {list(pedf.columns)}")
        pe_all=pd.DataFrame({
            "symbol":pedf[sc].astype(str).str.upper().str.strip(),
            "company_pe":num_series(pedf[pc])
        }).drop_duplicates("symbol",keep="last")
        report_diag["pe_file"]=pe_item.get("fileActlName")
        report_diag["pe_trading_date"]=pe_item.get("tradingDate")
        report_diag["pe_rows"]=len(pe_all)

        # Market-cap report is useful if currently exposed; otherwise derive candidate
        # market cap from official PE * TTM PAT below.
        mcap_item=None
        for keys in [["market","capital"],["mcap"]]:
            mcap_item=pick_report(meta,keys,asof)
            if mcap_item is not None:break
        if mcap_item is not None:
            try:
                mdf=read_csv_bytes(cl.download_report(mcap_item))
                ms=first_col(mdf,["Symbol"])
                mm=first_col(mdf,["Market Cap(Rs.)","Market Cap","Market Capitalisation"])
                ser=first_col(mdf,["Series"])
                if ms is not None and mm is not None:
                    raw=num_series(mdf[mm])
                    # MCAP report specification is Rs.; convert to crore.
                    mcap_all=pd.DataFrame({
                        "symbol":mdf[ms].astype(str).str.upper().str.strip(),
                        "market_cap_crore":raw/1e7,
                        "series":mdf[ser].astype(str).str.upper().str.strip() if ser is not None else ""
                    }).drop_duplicates("symbol",keep="last")
                    report_diag["mcap_file"]=mcap_item.get("fileActlName")
                    report_diag["mcap_rows"]=len(mcap_all)
            except Exception as e:
                errors.append({"stage":"mcap_report","error":repr(e)})
    except Exception as e:
        errors.append({"stage":"daily_reports","error":repr(e)})

    # Industry grouping from NSE financial-result metadata, then median positive PE
    # of all symbols in each industry. This reproduces the intent of PE < Industry PE
    # without relying on the fragile per-symbol quote endpoint.
    all_pe_symbols=set(pe_all["symbol"]) if len(pe_all) else set()
    industry=latest_industry_map(args.financial_metadata,asof,all_pe_symbols)
    pe_ind=pe_all.merge(industry,on="symbol",how="left")
    valid=pe_ind[(pe_ind["company_pe"]>0)&pe_ind["industry_basic"].notna()&pe_ind["industry_basic"].astype(str).ne("")]
    ind_median=(valid.groupby("industry_basic")["company_pe"].median().rename("industry_pe_reference").reset_index()
                if len(valid) else pd.DataFrame(columns=["industry_basic","industry_pe_reference"]))

    rows=pd.DataFrame({"symbol":syms})
    rows=rows.merge(pe_all,on="symbol",how="left")
    rows=rows.merge(industry,on="symbol",how="left")
    rows=rows.merge(ind_median,on="industry_basic",how="left")
    rows=rows.merge(mcap_all,on="symbol",how="left")

    # If daily MCAP report is absent, derive market cap from official PE and TTM PAT.
    mcap_source=np.where(rows["market_cap_crore"].notna(),"NSE daily MCAP report","")
    if args.quarterly and Path(args.quarterly).exists():
        q=pd.read_csv(args.quarterly)
        q["symbol"]=q["symbol"].astype(str).str.upper().str.strip()
        qp=q[["symbol"]+[c for c in ["latest_ttm_pat"] if c in q.columns]].copy()
        rows=rows.merge(qp,on="symbol",how="left")
        pat=pd.to_numeric(rows.get("latest_ttm_pat"),errors="coerce")
        derived=pd.to_numeric(rows["company_pe"],errors="coerce")*pat/1e7
        use=rows["market_cap_crore"].isna()&derived.gt(0)
        rows.loc[use,"market_cap_crore"]=derived[use]
        mcap_source=np.where(use,"derived from NSE PE x NSE TTM PAT",mcap_source)
    rows["mcap_source"]=mcap_source

    # V11/V10 market universe is EQ/BE/BZ only; SME SM/ST series are outside the
    # candidate universe by construction. Preserve explicit field for Rule 2.
    if "series" not in rows.columns:
        rows["series"]=""
    rows["is_sme"]=False
    rows["pe_lt_industry_pe"]=(
        pd.to_numeric(rows["company_pe"],errors="coerce").notna()
        & pd.to_numeric(rows["industry_pe_reference"],errors="coerce").notna()
        & (pd.to_numeric(rows["company_pe"],errors="coerce") < pd.to_numeric(rows["industry_pe_reference"],errors="coerce"))
    )
    rows["industry_sector"]=""
    rows["free_float_market_cap_crore"]=np.nan
    rows["valuation_source"]="NSE Daily PE Ratio + NSE financial metadata"
    rows["pe_report_date"]=report_diag.get("pe_trading_date")

    for c in OUT_COLS:
        if c not in rows.columns:rows[c]=np.nan
    rows=rows[OUT_COLS]

    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    rows.to_csv(out,index=False)
    err=pd.DataFrame(errors,columns=["stage","error"])
    err.to_csv(out.parent/"valuation_errors.csv",index=False)
    summary={
        "symbols_requested":len(syms),
        "symbols_extracted":int(len(rows)),
        "errors":len(errors),
        "pe_coverage":float(pd.to_numeric(rows["company_pe"],errors="coerce").notna().mean()) if len(rows) else 0,
        "industry_pe_coverage":float(pd.to_numeric(rows["industry_pe_reference"],errors="coerce").notna().mean()) if len(rows) else 0,
        "market_cap_coverage":float(pd.to_numeric(rows["market_cap_crore"],errors="coerce").notna().mean()) if len(rows) else 0,
        "sme_count":int(rows["is_sme"].fillna(False).sum()) if len(rows) else 0,
        "report_diagnostics":report_diag,
        "error_samples":errors[:10],
    }
    json.dump(summary,open(out.parent/"valuation_summary.json","w"),indent=2,default=str)
    print(json.dumps(summary,indent=2,default=str))

if __name__=="__main__":
    main()
