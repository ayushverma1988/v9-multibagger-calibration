from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

HOME="https://www.nseindia.com/"
API="https://www.nseindia.com/api/quote-equity"
HEADERS={
    "User-Agent":"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36",
    "Accept":"application/json,text/plain,*/*",
    "Accept-Language":"en-US,en;q=0.9",
    "Referer":"https://www.nseindia.com/get-quotes/equity",
}

class Client:
    def __init__(self):
        self.s=requests.Session();self.s.headers.update(HEADERS)
        try:self.s.get(HOME,timeout=20)
        except:pass
    def get_json(self,params):
        last=None
        for i in range(4):
            try:
                r=self.s.get(API,params=params,timeout=35)
                if r.status_code in (401,403):
                    try:self.s.get(HOME,timeout=20)
                    except:pass
                    r=self.s.get(API,params=params,timeout=35)
                r.raise_for_status();time.sleep(.12);return r.json()
            except Exception as e:
                last=e;time.sleep(min(8,1.5*(i+1)))
        raise RuntimeError(repr(last))

def num(v):
    if v is None:return np.nan
    try:return float(str(v).replace(",",""))
    except:return np.nan

def nested(d,path):
    cur=d
    for k in path:
        if not isinstance(cur,dict):return None
        cur=cur.get(k)
    return cur

def find_key(obj,keys):
    if isinstance(obj,dict):
        for k,v in obj.items():
            if str(k).lower() in keys and v not in (None,"","-"):
                return v
        for v in obj.values():
            z=find_key(v,keys)
            if z not in (None,"","-"):return z
    elif isinstance(obj,list):
        for v in obj:
            z=find_key(v,keys)
            if z not in (None,"","-"):return z
    return None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--symbols-file",required=True)
    ap.add_argument("--symbol-column",default="symbol")
    ap.add_argument("--output",required=True)
    ap.add_argument("--max-symbols",type=int,default=220)
    args=ap.parse_args()

    src=pd.read_csv(args.symbols_file)
    syms=list(dict.fromkeys(src[args.symbol_column].dropna().astype(str).str.upper().str.strip()))[:args.max_symbols]
    cl=Client();rows=[];errors=[]
    for i,sym in enumerate(syms,1):
        try:
            q=cl.get_json({"symbol":sym})
            t=cl.get_json({"symbol":sym,"section":"trade_info"})
            meta=q.get("metadata",{}) if isinstance(q,dict) else {}
            sec=q.get("securityInfo",{}) if isinstance(q,dict) else {}
            ind=q.get("industryInfo",{}) if isinstance(q,dict) else {}
            series=str(meta.get("series") or sec.get("series") or "").upper()
            symbol_pe=num(meta.get("pdSymbolPe"))
            sector_pe=num(meta.get("pdSectorPe"))
            if not np.isfinite(symbol_pe):
                symbol_pe=num(find_key(q,{"pdsymbolpe","symbolpe","pe"}))
            if not np.isfinite(sector_pe):
                sector_pe=num(find_key(q,{"pdsectorpe","sectorpe","industrype"}))
            mcap=find_key(t,{"totalmarketcap","marketcap","totalmarketcapitalization"})
            ff=find_key(t,{"ffmc","freefloatmarketcap","freefloatmarketcapitalization"})
            rows.append({
                "symbol":sym,
                "series":series,
                "is_sme":series in {"SM","ST"},
                "company_pe":symbol_pe,
                "industry_pe_reference":sector_pe,
                "pe_lt_industry_pe":bool(np.isfinite(symbol_pe) and np.isfinite(sector_pe) and symbol_pe<sector_pe),
                "market_cap_crore":num(mcap),
                "free_float_market_cap_crore":num(ff),
                "industry_basic":ind.get("basicIndustry") or ind.get("industry") or "",
                "industry_sector":ind.get("sector") or "",
                "valuation_source":"NSE quote-equity / trade_info",
            })
        except Exception as e:
            errors.append({"symbol":sym,"error":repr(e)})
        if i%25==0:print("valuation",i,"/",len(syms),"ok",len(rows),"errors",len(errors),flush=True)

    df=pd.DataFrame(rows)
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(out,index=False)
    pd.DataFrame(errors).to_csv(out.parent/"valuation_errors.csv",index=False)
    summary={
        "symbols_requested":len(syms),"symbols_extracted":len(df),"errors":len(errors),
        "pe_coverage":float(pd.to_numeric(df.get("company_pe"),errors="coerce").notna().mean()) if len(df) else 0,
        "industry_pe_coverage":float(pd.to_numeric(df.get("industry_pe_reference"),errors="coerce").notna().mean()) if len(df) else 0,
        "market_cap_coverage":float(pd.to_numeric(df.get("market_cap_crore"),errors="coerce").notna().mean()) if len(df) else 0,
        "sme_count":int(df.get("is_sme",pd.Series(dtype=bool)).fillna(False).sum()) if len(df) else 0,
    }
    json.dump(summary,open(out.parent/"valuation_summary.json","w"),indent=2)
    print(json.dumps(summary,indent=2))

if __name__=="__main__":
    main()
