"""Read-only pilot of NSE Quarterly financial filing INDEX with historical timestamps.

No model training, no inferred old values, no use of documents filed after a fold.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
import pandas as pd
import requests

ENDPOINT="https://www.nseindia.com/api/corporates-financial-results"

def as_ist(v):
    if not v: return pd.NaT
    dt=pd.to_datetime(v,errors="coerce",dayfirst=True)
    if pd.isna(dt):return pd.NaT
    return dt.tz_localize("Asia/Kolkata") if dt.tzinfo is None else dt.tz_convert("Asia/Kolkata")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--snapshot",required=True)
    ap.add_argument("--output",required=True)
    ap.add_argument("--sample-size",type=int,default=16)
    a=ap.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    snap=pd.read_parquet(a.snapshot,columns=["date","symbol"])
    snap["date"]=pd.to_datetime(snap["date"],errors="coerce").dt.normalize()
    snap["symbol"]=snap["symbol"].astype(str).str.upper().str.strip()
    syms=set()
    for fold in ("2018-06-29","2020-06-30","2021-12-31"):
        syms.update(snap.loc[snap["date"].eq(pd.Timestamp(fold)),"symbol"])
    # Fixed random-like sample across historical constituents, not today's shortlist.
    chosen=sorted(syms,key=lambda s:hashlib.sha256(s.encode()).hexdigest())[:a.sample_size]
    sess=requests.Session()
    sess.headers.update({
        "User-Agent":"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124 Safari/537.36",
        "Accept":"application/json,text/plain,*/*",
        "Referer":"https://www.nseindia.com/companies-listing/corporate-filings-financial-results",
    })
    try:sess.get("https://www.nseindia.com/",timeout=12)
    except requests.RequestException:pass
    rows=[];errors=[]
    for sym in chosen:
        try:
            r=sess.get(ENDPOINT,params={"index":"equities","symbol":sym,"period":"Quarterly"},timeout=18)
            if r.status_code==429:
                errors.append({"symbol":sym,"error":"HTTP 429; pause before repeating"});break
            r.raise_for_status()
            obj=r.json()
            if not isinstance(obj,list):raise ValueError("Unexpected annual response structure")
            for z in obj:
                if not isinstance(z,dict):continue
                availability=as_ist(z.get("broadCastDate") or z.get("filingDate"))
                end=as_ist(z.get("toDate"))
                if pd.isna(availability) or pd.isna(end) or availability<end:continue
                rows.append({
                    "symbol":sym,"fy_end":str(end.date()),
                    "available_at_utc":availability.tz_convert("UTC").isoformat(),
                    "xbrl_url":str(z.get("xbrl") or ""),
                    "filed_period":str(z.get("period") or ""),
                    "consolidated":str(z.get("consolidated") or ""),
                })
            print(f"{sym} quarterly_catalog_rows={sum(x['symbol']==sym for x in rows)}",flush=True)
        except (requests.RequestException,ValueError) as e:
            errors.append({"symbol":sym,"error":str(e)[:170]})
        time.sleep(0.5)
    d=pd.DataFrame(rows,columns=["symbol","fy_end","available_at_utc","xbrl_url","filed_period","consolidated"])
    d.to_csv(out/"annual_filings_source_index_PIT_CANDIDATES.csv",index=False)
    pd.DataFrame(errors,columns=["symbol","error"]).to_csv(out/"annual_catalog_errors.csv",index=False)
    good=d[pd.to_datetime(d["available_at_utc"],utc=True,errors="coerce").le(pd.Timestamp("2021-12-31T10:00:00Z"))]
    fetched=pd.to_datetime(d["available_at_utc"],utc=True,errors="coerce")
    period_end=pd.to_datetime(d["fy_end"],errors="coerce")
    asof=pd.Timestamp("2025-12-31T10:00:00Z")
    fresh=d[(fetched<=asof)&(period_end>=pd.Timestamp("2025-06-30"))]
    result={
        "sample_requested":len(chosen),"sample_completed":len(chosen)-len(errors),
        "quarterly_index_rows":len(d),
        "distinct_companies_quarterly_rows":d["symbol"].nunique(),
        "distinct_companies_with_pre_2022_filings":good["symbol"].nunique(),
        "pre_2022_filing_rows":len(good),
        "earliest_fy_end":str(d["fy_end"].min()) if len(d) else None,
        "status":"SOURCE_FEASIBILITY_ONLY_NOT_BACKTEST_ELIGIBLE",
        "source":"Official NSE quarterly financial filing catalog; retain timestamps and original URLs",
        "errors":errors,
    }
    (out/"annual_source_probe_summary.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)
    if d.empty:raise SystemExit("No usable quarterly financial catalog returned")
if __name__=="__main__":main()
