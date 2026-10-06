from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests

BASE="https://www.nseindia.com"
PAGE=BASE+"/companies-listing/corporate-filings-announcements"
API=BASE+"/api/corporate-announcements"
HEADERS={
    "user-agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/154.0 Safari/537.36",
    "accept-language":"en-US,en;q=0.9",
    "accept":"application/json,text/plain,*/*",
    "referer":PAGE,
}

def first(row,*names):
    for n in names:
        v=row.get(n)
        if v is not None and str(v).strip() not in {"","nan","None"}:
            return v
    return None

def session():
    s=requests.Session()
    r=s.get(PAGE,headers=HEADERS,timeout=45)
    r.raise_for_status()
    return s

def fetch_range(start,end,retries=5):
    params={"index":"equities","from_date":start.strftime("%d-%m-%Y"),"to_date":end.strftime("%d-%m-%Y")}
    last=None
    for attempt in range(1,retries+1):
        try:
            s=session()
            r=s.get(API,params=params,headers=HEADERS,timeout=90)
            if r.status_code in {401,403,429}:
                raise RuntimeError(f"NSE HTTP {r.status_code}")
            r.raise_for_status()
            data=r.json()
            if isinstance(data,dict) and "data" in data:
                data=data["data"]
            if not isinstance(data,list):
                raise RuntimeError(f"unexpected NSE payload {type(data).__name__}")
            return data
        except Exception as e:
            last=repr(e)
            time.sleep(min(30,2**attempt))
    raise RuntimeError(f"NSE primary fetch failed: {last}")

def parse_ts(v):
    t=pd.to_datetime(v,errors="coerce",dayfirst=True)
    if pd.isna(t): return pd.NaT
    if getattr(t,"tzinfo",None) is None:
        t=t.tz_localize("Asia/Kolkata")
    return t.tz_convert("UTC")

def normalize(rows):
    out=[]
    for r in rows:
        symbol=str(first(r,"symbol","sm_symbol") or "").strip().upper()
        if not symbol: continue
        isin=first(r,"sm_isin","isin","ISIN")
        isin=str(isin).strip().upper() if isin is not None else None
        headline=str(first(r,"desc","subject","SUBJECT") or "").strip()
        details=str(first(r,"attchmntText","details","DETAILS") or "").strip()
        ts=parse_ts(first(r,"an_dt","broadcastDateTime","sort_date","dt"))
        if pd.isna(ts): continue
        attach=first(r,"attchmntFile","attachment","ATTACHMENT")
        recid=first(r,"seq_id","csvName","bflag","orgid")
        raw="|".join(["NSE",str(recid),str(ts),symbol,headline,details[:250]])
        eid=hashlib.sha256(raw.encode()).hexdigest()
        out.append({
            "evidence_id":eid,
            "discovery_source":"NSE_PRIMARY",
            "query_family":"nse_corporate_announcement",
            "published_ts":ts,
            "symbol":symbol,
            "isin":isin,
            "title":headline,
            "details":details,
            "url":urljoin(BASE+"/",str(attach)) if attach else None,
            "domain":"nseindia.com",
            "source_tier":1,
            "source_trust":1.0,
            "source_record_id":str(recid) if recid is not None else eid,
        })
    return pd.DataFrame(out)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--output",required=True)
    ap.add_argument("--lookback-days",type=int,default=45)
    args=ap.parse_args()

    today=pd.Timestamp.now(tz="Asia/Kolkata").normalize().tz_localize(None)
    start=today-pd.Timedelta(days=int(args.lookback_days))
    parts=[]
    cur=start
    while cur<=today:
        end=min(cur+pd.Timedelta(days=29),today)
        parts.append(normalize(fetch_range(cur,end)))
        cur=end+pd.Timedelta(days=1)
    df=pd.concat(parts,ignore_index=True) if parts else pd.DataFrame()
    if len(df):
        df=df.drop_duplicates("evidence_id").sort_values("published_ts",ascending=False)
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
    df.to_parquet(out,index=False)
    summary={
        "status":"ok" if len(df) else "empty",
        "lookback_days":int(args.lookback_days),
        "rows":int(len(df)),
        "symbols":int(df["symbol"].nunique()) if len(df) else 0,
        "with_attachment":int(df["url"].notna().sum()) if len(df) else 0,
        "start":str(df["published_ts"].min()) if len(df) else None,
        "end":str(df["published_ts"].max()) if len(df) else None,
    }
    json.dump(summary,open(out.parent/"nse_primary_summary.json","w"),indent=2,default=str)
    print(json.dumps(summary,indent=2,default=str))
    if len(df)==0:
        raise SystemExit("No NSE primary announcements collected")

if __name__=="__main__":
    main()
