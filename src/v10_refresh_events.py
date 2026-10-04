from __future__ import annotations

import argparse, hashlib, json, time
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests

BASE="https://www.nseindia.com"
PAGE=BASE+"/companies-listing/corporate-filings-announcements"
API=BASE+"/api/corporate-announcements"
HEADERS={
    "user-agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0 Safari/537.36",
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

def parse_ts(v):
    t=pd.to_datetime(v,errors="coerce",dayfirst=True)
    if pd.isna(t): return pd.NaT
    if getattr(t,"tzinfo",None) is None: t=t.tz_localize("Asia/Kolkata")
    return t.tz_convert("UTC")

def session():
    s=requests.Session()
    r=s.get(PAGE,headers=HEADERS,timeout=45)
    r.raise_for_status()
    return s

def fetch_range(start:pd.Timestamp,end:pd.Timestamp,retries=5):
    params={"index":"equities","from_date":start.strftime("%d-%m-%Y"),"to_date":end.strftime("%d-%m-%Y")}
    last=None
    for attempt in range(1,retries+1):
        try:
            s=session()
            r=s.get(API,params=params,headers=HEADERS,timeout=90)
            if r.status_code in {401,403,429}: raise RuntimeError(f"NSE HTTP {r.status_code}")
            r.raise_for_status()
            data=r.json()
            if isinstance(data,dict) and "data" in data: data=data["data"]
            if not isinstance(data,list): raise RuntimeError(f"unexpected payload {type(data).__name__}")
            return data
        except Exception as exc:
            last=repr(exc); time.sleep(min(30,2**attempt))
    raise RuntimeError(f"recent NSE announcement refresh failed: {last}")

def classify(subject,details):
    text=(str(subject)+" "+str(details)).lower()
    ca=["split","bonus","merger","demerger","scheme of arrangement","name change","change in name","symbol change"]
    return "corporate_action" if any(x in text for x in ca) else "other"

def normalize(rows):
    out=[]
    for r in rows:
        symbol=str(first(r,"symbol","sm_symbol") or "").strip().upper()
        isin=first(r,"sm_isin","isin","ISIN")
        isin=str(isin).strip().upper() if isin is not None else None
        if isin in {"","NAN","NONE","NULL"}: isin=None
        headline=str(first(r,"desc","subject","SUBJECT") or "").strip()
        details=str(first(r,"attchmntText","details","DETAILS") or "").strip()
        ts=parse_ts(first(r,"an_dt","broadcastDateTime","sort_date","dt"))
        if pd.isna(ts): continue
        recid=first(r,"seq_id","csvName","bflag","orgid")
        raw="|".join(["NSE",str(recid),str(ts),symbol,headline])
        if recid is None: recid=hashlib.sha256(raw.encode()).hexdigest()
        attach=first(r,"attchmntFile","attachment","ATTACHMENT")
        out.append({
            "published_ts":ts,
            "source":"NSE",
            "source_record_id":str(recid),
            "symbol":symbol,
            "isin":isin,
            "headline":headline,
            "details":details,
            "event_type":classify(headline,details),
            "document_url":urljoin(BASE+"/",str(attach)) if attach else None,
        })
    return pd.DataFrame(out)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--base-events",required=True)
    ap.add_argument("--output",required=True)
    ap.add_argument("--lookback-days",type=int,default=120)
    args=ap.parse_args()

    base=pd.read_parquet(args.base_events).copy()
    base["published_ts"]=pd.to_datetime(base["published_ts"],utc=True,errors="coerce")
    today=pd.Timestamp.now(tz="Asia/Kolkata").normalize().tz_localize(None)
    start=today-pd.Timedelta(days=int(args.lookback_days))

    parts=[]
    s=start
    while s<=today:
        e=min(s+pd.Timedelta(days=29),today)
        parts.append(normalize(fetch_range(s,e)))
        s=e+pd.Timedelta(days=1)

    recent=pd.concat(parts,ignore_index=True) if parts else pd.DataFrame()
    if recent.empty:
        raise RuntimeError("Recent NSE event refresh returned zero rows; refusing to publish with unverified event freshness")

    # Add any missing canonical columns as nulls, then append/dedupe by source id.
    for c in base.columns:
        if c not in recent: recent[c]=pd.NA
    for c in recent.columns:
        if c not in base: base[c]=pd.NA
    allf=pd.concat([base,recent[base.columns]],ignore_index=True)
    allf["published_ts"]=pd.to_datetime(allf["published_ts"],utc=True,errors="coerce")
    allf=allf.dropna(subset=["published_ts","source","source_record_id"])
    allf=allf.sort_values(["published_ts","source","source_record_id"]).drop_duplicates(["source","source_record_id"],keep="last")

    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
    allf.to_parquet(out,index=False)
    summary={
        "base_rows":int(len(base)),
        "recent_rows_fetched":int(len(recent)),
        "merged_rows":int(len(allf)),
        "start":str(allf["published_ts"].min()),
        "end":str(allf["published_ts"].max()),
        "refresh_start":str(start.date()),
        "refresh_end":str(today.date()),
    }
    json.dump(summary,open(out.parent/"event_refresh_summary.json","w"),indent=2,default=str)
    print(json.dumps(summary,indent=2,default=str))

if __name__=="__main__":
    main()
