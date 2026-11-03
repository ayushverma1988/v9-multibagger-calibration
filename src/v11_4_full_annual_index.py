"""Marketwide annual NSE financial FILING INDEX for frozen point-in-time folds.

Source records remain unverified candidates, not numeric indicators.
Historical universe is the union of the original 18 folds (no current NSE list).
"""
from __future__ import annotations
import argparse,hashlib,json,time
from pathlib import Path
import pandas as pd
import requests

API="https://www.nseindia.com/api/corporates-financial-results"
HOME="https://www.nseindia.com/companies-listing/corporate-filings-financial-results"

def availability_timestamp(v, fallback=False):
    x=pd.to_datetime(v,dayfirst=True,errors="coerce")
    if pd.isna(x):return pd.NaT
    if x.tzinfo is None:
        # Without a published clock time, conservatively assume the close
        # of calendar day; cannot be used for that session's 15:30 cutoff.
        if fallback and len(str(v).strip())<=11:
            x=x.normalize()+pd.Timedelta(hours=23,minutes=59,seconds=59)
        x=x.tz_localize("Asia/Kolkata")
    return x.tz_convert("UTC")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--snapshot",required=True)
    p.add_argument("--folds",required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--shard-index",type=int,required=True)
    p.add_argument("--shard-count",type=int,default=8)
    p.add_argument("--limit",type=int,default=0)
    a=p.parse_args()
    if not (0<=a.shard_index<a.shard_count):raise SystemExit("bad shard index")
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    snap=pd.read_parquet(a.snapshot,columns=["date","symbol"])
    snap["date"]=pd.to_datetime(snap["date"],errors="coerce").dt.normalize()
    snap["symbol"]=snap["symbol"].astype(str).str.upper().str.strip()
    dates=pd.read_csv(a.folds)
    folds=pd.to_datetime(dates["date"],errors="coerce").dropna().dt.normalize()
    if len(folds)!=18:raise SystemExit(f"Expected 18 original folds; got {len(folds)}")
    eligible=set(snap.loc[snap["date"].isin(folds),"symbol"])
    eligible.discard("")
    universe=sorted(eligible,key=lambda x:hashlib.sha256(x.encode()).hexdigest())
    chosen=universe[a.shard_index::a.shard_count]
    if a.limit>0:chosen=chosen[:a.limit]
    session=requests.Session()
    session.headers.update({
        "User-Agent":"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36",
        "Accept":"application/json,text/plain,*/*",
        "Referer":HOME,
        "X-Requested-With":"XMLHttpRequest",
    })
    try:session.get("https://www.nseindia.com/",timeout=15)
    except requests.RequestException:pass
    rows=[];errors=[];statuses=[]
    for n,sym in enumerate(chosen,1):
        attempt_error=None
        for attempt in (1,2):
            try:
                res=session.get(API,params={"index":"equities","symbol":sym,"period":"Annual"},timeout=22)
                if res.status_code==429:
                    raise RuntimeError("NSE rate limit HTTP 429; stop this shard")
                res.raise_for_status()
                filings=res.json()
                if not isinstance(filings,list):raise ValueError("Unexpected annual results schema")
                accepted=0
                for z in filings:
                    if not isinstance(z,dict):continue
                    pub=availability_timestamp(z.get("broadCastDate"))
                    original="broadcast"
                    if pd.isna(pub):
                        pub=availability_timestamp(z.get("filingDate"),fallback=True)
                        original="date_only_conservative" if not pd.isna(pub) else "unknown"
                    end=pd.to_datetime(z.get("toDate"),errors="coerce",dayfirst=True)
                    if pd.isna(pub) or pd.isna(end):continue
                    period=pd.Timestamp(end).tz_localize("UTC")
                    if pub<period:continue
                    rows.append({
                        "symbol":sym,"fy_end":str(end.date()),"available_at_utc":pub.isoformat(),
                        "publication_precision":original,"consolidated":str(z.get("consolidated") or ""),
                        "xbrl_url":str(z.get("xbrl") or ""),"original_filing_date":str(z.get("filingDate") or ""),
                        "original_broadcast_date":str(z.get("broadCastDate") or ""),
                        "filing_period":str(z.get("period") or ""),"source":"nse_annual_financial_archive",
                    });accepted+=1
                statuses.append({"symbol":sym,"status":"retrieved","rows":accepted})
                break
            except (requests.RequestException,ValueError,RuntimeError) as exc:
                attempt_error=f"{type(exc).__name__}: {str(exc)[:160]}"
                if "429" in attempt_error:break
                if attempt==1:time.sleep(1.5)
        else:
            pass
        if not statuses or statuses[-1]["symbol"]!=sym:
            statuses.append({"symbol":sym,"status":"error","rows":0})
            errors.append({"symbol":sym,"error":attempt_error})
        if attempt_error and "429" in attempt_error:
            errors.append({"symbol":sym,"error":attempt_error})
            break
        if n%25==0:print(f"shard {a.shard_index} annual symbols {n}/{len(chosen)} indexed rows={len(rows)} errors={len(errors)}",flush=True)
        time.sleep(0.35)
    df=pd.DataFrame(rows,columns=[
        "symbol","fy_end","available_at_utc","publication_precision","consolidated","xbrl_url",
        "original_filing_date","original_broadcast_date","filing_period","source"])
    df=df.drop_duplicates(["symbol","fy_end","available_at_utc","xbrl_url"])
    df.to_parquet(out/"historical_annual_filing_index_shard.parquet",index=False)
    df.to_csv(out/"historical_annual_filing_index_shard.csv",index=False)
    summary={
        "scope":"HISTORICAL_ANNUAL_INDEX_PIT_SOURCE_ONLY",
        "shard_index":a.shard_index,"shard_count":a.shard_count,
        "historical_fold_union_n":len(universe),"requested":len(chosen),
        "attempted":len(statuses),"retrieved":sum(z["status"]=="retrieved" for z in statuses),
        "with_index_rows":sum(z["rows"]>0 for z in statuses),
        "filing_index_rows":len(df),"errors":errors,
        "source_timestamps_enforced":True,"numeric_facts_validated":False,
        "no_current_universe_substitution":True,
        "original_18_folds_unchanged":True,
    }
    (out/"annual_shard_summary.json").write_text(json.dumps(summary,indent=2))
    (out/"annual_shard_status.json").write_text(json.dumps(statuses,indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k!="errors"},indent=2),flush=True)
    if errors:raise SystemExit(f"Annual index shard {a.shard_index} has {len(errors)} source errors; retain artifact for targeted retry")
if __name__=="__main__":main()
