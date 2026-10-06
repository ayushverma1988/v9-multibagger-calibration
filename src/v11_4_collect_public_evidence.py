from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests

GDELT="https://api.gdeltproject.org/api/v2/doc/doc"
UA="Mozilla/5.0 (compatible; V11.4CatalystResearch/1.0; +https://github.com/)"


def fetch_gdelt(query:str,days:int,max_records:int,retries:int=4):
    params={
        "query":query,
        "mode":"artlist",
        "format":"json",
        "maxrecords":int(max_records),
        "timespan":f"{int(days)}d",
        "sort":"DateDesc",
    }
    last=None
    for i in range(retries):
        try:
            r=requests.get(GDELT,params=params,headers={"User-Agent":UA},timeout=60)
            r.raise_for_status()
            j=r.json()
            arts=j.get("articles",[]) if isinstance(j,dict) else []
            if not isinstance(arts,list):
                raise RuntimeError("unexpected GDELT payload")
            return arts
        except Exception as e:
            last=repr(e)
            time.sleep(min(20,2**i))
    raise RuntimeError(f"GDELT query failed: {last}")


def norm_article(a,family,query):
    url=str(a.get("url") or "").strip()
    title=str(a.get("title") or "").strip()
    dt=pd.to_datetime(a.get("seendate") or a.get("date"),utc=True,errors="coerce")
    domain=str(a.get("domain") or urlparse(url).netloc).lower()
    raw="|".join([family,url,title,str(dt)])
    return {
        "evidence_id":hashlib.sha256(raw.encode()).hexdigest(),
        "discovery_source":"GDELT",
        "query_family":family,
        "query":query,
        "published_ts":dt,
        "title":title,
        "url":url,
        "domain":domain,
        "language":a.get("language"),
        "source_country":a.get("sourcecountry"),
        "socialimage":a.get("socialimage"),
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",required=True)
    ap.add_argument("--output",required=True)
    ap.add_argument("--lookback-days",type=int,default=None)
    args=ap.parse_args()

    cfg=json.load(open(args.config))
    days=int(args.lookback_days or cfg.get("lookback_days_default",14))
    max_records=int(cfg.get("gdelt_max_records",250))
    rows=[]
    errors=[]
    for q in cfg["queries"]:
        family=q["family"]; query=q["query"]
        try:
            arts=fetch_gdelt(query,days,max_records)
            rows.extend(norm_article(a,family,query) for a in arts)
            print(f"{family}: {len(arts)} articles",flush=True)
        except Exception as e:
            errors.append({"family":family,"error":repr(e)})
            print(f"{family}: ERROR {e!r}",flush=True)

    df=pd.DataFrame(rows)
    if len(df):
        df=df.dropna(subset=["url"]).drop_duplicates("evidence_id").sort_values("published_ts",ascending=False)
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
    df.to_parquet(out,index=False)

    summary={
        "lookback_days":days,
        "rows":int(len(df)),
        "families":df["query_family"].value_counts().to_dict() if len(df) else {},
        "domains":df["domain"].value_counts().head(25).to_dict() if len(df) else {},
        "errors":errors,
        "status":"ok" if len(df)>0 else "empty",
    }
    json.dump(summary,open(out.parent/"public_evidence_collection_summary.json","w"),indent=2,default=str)
    print(json.dumps(summary,indent=2,default=str))
    if len(df)==0:
        raise SystemExit("No public evidence collected")


if __name__=="__main__":
    main()
