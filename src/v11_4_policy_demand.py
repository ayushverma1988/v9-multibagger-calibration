from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import feedparser
import pandas as pd
import requests

def fetch_body_text(url, headers):
    if not url:
        return ""
    try:
        r=requests.get(url,headers=headers,timeout=30)
        r.raise_for_status()
        txt=r.text
        txt=re.sub(r"(?is)<script.*?>.*?</script>"," ",txt)
        txt=re.sub(r"(?is)<style.*?>.*?</style>"," ",txt)
        txt=re.sub(r"(?s)<[^>]+>"," ",txt)
        txt=re.sub(r"\s+"," ",txt)
        return txt[:150000]
    except Exception:
        return ""

def theme_hits(text,cfg):
    t=str(text).lower()
    hits=[]
    for theme,words in cfg["themes"].items():
        if any(w.lower() in t for w in words):
            hits.append(theme)
    return hits

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--queries-config",required=True)
    ap.add_argument("--output",required=True)
    ap.add_argument("--lookback-days",type=int,default=30)
    args=ap.parse_args()

    cfg=json.load(open(args.queries_config))
    feeds=[
        "https://pib.gov.in/RssMain.aspx?ModId=6&Lang=1&Regid=1",
        "https://pib.gov.in/RssMain.aspx?ModId=6&Lang=1&Regid=3",
        "https://pib.gov.in/RssMain.aspx?ModId=6&Lang=1&Regid=48",
    ]
    headers={
        "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/134 Safari/537.36",
        "Accept":"application/rss+xml,application/xml,text/xml,text/html;q=0.9,*/*;q=0.8",
        "Referer":"https://www.pib.gov.in/ViewRss.aspx?lang=1&reg=3",
    }
    d=None; feed=None; fetch_errors=[]
    for candidate in feeds:
        try:
            r=requests.get(candidate,headers=headers,timeout=35)
            r.raise_for_status()
            parsed=feedparser.parse(r.content)
            if len(parsed.entries):
                d=parsed; feed=candidate; break
            fetch_errors.append({"feed":candidate,"status":r.status_code,"bytes":len(r.content),"entries":0})
        except Exception as exc:
            fetch_errors.append({"feed":candidate,"error":repr(exc)})
    if d is None:
        d=feedparser.FeedParserDict(entries=[])
        feed=feeds[0]
    now=pd.Timestamp.now(tz="UTC")
    rows=[]
    for e in d.entries:
        title=str(e.get("title",""))
        summary=str(e.get("summary",""))
        link=str(e.get("link",""))
        dt=pd.to_datetime(e.get("published") or e.get("updated"),utc=True,errors="coerce")
        if pd.isna(dt):
            continue
        age=max((now-dt).total_seconds()/86400,0)
        if age>args.lookback_days:
            continue
        body=fetch_body_text(link,headers)
        hits=theme_hits(title+" "+summary+" "+body,cfg)
        for theme in hits:
            freshness=math.exp(-age/14.0)
            rows.append({
                "published_ts":dt,
                "source":"PIB",
                "domain":"pib.gov.in",
                "title":title,
                "url":link,
                "theme":theme,
                "freshness":freshness,
                "primary_source":True,
                "body_scanned":bool(body),
            })
    df=pd.DataFrame(rows)
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
    df.to_parquet(out,index=False)

    if len(df):
        agg=df.groupby("theme").agg(
            pib_releases=("title","count"),
            distinct_releases=("url","nunique"),
            demand_score_raw=("freshness","sum"),
            latest=("published_ts","max"),
        ).reset_index()
        # bounded score: 1-exp(-sum), so repeated fresh releases saturate.
        agg["theme_demand_score"]=1.0-(-agg["demand_score_raw"]).map(math.exp)
    else:
        agg=pd.DataFrame(columns=["theme","pib_releases","distinct_releases","demand_score_raw","latest","theme_demand_score"])
    agg.to_csv(out.parent/"theme_demand_scores.csv",index=False)
    summary={
        "rss_entries":int(len(d.entries)),
        "matched_rows":int(len(df)),
        "themes":agg.sort_values("theme_demand_score",ascending=False).to_dict("records") if len(agg) else [],
        "feed":feed,
        "fetch_errors":fetch_errors,
    }
    json.dump(summary,open(out.parent/"policy_demand_summary.json","w"),indent=2,default=str)
    print(json.dumps(summary,indent=2,default=str))

if __name__=="__main__":
    main()
