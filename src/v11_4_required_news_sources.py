"""Required GDELT and Google News discovery, with honest collection status.

Discovery headlines do not establish a verified primary corporate catalyst.
Today's retrieved news is never backfilled into historical decisions.
Neither provider can replace the other when one request fails.
"""
from __future__ import annotations
from datetime import datetime,timezone
import hashlib,json,time,xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd
import requests

GDELT="https://api.gdeltproject.org/api/v2/doc/doc"
GOOGLE="https://news.google.com/rss/search"
QUERY='("capacity expansion" OR "commercial production" OR "order win" OR "promoter buying" OR demerger OR "regulatory approval")'

def fetch(url,params,session=requests):
    for attempt in range(2):
        try:
            response=session.get(url,params=params,
                headers={"User-Agent":"Mozilla/5.0 V11.4-research"},timeout=(10,40))
        except (requests.Timeout,requests.ConnectionError):
            if attempt==0:
                time.sleep(2);continue
            raise
        if response.status_code==429 and attempt==0:
            try:delay=float(response.headers.get("Retry-After",5))
            except (TypeError,ValueError):delay=5
            time.sleep(max(0,min(30,delay)));continue
        response.raise_for_status()
        if not response.content:raise ValueError("Empty required source response")
        if len(response.content)>5_000_000:raise ValueError("Unexpected oversized discovery response")
        return response
    raise ValueError("Required news source exhausted bounded retries")

def parse_gdelt(response):
    body=response.json()
    if not isinstance(body,dict) or not isinstance(body.get("articles"),list):
        raise ValueError("GDELT did not return an article-list response")
    return [{"provider":"GDELT","title":r.get("title"),"url":r.get("url"),
             "published_utc":pd.to_datetime(r.get("seendate"),format="%Y%m%dT%H%M%SZ",
                                          utc=True,errors="coerce")}
            for r in body["articles"] if isinstance(r,dict)]

def parse_google(response):
    root=ET.fromstring(response.content)
    if root.tag!="rss" or root.find("channel") is None:
        raise ValueError("Google News did not return its RSS channel")
    return [{"provider":"Google News","title":r.findtext("title"),
             "url":r.findtext("link"),"published_utc":pd.to_datetime(
                 r.findtext("pubDate"),utc=True,errors="coerce")}
            for r in root.findall("./channel/item")]

def collect(out,session=requests):
    dest=Path(out);dest.mkdir(parents=True,exist_ok=True)
    now=datetime.now(timezone.utc).isoformat();rows=[];status={}
    requests_to_make=[
        ("GDELT",GDELT,{"query":QUERY+" sourcecountry:India","mode":"artlist",
          "format":"json","maxrecords":100,"timespan":"14d","sort":"DateDesc"},parse_gdelt),
        ("Google News",GOOGLE,{"q":QUERY+" India when:14d","hl":"en-IN",
                              "gl":"IN","ceid":"IN:en"},parse_google)]
    for name,url,params,parser in requests_to_make:
        try:
            response=fetch(url,params,session);items=parser(response)
            status[name]={"status":"COLLECTED" if items else "COLLECTED_EMPTY",
                "rows":len(items),"response_SHA256":hashlib.sha256(response.content).hexdigest()}
            rows.extend(items)
        except Exception as exc:
            status[name]={"status":"BLOCKED","rows":0,"error":str(exc)[:400]}
    table=pd.DataFrame(rows,columns=["provider","title","url","published_utc"])
    table["primary_corporate_evidence_verified"]=False
    table["retrieved_at_utc"]=now
    table.to_csv(dest/"required_news_discovery_PRIVATE.csv",index=False)
    report={"scope":"CURRENT_DISCOVERY_NOT_HISTORICAL_PIT_MODEL_FEATURES",
        "required_sources":["GDELT","Google News"],"providers":status,
        "both_required_sources_collected":all(v["status"].startswith("COLLECTED") for v in status.values()),
        "primary_corporate_verification_complete":False,
        "headlines_do_not_prove_capacity_orders_or_promoter_buys":True,
        "historical_news_reconstruction_complete":False,"retrieved_at_utc":now}
    (dest/"required_news_source_status.json").write_text(json.dumps(report,indent=2))
    return report
