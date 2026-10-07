from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests

GDELT="https://api.gdeltproject.org/api/v2/doc/doc"
UA="Mozilla/5.0 (compatible; V11.4ExternalDemand/1.0)"

EVENTS=[
    ("procurement_or_award",1.00,re.compile(r"\b(procurement|tender|bid|award|purchase order|work order|contract awarded|l1 bidder|lowest bidder)\b",re.I)),
    ("deployment_or_commissioning",0.80,re.compile(r"\b(deployment|rollout|commissioned|commissioning|commercial production|operations commenced)\b",re.I)),
    ("capex_or_project",0.85,re.compile(r"\b(capex|investment|project cost|new plant|manufacturing facility|capacity addition|capacity expansion)\b",re.I)),
    ("policy_or_pli",0.70,re.compile(r"\b(pli|production linked incentive|scheme|approved|sanctioned|policy|subsidy|incentive)\b",re.I)),
]

def fetch(q,days,max_records,retries=4):
    params={"query":q,"mode":"artlist","format":"json","maxrecords":int(max_records),"timespan":f"{int(days)}d","sort":"DateDesc"}
    last=None
    for i in range(retries):
        try:
            r=requests.get(GDELT,params=params,headers={"User-Agent":UA},timeout=60)
            r.raise_for_status()
            j=r.json()
            a=j.get("articles",[]) if isinstance(j,dict) else []
            return a if isinstance(a,list) else []
        except Exception as e:
            last=repr(e); time.sleep(min(15,2**i))
    raise RuntimeError(last)

def body(url):
    try:
        r=requests.get(url,headers={"User-Agent":UA},timeout=20)
        if r.status_code!=200 or "text/html" not in r.headers.get("content-type","").lower():
            return ""
        x=re.sub(r"(?is)<script.*?>.*?</script>"," ",r.text)
        x=re.sub(r"(?is)<style.*?>.*?</style>"," ",x)
        x=re.sub(r"(?s)<[^>]+>"," ",x)
        return re.sub(r"\s+"," ",x)[:80000]
    except Exception:
        return ""

def event_type(text):
    for name,w,p in EVENTS:
        if p.search(text): return name,w
    return "general_demand",0.45

def theme_query(words,domains):
    # Keep query compact to avoid GDELT parser limits.
    tw=" OR ".join(f'"{w}"' if " " in w else w for w in words[:5])
    dw=" OR ".join(f"domain:{d}" for d in domains)
    return f"({tw}) (procurement OR tender OR investment OR capex OR project OR PLI OR deployment OR commissioning) ({dw})"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--queries-config",required=True)
    ap.add_argument("--source-config",required=True)
    ap.add_argument("--output",required=True)
    args=ap.parse_args()

    qcfg=json.load(open(args.queries_config))
    scfg=json.load(open(args.source_config))
    days=int(scfg.get("lookback_days",120))
    mx=int(scfg.get("max_records_per_query",60))
    official=set(scfg["official_domains"])
    secondary=set(scfg["secondary_domains"])
    rows=[]; errors=[]

    for theme,words in qcfg["themes"].items():
        for source_class,domains,trust in [
            ("official",scfg["official_domains"],1.0),
            ("secondary",scfg["secondary_domains"],0.60),
        ]:
            q=theme_query(words,domains)
            try:
                arts=fetch(q,days,mx)
            except Exception as e:
                errors.append({"theme":theme,"source_class":source_class,"error":repr(e)})
                continue
            # Fetch only a bounded set of newest pages per query.
            for a in arts[:30]:
                url=str(a.get("url") or "")
                domain=str(a.get("domain") or urlparse(url).netloc).lower().replace("www.","")
                title=str(a.get("title") or "")
                dt=pd.to_datetime(a.get("seendate") or a.get("date"),utc=True,errors="coerce")
                if pd.isna(dt): continue
                txt=title
                if source_class=="official":
                    txt+=" "+body(url)
                et,ew=event_type(txt)
                age=max((pd.Timestamp.now(tz="UTC")-dt).total_seconds()/86400.0,0)
                freshness=math.exp(-age/45.0)
                evidence=trust*ew*freshness
                rows.append({
                    "theme":theme,"published_ts":dt,"source_class":source_class,
                    "domain":domain,"title":title,"url":url,"event_type":et,
                    "trust":trust,"event_weight":ew,"freshness":freshness,
                    "evidence_weight":evidence
                })
            time.sleep(.15)

    df=pd.DataFrame(rows)
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
    if len(df):
        df=df.drop_duplicates(["theme","url"]).sort_values(["theme","published_ts"],ascending=[True,False])
    df.to_parquet(out,index=False)

    scores=[]
    for theme in qcfg["themes"]:
        g=df[df["theme"].eq(theme)] if len(df) else pd.DataFrame()
        if len(g):
            official_g=g[g["source_class"].eq("official")]
            secondary_g=g[g["source_class"].eq("secondary")]
            raw=float(g["evidence_weight"].sum())
            # Official evidence dominates; secondary evidence can add at most modest support.
            off=float(official_g["evidence_weight"].sum())
            sec=float(secondary_g["evidence_weight"].sum())
            combined=off + min(sec,1.0)
            score=1.0-math.exp(-combined/2.5)
            latest=str(g["published_ts"].max())
            distinct_domains=int(g["domain"].nunique())
            official_rows=int(len(official_g))
        else:
            raw=off=sec=score=0.0; latest=None; distinct_domains=official_rows=0
        scores.append({
            "theme":theme,
            "official_rows":official_rows,
            "distinct_domains":distinct_domains,
            "official_weight":off,
            "secondary_weight":sec,
            "demand_score_raw":raw,
            "theme_demand_score":float(max(0,min(score,1))),
            "latest":latest,
        })
    sdf=pd.DataFrame(scores).sort_values("theme_demand_score",ascending=False)
    sdf.to_csv(out.parent/"theme_demand_scores.csv",index=False)
    summary={
        "rows":int(len(df)),"themes_with_official_evidence":int((sdf["official_rows"]>0).sum()),
        "themes_with_any_evidence":int((sdf["distinct_domains"]>0).sum()),
        "top_themes":sdf.head(12).to_dict("records"),"errors":errors
    }
    json.dump(summary,open(out.parent/"external_demand_summary.json","w"),indent=2,default=str)
    print(json.dumps(summary,indent=2,default=str))
    if summary["themes_with_any_evidence"]<3:
        raise SystemExit("External demand coverage too sparse")

if __name__=="__main__":
    main()
