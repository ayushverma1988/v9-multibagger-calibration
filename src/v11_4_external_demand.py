from __future__ import annotations

import argparse
import json
import math
import re
import time
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests

UA="Mozilla/5.0 (compatible; V11.4ExternalDemand/1.2)"

EVENTS=[
    ("procurement_or_award",1.00,re.compile(r"\b(procurement|tender|bid|award|purchase order|work order|contract awarded|l1 bidder|lowest bidder|loa)\b",re.I)),
    ("deployment_or_commissioning",0.85,re.compile(r"\b(deployment|rollout|commissioned|commissioning|commercial production|operations commenced|go[- ]live)\b",re.I)),
    ("capex_or_project",0.85,re.compile(r"\b(capex|investment|project cost|new plant|manufacturing facility|capacity addition|capacity expansion|brownfield|greenfield)\b",re.I)),
    ("policy_or_pli",0.72,re.compile(r"\b(pli|production linked incentive|scheme|approved|sanctioned|policy|subsidy|incentive)\b",re.I)),
    ("qualification_or_approval",0.82,re.compile(r"\b(vendor approval|customer qualification|approved vendor|certification|regulatory approval|product approval)\b",re.I)),
]

def req(url,params=None,retries=4,timeout=60):
    last=None
    for i in range(retries):
        try:
            r=requests.get(url,params=params,headers={"User-Agent":UA},timeout=timeout,allow_redirects=True)
            if r.status_code==429:
                time.sleep(min(45,8*(i+1)))
                last=f"429 {r.url}"
                continue
            r.raise_for_status()
            return r
        except Exception as e:
            last=repr(e)
            time.sleep(min(20,3*(i+1)))
    raise RuntimeError(last)

def parse_rss(content,limit=None):
    root=ET.fromstring(content)
    out=[]
    for item in root.findall(".//item"):
        title=(item.findtext("title") or "").strip()
        link=(item.findtext("link") or "").strip()
        pub=(item.findtext("pubDate") or "").strip()
        source=item.find("source")
        source_name=(source.text or "").strip() if source is not None else ""
        source_url=(source.attrib.get("url","").strip() if source is not None else "")
        out.append({
            "title":title,"url":link,"published":pub,
            "source_name":source_name,"source_url":source_url
        })
        if limit and len(out)>=limit:
            break
    return out



class _TRParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_tr=False
        self.parts=[]
        self.rows=[]
    def handle_starttag(self,tag,attrs):
        if tag.lower()=="tr":
            self.in_tr=True
            self.parts=[]
    def handle_data(self,data):
        if self.in_tr:
            s=re.sub(r"\s+"," ",str(data)).strip()
            if s:self.parts.append(s)
    def handle_endtag(self,tag):
        if tag.lower()=="tr" and self.in_tr:
            txt=" ".join(self.parts).strip()
            if txt:self.rows.append(txt)
            self.in_tr=False
            self.parts=[]

def parse_html_rows(content):
    try:
        p=_TRParser()
        p.feed(content.decode("utf-8","ignore") if isinstance(content,(bytes,bytearray)) else str(content))
        return p.rows
    except Exception:
        return []

def fetch_pib_official(url,max_records=120):
    r=req(url,timeout=60)
    return parse_rss(r.content,limit=max_records)

def fetch_cppp_rows(url,max_rows=300):
    r=req(url,timeout=60)
    rows=parse_html_rows(r.content)
    out=[]
    for txt in rows[:max_rows]:
        # Skip navigation/boilerplate rows. Active tender rows generally contain a
        # title/reference and at least one date/time; bid-award rows contain award language.
        low=txt.lower()
        if len(txt)<30: continue
        if not (
            re.search(r"\b\d{1,2}[-/](?:[A-Za-z]{3}|\d{1,2})[-/]\d{2,4}\b",txt)
            or re.search(r"\b\d{1,2}-[A-Za-z]{3}-\d{4}\b",txt)
            or any(k in low for k in ["tender","bid award","procurement","purchase","project","supply","installation","construction"])
        ):
            continue
        out.append(txt[:2500])
    return out

def google_official_query(theme_words,days,domains):
    tw=" OR ".join(f'"{w}"' if " " in w else w for w in theme_words[:6])
    demand='procurement OR tender OR award OR order OR capex OR investment OR commissioning OR "commercial production" OR approval OR PLI OR sanctioned'
    dw=" OR ".join(f"site:{d}" for d in domains[:12])
    return f"({tw}) ({demand}) ({dw}) when:{int(days)}d"

def fetch_gdelt_feed(url,max_records=3000):
    r=req(url,timeout=75)
    return parse_rss(r.content,limit=max_records)

def fetch_google_news(url,q,max_records=60):
    r=req(url,params={"q":q,"hl":"en-IN","gl":"IN","ceid":"IN:en"},timeout=60)
    return parse_rss(r.content,limit=max_records)

def parse_ts(v):
    if not v:return pd.NaT
    try:return pd.Timestamp(parsedate_to_datetime(v)).tz_convert("UTC")
    except Exception:
        return pd.to_datetime(v,utc=True,errors="coerce")

def event_type(text):
    for name,w,p in EVENTS:
        if p.search(text): return name,w
    return "general_demand",0.45

def keyword_hit(text,term):
    t=str(term or "").strip().lower()
    if not t:return False
    pat=re.escape(t).replace(r"\ ",r"(?:\s|[-/])+")
    return re.search(r"(?<![A-Za-z0-9])"+pat+r"(?![A-Za-z0-9])",str(text),re.I) is not None

def themes_for(text,themes):
    return [k for k,words in themes.items() if any(keyword_hit(text,w) for w in words)]

def domain_matches(domain,domains):
    d=(domain or "").lower().replace("www.","").strip(".")
    for x in domains:
        x=x.lower().replace("www.","").strip(".")
        if d==x or d.endswith("."+x):
            return True
    return False

def source_class(domain,scfg):
    if domain_matches(domain,scfg.get("official_domains",[])):
        return "official",1.00
    if domain_matches(domain,scfg.get("secondary_domains",[])):
        return "secondary",0.72
    return "other_discovery",0.48

def article_domain(item):
    s=(item.get("source_url") or "").strip()
    if s:
        d=urlparse(s).netloc.lower().replace("www.","")
        if d:return d
    u=(item.get("url") or "").strip()
    return urlparse(u).netloc.lower().replace("www.","")

def resolve_url(url):
    if not url:return ""
    try:
        r=requests.get(url,headers={"User-Agent":UA},timeout=18,allow_redirects=True)
        return r.url or url
    except Exception:
        return url

def get_body(url):
    if not url:return ""
    try:
        r=requests.get(url,headers={"User-Agent":UA},timeout=18)
        if r.status_code!=200:return ""
        if "html" not in r.headers.get("content-type","").lower():return ""
        x=re.sub(r"(?is)<script.*?>.*?</script>"," ",r.text)
        x=re.sub(r"(?is)<style.*?>.*?</style>"," ",x)
        x=re.sub(r"(?s)<[^>]+>"," ",x)
        return re.sub(r"\s+"," ",x)[:60000]
    except Exception:
        return ""

def google_query(theme_words,days):
    tw=" OR ".join(f'"{w}"' if " " in w else w for w in theme_words[:6])
    demand='procurement OR tender OR award OR order OR capex OR investment OR commissioning OR "commercial production" OR approval OR PLI'
    return f"({tw}) ({demand}) when:{int(days)}d"

def norm_title(s):
    s=re.sub(r"[^a-z0-9 ]+"," ",str(s).lower())
    return re.sub(r"\s+"," ",s).strip()

def add_row(rows,theme,ts,discovery_source,item,domain,sclass,trust,text,scfg,resolved=False):
    et,ew=event_type(text)
    age=max((pd.Timestamp.now(tz="UTC")-ts).total_seconds()/86400.0,0.0)
    freshness=math.exp(-age/45.0)
    evidence=trust*ew*freshness
    rows.append({
        "theme":theme,
        "published_ts":ts,
        "discovery_source":discovery_source,
        "source_class":sclass,
        "domain":domain,
        "title":str(item.get("title") or ""),
        "url":str(item.get("url") or ""),
        "resolved_underlying":bool(resolved),
        "event_type":et,
        "trust":trust,
        "event_weight":ew,
        "freshness":freshness,
        "evidence_weight":evidence,
    })

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--queries-config",required=True)
    ap.add_argument("--source-config",required=True)
    ap.add_argument("--output",required=True)
    args=ap.parse_args()

    qcfg=json.load(open(args.queries_config))
    scfg=json.load(open(args.source_config))
    rows=[]; errors=[]
    source_health={"gdelt":False,"google_news":False,"pib_direct":False,"cppp_direct":False}
    source_items={"gdelt":0,"google_news":0,"pib_direct":0,"cppp_direct":0,"google_official":0}

    # 1) Mandatory GDELT raw/live article stream. This does not use the throttled DOC API.
    gdelt_url=scfg.get("gdelt",{}).get("live_article_feed","https://data.gdeltproject.org/gdeltv3/gal/feed.rss")
    try:
        arts=fetch_gdelt_feed(gdelt_url)
        source_items["gdelt"]=len(arts)
        source_health["gdelt"]=len(arts)>0
        for item in arts:
            title=str(item.get("title") or "")
            url=str(item.get("url") or "")
            d=article_domain(item)
            ts=parse_ts(item.get("published"))
            if pd.isna(ts):continue
            sclass,trust=source_class(d,scfg)
            text=title
            # GDELT GAL points directly to publisher URLs. For configured official/secondary
            # domains, attempt body extraction to improve theme/event precision.
            if sclass in {"official","secondary"}:
                b=get_body(url)
                if b:text+=" "+b
            hits=themes_for(text,qcfg["themes"])
            if not hits:continue
            for theme in hits:
                add_row(rows,theme,ts,"gdelt",item,d,sclass,trust,text,scfg,resolved=True)
    except Exception as e:
        errors.append({"source":"gdelt","error":repr(e)})

    # 2) Mandatory Google News broad recent-news discovery. Query each theme independently.
    google_url=scfg.get("google_news",{}).get("rss_search","https://news.google.com/rss/search")
    days=int(scfg.get("google_news",{}).get("lookback_days",scfg.get("lookback_days",120)))
    mx=int(scfg.get("google_news",{}).get("max_results_per_theme",60))
    google_total=0
    google_success=0
    for theme,words in qcfg["themes"].items():
        q=google_query(words,days)
        try:
            arts=fetch_google_news(google_url,q,mx)
            google_success+=1
            google_total+=len(arts)
        except Exception as e:
            errors.append({"source":"google_news","theme":theme,"error":repr(e)})
            continue
        for item in arts:
            title=str(item.get("title") or "")
            ts=parse_ts(item.get("published"))
            if pd.isna(ts):continue
            d=article_domain(item)
            sclass,trust=source_class(d,scfg)
            # The query establishes candidate theme relevance, but require at least one
            # explicit theme keyword in the title unless an underlying article can be resolved.
            title_hits=themes_for(title,qcfg["themes"])
            resolved=False
            text=title
            resolved_url=resolve_url(str(item.get("url") or ""))
            rd=urlparse(resolved_url).netloc.lower().replace("www.","")
            if rd and "news.google." not in rd and rd!="news.google.com":
                resolved=True
                if d in {"","news.google.com"}: d=rd
                item=dict(item); item["url"]=resolved_url
                sclass,trust=source_class(d,scfg)
                if sclass in {"official","secondary"}:
                    b=get_body(resolved_url)
                    if b:text+=" "+b
            hits=themes_for(text,qcfg["themes"])
            if theme not in hits and theme not in title_hits:
                continue
            add_row(rows,theme,ts,"google_news",item,d,sclass,trust,text,scfg,resolved=resolved)
        time.sleep(.35)

    source_items["google_news"]=google_total
    source_health["google_news"]=(google_success>=max(8,len(qcfg["themes"])//2) and google_total>0)

    # 3) Direct official-demand evidence: PIB release feeds.
    direct_cfg=scfg.get("official_direct",{})
    pib_count=0
    for feed_url in direct_cfg.get("pib_rss",[]):
        try:
            arts=fetch_pib_official(feed_url)
        except Exception as e:
            errors.append({"source":"pib_direct","url":feed_url,"error":repr(e)})
            continue
        for item in arts:
            title=str(item.get("title") or "")
            url=str(item.get("url") or "")
            ts=parse_ts(item.get("published"))
            if pd.isna(ts): continue
            text=title
            b=get_body(url)
            if b:text+=" "+b
            hits=themes_for(text,qcfg["themes"])
            if not hits: continue
            dom=urlparse(url).netloc.lower().replace("www.","") or "pib.gov.in"
            if not domain_matches(dom,["pib.gov.in"]): dom="pib.gov.in"
            for theme in hits:
                add_row(rows,theme,ts,"pib_direct",item,dom,"official",1.0,text,scfg,resolved=True)
                pib_count+=1
    source_items["pib_direct"]=pib_count
    source_health["pib_direct"]=pib_count>0

    # 4) Direct Government of India CPPP/eProcurement public tender listings.
    cppp_count=0
    for page_url in direct_cfg.get("cppp_pages",[]):
        try:
            tender_rows=fetch_cppp_rows(page_url)
        except Exception as e:
            errors.append({"source":"cppp_direct","url":page_url,"error":repr(e)})
            continue
        for txt in tender_rows:
            hits=themes_for(txt,qcfg["themes"])
            if not hits: continue
            item={"title":txt[:500],"url":page_url}
            ts=pd.Timestamp.now(tz="UTC")
            dom="eprocure.gov.in"
            for theme in hits:
                add_row(rows,theme,ts,"cppp_direct",item,dom,"official",1.0,txt,scfg,resolved=True)
                cppp_count+=1
    source_items["cppp_direct"]=cppp_count
    source_health["cppp_direct"]=cppp_count>0

    # 5) Dedicated official-domain discovery via Google News. Results count as
    # official only when the publisher/source domain is an approved official domain.
    google_official_count=0
    if direct_cfg.get("google_news_official_domain_discovery",False):
        for theme,words in qcfg["themes"].items():
            q=google_official_query(words,days,scfg.get("official_domains",[]))
            try:
                arts=fetch_google_news(google_url,q,min(mx,40))
            except Exception as e:
                errors.append({"source":"google_official","theme":theme,"error":repr(e)})
                continue
            for item in arts:
                ts=parse_ts(item.get("published"))
                if pd.isna(ts): continue
                d=article_domain(item)
                resolved_url=resolve_url(str(item.get("url") or ""))
                rd=urlparse(resolved_url).netloc.lower().replace("www.","")
                if rd and "news.google." not in rd and rd!="news.google.com":
                    d=rd
                    item=dict(item); item["url"]=resolved_url
                if not domain_matches(d,scfg.get("official_domains",[])):
                    continue
                text=str(item.get("title") or "")
                b=get_body(str(item.get("url") or ""))
                if b:text+=" "+b
                hits=themes_for(text,qcfg["themes"])
                if theme not in hits:
                    continue
                add_row(rows,theme,ts,"google_official",item,d,"official",1.0,text,scfg,resolved=True)
                google_official_count+=1
            time.sleep(.25)
    source_items["google_official"]=google_official_count

    df=pd.DataFrame(rows)
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
    cols=[
        "theme","published_ts","discovery_source","source_class","domain","title","url",
        "resolved_underlying","event_type","trust","event_weight","freshness","evidence_weight"
    ]
    if not len(df):
        df=pd.DataFrame(columns=cols)
    else:
        # Collapse duplicate stories across aggregators. A duplicate does not become two catalysts.
        df["title_norm"]=df["title"].map(norm_title)
        df["day"]=pd.to_datetime(df["published_ts"],utc=True,errors="coerce").dt.date.astype(str)
        df=df.sort_values("evidence_weight",ascending=False)
        df=df.drop_duplicates(["theme","domain","title_norm","day"])
        df=df.drop(columns=["title_norm","day"]).sort_values(["theme","published_ts"],ascending=[True,False])
    df.to_parquet(out,index=False)

    scores=[]
    for theme in qcfg["themes"]:
        g=df[df["theme"].eq(theme)].copy()
        if len(g):
            off=g[g["source_class"].eq("official")]
            sec=g[g["source_class"].eq("secondary")]
            oth=g[g["source_class"].eq("other_discovery")]
            gd=g[g["discovery_source"].eq("gdelt")]
            gn=g[g["discovery_source"].eq("google_news")]
            offw=float(off["evidence_weight"].sum())
            secw=float(sec["evidence_weight"].sum())
            othw=float(oth["evidence_weight"].sum())
            # Official evidence dominates. Secondary and other discovery are capped.
            combined=offw + min(secw,1.25) + min(othw,0.40)
            cross_bonus=0.15 if len(gd) and len(gn) else 0.0
            score=1.0-math.exp(-(combined+cross_bonus)/2.5)
            latest=str(g["published_ts"].max())
            distinct_domains=int(g["domain"].replace("",pd.NA).dropna().nunique())
        else:
            offw=secw=othw=score=0.0; latest=None; distinct_domains=0
            gd=gn=pd.DataFrame()
        scores.append({
            "theme":theme,
            "official_rows":int((g["source_class"]=="official").sum()) if len(g) else 0,
            "secondary_rows":int((g["source_class"]=="secondary").sum()) if len(g) else 0,
            "gdelt_rows":int((g["discovery_source"]=="gdelt").sum()) if len(g) else 0,
            "google_news_rows":int((g["discovery_source"]=="google_news").sum()) if len(g) else 0,
            "pib_direct_rows":int((g["discovery_source"]=="pib_direct").sum()) if len(g) else 0,
            "cppp_direct_rows":int((g["discovery_source"]=="cppp_direct").sum()) if len(g) else 0,
            "google_official_rows":int((g["discovery_source"]=="google_official").sum()) if len(g) else 0,
            "distinct_domains":distinct_domains,
            "official_weight":offw,
            "secondary_weight":secw,
            "other_weight":othw,
            "theme_demand_score":float(max(0,min(score,1))),
            "latest":latest,
        })

    sdf=pd.DataFrame(scores).sort_values("theme_demand_score",ascending=False)
    sdf.to_csv(out.parent/"theme_demand_scores.csv",index=False)

    reqs=scfg.get("required_discovery_sources",{"gdelt":True,"google_news":True})
    required_ok=all((not bool(reqs.get(k))) or bool(source_health.get(k)) for k in reqs)
    summary={
        "required_sources":reqs,
        "source_health":source_health,
        "source_items":source_items,
        "required_sources_ok":required_ok,
        "rows":int(len(df)),
        "themes_with_official_evidence":int((sdf["official_rows"]>0).sum()),
        "themes_with_direct_official_evidence":int(((sdf.get("pib_direct_rows",0)>0)|(sdf.get("cppp_direct_rows",0)>0)|(sdf.get("google_official_rows",0)>0)).sum()),
        "themes_with_any_evidence":int((sdf["distinct_domains"]>0).sum()),
        "themes_with_cross_source_corroboration":int(((sdf["gdelt_rows"]>0)&(sdf["google_news_rows"]>0)).sum()),
        "top_themes":sdf.head(12).to_dict("records"),
        "errors":errors,
        "note":"GDELT and Google News are mandatory discovery layers. Aggregator evidence alone cannot create stock eligibility."
    }
    json.dump(summary,open(out.parent/"external_demand_summary.json","w"),indent=2,default=str)
    print(json.dumps(summary,indent=2,default=str))

    if not required_ok:
        raise SystemExit("Required discovery source health failed")
    if summary["themes_with_any_evidence"]<3:
        raise SystemExit("External demand coverage too sparse")
    min_official=int(direct_cfg.get("minimum_themes_with_official_evidence",3))
    if summary["themes_with_official_evidence"]<min_official:
        raise SystemExit(f"Direct/official demand coverage too sparse: {summary['themes_with_official_evidence']} < {min_official}")

if __name__=="__main__":
    main()
