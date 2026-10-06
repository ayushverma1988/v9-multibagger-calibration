from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests

import v11_4_catalyst_graph as cg

BASE_API="https://api.bseindia.com/BseIndiaAPI/api/AnnSubCategoryGetData/w"
PAGE="https://www.bseindia.com/corporates/ann.html"
ATTACH="https://www.bseindia.com/xml-data/corpfiling/AttachLive/"
HEADERS={
    "Host":"api.bseindia.com",
    "Referer":PAGE,
    "User-Agent":"Mozilla/5.0 (Windows NT 11.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.6998.166 Safari/537.36",
    "Sec-CH-UA":'"Google Chrome";v="134", "Chromium";v="134", "Not?A_Brand";v="99"',
    "Sec-CH-UA-Mobile":"?0",
    "Sec-CH-UA-Platform":'"Windows"',
    "DNT":"1",
    "Accept":"application/json, text/plain, */*",
    "Accept-Encoding":"gzip, deflate, br",
    "Accept-Language":"en-US,en;q=0.9",
    "Cache-Control":"no-cache",
    "Connection":"keep-alive",
}

def first(d,*names):
    for n in names:
        v=d.get(n)
        if v is not None and str(v).strip() not in {"","nan","None","NULL"}:
            return v
    return None

def fetch_page(start,end,page,retries=4):
    params={
        "pageno":int(page),
        "strType":"C",
        "strSearch":"P",
        "strPrevDate":start.strftime("%Y%m%d"),
        "strToDate":end.strftime("%Y%m%d"),
    }
    last=None
    for i in range(retries):
        try:
            s=requests.Session()
            s.headers.update(HEADERS)
            # Do not pre-visit the main BSE page; it currently returns 403
            # from cloud runners while the JSON endpoint can still work directly.
            r=s.get(BASE_API,params=params,timeout=60)
            r.raise_for_status()
            j=r.json()
            table=j.get("Table",[]) if isinstance(j,dict) else []
            total=None
            if isinstance(j,dict) and j.get("Table1"):
                total=int(first(j["Table1"][0],"ROWCNT","RowCnt","rowcnt") or 0)
            return table,total
        except Exception as e:
            last=repr(e)
            time.sleep(min(15,2**i))
    raise RuntimeError(f"BSE announcements failed: {last}")

def norm_name(s):
    return cg.norm_name(s)

def nse_master(url):
    r=requests.get(url,headers={"User-Agent":"Mozilla/5.0"},timeout=60)
    r.raise_for_status()
    from io import StringIO
    df=pd.read_csv(StringIO(r.text))
    cols={c.strip().upper():c for c in df.columns}
    out=pd.DataFrame({
        "symbol":df[cols["SYMBOL"]].astype(str).str.upper().str.strip(),
        "company_name":df[cols["NAME OF COMPANY"]].astype(str).str.strip(),
        "isin":df[cols["ISIN NUMBER"]].astype(str).str.upper().str.strip() if "ISIN NUMBER" in cols else None,
    })
    out["key"]=out["company_name"].map(norm_name)
    return out

def match_name(name,master):
    k=norm_name(name)
    if not k:
        return None
    exact=master[master["key"].eq(k)]
    if len(exact):
        r=exact.iloc[0]
        return r["symbol"],r["isin"]
    # conservative token overlap; do not fuzzy-match very short names
    kt=set(k.split())
    if len(kt)<2:
        return None
    best=None
    for r in master.itertuples(index=False):
        rt=set(r.key.split())
        if not rt: continue
        jac=len(kt&rt)/len(kt|rt)
        if jac>=0.75 and (best is None or jac>best[0]):
            best=(jac,r.symbol,r.isin)
    return (best[1],best[2]) if best else None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--output",required=True)
    ap.add_argument("--queries-config",required=True)
    ap.add_argument("--lookback-days",type=int,default=14)
    ap.add_argument("--max-pages",type=int,default=25)
    ap.add_argument("--master-url",default="https://archives.nseindia.com/content/equities/EQUITY_L.csv")
    args=ap.parse_args()

    cfg=json.load(open(args.queries_config))
    end=pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None).normalize()
    start=end-pd.Timedelta(days=int(args.lookback_days))
    master=nse_master(args.master_url)

    raw=[]
    total=None
    for p in range(1,int(args.max_pages)+1):
        rows,total=fetch_page(start,end,p)
        raw.extend(rows)
        if not rows: break
        if total and len(raw)>=total: break

    outrows=[]
    for r in raw:
        name=str(first(r,"SLONGNAME","LONG_NAME","COMPANYNAME","SCRIP_NAME") or "").strip()
        matched=match_name(name,master)
        if not matched: continue
        symbol,isin=matched
        title=str(first(r,"NEWSSUB","HEADLINE","NEWS_SUBJECT","SUBJECT") or "").strip()
        details=str(first(r,"MORE","HEADLINE","NEWSSUB") or "").strip()
        text=(title+" "+details).strip()
        cts=cg.catalyst_types(text)
        st,sw=cg.stage(text)
        # Keep all mapped records but high-confidence board will filter.
        th=cg.themes(text,cfg)
        neg=bool(cg.NEGATIVE_PATTERNS.search(text))
        dt=pd.to_datetime(first(r,"NEWS_DT","NEWS_DATE","DT_TM"),dayfirst=True,errors="coerce")
        if pd.notna(dt) and dt.tzinfo is None:
            dt=dt.tz_localize("Asia/Kolkata").tz_convert("UTC")
        att=str(first(r,"ATTACHMENTNAME","ATTACHMENT","NSURL") or "").strip()
        url=att if att.startswith("http") else (urljoin(ATTACH,att) if att else "")
        rid=str(first(r,"NEWSID","SCRIP_CD","ID") or "")
        eid=hashlib.sha256("|".join(["BSE",rid,symbol,str(dt),title]).encode()).hexdigest()
        outrows.append({
            "evidence_id":eid,
            "published_ts":dt,
            "symbol":symbol,
            "company_name":name,
            "isin":isin,
            "domain":"bseindia.com",
            "url":url,
            "title":title,
            "query_family":"bse_primary",
            "source_tier":1,
            "source_trust":1.0,
            "catalyst_types":json.dumps(cts),
            "themes":json.dumps(th),
            "stage":st,
            "stage_weight":sw,
            "money_crore_max":cg.extract_money_crore(text),
            "capacity_pct_max":cg.extract_capacity_pct(text),
            "negative_flag":neg,
            "corroboration_count":1,
            "evidence_confidence":cg.event_confidence(1.0,sw,cts,neg,1),
        })

    df=pd.DataFrame(outrows)
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
    df.to_parquet(out,index=False)
    meaningful=df[(df["stage_weight"]>=0.40)|df["catalyst_types"].ne("[]")] if len(df) else df
    summary={
        "date_start":str(start.date()),"date_end":str(end.date()),
        "raw_announcements":int(len(raw)),
        "mapped_rows":int(len(df)),
        "meaningful_rows":int(len(meaningful)),
        "companies":int(meaningful["symbol"].nunique()) if len(meaningful) else 0,
        "reported_total":total,
    }
    json.dump(summary,open(out.parent/"bse_primary_summary.json","w"),indent=2,default=str)
    print(json.dumps(summary,indent=2,default=str))

if __name__=="__main__":
    main()
