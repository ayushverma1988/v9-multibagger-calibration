from __future__ import annotations

import argparse
import io
import json
import re
import time
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
import requests

API="https://www.nseindia.com/api/integrated-filing-results"
HOME="https://www.nseindia.com/"
REFERER="https://www.nseindia.com/companies-listing/corporate-integrated-filing"
TYPE="Integrated Filing- Financials"

HEADERS={
    "User-Agent":"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept":"application/json,text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language":"en-US,en;q=0.9",
    "Referer":REFERER,
}

TAG_GROUPS={
    "revenue":[
        "RevenueFromOperations",
        "RevenueFromOperationsNet",
        "RevenueFromOperationsNetOfExciseDuty",
        "SalesRevenue",
        "Revenue",
    ],
    "total_income":[
        "TotalIncome",
        "RevenueAndOtherIncome",
    ],
    "pbt":[
        "ProfitBeforeTax",
        "ProfitLossBeforeTax",
        "ProfitBeforeExceptionalItemsAndTax",
    ],
    "pat":[
        "ProfitLossForPeriod",
        "ProfitLossForPeriodFromContinuingOperations",
        "ProfitLoss",
        "ProfitAfterTax",
    ],
    "finance_cost":[
        "FinanceCosts",
        "FinanceCost",
    ],
    "depreciation":[
        "DepreciationAndAmortisationExpense",
        "DepreciationDepletionAndAmortisation",
        "DepreciationAmortisationAndImpairmentExpense",
    ],
}

def local(tag):
    return tag.rsplit("}",1)[-1]

def clean(v):
    return re.sub(r"\s+"," ",str(v or "").replace("\xa0"," ")).strip()

def parse_num(v):
    s=clean(v)
    if not s or s.lower() in {"na","nan","nil","-","--"}:
        return None
    s=s.replace(",","").replace("(","-").replace(")","")
    m=re.search(r"[-+]?\d+(?:\.\d+)?",s)
    if not m: return None
    try:return float(m.group(0))
    except:return None

def parse_date(v):
    s=clean(v).title()
    for fmt in ("%d-%b-%Y","%d-%B-%Y","%d-%m-%Y","%Y-%m-%d"):
        try:return datetime.strptime(s,fmt).date()
        except:pass
    return None

def parse_dt(v):
    s=clean(v).title()
    for fmt in ("%d-%b-%Y %H:%M:%S","%d-%B-%Y %H:%M:%S"):
        try:return datetime.strptime(s,fmt)
        except:pass
    return datetime.min

class Client:
    def __init__(self):
        self.s=requests.Session(); self.s.headers.update(HEADERS)
        try:self.s.get(HOME,timeout=20)
        except:pass
    def get(self,url,**kwargs):
        last=None
        for i in range(4):
            try:
                r=self.s.get(url,timeout=45,**kwargs)
                if r.status_code in (401,403):
                    try:self.s.get(HOME,timeout=20)
                    except:pass
                    r=self.s.get(url,timeout=45,**kwargs)
                r.raise_for_status()
                time.sleep(.15)
                return r
            except Exception as e:
                last=e; time.sleep(min(8,1.5*(i+1)))
        raise RuntimeError(f"GET failed {url}: {last!r}")

def fetch_rows(client,symbol,max_pages=3,size=20):
    rows=[]
    total=None
    for page in range(1,max_pages+1):
        r=client.get(API,params={"symbol":symbol,"type":TYPE,"page":page,"size":size})
        j=r.json()
        batch=j.get("data",[]) if isinstance(j,dict) else []
        if not isinstance(batch,list): break
        rows.extend(x for x in batch if isinstance(x,dict))
        try: total=int(j.get("totalCount"))
        except: total=None
        if total is not None and len(rows)>=total:break
        if len(batch)<size:break
    return rows

def context_map(root):
    out={}
    for e in root.iter():
        if local(e.tag)!="context":continue
        cid=e.attrib.get("id")
        if not cid:continue
        start=end=instant=None
        for ch in e.iter():
            n=local(ch.tag)
            if n=="startDate":start=parse_date(ch.text)
            elif n=="endDate":end=parse_date(ch.text)
            elif n=="instant":instant=parse_date(ch.text)
        dur=(end-start).days+1 if start and end else None
        out[cid]={"start":start,"end":end or instant,"duration":dur}
    return out

def fact_score(ref,contexts,qe):
    if not ref:return 0
    c=contexts.get(ref,{})
    score=0
    if c.get("end")==qe:score+=1000
    d=c.get("duration")
    if d is not None:
        if 70<=d<=110:score+=500
        elif 55<=d<=130:score+=200
        else:score-=min(abs(d-91),300)
    lr=ref.lower()
    if "segment" in lr:score-=250
    return score

def choose_fact(root,contexts,tags,qe):
    candidates=[]
    wanted=set(tags)
    for e in root.iter():
        tag=local(e.tag)
        if tag not in wanted:continue
        val=parse_num(e.text)
        if val is None:continue
        ref=e.attrib.get("contextRef")
        candidates.append((fact_score(ref,contexts,qe),val,tag,ref))
    if not candidates:return None,None,None
    candidates.sort(key=lambda z:z[0],reverse=True)
    return candidates[0][1],candidates[0][2],candidates[0][3]

def extract_xbrl(client,url,qe):
    if not url:return None
    try:
        b=client.get(url).content
        root=ET.fromstring(b)
        contexts=context_map(root)
        facts={}
        audit={}
        for k,tags in TAG_GROUPS.items():
            v,tag,ref=choose_fact(root,contexts,tags,qe)
            facts[k]=v; audit[k]={"tag":tag,"context":ref}
        facts["ebitda_proxy"]=(
            facts["pbt"] + (facts["finance_cost"] or 0) + (facts["depreciation"] or 0)
            if facts["pbt"] is not None else None
        )
        facts["_audit"]=audit
        return facts
    except Exception as e:
        return {"_error":repr(e)}

def choose_filings(rows):
    parsed=[]
    for r in rows:
        qe=parse_date(r.get("qe_Date"))
        if not qe:continue
        typ=clean(r.get("consolidated")).lower()
        parsed.append({
            "raw":r,"qe":qe,"consolidated":typ,
            "ts":parse_dt(r.get("creation_Date") or r.get("broadcast_Date")),
        })
    # Prefer consolidated if the company has enough consolidated history;
    # otherwise use standalone. Never mix within the same quarter.
    cons=[x for x in parsed if "consolidated" in x["consolidated"] and "standalone" not in x["consolidated"]]
    mode="consolidated" if len({x["qe"] for x in cons})>=3 else "standalone"
    pool=cons if mode=="consolidated" else [x for x in parsed if "standalone" in x["consolidated"]]
    by={}
    for x in pool:
        if x["qe"] not in by or x["ts"]>by[x["qe"]]["ts"]:by[x["qe"]]=x
    return mode,sorted(by.values(),key=lambda z:z["qe"],reverse=True)

def pct(a,b):
    if a is None or b is None or abs(b)<1e-12:return None
    return a/b-1.0

def margin(v,rev):
    if v is None or rev is None or abs(rev)<1e-12:return None
    return v/rev

def compute_metrics(q):
    # q newest first
    out={}
    if not q:return out
    latest=q[0]
    out["latest_qe"]=str(latest["qe"])
    for k in ["revenue","total_income","pbt","pat","finance_cost","depreciation","ebitda_proxy"]:
        out[f"latest_{k}"]=latest["facts"].get(k)
    out["latest_ebitda_margin"]=margin(latest["facts"].get("ebitda_proxy"),latest["facts"].get("revenue"))
    out["latest_pat_margin"]=margin(latest["facts"].get("pat"),latest["facts"].get("revenue"))
    if len(q)>=2:
        prev=q[1]
        out["revenue_qoq"]=pct(latest["facts"].get("revenue"),prev["facts"].get("revenue"))
        out["pat_qoq"]=pct(latest["facts"].get("pat"),prev["facts"].get("pat"))
        out["ebitda_qoq"]=pct(latest["facts"].get("ebitda_proxy"),prev["facts"].get("ebitda_proxy"))
        pm=margin(prev["facts"].get("ebitda_proxy"),prev["facts"].get("revenue"))
        out["ebitda_margin_qoq_change"]=(out["latest_ebitda_margin"]-pm) if out["latest_ebitda_margin"] is not None and pm is not None else None
    if len(q)>=5:
        py=q[4]
        out["revenue_yoy"]=pct(latest["facts"].get("revenue"),py["facts"].get("revenue"))
        out["pat_yoy"]=pct(latest["facts"].get("pat"),py["facts"].get("pat"))
        out["ebitda_yoy"]=pct(latest["facts"].get("ebitda_proxy"),py["facts"].get("ebitda_proxy"))
        pym=margin(py["facts"].get("ebitda_proxy"),py["facts"].get("revenue"))
        out["ebitda_margin_yoy_change"]=(out["latest_ebitda_margin"]-pym) if out["latest_ebitda_margin"] is not None and pym is not None else None
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--symbols-file",required=True)
    ap.add_argument("--symbol-column",default="symbol")
    ap.add_argument("--output",required=True)
    ap.add_argument("--max-symbols",type=int,default=150)
    args=ap.parse_args()

    src=pd.read_csv(args.symbols_file)
    symbols=list(dict.fromkeys(src[args.symbol_column].dropna().astype(str).str.upper().str.strip()))[:args.max_symbols]
    client=Client()
    rows=[]; details=[]; errors=[]
    for i,sym in enumerate(symbols,1):
        try:
            filings=fetch_rows(client,sym)
            mode,chosen=choose_filings(filings)
            qs=[]
            for x in chosen[:8]:
                raw=x["raw"]
                facts=extract_xbrl(client,clean(raw.get("xbrl")),x["qe"])
                if not facts or facts.get("_error"):
                    continue
                qs.append({"qe":x["qe"],"facts":facts,"broadcast":raw.get("broadcast_Date"),"xbrl":raw.get("xbrl")})
            m=compute_metrics(qs)
            m.update({"symbol":sym,"filing_mode":mode,"quarters_extracted":len(qs),"filings_seen":len(filings)})
            rows.append(m)
            details.append({"symbol":sym,"mode":mode,"quarters":[{"qe":str(q["qe"]),"facts":q["facts"],"broadcast":q["broadcast"],"xbrl":q["xbrl"]} for q in qs]})
        except Exception as e:
            errors.append({"symbol":sym,"error":repr(e)})
        if i%20==0:print("financials",i,"/",len(symbols),"ok",len(rows),"errors",len(errors),flush=True)

    df=pd.DataFrame(rows)
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(out,index=False)
    json.dump(details,open(out.parent/"financial_detail.json","w"),indent=2,default=str)
    pd.DataFrame(errors).to_csv(out.parent/"financial_errors.csv",index=False)
    summary={
        "symbols_requested":len(symbols),
        "symbols_extracted":len(df),
        "symbols_with_5q":int((pd.to_numeric(df.get("quarters_extracted"),errors="coerce")>=5).sum()) if len(df) else 0,
        "errors":len(errors),
        "coverage":float(len(df)/len(symbols)) if symbols else 0,
        "columns":list(df.columns),
    }
    json.dump(summary,open(out.parent/"financial_summary.json","w"),indent=2,default=str)
    print(json.dumps(summary,indent=2,default=str))

if __name__=="__main__":
    main()
