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
LEGACY_API="https://www.nseindia.com/api/corporates-financial-results"
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

def fetch_integrated_rows(client,symbol,max_pages=3,size=20):
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

def fetch_legacy_rows(client,symbol):
    # Historical quarterly filings before/around the Integrated Filing migration.
    # NSE returns the symbol's filing catalog with point-in-time broadcast dates
    # and direct XBRL links.
    r=client.get(LEGACY_API,params={
        "index":"equities","symbol":symbol,"period":"Quarterly"
    })
    j=r.json()
    return [x for x in j if isinstance(x,dict)] if isinstance(j,list) else []

def _mode_label(v):
    s=clean(v).lower().replace("_"," ")
    if "non-consolidated" in s or "non consolidated" in s or "standalone" in s:
        return "standalone"
    if "consolidated" in s:
        return "consolidated"
    return "unknown"

def normalize_filings(integrated_rows,legacy_rows):
    out=[]
    for r in integrated_rows:
        qe=parse_date(r.get("qe_Date"))
        if not qe: continue
        out.append({
            "raw":r,
            "qe":qe,
            "mode":_mode_label(r.get("consolidated")),
            "ts":parse_dt(r.get("broadcast_Date") or r.get("creation_Date")),
            "xbrl":clean(r.get("xbrl")),
            "broadcast":r.get("broadcast_Date") or r.get("creation_Date"),
            "source":"integrated",
        })
    for r in legacy_rows:
        qe=parse_date(r.get("toDate"))
        if not qe: continue
        out.append({
            "raw":r,
            "qe":qe,
            "mode":_mode_label(r.get("consolidated")),
            "ts":parse_dt(r.get("broadCastDate") or r.get("filingDate")),
            "xbrl":clean(r.get("xbrl")),
            "broadcast":r.get("broadCastDate") or r.get("filingDate"),
            "source":"legacy",
        })
    return out

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

def choose_filings(integrated_rows,legacy_rows):
    parsed=normalize_filings(integrated_rows,legacy_rows)

    # Use one accounting basis across the time series. Prefer the mode with the
    # broadest quarterly coverage; break close ties in favour of consolidated.
    cons=[x for x in parsed if x["mode"]=="consolidated"]
    stand=[x for x in parsed if x["mode"]=="standalone"]
    nc=len({x["qe"] for x in cons})
    ns=len({x["qe"] for x in stand})
    mode="consolidated" if nc>=max(4,ns-1) else "standalone"
    pool=cons if mode=="consolidated" else stand
    if not pool:
        pool=parsed
        mode="mixed_unknown"

    # Same quarter can exist in both old and new filing systems. Prefer the
    # Integrated Filing version when available, then the latest broadcast.
    by={}
    for x in pool:
        cur=by.get(x["qe"])
        if cur is None:
            by[x["qe"]]=x
            continue
        cur_pref=(1 if cur["source"]=="integrated" else 0,cur["ts"])
        new_pref=(1 if x["source"]=="integrated" else 0,x["ts"])
        if new_pref>cur_pref:
            by[x["qe"]]=x
    return mode,sorted(by.values(),key=lambda z:z["qe"],reverse=True)

def pct(a,b):
    if a is None or b is None or abs(b)<1e-12:return None
    return a/b-1.0

def profit_growth(a,b):
    """Growth only when the prior base is positive; sign changes are separate flags."""
    if a is None or b is None or b<=0:
        return None
    return a/b-1.0

def turnaround(a,b):
    if a is None or b is None:
        return None
    if b<=0 and a>0:
        return 1.0
    if b>0 and a<=0:
        return -1.0
    return 0.0

def margin(v,rev):
    if v is None or rev is None or abs(rev)<1e-12:return None
    return v/rev

def _sum_metric(q,start,end,key):
    vals=[]
    for z in q[start:end]:
        v=z["facts"].get(key)
        if v is None:
            return None
        vals.append(v)
    return sum(vals) if vals else None

def _block_growth(q,a0,a1,b0,b1,key,profit=False):
    a=_sum_metric(q,a0,a1,key)
    b=_sum_metric(q,b0,b1,key)
    return profit_growth(a,b) if profit else pct(a,b)

def _block_margin(q,start,end,num_key):
    n=_sum_metric(q,start,end,num_key)
    r=_sum_metric(q,start,end,"revenue")
    return margin(n,r)

def compute_metrics(q):
    # q newest first. V11.4 uses up to 12 quarters (~3 years) as context.
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
        out["pat_qoq"]=profit_growth(latest["facts"].get("pat"),prev["facts"].get("pat"))
        out["pat_turnaround_qoq"]=turnaround(latest["facts"].get("pat"),prev["facts"].get("pat"))
        out["ebitda_qoq"]=profit_growth(latest["facts"].get("ebitda_proxy"),prev["facts"].get("ebitda_proxy"))
        out["ebitda_turnaround_qoq"]=turnaround(latest["facts"].get("ebitda_proxy"),prev["facts"].get("ebitda_proxy"))
        pm=margin(prev["facts"].get("ebitda_proxy"),prev["facts"].get("revenue"))
        out["ebitda_margin_qoq_change"]=(out["latest_ebitda_margin"]-pm) if out["latest_ebitda_margin"] is not None and pm is not None else None

    if len(q)>=3:
        q1=q[1]; q2=q[2]
        out["revenue_vs_2q_back"]=pct(latest["facts"].get("revenue"),q2["facts"].get("revenue"))
        p0=latest["facts"].get("pat"); p1=q1["facts"].get("pat"); p2=q2["facts"].get("pat")
        out["pat_latest"]=p0
        out["pat_preceding"]=p1
        out["pat_2q_back"]=p2
        out["pat_latest_gt_preceding"]=bool(p0 is not None and p1 is not None and p0>p1)
        out["pat_preceding_gt_2q_back"]=bool(p1 is not None and p2 is not None and p1>p2)

    if len(q)>=5:
        py=q[4]
        out["revenue_yoy"]=pct(latest["facts"].get("revenue"),py["facts"].get("revenue"))
        out["pat_yoy"]=profit_growth(latest["facts"].get("pat"),py["facts"].get("pat"))
        out["pat_turnaround_yoy"]=turnaround(latest["facts"].get("pat"),py["facts"].get("pat"))
        out["ebitda_yoy"]=profit_growth(latest["facts"].get("ebitda_proxy"),py["facts"].get("ebitda_proxy"))
        out["ebitda_turnaround_yoy"]=turnaround(latest["facts"].get("ebitda_proxy"),py["facts"].get("ebitda_proxy"))
        pym=margin(py["facts"].get("ebitda_proxy"),py["facts"].get("revenue"))
        out["ebitda_margin_yoy_change"]=(out["latest_ebitda_margin"]-pym) if out["latest_ebitda_margin"] is not None and pym is not None else None

    # Latest two quarters vs same two quarters last year: highest short-horizon relevance.
    if len(q)>=6:
        out["recent2_revenue_yoy"]=_block_growth(q,0,2,4,6,"revenue")
        out["recent2_ebitda_yoy"]=_block_growth(q,0,2,4,6,"ebitda_proxy",profit=True)
        out["recent2_pat_yoy"]=_block_growth(q,0,2,4,6,"pat",profit=True)
        a=_sum_metric(q,0,2,"pat"); b=_sum_metric(q,4,6,"pat")
        out["recent2_pat_turnaround"]=turnaround(a,b)
        a=_sum_metric(q,0,2,"ebitda_proxy"); b=_sum_metric(q,4,6,"ebitda_proxy")
        out["recent2_ebitda_turnaround"]=turnaround(a,b)
        m0=_block_margin(q,0,2,"ebitda_proxy"); m1=_block_margin(q,4,6,"ebitda_proxy")
        out["recent2_ebitda_margin_yoy_change"]=(m0-m1) if m0 is not None and m1 is not None else None

    # Previous two quarters vs their year-ago comparables: tells whether acceleration was already forming.
    if len(q)>=8:
        out["prev2_revenue_yoy"]=_block_growth(q,2,4,6,8,"revenue")
        out["prev2_ebitda_yoy"]=_block_growth(q,2,4,6,8,"ebitda_proxy",profit=True)
        out["prev2_pat_yoy"]=_block_growth(q,2,4,6,8,"pat",profit=True)
        m0=_block_margin(q,2,4,"ebitda_proxy"); m1=_block_margin(q,6,8,"ebitda_proxy")
        out["prev2_ebitda_margin_yoy_change"]=(m0-m1) if m0 is not None and m1 is not None else None

    # Three annual blocks from 12 quarters. This is context, not a long-history momentum signal.
    if len(q)>=8:
        out["latest_ttm_revenue"]=_sum_metric(q,0,4,"revenue")
        out["previous_ttm_revenue"]=_sum_metric(q,4,8,"revenue")
        out["latest_ttm_ebitda"]=_sum_metric(q,0,4,"ebitda_proxy")
        out["previous_ttm_ebitda"]=_sum_metric(q,4,8,"ebitda_proxy")
        out["latest_ttm_pat"]=_sum_metric(q,0,4,"pat")
        out["previous_ttm_pat"]=_sum_metric(q,4,8,"pat")
        out["ttm_revenue_growth"]=pct(out["latest_ttm_revenue"],out["previous_ttm_revenue"])
        out["ttm_ebitda_growth"]=profit_growth(out["latest_ttm_ebitda"],out["previous_ttm_ebitda"])
        out["ttm_pat_growth"]=profit_growth(out["latest_ttm_pat"],out["previous_ttm_pat"])
        out["ttm_pat_turnaround"]=turnaround(out["latest_ttm_pat"],out["previous_ttm_pat"])
        out["ttm_ebitda_margin"] = margin(out["latest_ttm_ebitda"],out["latest_ttm_revenue"])
        prev_m=margin(out["previous_ttm_ebitda"],out["previous_ttm_revenue"])
        out["ttm_ebitda_margin_change"]=(out["ttm_ebitda_margin"]-prev_m) if out["ttm_ebitda_margin"] is not None and prev_m is not None else None

    if len(q)>=12:
        out["third_ttm_revenue"]=_sum_metric(q,8,12,"revenue")
        out["third_ttm_ebitda"]=_sum_metric(q,8,12,"ebitda_proxy")
        out["third_ttm_pat"]=_sum_metric(q,8,12,"pat")
        out["previous_ttm_revenue_growth"]=pct(out.get("previous_ttm_revenue"),out["third_ttm_revenue"])
        out["previous_ttm_ebitda_growth"]=profit_growth(out.get("previous_ttm_ebitda"),out["third_ttm_ebitda"])
        out["previous_ttm_pat_growth"]=profit_growth(out.get("previous_ttm_pat"),out["third_ttm_pat"])
        out["revenue_growth_acceleration_3y"]=(
            out["ttm_revenue_growth"]-out["previous_ttm_revenue_growth"]
            if out.get("ttm_revenue_growth") is not None and out.get("previous_ttm_revenue_growth") is not None else None
        )
        prev_m=margin(out.get("previous_ttm_ebitda"),out.get("previous_ttm_revenue"))
        third_m=margin(out["third_ttm_ebitda"],out["third_ttm_revenue"])
        out["previous_ttm_ebitda_margin_change"]=(prev_m-third_m) if prev_m is not None and third_m is not None else None

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
            integrated=fetch_integrated_rows(client,sym)
            legacy=fetch_legacy_rows(client,sym)
            mode,chosen=choose_filings(integrated,legacy)
            qs=[]
            for x in chosen[:12]:
                facts=extract_xbrl(client,x["xbrl"],x["qe"])
                if not facts or facts.get("_error"):
                    continue
                qs.append({
                    "qe":x["qe"],"facts":facts,"broadcast":x["broadcast"],
                    "xbrl":x["xbrl"],"source":x["source"]
                })
            m=compute_metrics(qs)
            m.update({
                "symbol":sym,
                "filing_mode":mode,
                "quarters_extracted":len(qs),
                "integrated_quarters":sum(1 for q in qs if q["source"]=="integrated"),
                "legacy_quarters":sum(1 for q in qs if q["source"]=="legacy"),
                "integrated_filings_seen":len(integrated),
                "legacy_filings_seen":len(legacy),
                "filings_seen":len(integrated)+len(legacy),
            })
            rows.append(m)
            details.append({
                "symbol":sym,"mode":mode,
                "quarters":[
                    {"qe":str(q["qe"]),"facts":q["facts"],"broadcast":q["broadcast"],"xbrl":q["xbrl"],"source":q["source"]}
                    for q in qs
                ]
            })
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
        "symbols_with_8q":int((pd.to_numeric(df.get("quarters_extracted"),errors="coerce")>=8).sum()) if len(df) else 0,
        "symbols_with_12q":int((pd.to_numeric(df.get("quarters_extracted"),errors="coerce")>=12).sum()) if len(df) else 0,
        "median_quarters":float(pd.to_numeric(df.get("quarters_extracted"),errors="coerce").median()) if len(df) else 0,
        "symbols_using_legacy_history":int((pd.to_numeric(df.get("legacy_quarters"),errors="coerce").fillna(0)>0).sum()) if len(df) else 0,
        "errors":len(errors),
        "coverage":float(len(df)/len(symbols)) if symbols else 0,
        "columns":list(df.columns),
    }
    json.dump(summary,open(out.parent/"financial_summary.json","w"),indent=2,default=str)
    print(json.dumps(summary,indent=2,default=str))

if __name__=="__main__":
    main()
