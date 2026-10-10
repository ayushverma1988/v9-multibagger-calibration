from __future__ import annotations

import argparse
import json
import math
import re
import time
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
import requests

API="https://www.nseindia.com/api/corporates-financial-results"
HOME="https://www.nseindia.com/"
HEADERS={
    "User-Agent":"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36",
    "Accept":"application/json,text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language":"en-US,en;q=0.9",
    "Referer":"https://www.nseindia.com/companies-listing/corporate-filings-financial-results",
}

TAG_GROUPS={
    "revenue":[
        "RevenueFromOperations","RevenueFromOperationsNet","RevenueFromOperationsNetOfExciseDuty",
        "SalesRevenue","Revenue"
    ],
    "pbt":["ProfitBeforeTax","ProfitLossBeforeTax","ProfitBeforeExceptionalItemsAndTax"],
    "pat":["ProfitLossForPeriod","ProfitLossForPeriodFromContinuingOperations","ProfitLoss","ProfitAfterTax"],
    "finance_cost":["FinanceCosts","FinanceCost"],
    "depreciation":["DepreciationAndAmortisationExpense","DepreciationDepletionAndAmortisation","DepreciationAmortisationAndImpairmentExpense"],
    "equity_total":["TotalEquity","Equity","NetWorth","ShareholdersFunds"],
    "equity_share_capital":["EquityShareCapital","PaidUpEquityShareCapital"],
    "other_equity":["OtherEquity","ReservesAndSurplus","OtherReserves"],
    "borrowings_total":["Borrowings","TotalBorrowings","Debt"],
    "borrowings_noncurrent":["NoncurrentBorrowings","LongTermBorrowings","BorrowingsNonCurrent"],
    "borrowings_current":["CurrentBorrowings","ShortTermBorrowings","BorrowingsCurrent"],
    "trade_receivables":["TradeReceivables","TradeReceivablesCurrent","TradeReceivablesNonCurrent"],
    "ppe":["PropertyPlantAndEquipment","PropertyPlantAndEquipmentAndIntangibleAssets","TangibleAssets","FixedAssets"],
    "cash":["CashAndCashEquivalents","CashCashEquivalentsAndBankBalances"],
    "cfo":["CashFlowsFromUsedInOperatingActivities","NetCashFlowsFromUsedInOperatingActivities","CashGeneratedFromOperations"],
}

def local(tag): return tag.rsplit("}",1)[-1]

def clean(v): return re.sub(r"\s+"," ",str(v or "").replace("\xa0"," ")).strip()

def parse_num(v):
    s=clean(v)
    if not s or s.lower() in {"na","nan","nil","-","--"}: return None
    s=s.replace(",","").replace("(","-").replace(")","")
    m=re.search(r"[-+]?\d+(?:\.\d+)?",s)
    if not m:return None
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
                time.sleep(.12)
                return r
            except Exception as e:
                last=e;time.sleep(min(8,1.5*(i+1)))
        raise RuntimeError(f"GET failed {url}: {last!r}")

def fetch_annual_catalog(client,symbol):
    r=client.get(API,params={"index":"equities","symbol":symbol,"period":"Annual"})
    j=r.json()
    return [x for x in j if isinstance(x,dict)] if isinstance(j,list) else []

def mode_label(v):
    s=clean(v).lower().replace("_"," ")
    if "non-consolidated" in s or "standalone" in s:return "standalone"
    if "consolidated" in s:return "consolidated"
    return "unknown"

def normalize_catalog(rows):
    out=[]
    for r in rows:
        fy_end=parse_date(r.get("toDate"))
        if not fy_end:continue
        out.append({
            "raw":r,"fy_end":fy_end,"mode":mode_label(r.get("consolidated")),
            "ts":parse_dt(r.get("broadCastDate") or r.get("filingDate")),
            "xbrl":clean(r.get("xbrl")),
            "broadcast":r.get("broadCastDate") or r.get("filingDate"),
        })
    return out

def choose_annual(rows,max_years=8):
    parsed=normalize_catalog(rows)
    cons=[x for x in parsed if x["mode"]=="consolidated"]
    stand=[x for x in parsed if x["mode"]=="standalone"]
    nc=len({x["fy_end"] for x in cons}); ns=len({x["fy_end"] for x in stand})
    mode="consolidated" if nc>=max(3,ns-1) else "standalone"
    pool=cons if mode=="consolidated" else stand
    if not pool: pool=parsed; mode="mixed_unknown"
    by={}
    for x in pool:
        cur=by.get(x["fy_end"])
        if cur is None or x["ts"]>cur["ts"]:by[x["fy_end"]]=x
    return mode,sorted(by.values(),key=lambda z:z["fy_end"],reverse=True)[:max_years]

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

def fact_score(ref,contexts,fy_end):
    if not ref:return 0
    c=contexts.get(ref,{})
    score=0
    if c.get("end")==fy_end:score+=1000
    d=c.get("duration")
    if d is None:
        score+=450
    elif 300<=d<=400:
        score+=550
    elif 250<=d<=430:
        score+=200
    else:
        score-=min(abs(d-365),350)
    lr=ref.lower()
    if "segment" in lr:score-=300
    return score

def choose_fact(root,contexts,tags,fy_end):
    wanted=set(tags);cand=[]
    for e in root.iter():
        tag=local(e.tag)
        if tag not in wanted:continue
        val=parse_num(e.text)
        if val is None:continue
        ref=e.attrib.get("contextRef")
        cand.append((fact_score(ref,contexts,fy_end),val,tag,ref))
    if not cand:return None,None,None
    cand.sort(key=lambda z:z[0],reverse=True)
    return cand[0][1],cand[0][2],cand[0][3]

def extract_annual(client,url,fy_end):
    if not url:return None
    try:
        root=ET.fromstring(client.get(url).content)
        ctx=context_map(root)
        facts={};audit={}
        for k,tags in TAG_GROUPS.items():
            v,tag,ref=choose_fact(root,ctx,tags,fy_end)
            facts[k]=v;audit[k]={"tag":tag,"context":ref}
        if facts["borrowings_total"] is None:
            xs=[facts.get("borrowings_noncurrent"),facts.get("borrowings_current")]
            if any(v is not None for v in xs):facts["borrowings_total"]=sum(v or 0 for v in xs)
        if facts["equity_total"] is None:
            xs=[facts.get("equity_share_capital"),facts.get("other_equity")]
            if any(v is not None for v in xs):facts["equity_total"]=sum(v or 0 for v in xs)
        if facts["pbt"] is not None:
            facts["ebit"]=facts["pbt"]+(facts["finance_cost"] or 0)
            facts["ebitda_proxy"]=facts["ebit"]+(facts["depreciation"] or 0)
        else:
            facts["ebit"]=None;facts["ebitda_proxy"]=None
        facts["_audit"]=audit
        return facts
    except Exception as e:
        return {"_error":repr(e)}

def ratio(a,b):
    if a is None or b is None or abs(b)<1e-12:return None
    return a/b

def cagr(new,old,years):
    if new is None or old is None or new<=0 or old<=0 or years<=0:return None
    return (new/old)**(1.0/years)-1.0

def avg(vals):
    z=[v for v in vals if v is not None and np.isfinite(v)]
    return float(np.mean(z)) if z else None

def metrics(years):
    out={}
    if not years:return out
    years=sorted(years,key=lambda z:z["fy_end"],reverse=True)
    latest=years[0]["facts"]
    out["annual_years_extracted"]=len(years)
    out["latest_fy_end"]=str(years[0]["fy_end"])
    out["opm_current"]=ratio(latest.get("ebitda_proxy"),latest.get("revenue"))
    out["debt_to_equity"]=ratio(latest.get("borrowings_total"),latest.get("equity_total"))
    out["interest_coverage"]=ratio(latest.get("ebit"),latest.get("finance_cost"))
    out["cfo_to_pat_last_year"]=ratio(latest.get("cfo"),latest.get("pat"))
    out["receivables_to_pat"]=ratio(latest.get("trade_receivables"),latest.get("pat"))
    out["receivables_to_sales"]=ratio(latest.get("trade_receivables"),latest.get("revenue"))
    ce=(latest.get("equity_total") or 0)+(latest.get("borrowings_total") or 0)-(latest.get("cash") or 0)
    out["capital_employed_latest"]=ce if ce else None
    out["sales_to_capital_employed"]=ratio(latest.get("revenue"),out["capital_employed_latest"])
    out["reserves_gt_borrowings"]=bool(
        latest.get("other_equity") is not None and latest.get("borrowings_total") is not None
        and latest["other_equity"]>latest["borrowings_total"]
    ) if latest.get("other_equity") is not None and latest.get("borrowings_total") is not None else None

    roes=[];roces=[];opms=[]
    for i,y in enumerate(years):
        f=y["facts"]
        opms.append(ratio(f.get("ebitda_proxy"),f.get("revenue")))
        if i+1<len(years):
            prev=years[i+1]["facts"]
            avgeq=avg([f.get("equity_total"),prev.get("equity_total")])
            curr_ce=(f.get("equity_total") or 0)+(f.get("borrowings_total") or 0)-(f.get("cash") or 0)
            prev_ce=(prev.get("equity_total") or 0)+(prev.get("borrowings_total") or 0)-(prev.get("cash") or 0)
            avgce=avg([curr_ce if curr_ce else None,prev_ce if prev_ce else None])
        else:
            avgeq=f.get("equity_total")
            curr_ce=(f.get("equity_total") or 0)+(f.get("borrowings_total") or 0)-(f.get("cash") or 0)
            avgce=curr_ce if curr_ce else None
        roes.append(ratio(f.get("pat"),avgeq))
        roces.append(ratio(f.get("ebit"),avgce))

    out["roe_current"]=roes[0] if roes else None
    out["roe_3y_avg"]=avg(roes[:3])
    out["roe_5y_avg"]=avg(roes[:5])
    out["roce_3y_avg"]=avg(roces[:3])
    out["roce_7y_avg"]=avg(roces[:7])
    out["opm_5y_avg"]=avg(opms[:5])

    # Use actual FY distance, not row count, to avoid gaps pretending to be years.
    for n in [3,5,7]:
        if len(years)>n:
            newest=years[0];old=years[n]
            dy=max((newest["fy_end"]-old["fy_end"]).days/365.25,0)
            if abs(dy-n)<=1.2:
                out[f"sales_growth_{n}y"]=cagr(newest["facts"].get("revenue"),old["facts"].get("revenue"),dy)
                out[f"profit_growth_{n}y"]=cagr(newest["facts"].get("pat"),old["facts"].get("pat"),dy)

    if len(years)>=2:
        p0=years[0]["facts"].get("ppe");p1=years[1]["facts"].get("ppe")
        out["fixed_assets_yoy_growth"]=ratio(p0,p1)-1 if ratio(p0,p1) is not None else None
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--symbols-file",required=True)
    ap.add_argument("--symbol-column",default="symbol")
    ap.add_argument("--output",required=True)
    ap.add_argument("--max-symbols",type=int,default=80)
    ap.add_argument("--max-years",type=int,default=8)
    args=ap.parse_args()

    src=pd.read_csv(args.symbols_file)
    syms=list(dict.fromkeys(src[args.symbol_column].dropna().astype(str).str.upper().str.strip()))[:args.max_symbols]
    client=Client();rows=[];details=[];errors=[]
    for i,sym in enumerate(syms,1):
        try:
            cat=fetch_annual_catalog(client,sym)
            mode,chosen=choose_annual(cat,args.max_years)
            ys=[]
            for x in chosen:
                f=extract_annual(client,x["xbrl"],x["fy_end"])
                if not f or f.get("_error"):continue
                ys.append({"fy_end":x["fy_end"],"facts":f,"broadcast":x["broadcast"],"xbrl":x["xbrl"]})
            m=metrics(ys)
            m.update({"symbol":sym,"annual_mode":mode,"annual_catalog_rows":len(cat)})
            rows.append(m)
            details.append({"symbol":sym,"mode":mode,"years":[{"fy_end":str(y["fy_end"]),"broadcast":y["broadcast"],"xbrl":y["xbrl"],"facts":y["facts"]} for y in ys]})
        except Exception as e:
            errors.append({"symbol":sym,"error":repr(e)})
        if i%10==0:print("annual",i,"/",len(syms),"ok",len(rows),"errors",len(errors),flush=True)

    df=pd.DataFrame(rows)
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(out,index=False)
    json.dump(details,open(out.parent/"annual_fundamental_detail.json","w"),indent=2,default=str)
    pd.DataFrame(errors).to_csv(out.parent/"annual_fundamental_errors.csv",index=False)
    coverage={}
    for c in [
        "sales_growth_3y","sales_growth_5y","profit_growth_5y","profit_growth_7y","opm_current","opm_5y_avg",
        "roce_3y_avg","roce_7y_avg","roe_3y_avg","roe_5y_avg","debt_to_equity","interest_coverage",
        "cfo_to_pat_last_year","receivables_to_pat","sales_to_capital_employed","fixed_assets_yoy_growth"
    ]:
        coverage[c]=float(pd.to_numeric(df.get(c),errors="coerce").notna().mean()) if len(df) and c in df else 0.0
    summary={
        "symbols_requested":len(syms),"symbols_extracted":len(df),"errors":len(errors),
        "symbols_with_5y":int((pd.to_numeric(df.get("annual_years_extracted"),errors="coerce")>=6).sum()) if len(df) else 0,
        "symbols_with_7y":int((pd.to_numeric(df.get("annual_years_extracted"),errors="coerce")>=8).sum()) if len(df) else 0,
        "field_coverage":coverage,
    }
    json.dump(summary,open(out.parent/"annual_fundamental_summary.json","w"),indent=2)
    print(json.dumps(summary,indent=2))

if __name__=="__main__":
    main()
