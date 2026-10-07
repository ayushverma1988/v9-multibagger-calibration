from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

PAGE="https://www.nseindia.com/companies-listing/corporate-filings-financial-results"
API="https://www.nseindia.com/api/results-comparision"
HEADERS={
    "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/134 Safari/537.36",
    "Accept":"application/json,text/plain,*/*",
    "Accept-Language":"en-US,en;q=0.9",
    "Referer":PAGE,
}


def num(v):
    try:
        if v is None or str(v).strip() in {"","-","None","nan","null"}:
            return np.nan
        return float(str(v).replace(",","").strip())
    except Exception:
        return np.nan


def session():
    s=requests.Session()
    s.headers.update(HEADERS)
    try:
        s.get(PAGE,timeout=20)
    except Exception:
        pass
    return s


def get_symbol(s,symbol,retries=4):
    last=None
    for i in range(retries):
        try:
            r=s.get(API,params={"symbol":symbol},timeout=45)
            if r.status_code in {401,403,429}:
                try:s.get(PAGE,timeout=20)
                except Exception:pass
                if r.status_code==429:
                    time.sleep(10*(i+1))
                raise RuntimeError(f"NSE HTTP {r.status_code}")
            r.raise_for_status()
            j=r.json()
            rows=j.get("resCmpData",[]) if isinstance(j,dict) else []
            if not isinstance(rows,list):
                rows=[]
            return rows,j.get("bankNonBnking") if isinstance(j,dict) else None,None
        except Exception as e:
            last=repr(e)
            time.sleep(min(12,2**i))
    return [],None,last


def quarter_rows(rows):
    q=[]
    if not isinstance(rows,list):
        rows=[]
    for r in rows:
        if not isinstance(r,dict):
            continue
        fr=pd.to_datetime(r.get("re_from_dt"),errors="coerce",dayfirst=True)
        to=pd.to_datetime(r.get("re_to_dt"),errors="coerce",dayfirst=True)
        if pd.isna(fr) or pd.isna(to):
            continue
        days=(to-fr).days+1
        if days<70 or days>115:
            continue
        z={
            "from":fr,"to":to,
            "sales":num(r.get("re_net_sale")),
            "total_income":num(r.get("re_total_inc") or r.get("re_tot_inc")),
            "net_profit":num(r.get("re_net_profit") or r.get("re_con_pro_loss")),
            "pbt":num(r.get("re_pro_loss_bef_tax") or r.get("re_pro_loss_bef_tax_sum")),
            "interest":num(r.get("re_int_new") or r.get("re_int_expd")),
            "depreciation":num(r.get("re_depr_und_exp")),
            "debt_equity":num(r.get("re_debt_eqt_rat")),
            "eps":num(r.get("re_basic_eps_for_cont_dic_opr") or r.get("re_basic_eps")),
            "created":pd.to_datetime(r.get("re_create_dt"),errors="coerce",dayfirst=True),
            "notes":str(r.get("re_desc_note_fin") or "")[:4000],
        }
        # NSE result values are typically reported in lakh rupees. Convert to crore.
        for c in ["sales","total_income","net_profit","pbt","interest","depreciation"]:
            if pd.notna(z[c]):
                z[c]=z[c]/100.0
        q.append(z)
    if not q:
        return pd.DataFrame()
    d=pd.DataFrame(q).sort_values(["to","created"]).drop_duplicates("to",keep="last").sort_values("to")
    return d


def safe_growth(a,b):
    if pd.isna(a) or pd.isna(b) or b<=0:
        return np.nan
    return a/b-1.0


def clip01(x,lo,hi):
    if pd.isna(x):
        return np.nan
    return float(np.clip((x-lo)/(hi-lo),0,1))


def feature_row(symbol,df,bank_flag,error):
    base={"symbol":symbol,"fetch_error":error,"bank_nonbanking":bank_flag}
    if df.empty:
        base.update({"financial_available":False,"financial_score_raw":np.nan})
        return base

    latest=df.iloc[-1]
    prev=df.iloc[-2] if len(df)>=2 else None
    yoy=df.iloc[-5] if len(df)>=5 else None
    prev_yoy=df.iloc[-6] if len(df)>=6 else None

    sales_yoy=safe_growth(latest["sales"],yoy["sales"]) if yoy is not None else np.nan
    profit_yoy=safe_growth(latest["net_profit"],yoy["net_profit"]) if yoy is not None else np.nan
    sales_qoq=safe_growth(latest["sales"],prev["sales"]) if prev is not None else np.nan
    latest_margin=latest["net_profit"]/latest["sales"] if pd.notna(latest["net_profit"]) and pd.notna(latest["sales"]) and latest["sales"]>0 else np.nan
    yoy_margin=yoy["net_profit"]/yoy["sales"] if yoy is not None and pd.notna(yoy["net_profit"]) and pd.notna(yoy["sales"]) and yoy["sales"]>0 else np.nan
    margin_delta=latest_margin-yoy_margin if pd.notna(latest_margin) and pd.notna(yoy_margin) else np.nan

    prev_sales_yoy=safe_growth(prev["sales"],prev_yoy["sales"]) if prev is not None and prev_yoy is not None else np.nan
    sales_accel=sales_yoy-prev_sales_yoy if pd.notna(sales_yoy) and pd.notna(prev_sales_yoy) else np.nan

    pbt_margin=latest["pbt"]/latest["sales"] if pd.notna(latest["pbt"]) and pd.notna(latest["sales"]) and latest["sales"]>0 else np.nan
    interest_ratio=latest["interest"]/latest["sales"] if pd.notna(latest["interest"]) and pd.notna(latest["sales"]) and latest["sales"]>0 else np.nan
    yoy_interest_ratio=(yoy["interest"]/yoy["sales"]) if yoy is not None and pd.notna(yoy["interest"]) and pd.notna(yoy["sales"]) and yoy["sales"]>0 else np.nan
    interest_improvement=yoy_interest_ratio-interest_ratio if pd.notna(interest_ratio) and pd.notna(yoy_interest_ratio) else np.nan

    annualized_sales=latest["sales"]*4 if pd.notna(latest["sales"]) else np.nan
    stale_days=(pd.Timestamp.now().normalize()-latest["to"]).days
    freshness=float(np.exp(-max(stale_days,0)/240.0))

    parts=[]
    weights=[]
    for val,w in [
        (clip01(sales_yoy,-0.05,0.45),0.25),
        (clip01(profit_yoy,-0.10,0.80),0.25),
        (clip01(margin_delta,-0.03,0.06),0.18),
        (clip01(sales_accel,-0.15,0.25),0.12),
        (clip01(interest_improvement,-0.02,0.03),0.08),
        (clip01(sales_qoq,-0.15,0.25),0.12),
    ]:
        if pd.notna(val):
            parts.append(val*w); weights.append(w)
    score=sum(parts)/sum(weights) if weights else np.nan
    if pd.notna(score):
        score*=freshness

    base.update({
        "financial_available":True,
        "latest_quarter_end":str(latest["to"].date()),
        "financial_stale_days":int(stale_days),
        "financial_freshness":freshness,
        "quarters_available":int(len(df)),
        "latest_quarter_sales_crore":latest["sales"],
        "annualized_sales_crore":annualized_sales,
        "latest_net_profit_crore":latest["net_profit"],
        "sales_yoy":sales_yoy,
        "profit_yoy":profit_yoy,
        "sales_qoq":sales_qoq,
        "sales_growth_acceleration":sales_accel,
        "net_margin":latest_margin,
        "net_margin_yoy_delta":margin_delta,
        "pbt_margin":pbt_margin,
        "interest_to_sales":interest_ratio,
        "interest_burden_improvement":interest_improvement,
        "debt_equity_reported":latest["debt_equity"],
        "financial_score_raw":score,
    })
    return base


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--companies",required=True)
    ap.add_argument("--output",required=True)
    ap.add_argument("--max-symbols",type=int,default=250)
    ap.add_argument("--sleep",type=float,default=0.35)
    args=ap.parse_args()

    src=pd.read_csv(args.companies)
    if "symbol" not in src:
        raise RuntimeError("companies file needs symbol")
    if "max_linked_evidence_score" in src:
        src=src.sort_values("max_linked_evidence_score",ascending=False)
    symbols=list(dict.fromkeys(src["symbol"].dropna().astype(str).str.upper().str.strip()))[:int(args.max_symbols)]

    s=session()
    feats=[]; raw_summary=[]
    for i,sym in enumerate(symbols,1):
        rows,bank,err=get_symbol(s,sym)
        q=quarter_rows(rows)
        feats.append(feature_row(sym,q,bank,err))
        raw_summary.append({"symbol":sym,"raw_rows":len(rows) if isinstance(rows,list) else 0,"quarter_rows":len(q),"error":err})
        if i%25==0:
            print(f"financial {i}/{len(symbols)}",flush=True)
        time.sleep(max(0,float(args.sleep)))

    f=pd.DataFrame(feats)
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
    f.to_csv(out,index=False)
    summary={
        "symbols_requested":len(symbols),
        "financial_available":int(f.get("financial_available",pd.Series(dtype=bool)).fillna(False).sum()) if len(f) else 0,
        "fresh_within_240d":int((pd.to_numeric(f.get("financial_stale_days"),errors="coerce")<=240).sum()) if len(f) else 0,
        "median_stale_days":float(pd.to_numeric(f.get("financial_stale_days"),errors="coerce").median()) if len(f) else None,
        "errors":sum(1 for x in raw_summary if x["error"]),
    }
    json.dump(summary,open(out.parent/"financial_inflection_summary.json","w"),indent=2,default=str)
    print(json.dumps(summary,indent=2,default=str))

if __name__=="__main__":
    main()
