from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
import requests

HEADERS={"User-Agent":"Mozilla/5.0","Accept":"application/xml,text/xml,text/html,*/*"}

def num(v):
    if v is None:return np.nan
    s=str(v).strip().replace(",","").replace("%","")
    if not s or s.lower() in {"nan","na","none","-"}:return np.nan
    try:return float(s)
    except:
        m=re.search(r"[-+]?\d+(?:\.\d+)?",s)
        return float(m.group()) if m else np.nan

def pct_norm(v):
    x=num(v)
    if not np.isfinite(x):return np.nan
    return x/100.0 if abs(x)>1.5 else x

def local(tag):return tag.rsplit("}",1)[-1]

def fetch_pledge(url):
    if not url:return (np.nan,0.0,"")
    try:
        r=requests.get(url,headers=HEADERS,timeout=30)
        r.raise_for_status()
        root=ET.fromstring(r.content)
        cand=[]
        for e in root.iter():
            tag=local(e.tag)
            tl=tag.lower()
            if not (("pledge" in tl or "encumber" in tl) and ("percent" in tl or "percentage" in tl)):
                continue
            v=num(e.text)
            if not np.isfinite(v) or v<0 or v>100:continue
            score=0.45
            if "promoter" in tl:score+=0.35
            if "share" in tl:score+=0.10
            if "total" in tl:score+=0.05
            cand.append((score,v,tag))
        if not cand:return (np.nan,0.0,"")
        cand.sort(reverse=True)
        score,v,tag=cand[0]
        return (v/100.0,float(min(score,1.0)),tag)
    except Exception:
        return (np.nan,0.0,"")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--shareholding",required=True)
    ap.add_argument("--insider",required=True)
    ap.add_argument("--symbols-file",required=True)
    ap.add_argument("--symbol-column",default="symbol")
    ap.add_argument("--asof",default="")
    ap.add_argument("--output",required=True)
    ap.add_argument("--fetch-pledge-xbrl",action="store_true")
    args=ap.parse_args()

    asof=pd.Timestamp(args.asof, tz="UTC") if args.asof else pd.Timestamp.now(tz="UTC")
    syms=set(pd.read_csv(args.symbols_file)[args.symbol_column].dropna().astype(str).str.upper().str.strip())
    sh=pd.read_parquet(args.shareholding)
    ins=pd.read_parquet(args.insider)

    scol="symbol_norm" if "symbol_norm" in sh.columns else "_query_symbol" if "_query_symbol" in sh.columns else "symbol"
    sh["_sym"]=sh[scol].astype(str).str.upper().str.strip()
    sh=sh[sh["_sym"].isin(syms)].copy()
    sh["_date"]=pd.to_datetime(sh.get("date"),utc=True,errors="coerce")
    sh["_broadcast"]=pd.to_datetime(sh.get("broadcastDate",sh.get("submissionDate")),utc=True,errors="coerce")
    sh=sh[(sh["_date"]<=asof)&(sh["_broadcast"].fillna(sh["_date"])<=asof)].copy()

    rows=[]
    for sym,g in sh.groupby("_sym"):
        g=g.sort_values(["_date","_broadcast"])
        latest=g.iloc[-1]
        prev=g.iloc[-2] if len(g)>=2 else None
        ph=pct_norm(latest.get("pr_and_prgrp"))
        prevph=pct_norm(prev.get("pr_and_prgrp")) if prev is not None else np.nan
        pledge=np.nan;pledge_conf=0.0;pledge_tag=""
        if args.fetch_pledge_xbrl:
            pledge,pledge_conf,pledge_tag=fetch_pledge(str(latest.get("xbrl") or ""))
            time.sleep(.08)
        rows.append({
            "symbol":sym,
            "shareholding_date":str(latest["_date"].date()) if pd.notna(latest["_date"]) else "",
            "promoter_holding":ph,
            "promoter_delta_qoq":ph-prevph if np.isfinite(ph) and np.isfinite(prevph) else np.nan,
            "pledged_pct":pledge,
            "pledged_pct_confidence":pledge_conf,
            "pledged_pct_tag":pledge_tag,
            "shareholding_xbrl":str(latest.get("xbrl") or ""),
        })

    own=pd.DataFrame(rows)

    icol="symbol_norm" if "symbol_norm" in ins.columns else "symbol"
    ins["_sym"]=ins[icol].astype(str).str.upper().str.strip()
    ins=ins[ins["_sym"].isin(syms)].copy()
    ins["_date"]=pd.to_datetime(ins.get("date"),utc=True,errors="coerce")
    win=ins[(ins["_date"]<=asof)&(ins["_date"]>=asof-pd.Timedelta(days=180))].copy()
    cat=win.get("personCategory",pd.Series("",index=win.index)).astype(str).str.lower()
    mode=win.get("acqMode",pd.Series("",index=win.index)).astype(str).str.lower()
    prom=cat.str.contains("promoter",na=False)
    openm=mode.str.contains("open",na=False)&mode.str.contains("market",na=False)
    w=win[prom&openm].copy()
    w["_buy"]=pd.to_numeric(w.get("buyValue"),errors="coerce").fillna(0)
    w["_sell"]=pd.to_numeric(w.get("sellValue"),errors="coerce").fillna(0)
    if len(w):
        ag=w.groupby("_sym").agg(
            promoter_open_market_buy_value_180=("_buy","sum"),
            promoter_open_market_sell_value_180=("_sell","sum"),
            promoter_open_market_trade_count_180=("_sym","size"),
        ).reset_index().rename(columns={"_sym":"symbol"})
        ag["promoter_open_market_net_value_180"]=ag["promoter_open_market_buy_value_180"]-ag["promoter_open_market_sell_value_180"]
        own=own.merge(ag,on="symbol",how="outer")
    else:
        own["promoter_open_market_buy_value_180"]=0.0
        own["promoter_open_market_sell_value_180"]=0.0
        own["promoter_open_market_trade_count_180"]=0
        own["promoter_open_market_net_value_180"]=0.0

    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    own.to_csv(out,index=False)
    summary={
        "symbols_requested":len(syms),
        "symbols_with_shareholding":int(own["promoter_holding"].notna().sum()) if len(own) else 0,
        "promoter_holding_coverage":float(own["promoter_holding"].notna().sum()/max(len(syms),1)) if len(own) else 0,
        "pledge_coverage_high_conf":float(((pd.to_numeric(own["pledged_pct_confidence"],errors="coerce")>=0.70)&own["pledged_pct"].notna()).sum()/max(len(syms),1)) if len(own) else 0,
        "promoter_open_market_buyers_180":int((pd.to_numeric(own.get("promoter_open_market_net_value_180"),errors="coerce").fillna(0)>0).sum()) if len(own) else 0,
    }
    json.dump(summary,open(out.parent/"ownership_summary.json","w"),indent=2)
    print(json.dumps(summary,indent=2))

if __name__=="__main__":
    main()
