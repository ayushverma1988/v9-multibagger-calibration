"""Research-only discovery of fresh official NSE daily PE and market-cap reports.

No inferred valuation figures, no standalone model weights, no effect on
already sealed six-month picks. Only discover availability and source dates
to establish whether exact user's condition 2 and 3 can be sourced.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from datetime import datetime,timezone
from urllib.parse import urlparse
import pandas as pd
import requests

API="https://www.nseindia.com/api/daily-reports"
HOME="https://www.nseindia.com/"
HEADERS={"user-agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/129 Safari/537.36",
         "accept-language":"en-US,en;q=0.9","accept":"application/json,text/plain,*/*",
         "referer":"https://www.nseindia.com/all-reports"}
def get_catalog(asof):
 sess=requests.Session()
 sess.headers.update(HEADERS)
 first=sess.get(HOME,timeout=25);first.raise_for_status()
 r=sess.get(API,params={"key":"CM"},timeout=60)
 r.raise_for_status()
 data=r.json()
 if not isinstance(data,dict):
  raise ValueError("Unrecognized official NSE report catalog structure")
 items=[]
 for bucket in ("CurrentDay","PreviousDay"):
  rows=data.get(bucket,[])
  if not isinstance(rows,list):continue
  for rec in rows:
   if not isinstance(rec,dict):continue
   actual=str(rec.get("fileActlName") or "")
   label=str(rec.get("displayName") or "")
   date=pd.to_datetime(rec.get("tradingDate"),errors="coerce",dayfirst=True)
   # No future official prices can be used in historic reconstruction.
   if pd.notna(date) and date.normalize()>pd.Timestamp(asof).normalize():continue
   key=(label+" "+actual).lower()
   category=("PE_ratio" if ("p/e" in key or "pe ratio" in key or "price earning" in key or "price to earning" in key)
       else "Market_Capitalization" if ("market cap" in key or "mcap" in key or "market capital" in key)
       else "Industry_Sector_Classification" if ("industry" in key or "sector classification" in key)
       else "Other")
   items.append({"bucket":bucket,"category":category,"report_label":label[:140],
                 "file_name":actual[:150],"reported_trading_date":str(date.date()) if pd.notna(date) else None,
                 "source_host":urlparse(str(rec.get("filePath") or HOME)).hostname,
                 "has_file_reference":bool(actual)})
 return items
def main():
 a=argparse.ArgumentParser()
 a.add_argument("--asof",required=True)
 a.add_argument("--output",required=True)
 p=a.parse_args()
 out=Path(p.output);out.mkdir(parents=True,exist_ok=True)
 error=None
 try:rows=get_catalog(p.asof)
 except Exception as exc:
  rows=[]
  error=str(exc)[:300]
 matches=[x for x in rows if x["category"]!="Other"]
 summary={"scope":"OFFICIAL_NSE_DAILY_REPORT_CATALOG_READINESS_ONLY",
          "asof":p.asof,"original_official_catalog_url":API,
          "official_report_count":len(rows),
          "PE_ratio_catalog_file_count":sum(x["category"]=="PE_ratio" for x in rows),
          "market_cap_catalog_file_count":sum(x["category"]=="Market_Capitalization" for x in rows),
          "industry_sector_catalog_file_count":sum(x["category"]=="Industry_Sector_Classification" for x in rows),
          "available_relevant_reports":matches[:40],
          "source_api_error":error,
          "report_value_XBRL_units_verified":False,
          "user_value_rule_integrated_into_frozen_predictions":False,
          "daily_valuation_source_ready_to_extract":bool(matches and not error)}
 (out/"official_NSE_PE_marketcap_catalog_status.json").write_text(json.dumps(summary,indent=2))
 print(json.dumps(summary,indent=2),flush=True)
if __name__=="__main__":main()
