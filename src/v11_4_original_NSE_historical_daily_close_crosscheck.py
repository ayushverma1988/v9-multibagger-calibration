"""Official NSE historical fold-close verifier for original RSI>70 market data.

Requires historic-date NSE official UDiFF/legacy cash-market bhavcopy and
independent official sec_bhavdata_full, where available. The 8 target dates
have already been frozen, and this checks exchange-supplied closing prices
ONLY; it does not claim the preceding 120 RSI calculation sessions were
independently certified or impute unobserved numbers. Fail closed on unavailable
exchange downloads, invalid quote formats or conflicting official providers.
"""
from __future__ import annotations
import argparse,hashlib,io,json,os,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import requests
DATES=("2022-06-30","2022-12-30","2023-06-30","2023-12-29",
       "2024-06-28","2024-12-31","2025-06-30","2025-12-31")
COUNTS=dict(zip(DATES,(1087,1123,1147,1276,1322,1345,1298,1314)))
MAIN_SERIES=("EQ","BE","BZ")
USERAGENT="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/129.0.0.0 Safari/537.36"
NSE_ARCHIVE="https://archives.nseindia.com"

def require(frame,*names):
 lookup={str(k).upper().strip():k for k in frame.columns}
 for name in names:
  if name.upper() in lookup:return lookup[name.upper()]
 raise ValueError("Original NSE price tape missing expected quote field: "+"/".join(names))
def quote_frame(data,kind):
 x=pd.read_csv(io.BytesIO(data),low_memory=False,skipinitialspace=True)
 if kind=="udiff":
  sym=require(x,"TckrSymb","SYMBOL");series=require(x,"SctySrs","SERIES")
  close=require(x,"ClsPric","CLOSE_PRICE")
 elif kind=="oldzip":
  sym=require(x,"SYMBOL");series=require(x,"SERIES")
  close=require(x,"CLOSE","CLOSE_PRICE")
 elif kind=="full":
  sym=require(x,"SYMBOL");series=require(x,"SERIES")
  close=require(x,"CLOSE_PRICE","CLOSE")
 else:raise ValueError("Unsupported original official NSE source family")
 z=pd.DataFrame({"symbol":x[sym].astype(str).str.upper().str.strip(),
  "series":x[series].astype(str).str.upper().str.strip(),
  "official_close_INR":pd.to_numeric(x[close],errors="coerce")})
 z=z[z["series"].isin(MAIN_SERIES)&z["official_close_INR"].gt(0)].copy()
 z["priority"]=z["series"].map({"EQ":0,"BE":1,"BZ":2})
 z=z.sort_values(["symbol","priority"])
 duplicate=z.duplicated(["symbol","series"],keep=False)
 invalid=set(z.loc[duplicate].groupby("symbol")["official_close_INR"].nunique().loc[lambda t:t>1].index)
 z=z[~z["symbol"].isin(invalid)].drop_duplicates(["symbol"],keep="first")
 if len(z)<500:raise ValueError(f"Inadequate original official NSE {kind} quote rows {len(z)}")
 return z.drop(columns=["series","priority"]).reset_index(drop=True)
def zip_csv(data):
 with zipfile.ZipFile(io.BytesIO(data)) as z:
  items=[q for q in z.namelist() if q.lower().endswith(".csv")]
  if len(items)!=1:raise ValueError("Original NSE tape archive zip has no unique CSV")
  return z.read(items[0])
def urls(day):
 t=pd.Timestamp(day)
 mon=t.strftime("%b").upper()
 return [
 ("udiff",NSE_ARCHIVE+"/content/cm/BhavCopy_NSE_CM_0_0_0_"+t.strftime("%Y%m%d")+"_F_0000.csv.zip"),
 ("oldzip",NSE_ARCHIVE+"/content/historical/EQUITIES/"+t.strftime("%Y")+"/"+mon+"/cm"+t.strftime("%d")+mon+t.strftime("%Y")+"bhav.csv.zip"),
 ("full",NSE_ARCHIVE+"/products/content/sec_bhavdata_full_"+t.strftime("%d%m%Y")+".csv")
 ]
def download(day,folder,session=None):
 s=session or requests.Session()
 headers={"User-Agent":USERAGENT,"Accept":"text/csv,application/zip,*/*",
          "Accept-Language":"en-US,en;q=0.9","Referer":"https://www.nseindia.com/"}
 variants=urls(day);results={};errors={}
 for typ,url in variants:
  try:
   r=s.get(url,headers=headers,timeout=35)
   r.raise_for_status()
   if len(r.content)<1000:raise ValueError("Unexpected tiny original NSE file")
   csv=zip_csv(r.content) if typ!="full" else r.content
   table=quote_frame(csv,typ)
   digest=hashlib.sha256(r.content).hexdigest()
   results[typ]=(table,url,digest)
   folder.mkdir(parents=True,exist_ok=True)
   (folder/f"{day}_{typ}.sha256.json").write_text(json.dumps(
      {"official_url":url,"response_bytes":len(r.content),"sha256":digest,
       "parsed_equity_quote_count":len(table)},indent=2))
  except (requests.RequestException,OSError,ValueError,zipfile.BadZipFile) as exc:
   errors[typ]={"official_original_market_URL":url,
                "status":"OFFICIAL_SOURCE_UNAVAILABLE_OR_BAD_FORMAT",
                "error_type":type(exc).__name__,
                "detail":str(exc)[:190]}
 # Prefer original dated cash-market tape as primary. Official full is an
 # independent archive family. Same NSE operator, not independent company.
 primary=results.get("udiff") or results.get("oldzip")
 secondary=results.get("full")
 return primary,secondary,errors

def validate(date,frozen,primary,secondary):
 """No future outcomes; retain original denominator and count unknowns."""
 if date not in COUNTS:raise ValueError("Unregistered original fold date")
 s=frozen.copy()
 if {"y6","y12","y24","dd30_6m","y6_mature_date"}&set(s):
  raise ValueError("No future six-month stock outcomes in original quote audit")
 if not {"date","symbol","close"}.issubset(s):
  raise ValueError("Original frozen price source missing stock-date-close")
 s["date"]=pd.to_datetime(s["date"],errors="raise").dt.strftime("%Y-%m-%d")
 s["symbol"]=s["symbol"].astype(str).str.upper().str.strip()
 s=s[s["date"].eq(date)].copy()
 if len(s)!=COUNTS[date] or s["symbol"].duplicated().any():
  raise ValueError("Original frozen eightfold company identities drifted")
 s["close"]=pd.to_numeric(s["close"],errors="coerce")
 if s["close"].isna().any() or s["close"].le(0).any():
  raise ValueError("Original market closes invalid")
 if primary is None:
  return {"fold":date,"original_stock_universe":len(s),
   "strict_two_exchange_file_family_match_count":0,
   "single_official_quote_reference_matched_count":0,
   "source_unavailable_unknown_stocks":len(s),
   "source_integrity_status":"ORIGINAL_OFFICIAL_PRIMARY_MISSING_NO_VALIDATION_CLAIM",
   "raw_RSI_120_day_history_source_independently_certified":False}
 first=primary.copy().rename(columns={"official_close_INR":"primary_close"})
 if first["symbol"].duplicated().any():raise ValueError("Ambiguous NSE primary company quote")
 joined=s[["symbol","close"]].merge(first,on="symbol",how="left",validate="1:1")
 if secondary is not None:
  second=secondary.rename(columns={"official_close_INR":"independent_full_close"})
  if second["symbol"].duplicated().any():raise ValueError("Ambiguous NSE secondary company quote")
  joined=joined.merge(second,on="symbol",how="left",validate="1:1")
  agree=np.isclose(joined["primary_close"],joined["independent_full_close"],
             rtol=0,atol=.10,equal_nan=False)
 else:agree=pd.Series(False,index=joined.index)
 pgood=np.isclose(joined["close"],joined["primary_close"],
             rtol=.002,atol=.10,equal_nan=False)
 matched=pgood&agree
 missing=int(joined["primary_close"].isna().sum())
 if secondary is not None:
  missing_secondary=int(joined["independent_full_close"].isna().sum())
 else:missing_secondary=len(joined)
 return {"fold":date,"original_stock_universe":len(joined),
  "official_primary_quote_overlap":len(joined)-missing,
  "official_full_quote_overlap":len(joined)-missing_secondary,
  "original_frozen_close_matches_official_primary":int(pgood.sum()),
  "official_two_archive_families_close_consistent":int(agree.sum()),
  "strict_two_exchange_file_family_match_count":int(matched.sum()),
  "single_official_quote_reference_matched_count":int(pgood.sum()) if secondary is None else 0,
  "original_market_price_official_dual_archive_validation_pct":round(100*int(matched.sum())/len(s),2),
  "source_unavailable_unknown_stocks":int((~matched).sum()),
  "source_integrity_status":("TWO_OFFICIAL_ARCHIVE_FILE_FAMILIES_CONSISTENT_PARTIAL_DAY_VERIFICATION"
     if secondary is not None else "ONE_OFFICIAL_FILE_FAMILY_ONLY_PARTIAL_VERIFICATION"),
  "raw_RSI_120_day_history_source_independently_certified":False,
  "no_prediction_accuracy_inference_from_price_source_agreement":True}

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--snapshot",required=True);p.add_argument("--out",required=True)
 a=p.parse_args()
 folder=Path(a.out);folder.mkdir(parents=True,exist_ok=True)
 original=pd.read_parquet(a.snapshot,columns=["date","symbol","close"])
 data=[];details=[]
 with requests.Session() as session:
  for date in DATES:
   primary,secondary,errors=download(date,folder/"raw_archive_SHA256",session)
   psrc=primary[0] if primary is not None else None
   ssrc=secondary[0] if secondary is not None else None
   result=validate(date,original,psrc,ssrc)
   result.update({"exchange_price_source_URL_primary":primary[1] if primary else None,
     "exchange_secondary_full_URL":secondary[1] if secondary else None,
     "independent_exchange_file_archive_families_retrieved":int(primary is not None)+int(secondary is not None),
     "official_exchange_download_errors":errors})
   details.append(result)
   print("OFFICIAL_NSE_FOLD_CLOSE",json.dumps({k:result[k] for k in (
    "fold","original_stock_universe","strict_two_exchange_file_family_match_count",
    "single_official_quote_reference_matched_count","source_integrity_status")}),flush=True)
 out={"scope":"OFFICIAL_NSE_EIGHT_FROZEN_FOLD_DAY_CLOSE_DUAL_ARCHIVE_PARTIAL_VERIFICATION_NOT_120_RSI_DAY_CERTIFICATION",
  "original_8_2022_2025_decision_dates":len(details),
  "original_market_stockdate_total":sum(x["original_stock_universe"] for x in details),
  "official_archives_required_at_original_market_date_not_future_equity_prices":True,
  "exact_NSE_official_archive_families":"cash market UDiFF/legacy equity Bhavcopy + sec_bhavdata_full",
  "genuinely_distinct_exchange_operators_compared":False,
  "off_exchange_archive_full_daily_120_bar_provenance_validated":False,
  "six_month_multibagger_precision_recomputed":False,
  "no_training_calibration_or_original_2026_stocks_changed":True,
  "official_NSE_two_source_day_count":sum(x["independent_exchange_file_archive_families_retrieved"]==2 for x in details),
  "official_NSE_one_source_day_count":sum(x["independent_exchange_file_archive_families_retrieved"]==1 for x in details),
  "official_NSE_unavailable_day_count":sum(x["independent_exchange_file_archive_families_retrieved"]==0 for x in details),
  "strict_exchange_bhavcopy_consistent_original_day_stockdates":sum(x["strict_two_exchange_file_family_match_count"] for x in details),
  "per_date":details}
 (folder/"original_NSE_eightfold_exchange_day_close_audit.json").write_text(json.dumps(out,indent=2))
 print(json.dumps({k:v for k,v in out.items() if k!="per_date"},indent=2),flush=True)
if __name__=="__main__":main()
