"""Read-only official NSE archived valuation source probe.

Does NOT call blocked www.nseindia.com API; tests exact exchange archive
URLs in separate dimensions:
 * company P/E details (individual securities), unknown schema until verified
 * INDEX P/E, P/B, dividend yields (never assert company P/E)
 * cash security master, limited to actually reported market-cap attributes
 * official index daily snapshots for naming verification
All material raw bytes are source-SHA256 registered, never infer stock P/E
from an index line or invent a market cap from price alone.
"""
from __future__ import annotations
import argparse,hashlib,io,json,gzip
from pathlib import Path
from datetime import datetime,timezone
import pandas as pd,requests

HOSTS=("https://archives.nseindia.com","https://nsearchives.nseindia.com")
HEADERS={"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/129 Safari/537.36",
         "Accept":"text/csv,application/octet-stream,*/*","Accept-Language":"en-US,en;q=0.9"}
def sha(b):return hashlib.sha256(b).hexdigest()
def urls(dt):
 day=pd.Timestamp(dt).normalize()
 six=day.strftime("%d%m%y"); eight=day.strftime("%d%m%Y")
 return {
  "individual_security_PE":{"path":f"/content/equities/peDetail/PE_{six}.csv","expected":"per_security"},
  "index_PE_PB_dividend":{"path":f"/archives/equities/mkt/PE_{six}.csv","expected":"index_only"},
  "cash_security_master":{"path":f"/content/cm/NSE_CM_security_{eight}.csv.gz","expected":"security_master"},
  "index_close_daily":{"path":f"/content/indices/ind_close_all_{eight}.csv","expected":"index_only"}
 }
def looks_like_document(data):
 if len(data)<50:return False
 x=data[:400].lstrip().lower()
 return not(x.startswith(b"<!doctype html") or x.startswith(b"<html") or
            x.startswith(b"<?xml") or b"access denied" in x[:100] or
            b"request blocked" in x[:100])
def decode_tabular(raw,filename):
 b=gzip.decompress(raw) if filename.endswith(".gz") else raw
 if not looks_like_document(b):raise ValueError("HTML/blocked or non-tabular response")
 last=None
 for header in (0,1,2):
  try:
   x=pd.read_csv(io.BytesIO(b),skiprows=header,encoding="utf-8-sig",on_bad_lines="skip")
   if x.shape[0] and x.shape[1]>=2:return x
  except Exception as e:last=e
 raise ValueError(f"No tabular CSV schema {last}")
def classify_columns(frame,purpose):
 cols=[str(c).strip() for c in frame.columns]
 normalized=[c.lower().replace(" ","").replace("_","") for c in cols]
 is_symbol=any(n in {"symbol","tckrsymb","ticker","securitysymbol","sym"} for n in normalized)
 is_index=any("index" in n for n in normalized)
 has_pe=any("p/e" in c.lower() or n in {"pe","peratio","pricetoearnings"} for c,n in zip(cols,normalized))
 has_mcap=any("marketcap" in n or "marketcapital" in n for n in normalized)
 has_shares=any("issuedshare" in n or "listedshare" in n or "outstandingshare" in n for n in normalized)
 if purpose=="per_security":
  verified=is_symbol and has_pe and not is_index
 elif purpose=="index_only":
  verified=is_index and has_pe if "PE" in str(purpose) else is_index
 else:verified=is_symbol and (has_mcap or has_shares)
 return {"rows":int(len(frame)),"columns":cols[:65],
         "symbol_column_detected":is_symbol,"index_column_detected":is_index,
         "PE_column_detected":has_pe,"market_cap_column_detected":has_mcap,
         "issued_shares_column_detected":has_shares,
         "company_PE_semantics_verified":bool(purpose=="per_security" and is_symbol and has_pe and not is_index),
         "value_market_cap_semantics_verified":bool(purpose=="security_master" and is_symbol and has_mcap)}
def probe(asof,out):
 dest=Path(out);dest.mkdir(parents=True,exist_ok=True)
 res={"snapshot_date":str(pd.Timestamp(asof).date()),
      "scope":"NSE_DIRECT_ARCHIVE_PRIMARY_SOURCE_PROBE_NO_PRICE_MODEL_CHANGES",
      "attempted":{},"reliably_verified_original_company_PE":False,
      "index_PE_will_never_replace_company_PE":True,
      "original_individual_security_valuation_ready":False,
      "original_market_cap_readiness":False,
      "not_production_approved":True,
      "not_a_stock_selection":True,
      "created_utc":datetime.now(timezone.utc).isoformat()}
 for kind,entry in urls(asof).items():
  attempts=[]
  for host in HOSTS:
   url=host+entry["path"]
   r={"url":url,"expected":entry["expected"]}
   try:
    resp=requests.get(url,headers=HEADERS,timeout=40)
    r["HTTP_status"]=resp.status_code
    if resp.status_code!=200:raise ValueError(f"HTTP {resp.status_code}")
    raw=resp.content
    r["byte_count"]=len(raw)
    r["source_bytes_SHA256"]=sha(raw)
    x=decode_tabular(raw,entry["path"])
    diag=classify_columns(x,entry["expected"])
    r.update(diag)
    # Save only original verified CSV values to private runtime output on valid schema.
    # Company symbols and private selection list are not part of this probe.
    if kind=="individual_security_PE" and diag["company_PE_semantics_verified"]:
     x.to_csv(dest/"official_security_PE_original_schema.csv",index=False)
     res["reliably_verified_original_company_PE"]=True
     res["individual_security_pe_url"]=url
    if kind=="cash_security_master" and diag["value_market_cap_semantics_verified"]:
     res["original_market_cap_readiness"]=True
    r["source_readable"]=True
    attempts.append(r)
    # Do not try more URLs after verified original publication and schema.
    break
   except Exception as exc:
    r["source_readable"]=False;r["error"]=str(exc)[:280]
    attempts.append(r)
  res["attempted"][kind]=attempts
 res["original_individual_security_valuation_ready"]=res["reliably_verified_original_company_PE"]
 (dest/"NSE_primary_archive_source_diagnostic.json").write_text(json.dumps(res,indent=2))
 print(json.dumps(res,indent=2),flush=True)
 return res
def main():
 p=argparse.ArgumentParser()
 p.add_argument("--asof",required=True)
 p.add_argument("--output",required=True)
 a=p.parse_args();probe(a.asof,a.output)
if __name__=="__main__":main()
