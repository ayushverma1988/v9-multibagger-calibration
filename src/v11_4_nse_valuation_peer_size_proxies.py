"""NSE origin sector-peer valuation and issued-capital market-size proxies.

IMPORTANT: CURRENT undated index members are NOT a 15:30 IST PIT source.
Only append to a separate after-close research dataset. No current or frozen
22-variable predictor input or first-seen 6-month stock rankings modified.
"""
from __future__ import annotations
import argparse,hashlib,io,gzip,json
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import requests
from v11_4_nse_archive_valuation_probe import HEADERS

HOST="https://archives.nseindia.com"
INDEX_LIST_PATH="/content/indices/ind_niftytotalmarket_list.csv"
MIN_PEERS=5
def get_bytes(url):
 r=requests.get(url,headers=HEADERS,timeout=45);r.raise_for_status()
 if len(r.content)<300:return None
 return r.content
def normsymbol(s):
 return s.astype(str).str.upper().str.strip()
def normisin(s):
 return s.astype(str).str.upper().str.replace(r"\s+","",regex=True)
def peer_reference(company,member,min_peers=MIN_PEERS):
 required={"symbol","isin","company_pe"}
 if not required.issubset(company):raise ValueError("Original company PE market-ISIN source incomplete")
 if not {"symbol","isin","industry"}.issubset(member):raise ValueError("Official constituent file lacks industry and ISIN")
 c=company.copy();m=member.copy()
 for x in (c,m):
  x["symbol"]=normsymbol(x["symbol"]);x["isin"]=normisin(x["isin"])
 m["industry"]=m["industry"].astype(str).str.strip()
 m=m[~m["industry"].isin(["","nan","None"])].copy()
 if m.duplicated(["symbol","isin"]).any():raise ValueError("Multiple sectors for same constituent")
 c=c.merge(m[["symbol","isin","industry"]],on=["symbol","isin"],how="left",validate="1:1")
 c["company_pe"]=pd.to_numeric(c["company_pe"],errors="coerce")
 peers=c[c["company_pe"].gt(0)].groupby("industry")["company_pe"].agg(["median","count"])
 eligible=peers.loc[peers["count"].ge(min_peers)].copy()
 c["industry_peer_median_PE_proxy"]=c["industry"].map(eligible["median"])
 c["industry_peer_companies_with_positive_pe"]=c["industry"].map(eligible["count"])
 c["company_PE_below_own_current_sector_peer_median_proxy"]=(
  c["company_pe"]<c["industry_peer_median_PE_proxy"])
 c.loc[c[["company_pe","industry_peer_median_PE_proxy"]].isna().any(axis=1),
       "company_PE_below_own_current_sector_peer_median_proxy"]=pd.NA
 c["company_PE_below_own_current_sector_peer_median_proxy"]=c[
  "company_PE_below_own_current_sector_peer_median_proxy"].astype("boolean")
 return c
def security_issued_shares_estimate(master,company):
 req={"TckrSymb","SctySrs","ISIN","IssdCptl","ParVal"}
 if not req.issubset(master):raise ValueError("Original NSE security master missing issue size")
 m=master[master["SctySrs"].astype(str).isin(["EQ","BE","BZ"])].copy()
 m["symbol"]=normsymbol(m["TckrSymb"]);m["isin"]=normisin(m["ISIN"])
 m["priority"]=m["SctySrs"].map({"EQ":0,"BE":1,"BZ":2})
 m=m.sort_values(["symbol","isin","priority"]).drop_duplicates(["symbol","isin"])
 m["issued_security_count_NSE_raw"]=pd.to_numeric(m["IssdCptl"],errors="coerce")
 m["par_value_exchange_raw"]=pd.to_numeric(m["ParVal"],errors="coerce")
 c=company.copy();c["symbol"]=normsymbol(c["symbol"]);c["isin"]=normisin(c["isin"])
 c=c.merge(m[["symbol","isin","issued_security_count_NSE_raw",
              "par_value_exchange_raw"]],on=["symbol","isin"],how="left",validate="1:1")
 count=c["issued_security_count_NSE_raw"]
 close=pd.to_numeric(c["close"],errors="coerce")
 guess=count*close/1e7
 c["issued_security_count_proxy_market_cap_crore"]=guess.where(
  count.gt(0) & close.gt(0) & guess.between(.01,20000000))
 c["issued_security_count_proxy_is_independent_certified_market_cap"]=False
 return c
def run(source,asof,out):
 date=str(pd.Timestamp(asof).date())
 src=Path(source);c=pd.read_csv(src)
 if not {"date","symbol","isin","close","company_pe","company_pe_source_file_sha256"}.issubset(c):
  raise ValueError("Only original verified NSE P/E-and-two-source day match is acceptable")
 if not pd.to_datetime(c["date"]).dt.strftime("%Y-%m-%d").eq(date).all():
  raise ValueError("Wrong NSE stock date")
 member_url=HOST+INDEX_LIST_PATH
 b=get_bytes(member_url)
 if b is None:raise ValueError("NSE index constituent file missing")
 mem=pd.read_csv(io.BytesIO(b),encoding="utf-8-sig")
 if len(mem)<500:raise ValueError("Too few original NSE Total Market constituents")
 fields={"Symbol":"symbol","ISIN Code":"isin","Industry":"industry"}
 if not set(fields).issubset(mem):
  raise ValueError(f"Unrecognized NSE index member columns: {list(mem)}")
 m=mem[list(fields)].rename(columns=fields)
 x=peer_reference(c,m)
 master_url=HOST+f"/content/cm/NSE_CM_security_{pd.Timestamp(asof).strftime('%d%m%Y')}.csv.gz"
 raw=get_bytes(master_url)
 if raw is None or not raw.startswith(b"\x1f\x8b"):
  raise ValueError("NSE cash-market security master missing or not gzip")
 master=pd.read_csv(io.BytesIO(gzip.decompress(raw)),low_memory=False)
 x=security_issued_shares_estimate(master,x)
 dest=Path(out);dest.mkdir(parents=True,exist_ok=True)
 x.to_csv(dest/"official_NSE_industry_peer_and_issued_capital_proxies_RESEARCH.csv",index=False)
 cov={
  "scope":"NON_PIT_AFTER_CLOSE_NSE_INDUSTRY_MEDIAN_AND_ISSUE_SHARECOUNT_PROXIES",
  "date_of_original_closing_prices":date,
  "original_NSE_current_index_members_url":member_url,
  "current_index_list_bytes_sha256":hashlib.sha256(b).hexdigest(),
  "current_constituents_not_reconstructable_as_at_1530_original_date":True,
  "original_master_same_trade_date":master_url,
  "original_security_master_sha256":hashlib.sha256(raw).hexdigest(),
  "verified_NSE_individual_PE_joined_original_market_company_rows":len(x),
  "original_total_market_constituent_rows":len(mem),
  "companies_with_sector_membership_match":int(x["industry"].notna().sum()),
  "companies_with_peer_median_PE_proxy":int(x["industry_peer_median_PE_proxy"].notna().sum()),
  "companies_below_sector_peer_PE_proxy":int(x["company_PE_below_own_current_sector_peer_median_proxy"].fillna(False).sum()),
  "companies_with_issued_security_count_market_cap_proxy":int(x["issued_security_count_proxy_market_cap_crore"].notna().sum()),
  "verified_exchange_certified_market_cap":False,
  "verified_official_issuer_industry_PE":False,
  "no_model_rank_or_training_changed":True,
  "no_original_Oct8_historical_filing_rebuilt":True,
  "created_utc":datetime.now(timezone.utc).isoformat()}
 (dest/"nse_valuation_peer_size_proxy_coverage.json").write_text(json.dumps(cov,indent=2))
 print(json.dumps(cov,indent=2),flush=True)
 return cov
def main():
 p=argparse.ArgumentParser()
 p.add_argument("--source",required=True)
 p.add_argument("--asof",required=True)
 p.add_argument("--output",required=True)
 a=p.parse_args();run(a.source,a.asof,a.output)
if __name__=="__main__":main()
