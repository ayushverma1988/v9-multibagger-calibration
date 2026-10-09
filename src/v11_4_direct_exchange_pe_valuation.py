"""Original NSE direct-archive company P/E valuation companion (research only).

Read the actual NSE single-stock PE report, not the blocked web JSON API.
The separate index CSV is not a stock valuation substitute. Match the exact
original NSE trade date and independently verify it against TWO NSE official
bhavcopy feeds; preserve the original file SHA256 for reproducibility.
Do not assert market cap or industry comparisons without verified units and
publication timestamps. No updates to the frozen 22-predictor model.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import requests
from v11_4_nse_archive_valuation_probe import decode_tabular,HEADERS
from v11_4_live_nse_market_catalyst import exchange_primary_for_day

HOST="https://archives.nseindia.com"
def digest(raw):return hashlib.sha256(raw).hexdigest()
def parse_positive_pe(values):
 # NSE publishes non-numeric P/E for companies with losses/no coverage:
 # those are UNKNOWN, not a negative valuation pass.
 z=pd.to_numeric(pd.Series(values).astype(str).str.replace(",","",regex=False),errors="coerce")
 return z.where(z.gt(0)&z.lt(20000))
def normalize_company_pe(frame):
 required={"SYMBOL","SYMBOL P/E","ADJUSTED P/E"}
 if not required.issubset(frame.columns):
  raise ValueError("NSE securities P/E report lost original three-field company schema")
 x=frame[list(required)].copy()
 x.columns=[str(c).strip() for c in x.columns]
 x["symbol"]=x["SYMBOL"].astype(str).str.upper().str.strip()
 if x["symbol"].eq("").any() or x["symbol"].eq("NAN").any():
  x=x[~x["symbol"].isin(["","NAN"])].copy()
 if x["symbol"].duplicated().any():
  raise ValueError("Exchange P/E report has conflicting duplicate security symbol")
 x["company_pe"]=parse_positive_pe(x["SYMBOL P/E"]).to_numpy()
 x["company_pe_adjusted_exchange"]=parse_positive_pe(x["ADJUSTED P/E"]).to_numpy()
 if len(x)<1200:raise ValueError("Incomplete original NSE company P/E report")
 if x["company_pe"].notna().sum()<350:raise ValueError("Too few numeric positive securities P/E")
 return x[["symbol","company_pe","company_pe_adjusted_exchange"]]
def normalize_day_market(primary):
 # Official exchange day already checked against independent second bhavcopy.
 x=primary[primary["series"].isin(["EQ","BE","BZ"])].copy()
 # Prefer common EQ series, never share duplicate symbols with other series.
 x["series_priority"]=x["series"].map({"EQ":0,"BE":1,"BZ":2})
 x=x.sort_values(["symbol","series_priority"]).drop_duplicates("symbol")
 x=x[x["isin"].astype(str).str.startswith("INE")].copy()
 return x[["date","symbol","series","isin","close"]]
def run(date,out):
 target=pd.Timestamp(date).normalize()
 official,market_proof=exchange_primary_for_day(target)
 m=normalize_day_market(official)
 url=f"{HOST}/content/equities/peDetail/PE_{target.strftime('%d%m%y')}.csv"
 r=requests.get(url,headers=HEADERS,timeout=45)
 r.raise_for_status()
 original=r.content
 if len(original)<20000:raise ValueError("NSE company P/E file suspiciously small")
 exchange=decode_tabular(original,url)
 p=normalize_company_pe(exchange)
 full=m.merge(p,on="symbol",how="left",validate="1:1",indicator=True)
 both=int(full["_merge"].eq("both").sum())
 if both<500:raise ValueError("Insufficient same-day independently verified NSE market/P-E company match")
 full=full.drop(columns="_merge")
 full["company_pe_source_file_sha256"]=digest(original)
 full["company_pe_source_url"]=url
 # Diagnostic-only source fetched timestamp. DO NOT misstate that NSE
 # published the company P/E values by the 15:30 cash-market cutoff.
 full["pe_source_pit_at_1530_IST_verified"]=False
 full["is_sme_official_mainboard_series"]=False
 # Deliberately leave unverified competitor and capitalization values NULL.
 full["market_cap_crore"]=np.nan
 full["industry_pe_reference"]=np.nan
 full["pe_lt_industry_pe"]=pd.NA
 d=Path(out);d.mkdir(parents=True,exist_ok=True)
 full.to_csv(d/"nse_original_company_pe_same_day_research.csv",index=False)
 summary={
  "scope":"ORIGINAL_NSE_COMPANY_PE_DATE_JOIN_NON_PREDICTIVE_RESEARCH_OVERLAY",
  "asof_date_IST":str(target.date()),"official_company_pe_url":url,
  "official_company_pe_original_bytes_sha256":digest(original),
  "exact_day_bhavcopy_independently_cross_checked":bool(market_proof["official_primary_used"]),
  "independent_verified_market_rows":len(m),
  "original_company_PE_report_securities":len(p),
  "same_day_market_and_company_PE_matches":both,
  "individual_security_PE_usable":int(full["company_pe"].notna().sum()),
  "individual_security_PE_adjusted_usable":int(full["company_pe_adjusted_exchange"].notna().sum()),
  "source_market_validation":market_proof,
  "PE_data_publication_at_1530_IST_verified":False,
  "used_index_PE_as_individual_security_PE":False,
  "valid_market_capitalization":False,
  "valid_industry_peer_PE":False,
  "four_screen_full_condition_2_not_yet_validated":True,
  "model_train_or_rank_modified":False,"future_six_month_labels_used":False,
  "created_utc":datetime.now(timezone.utc).isoformat()}
 (d/"nse_direct_company_pe_coverage_audit.json").write_text(json.dumps(summary,indent=2))
 print(json.dumps(summary,indent=2),flush=True)
 return summary
def main():
 p=argparse.ArgumentParser()
 p.add_argument("--asof",required=True);p.add_argument("--output",required=True)
 a=p.parse_args();run(a.asof,a.output)
if __name__=="__main__":main()
