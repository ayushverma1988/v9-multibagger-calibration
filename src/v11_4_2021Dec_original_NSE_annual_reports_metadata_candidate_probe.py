"""Investigate old missing FY2019/20/21 via ORIGINAL NSE annual-report metadata API.

Pre-registered company sample is selected only by whether the official 2021
Dec NSE financial XBRL index lacked one same-mode FY (never by returns or
winners). NSE annual-report PDF/ZIP metadata is another source *candidate*,
not audited 3FY INR numeric revenue/PAT. Reject documents disseminated
after December 31, 2021 15:30 IST. No present-day market fundamentals,
no 2025 outcomes, no new calibrated probabilities, no claims of 70% repair.
"""
from __future__ import annotations
import argparse,hashlib,json,math,re,urllib.parse
from pathlib import Path
import pandas as pd
import requests

DATE="2021-12-31"
CUTOFF=pd.Timestamp("2021-12-31T10:00:00Z")
HOST="https://www.nseindia.com"
URL=HOST+"/api/annual-reports"
REQUIRED={"date","symbol","missing_original_FY_under_same_mode",
 "asof_best_same_mode_years","failure_reason"}
APPROVED_HOST="nsearchives.nseindia.com"
SECTOR="equities"

def timeval(a):
 if a is None or str(a).strip() in ("","None","nan","NaT"):return pd.NaT
 try:q=pd.Timestamp(a)
 except (TypeError,ValueError):return pd.NaT
 if pd.isna(q):return pd.NaT
 if q.tzinfo is None:q=q.tz_localize("Asia/Kolkata")
 return q.tz_convert("UTC")
def strict_original_reports(records,symbol,needed_fys):
 """Return only original NSE annual document links broadcast before 2021-12-31."""
 if isinstance(records,dict):records=records.get("data",records.get("reports",[]))
 if not isinstance(records,list):raise ValueError("Unknown NSE original annual reports API payload")
 found=[];late=0;wrongyear=0;bad=0
 for v in records:
  if not isinstance(v,dict):continue
  try:fy=int(float(v.get("toYr")))
  except (TypeError,ValueError):bad+=1;continue
  if fy not in needed_fys:wrongyear+=1;continue
  start=v.get("fromYr")
  if start is not None:
   try:
    if int(float(start))!=fy-1:
     bad+=1;continue
   except (TypeError,ValueError):
    bad+=1;continue
  # The exchange may publish and disseminate at different clocks.
  stamps=[timeval(v.get(k)) for k in ("broadcast_dttm","disseminationDateTime")]
  if any(pd.isna(t) for t in stamps):
   bad+=1;continue
  available=max(stamps)
  if available>CUTOFF:late+=1;continue
  link=str(v.get("fileName","")).strip()
  if link.startswith("/"):link=urllib.parse.urljoin(HOST,link)
  parse=urllib.parse.urlparse(link)
  if parse.scheme!="https" or parse.hostname!=APPROVED_HOST or not re.search(r"\.(pdf|zip|xml)$",parse.path,re.I):
   bad+=1;continue
  found.append({"date":DATE,"symbol":symbol,"fiscal_year":fy,
   "official_NSE_annual_report_original_asof_URL":link,
   "broadcast_or_dissemination_AVAILABLE_UTC":available.isoformat(),
   "original_NSE_annual_report_reporting_mode_numeric_verified":False,
   "original_NSE_annual_report_revenue_PAT_INR_numeric_verified":False,
   "original_year_end_is_actual_March31_assumed_only_cannot_claim_numeric":True})
 return found,{"returned_raw_source_records":len(records),
   "target_FY_late_publication_rejected":late,
   "other_FY_ignored":wrongyear,
   "invalid_PIT_metadata_or_host_rejected":bad}

def inspect(original_missing,limit=48,session=None):
 if limit<1 or limit>96:raise ValueError("Conservative original NSE historical source API sample size 1..96 only")
 if not REQUIRED.issubset(original_missing):raise ValueError("Original private missing-FY source worklist incomplete")
 if {"y6","y12","y24","dd30_6m","y6_mature_date"}&set(original_missing):
  raise ValueError("Six-month winners contaminated source-only missing filing list")
 a=original_missing.loc[
   original_missing["date"].eq(DATE)&
   original_missing["asof_best_same_mode_years"].eq(2)&
   original_missing["failure_reason"].eq("ONLY_TWO_EXACT_FYS_IN_SAME_MODE_ASOF_CLOSE")].copy()
 if a["symbol"].duplicated().any() or len(a)<20:
  raise ValueError("Original historical December 2021 one-missing-FY candidate population wrong")
 a["order_hash"]=a["symbol"].astype(str).map(
   lambda z:hashlib.sha256(z.encode("utf-8")).hexdigest())
 a=a.sort_values(["order_hash","symbol"]).head(limit)
 agent=session or requests.Session()
 headers={"User-Agent":"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/129.0.0.0 Safari/537.36",
          "Accept":"application/json,text/plain,*/*","Accept-Language":"en-US,en;q=0.9",
          "Referer":"https://www.nseindia.com/companies-listing/corporate-filings-annual-reports"}
 samples=[];events=[];failures=[]
 try:
  agent.get(HOST+"/companies-listing/corporate-filings-annual-reports",
    headers=headers,timeout=30)
 except requests.RequestException:
  pass
 for i,row in enumerate(a.to_dict("records"),1):
  symbol=str(row["symbol"]).upper()
  fy=row["missing_original_FY_under_same_mode"]
  if hasattr(fy,"tolist"):fy=fy.tolist()
  if not isinstance(fy,(list,tuple)):raise ValueError("Missing exact annual financial year not parsed")
  missing_fys={int(year) for year in fy}
  if not missing_fys or not missing_fys.issubset({2019,2020,2021}):
   raise ValueError("Unknown exact missing historical annual FY for frozen 2021-12-31")
  source_meta={"date":DATE,"symbol":symbol,"original_missing_fiscal_years":sorted(missing_fys),
    "historical_input_Dec2021_NSE_fiscal_annual_links_verified":False,
    "NSE_alternate_annual_source_metadata_checked":False,
    "annual_statement_reporting_mode_not_yet_verified":True,
    "no_actual_new_3FY_annual_numbers_confirmed":True}
  try:
   reply=agent.get(URL,params={"index":SECTOR,"symbol":symbol},headers=headers,timeout=30)
   reply.raise_for_status()
   records=reply.json()
   accepted,audit=strict_original_reports(records,symbol,missing_fys)
   samples.extend(accepted)
   source_meta.update(audit)
   source_meta["NSE_alternate_annual_source_metadata_checked"]=True
   source_meta["historical_ASOF_report_links_found_for_missing_FY"]=len(set(x["fiscal_year"] for x in accepted))
  except (requests.RequestException,ValueError,TypeError) as exc:
   source_meta.update({"metadata_API_verification_error_type":type(exc).__name__,
                       "metadata_API_verification_error_excerpt":str(exc)[:120]})
   failures.append({"symbol":symbol,"error_type":type(exc).__name__})
  events.append(source_meta)
  if i%12==0:
   print("HISTORICAL_NSE_ANNUAL_REPORT_METADATA_AUDIT",
    json.dumps({"company_records_checked":i,
       "official_pre_Dec2021_other_years_NOT_used":True,
       "company_count_with_official_asof_missing_FY_annual_metadata":len(set(x["symbol"] for x in samples)),
       "source_api_errors_to_date":len(failures),
       "new_original_threeFY_numeric_recovered":0}),flush=True)
 accepted={z["symbol"] for z in samples}
 report={"scope":"SOURCE_ONLY_PIT_NSE_2021_DEC_ANNUAL_REPORT_METADATA_ALTERNATE_3FY_ONE_MISSING_FY_PILOT",
  "historical_original_frozen_fold":DATE,
  "max_sample_48_deterministic_missing_annual_NSE_XBRL_issuers_not_selected_by_past_returns":len(a),
  "original_2021_Dec_3FY_NSE_financial_XML_link_sets_without_reconstruction":751,
  "original_2021_Dec_total_frozen_equities":1079,
  "needed_additional_genuinely_usable_full_3FY_company_sources_to_70pct":5,
  "new_original_3FY_numeric_company_years_verified":0,
  "annual_report_metadata_issuer_FY_candidates_available_before_historical_asof":len(accepted),
  "annual_report_metadata_original_FY_source_link_count":len(samples),
  "original_annual_report_candidate_can_be_PDF_not_same_mode_3FY_facts":True,
  "annual_report_original_source_links_still_need_document_and_revenue_PAT_verification":True,
  "access_errors_count":len(failures),
  "NSE_annual_report_original_API":URL,
  "new_original_3FY_scientifically_validated_2021_Dec_fold":False,
  "no_original_2025_six_month_outcomes_or_current_stock_picker_read":True,
  "original_frozen_2026_V11_4_model_rankings_unchanged":True}
 return report,pd.DataFrame(events),pd.DataFrame(samples),pd.DataFrame(failures)

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--private-missing-original",required=True)
 p.add_argument("--max-companies",type=int,default=48)
 p.add_argument("--out",required=True)
 a=p.parse_args()
 raw=pd.read_parquet(a.private_missing_original)
 result,events,found,errors=inspect(raw,a.max_companies)
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 events.to_parquet(out/"PRIVATE_2021Dec_missing_FY_NSE_annual_reports_metadata_status.parquet",index=False)
 found.to_parquet(out/"PRIVATE_2021Dec_valid_preclose_missingFY_ANNUALREPORT_links_cANDIDATE_notnumeric.parquet",index=False)
 errors.to_csv(out/"PRIVATE_NSE_annual_report_API_error_metadata.csv",index=False)
 (out/"2021Dec_NSE_original_asof_old_annualreport_api_metadata_pilot_aggregate.json").write_text(json.dumps(result,indent=2))
 print(json.dumps(result,indent=2),flush=True)
if __name__=="__main__":main()
