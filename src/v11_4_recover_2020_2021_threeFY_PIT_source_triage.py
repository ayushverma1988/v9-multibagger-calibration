"""Prioritize genuinely *historically published* NSE 2020/21 three-FY gaps.

This is source INDEX TRIAGE, not historical numerical reconstruction, not
a model backtest, and never a promise that a late filing existed on a
historical decision date. Keep frozen company denominators and reporting
modes; confidential company-target source gap details go to PRIVATE archive.
"""
from __future__ import annotations
import argparse,json,math,re
from pathlib import Path
import pandas as pd
TARGETS={
 "2020-06-30":{"fys":(2018,2019,2020),"universe":705,"original_link_sets":393},
 "2020-12-31":{"fys":(2018,2019,2020),"universe":863,"original_link_sets":565},
 "2021-06-30":{"fys":(2019,2020,2021),"universe":1046,"original_link_sets":704},
 "2021-12-31":{"fys":(2019,2020,2021),"universe":1079,"original_link_sets":751}}
MODES=("consolidated","standalone")
def mode(raw):
 value=str(raw).lower()
 if "standalone" in value or "non-consolidated" in value:return "standalone"
 if "consolidated" in value:return "consolidated"
 return "unknown"
def start_cutoff(day):
 return (pd.Timestamp(day).tz_localize("Asia/Kolkata")
          +pd.Timedelta(hours=15,minutes=30)).tz_convert("UTC")
def triage(catalog,original):
 req={"symbol","fy_end","available_at_utc","consolidated","xbrl_url"}
 if not req.issubset(catalog):raise ValueError("Annual NSE index lacks official source fields")
 if not {"date","symbol"}.issubset(original):raise ValueError("Frozen market identity fields missing")
 if {"y6","y12","y24","y6_mature_date","dd30_6m"}&set(original):
  raise ValueError("No future stock prices or outcomes in source triage")
 stock=original.copy()
 stock["date"]=pd.to_datetime(stock["date"],errors="raise").dt.strftime("%Y-%m-%d")
 stock["symbol"]=stock["symbol"].astype(str).str.upper().str.strip()
 stock=stock.loc[stock["date"].isin(TARGETS)].copy()
 if stock[["date","symbol"]].duplicated().any():
  raise ValueError("Duplicate original market security")
 z=catalog.copy()
 z["symbol"]=z["symbol"].astype(str).str.upper().str.strip()
 z["fy"]=pd.to_datetime(z["fy_end"],errors="coerce",utc=True,format="mixed")
 z["published"]=pd.to_datetime(z["available_at_utc"],errors="coerce",utc=True,format="mixed")
 z["mode"]=z["consolidated"].map(mode)
 z["link"]=z["xbrl_url"].astype(str)
 z["source_valid"]=(
   z["link"].str.match(r"^https://nsearchives\.nseindia\.com/[^?#]+\.xml$",case=False,na=False)&
   ~z["link"].str.contains(r"/(?:INTEGRATED_FILING_)?BANKING_",case=False,regex=True,na=False)&
   z["mode"].isin(MODES)&z["published"].notna()&z["fy"].notna()&
   z["published"].ge(z["fy"]))
 # Invalid source rows do not become eligible after weakening provenance.
 z=z.loc[z["source_valid"]].copy()
 results=[];missing=[]
 for day,meta in TARGETS.items():
  cut=start_cutoff(day);years=set(meta["fys"])
  f=stock.loc[stock["date"].eq(day),"symbol"].tolist()
  if len(f)!=meta["universe"]:
   raise ValueError(f"Original historical {day} market denominator drift")
  yrs={pd.Timestamp(f"{yr}-03-31T00:00:00Z") for yr in years}
  fset=set(f)
  docs=z.loc[z["symbol"].isin(fset)&z["fy"].isin(yrs)].copy()
  available=docs.loc[docs["published"].le(cut)].copy()
  # Original 18fold proof always uses earliest eligible source per company/mode/FY.
  available=(available.sort_values(["published","link"])
                     .drop_duplicates(["symbol","mode","fy"],keep="first"))
  c={}
  for (symbol,statement_mode),part in available.groupby(["symbol","mode"]):
   c.setdefault(symbol,{})[statement_mode]=set(part["fy"].dt.year)
  valid={sym for sym in f if any(c.get(sym,{}).get(m,set())==years for m in MODES)}
  if len(valid)!=meta["original_link_sets"]:
   raise ValueError(f"{day}: index candidate count changed: {len(valid)} != {meta['original_link_sets']}")
  status={}
  records=[]
  for sym in sorted(fset-valid):
   modes=c.get(sym,{})
   best=max((len(modes.get(m,set())) for m in MODES),default=0)
   combined=set().union(*[modes.get(m,set()) for m in MODES])
   after=docs.loc[docs["symbol"].eq(sym)&docs["published"].gt(cut)]
   if best==2:reason="ONLY_TWO_EXACT_FYS_IN_SAME_MODE_ASOF_CLOSE"
   elif best==1:reason="ONLY_ONE_EXACT_FY_IN_SAME_MODE_ASOF_CLOSE"
   elif len(combined)==3:reason="ALL_FYS_PRESENT_ONLY_MIXED_MODES_REJECT"
   else:reason="NO_OR_INSUFFICIENT_ORIGINAL_THREEFY_LINKS_ASOF_CLOSE"
   later=bool(len(after)>0)
   records.append({"date":day,"symbol":sym,
    "missing_original_FY_under_same_mode":[y for y in sorted(years)
        if y not in max((modes.get(mode,set()) for mode in MODES),key=len,default=set())],
    "asof_best_same_mode_years":best,
    "failure_reason":reason,
    "official_NSE_FY_filing_first_found_only_after_decision":later,
    "later_docs_not_usable_for_historical_selection":True,
    "NSE_alternate_annual_report_XBRL_numerics_not_yet_verified":True})
   status[reason]=status.get(reason,0)+1
  missing.extend(records)
  needed=max(0,math.ceil(0.70*len(f)-1e-10)-len(valid))
  gap_now=len(f)-len(valid)
  count_late=sum(1 for row in records if row["official_NSE_FY_filing_first_found_only_after_decision"])
  results.append({"original_fold":day,
   "original_unchanged_stock_universe":len(f),
   "original_eligible_same_mode_threeFY_NSE_XML_link_sets":len(valid),
   "original_link_coverage_percent":round(len(valid)/len(f)*100,2),
   "remaining_original_annual_link_gap":gap_now,
   "additional_strict_same_mode_preclose_original_annual_link_sets_needed_for_70pct":needed,
   "link_source_coverage_met_70pct":len(valid)/len(f)>=.70,
   "missing_companies_with_later_dated_official_NSE_filing_candidate_NON_ELIGIBLE":count_late,
   "missing_original_annual_source_reason_counts":status,
   "fiscal_years":sorted(years),
   "some_2020_market_dates_span_COVID_period_and_not_new_production_training":day.startswith("2020"),
   "possible_NSE_annual_report_XBRL_or_BSE_official_issuer_backfill_not_yet_examined":True,
   "threeFY_revenue_PAT_numeric_verified_from_new_old_document":False})
 report={"scope":"PIT_2020_2021_ORIGINAL_NSE_ANNUAL_3FY_SOURCE_DEFICIT_TRIAGE_NOT_RECONSTRUCTED",
  "four_frozen_historical_pre2022_folds":4,
  "original_exchange_market_snapshots":sum(x["original_unchanged_stock_universe"] for x in results),
  "source_link_matched_3FY_company_fold_sets":sum(x["original_eligible_same_mode_threeFY_NSE_XML_link_sets"] for x in results),
  "minimum_additional_3FY_original_same_mode_XML_sources_for_70pct_each_fold":sum(
   x["additional_strict_same_mode_preclose_original_annual_link_sets_needed_for_70pct"] for x in results),
  "source_link_percent_not_equivalent_to_numeric_coverage":True,
  "no_futures_no_2025_outcomes_no_model_training":True,
  "historical_publication_clock_and_issuer_mode_not_weakened":True,
  "2020_research_dates_not_promoted_to_production":True,
  "per_date":results}
 return report,pd.DataFrame(missing)
def main():
 p=argparse.ArgumentParser()
 p.add_argument("--official-annual-index",required=True)
 p.add_argument("--frozen-snapshot",required=True)
 p.add_argument("--out",required=True)
 a=p.parse_args()
 annual=pd.read_csv(a.official_annual_index,dtype=str).fillna("")
 frozen=pd.read_parquet(a.frozen_snapshot,columns=["date","symbol"])
 result,missing=triage(annual,frozen)
 folder=Path(a.out);folder.mkdir(parents=True,exist_ok=True)
 missing.to_parquet(folder/"PRIVATE_original_2020_2021_missing_3FY_filing_link_candidates.parquet",index=False)
 (folder/"2020_2021_original_NSE_threeFY_missing_preclose_source_triage_aggregate.json").write_text(json.dumps(result,indent=2))
 print(json.dumps(result,indent=2),flush=True)
if __name__=="__main__":main()
