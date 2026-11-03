"""Inventory full 18 independent decision dates for ORIGINAL 3-FY NSE source coverage.

Financial XML links are not numeric extraction or predictive training.
Data must predate stock snapshot at 15:30 IST, same company and reporting mode.
The 2025 Integrated March Q4 FourD *index* is a source candidate not proof
every document contains full-year FY2025 values. Covid folds reported but not
promoted for user's requested non-Covid exploration; unchanged dates remain.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd
from v11_4_fy2023_historical_annual_pilot import mode
from v11_4_strict_annual_numeric_features import fold_close

def eligible_three_fy(filings, snapshot, dates):
 needed={"symbol","fy_end","available_at_utc","consolidated","xbrl_url"}
 if not needed.issubset(filings):raise ValueError("Original NSE annual filing index missing fields")
 if not {"date","symbol"}.issubset(snapshot):raise ValueError("Missing historical original market identities")
 folds=pd.to_datetime(dates["date"],errors="raise").dt.normalize()
 if len(folds)!=18 or folds.duplicated().any():raise ValueError("Expected exactly 18 distinct preregistered historical dates")
 stock=snapshot.copy();stock["date"]=pd.to_datetime(stock["date"],errors="raise").dt.normalize()
 stock["symbol"]=stock["symbol"].astype(str).str.upper().str.strip()
 if stock[["date","symbol"]].duplicated().any():raise ValueError("Repeated historical security observation")
 z=filings.copy()
 z["symbol"]=z["symbol"].astype(str).str.upper().str.strip()
 z["fy"]=pd.to_datetime(z["fy_end"],errors="coerce",utc=True,format="mixed")
 z["pub"]=pd.to_datetime(z["available_at_utc"],errors="coerce",utc=True,format="mixed")
 z["mode"]=z["consolidated"].map(mode)
 good=z["xbrl_url"].astype(str).str.match(r"^https://nsearchives\.nseindia\.com/[^?#]+\.xml$",case=False,na=False)
 bank=z["xbrl_url"].astype(str).str.contains(r"/(?:INTEGRATED_FILING_)?BANKING_",case=False,regex=True,na=False)
 z=z[good&~bank&z["mode"].isin(["consolidated","standalone"])&
     z["pub"].notna()&z["fy"].notna()&(z["pub"]>=z["fy"])].copy()
 z=z.sort_values(["pub","xbrl_url"]).drop_duplicates(["symbol","mode","fy"],keep="first")
 reports=[];chosen=[]
 for day in folds:
  yr=int(day.year)
  years=(yr-2,yr-1,yr)
  universe=set(stock.loc[stock["date"].eq(day),"symbol"])
  if len(universe)<300:
   raise ValueError(f"Historical frozen stock universe inadequate {day.date()}: {len(universe)}")
  cut=fold_close(str(day.date()))
  fy=[pd.Timestamp(f"{y}-03-31T00:00:00Z") for y in years]
  x=z[z["symbol"].isin(universe)&z["fy"].isin(fy)&z["pub"].le(cut)].copy()
  valid={}
  for sym,g in x.groupby("symbol"):
   for m in ("consolidated","standalone"):
    if set(g.loc[g["mode"].eq(m),"fy"].dt.year)==set(years):
     valid[sym]=m;break
  selected=x.loc[x["symbol"].map(valid).eq(x["mode"])].copy()
  if selected.duplicated(["symbol","fy"]).any() or len(selected)!=len(valid)*3:
   raise ValueError("Original company fiscal-year pairing drift")
  selected["fold"]=str(day.date())
  selected["fiscal_year"]=selected["fy"].dt.year
  chosen.append(selected[["fold","symbol","mode","fiscal_year","pub","xbrl_url"]])
  reports.append({
   "historical_fold":str(day.date()),
   "original_stock_universe":len(universe),
   "three_consecutive_fiscal_years":"|".join(map(str,years)),
   "source_filing_link_sets_before_fold_close":len(valid),
   "point_in_time_source_url_coverage_pct":round(100*len(valid)/len(universe),2),
   "three_fy_2025_integrated_candidate_flag":yr==2025,
   "covid_period_2020_included_for_coverage_diagnostics_only":day.year==2020,
   "source_numeric_parser_has_not_verified_all_links":True,
   "matured_six_month_labels_not_yet_validated":True})
 return pd.DataFrame(reports),pd.concat(chosen,ignore_index=True)
def main():
 p=argparse.ArgumentParser()
 p.add_argument("--annual-index",required=True);p.add_argument("--snapshot",required=True)
 p.add_argument("--folds",required=True);p.add_argument("--out",required=True)
 a=p.parse_args()
 report,listing=eligible_three_fy(pd.read_csv(a.annual_index,dtype=str).fillna(""),
  pd.read_parquet(a.snapshot,columns=["date","symbol"]),pd.read_csv(a.folds))
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 report.to_csv(out/"all_18_folds_PIT_original_three_fiscal_year_source_inventory.csv",index=False)
 listing.to_csv(out/"all_18_folds_orig_NSE_3FY_paired_source_links_RESEARCH_ONLY.csv",index=False)
 summary={
 "scope":"ORIGINAL_NSE_THREE_FISCAL_YEAR_PIT_URL_COVERAGE_ALL_18_FOLDS",
 "original_18_preregistered_folds_count":len(report),
 "noncovid_folds_with_70pct_original_source_URL_coverage":int(report.loc[~report["historical_fold"].str.startswith("2020"),"point_in_time_source_url_coverage_pct"].ge(70).sum()),
 "all_folds_with_70pct_URL_coverage":int(report["point_in_time_source_url_coverage_pct"].ge(70).sum()),
 "earliest_fiscal_link_year":int(listing["fiscal_year"].min()) if len(listing) else None,
 "later_year_full_numeric_verification_not_implicit":True,
 "historical_original_past_covid_folds_not_removed_from_market_identity":True,
 "no_model_fit_no_outcome_labels_no_calibration":True,
 "only_two_year_CAGR_possible_from_three_fiscal_year_endpoints":True}
 (out/"all_18_folds_3FY_original_NSE_PIT_source_feasibility_summary.json").write_text(json.dumps(summary,indent=2))
 print(json.dumps(summary,indent=2),flush=True)
 print(report.to_string(index=False),flush=True)
if __name__=="__main__":main()
