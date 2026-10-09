"""Original NSE three-financial-year PIT index audit; research only.

Three fiscal annual points FY2021/22/23 support two successive YoY changes
and two-year CAGR, NOT a 3-year CAGR. The latter needs FY2020 too.
No latest website ratio used as a 2023 historical feature.
"""
from __future__ import annotations
import argparse,json,re
from pathlib import Path
import pandas as pd
from v11_4_fy2023_historical_annual_pilot import select_2023_fold,mode
from v11_4_strict_annual_numeric_features import fold_close
FY=(2021,2022,2023)
HOST=r"^https://nsearchives\.nseindia\.com/[^?#]+\.xml$"

def candidates(index,snapshot,fy=FY):
 fold,universe=select_2023_fold(snapshot)
 z=index.copy()
 if not {"symbol","fy_end","available_at_utc","consolidated","xbrl_url"}.issubset(z):
  raise ValueError("Missing original historical NSE financial filing catalog fields")
 z["symbol"]=z["symbol"].astype(str).str.upper().str.strip()
 z["end"]=pd.to_datetime(z["fy_end"],utc=True,errors="coerce")
 z["pub"]=pd.to_datetime(z["available_at_utc"],utc=True,errors="coerce",format="mixed")
 z["mode"]=z["consolidated"].map(mode)
 close=fold_close(fold)
 good=z["xbrl_url"].str.match(HOST,case=False,na=False)
 nonbank=~z["xbrl_url"].str.contains(r"/(?:INTEGRATED_FILING_)?BANKING_",case=False,regex=True,na=False)
 fiscal=[pd.Timestamp(f"{year}-03-31T00:00:00Z") for year in fy]
 z=z[z["symbol"].isin(universe)&z["end"].isin(fiscal)&
     z["pub"].notna()&(z["pub"]>=z["end"])&(z["pub"]<=close)&
     z["mode"].isin(["consolidated","standalone"])&good&nonbank].copy()
 z=z.sort_values(["pub","xbrl_url"]).drop_duplicates(["symbol","mode","end"],keep="first")
 found={}
 counts={}
 for year in fy:
  counts[str(year)]=int(z["end"].eq(pd.Timestamp(f"{year}-03-31T00:00:00Z")).sum())
 sets={m:{} for m in ("consolidated","standalone")}
 for m,g in z.groupby("mode"):
  for symbol,t in g.groupby("symbol"):
   years={x.year for x in t["end"].dt.date}
   sets[m][symbol]=years
 unique={}
 for symbol in sorted(universe):
  matches=[]
  for m in ("consolidated","standalone"):
   years=sets[m].get(symbol,set())
   if set(fy).issubset(years):matches.append(m)
  if matches:unique[symbol]=matches[0]
 two=sum(any({fy[-2],fy[-1]}.issubset(sets[m].get(s,set())) for m in sets) for s in universe)
 report={
   "scope":"2023_FOLD_THREE_FINANCIAL_YEARS_ORIGINAL_NSE_INDEX_NOT_MODEL_TRAINING",
   "historical_decision_date_IST":fold,
   "original_stocks":len(universe),
   "FY2021_FY2022_FY2023_publicly_available_asof_2023_close":counts,
   "original_FY2022_FY2023_source_pairs":int(two),
   "original_three_consecutive_FY_source_URL_sets":len(unique),
   "three_FY_source_URL_coverage_of_original_universe":len(unique)/len(universe),
   "allows_two_annual_yoy_changes":True,
   "allows_2y_cagr_from_FY2021_to_FY2023":True,
   "allows_true_3y_sales_cagr":False,
   "true_3y_cagr_requires_fy2020":True,
   "allows_3y_roe_avg_without_balance_sheet_three_years":False,
   "strict_xml_numeric_verified_at_index_stage":False,
   "no_future_financial_publication":True,
   "no_stale_2026_website_ratios_as_historical_training":True}
 selected=z[z.apply(lambda r:unique.get(r["symbol"])==r["mode"],axis=1)].copy()
 selected=selected[selected["symbol"].isin(unique)].copy()
 selected["fy"]=selected["end"].dt.year
 if selected.duplicated(["symbol","fy"]).any():
  raise ValueError("Multiple source annual reports per original company/fy")
 return selected,report

def run(index,snapshot,out):
 source,report=candidates(index,snapshot)
 target=Path(out);target.mkdir(parents=True,exist_ok=True)
 cols=["symbol","mode","fy","pub","xbrl_url"]
 source[cols].to_csv(target/"three_FY_original_NSE_2021_2022_2023_index_RESEARCH_ONLY.csv",index=False)
 (target/"three_FY_original_NSE_PIT_source_inventory.json").write_text(json.dumps(report,indent=2))
 print(json.dumps(report,indent=2),flush=True)
 return report

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--annual-index",required=True);p.add_argument("--snapshot",required=True);p.add_argument("--out",required=True)
 a=p.parse_args()
 run(pd.read_csv(a.annual_index,dtype=str).fillna(""),pd.read_parquet(a.snapshot,columns=["date","symbol"]),a.out)
if __name__=="__main__":main()
