"""V11.4 three-fiscal-year official NSE FY2022/23/24 link inventory.

Historical 2024-12-31 stock universe only, original NSE XML, consistent mode,
record available before 15:30 IST. Fiscal annual 2022,2023,2024 is only two
successive annual growth intervals; NOT a true 3-year CAGR.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd
from v11_4_fy2023_historical_annual_pilot import mode
from v11_4_strict_annual_numeric_features import fold_close

YEARS=(2022,2023,2024)
FOLD="2024-12-31"

def three_fy_2024_index(catalog,snapshot):
 original=snapshot.copy()
 original["date"]=pd.to_datetime(original["date"],errors="raise").dt.normalize()
 original["symbol"]=original["symbol"].astype(str).str.upper().str.strip()
 if original[["date","symbol"]].duplicated().any():raise ValueError("Original historic stock universe has duplicates")
 universe=set(original.loc[original["date"].eq(pd.Timestamp(FOLD)),"symbol"])
 if len(universe)<500:raise ValueError("No original Dec 2024 trading universe")
 z=catalog.copy()
 fields={"symbol","fy_end","available_at_utc","consolidated","xbrl_url"}
 if not fields.issubset(z):raise ValueError("Incomplete original NSE annual filing index")
 z["symbol"]=z["symbol"].astype(str).str.upper().str.strip()
 z["period"]=pd.to_datetime(z["fy_end"],utc=True,errors="coerce",format="mixed")
 z["pub"]=pd.to_datetime(z["available_at_utc"],utc=True,errors="coerce",format="mixed")
 z["mode"]=z["consolidated"].map(mode)
 cutoff=fold_close(FOLD)
 fiscal=[pd.Timestamp(f"{y}-03-31T00:00:00Z") for y in YEARS]
 z=z[z["symbol"].isin(universe)&z["period"].isin(fiscal)&
   z["pub"].notna()&(z["pub"]>=z["period"])&(z["pub"]<=cutoff)&
   z["mode"].isin(("consolidated","standalone"))&
   z["xbrl_url"].str.match(r"^https://nsearchives\.nseindia\.com/[^?#]+\.xml$",case=False,na=False)&
   ~z["xbrl_url"].str.contains(r"/(?:INTEGRATED_FILING_)?BANKING_",case=False,regex=True,na=False)].copy()
 z=z.sort_values(["pub","xbrl_url"]).drop_duplicates(["symbol","mode","period"],keep="first")
 group={}
 for sym,g in z.groupby("symbol"):
  for statement in ("consolidated","standalone"):
   sub=g[g["mode"].eq(statement)]
   if set(sub["period"].dt.year)==set(YEARS):
    group[sym]=statement
    break
 complete=z[z.apply(lambda r:group.get(r["symbol"])==r["mode"],axis=1)].copy()
 complete["fy"]=complete["period"].dt.year
 if complete.duplicated(["symbol","fy"]).any() or len(complete)!=3*len(group):
  raise ValueError("Three-year matched original issuer fiscal identity broken")
 report={
  "scope":"V11_4_ORIGINAL_2024_FOLD_NSE_FY2022_2023_2024_ANNUAL_FILINGS_INVENTORY",
  "original_historical_date_IST":FOLD,"original_stock_universe":len(universe),
  "original_2022_2023_2024_source_links":{str(y):int(z["period"].dt.year.eq(y).sum()) for y in YEARS},
  "companies_with_all_three_original_annual_source_links":len(group),
  "three_FY_original_filing_URL_coverage_fraction":len(group)/len(universe),
  "no_file_after_2024_decision_cutoff":True,
  "no_mixed_standalone_consolidated":True,"not_yet_numeric_extracted":True,
  "two_year_CAGR_not_three_year":True,"model_training_approved":False}
 return complete,report

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--annual-index",required=True);p.add_argument("--snapshot",required=True);p.add_argument("--out",required=True)
 a=p.parse_args()
 idx,r=three_fy_2024_index(pd.read_csv(a.annual_index,dtype=str).fillna(""),
                            pd.read_parquet(a.snapshot,columns=["date","symbol"]))
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 idx[["symbol","mode","fy","pub","xbrl_url"]].to_csv(out/"FY2022_FY2023_FY2024_original_asof_2024_three_filing_index.csv",index=False)
 (out/"FY2024_three_fiscal_year_source_inventory.json").write_text(json.dumps(r,indent=2))
 print(json.dumps(r,indent=2),flush=True)
if __name__=="__main__":main()
