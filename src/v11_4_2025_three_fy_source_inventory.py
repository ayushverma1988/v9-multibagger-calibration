"""Audit 2025 frozen historical stock universe for FY2023/24/25 three FY XBRL.

Original NSE sources only, no 2026 restated figures, no market labels, and no
misstatement that three yearly observations provide a 3-year CAGR.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd
from v11_4_fy2023_historical_annual_pilot import mode
from v11_4_strict_annual_numeric_features import fold_close
YEARS=(2023,2024,2025)
def original_2025_three_fy(catalog,snapshot):
 x=snapshot.copy()
 x["date"]=pd.to_datetime(x["date"],errors="raise").dt.normalize()
 x["symbol"]=x["symbol"].astype(str).str.upper().str.strip()
 dec=sorted(x.loc[(x["date"].dt.year==2025)&(x["date"].dt.month==12),"date"].unique())
 if len(dec)!=1:raise ValueError(f"Expected exactly one Dec2025 frozen historical fold, found {len(dec)}")
 date=pd.Timestamp(dec[0]);symbols=set(x.loc[x["date"].eq(date),"symbol"])
 if len(symbols)<500:raise ValueError("Insufficient original Dec2025 stock universe")
 c=catalog.copy()
 c["symbol"]=c["symbol"].astype(str).str.upper().str.strip()
 c["fy_end_ts"]=pd.to_datetime(c["fy_end"],utc=True,errors="coerce")
 c["published_at"]=pd.to_datetime(c["available_at_utc"],utc=True,errors="coerce",format="mixed")
 c["mode"]=c["consolidated"].map(mode)
 cut=fold_close(date)
 c=c[
  c["symbol"].isin(symbols)&
  c["fy_end_ts"].isin([pd.Timestamp(f"{yr}-03-31T00:00:00Z") for yr in YEARS])&
  c["published_at"].notna()&(c["published_at"]>=c["fy_end_ts"])&(c["published_at"]<=cut)&
  c["mode"].isin(["consolidated","standalone"])&
  c["xbrl_url"].str.match(r"^https://nsearchives\.nseindia\.com/[^?#]+\.xml$",case=False,na=False)&
  ~c["xbrl_url"].str.contains(r"/(?:INTEGRATED_FILING_)?BANKING_",case=False,regex=True,na=False)
 ].copy()
 c=c.sort_values(["published_at","xbrl_url"]).drop_duplicates(["symbol","mode","fy_end_ts"],keep="first")
 years={z:c[c["fy_end_ts"].dt.year.eq(z)] for z in YEARS}
 pairs={}
 for symbol,g in c.groupby("symbol"):
  modes=set(g["mode"])
  for m in ("consolidated","standalone"):
   if m in modes and set(g[g["mode"].eq(m)]["fy_end_ts"].dt.year)==set(YEARS):
    pairs[symbol]=m;break
 out=c[c.apply(lambda row:pairs.get(row["symbol"])==row["mode"],axis=1)].copy()
 if out.duplicated(["symbol","fy_end_ts"]).any():raise ValueError("Duplicate authoritative 2025 original XBRL")
 report={
  "scope":"ORIGINAL_NSE_2025_FOLD_FY2023_FY2024_FY2025_THREE_FINANCIAL_YEARS_LINKS_ONLY",
  "original_dec_2025_fold_IST":str(date.date()),"original_stock_universe_count":len(symbols),
  "FY2023_Y24_Y25_separate_filing_links":{str(y):len(years[y]) for y in YEARS},
  "companies_with_all_three_FY_source_URLs_published_before_2025_close":len(pairs),
  "all_three_FY_URL_coverage":len(pairs)/len(symbols),
  "strict_xml_numeric_extraction_not_done":True,
  "latest_2026_current_edition_financial_source_not_substituted":True,
  "source_short_history_growth_duration_is_two_intervals":True,
  "2025_external_four_family_financial_model_not_retrained":True
 }
 return out,report
def main():
 p=argparse.ArgumentParser()
 p.add_argument("--annual-index",required=True);p.add_argument("--snapshot",required=True);p.add_argument("--out",required=True)
 a=p.parse_args()
 result,report=original_2025_three_fy(pd.read_csv(a.annual_index,dtype=str).fillna(""),
                                     pd.read_parquet(a.snapshot,columns=["date","symbol"]))
 dest=Path(a.out);dest.mkdir(parents=True,exist_ok=True)
 result.to_csv(dest/"original_2025_fold_three_FY_2023_2025_PIT_urls_RESEARCH.csv",index=False)
 (dest/"original_2025_three_FY_XBRL_source_inventory.json").write_text(json.dumps(report,indent=2))
 print(json.dumps(report,indent=2))
if __name__=="__main__":main()
