"""Historical Dec2024 NSE three-fiscal-year FY2022-24 strict numeric recovery.

Source-published-before original 2024-12-31 15:30 IST; strictly same mode,
original XML sha256 and 12-month revenue / PAT. No financial model fitting.
"""
from __future__ import annotations
import argparse,hashlib,json,math
from pathlib import Path
import pandas as pd
from v11_4_longterm_fundamentals import Client
from v11_4_recover_legacy_FY2022_FourD import extract_2022_legacy_fourd
from v11_4_strict_annual_numeric_features import strict_annual_facts,fold_close

FOLD="2024-12-31"
YEARS=(2022,2023,2024)

def extract_original_financials(index,out,shard=0,shards=1,limit=128):
 z=index.copy()
 keys={"symbol","mode","fy","pub","xbrl_url"}
 if not keys.issubset(z):raise ValueError("Missing original NSE 2024 financial source provenance")
 z["fy"]=pd.to_numeric(z["fy"],errors="raise").astype(int)
 if z.duplicated(["symbol","fy"]).any():raise ValueError("Duplicate FY of same company")
 groups=list(z.groupby("symbol",sort=True))
 if not 0<=shard<shards<=24 or not 1<=limit<=256:raise ValueError("Invalid original 2024 fiscal source shard")
 chosen=groups[shard::shards][:limit]
 if not chosen:raise ValueError("No filing company for shard")
 cut=fold_close(FOLD)
 root=Path(out);root.mkdir(parents=True,exist_ok=True)
 client=Client()
 ok=[];rejections=[]
 for num,(symbol,g) in enumerate(chosen,1):
  record={"symbol":symbol,"original_decision_date":FOLD,
          "original_reporting_mode":g["mode"].iloc[0],
          "annual_fiscal_years":"2022|2023|2024",
          "model_training_authorized":False}
  try:
   if len(g)!=3 or set(g["fy"])!=set(YEARS):raise ValueError("Missing exactly three consecutive fiscal annual years")
   if g["mode"].nunique()!=1:raise ValueError("Consolidated/standalone mode switched between years")
   metrics={}
   for r in g.itertuples(index=False):
    fy=int(r.fy);fyend=f"{fy}-03-31"
    pub=pd.to_datetime(r.pub,utc=True,errors="coerce")
    if pd.isna(pub) or pub>cut or pub<pd.Timestamp(fyend,tz="UTC"):
     raise ValueError("Future or invalid pre-decision original NSE fiscal source publication")
    if not str(r.xbrl_url).startswith("https://nsearchives.nseindia.com/") or not str(r.xbrl_url).lower().endswith(".xml"):
     raise ValueError("Nonoriginal NSE annual XML source")
    raw=client.get(str(r.xbrl_url)).content
    if fy==2022:
     values,audit=extract_2022_legacy_fourd(raw,fyend,g["mode"].iloc[0],source_is_annual=True)
    else:
     values,audit=strict_annual_facts(raw,fyend,source_is_annual=True)
    if any(values.get(k) is None or not math.isfinite(float(values[k])) for k in ("revenue","pat")):
     raise ValueError("Original NSE source does not contain both finite annual revenue and PAT")
    record[f"FY{fy}_revenue_INR"]=float(values["revenue"])
    record[f"FY{fy}_PAT_INR"]=float(values["pat"])
    record[f"FY{fy}_source_sha256"]=hashlib.sha256(raw).hexdigest()
    record[f"FY{fy}_source_url"]=str(r.xbrl_url)
    record[f"FY{fy}_published_utc"]=pub.isoformat()
    record[f"FY{fy}_annual_context_audit"]=json.dumps(audit,sort_keys=True)
    metrics[fy]=values
   revenue=[float(metrics[y]["revenue"]) for y in YEARS]
   profit=[float(metrics[y]["pat"]) for y in YEARS]
   if min(revenue)<=0:raise ValueError("Invalid nonpositive financial baseline")
   record["FY2024_vs_2023_revenue_yoy"]=revenue[2]/revenue[1]-1
   record["FY2023_vs_2022_revenue_yoy"]=revenue[1]/revenue[0]-1
   record["FY2022_to_FY2024_sales_two_year_CAGR"]=(revenue[2]/revenue[0])**.5-1
   record["FY2022_to_FY2024_PAT_two_year_CAGR"]=(profit[2]/profit[0])**.5-1 if profit[0]>0 and profit[2]>0 else None
   record["no_3year_CAGR_from_only_three_FY_observations"]=True
   ok.append(record)
  except Exception as ex:
   record["error"]=str(ex)[:220]
   rejections.append(record)
  if num%8==0:print("FY2024 original three-FY shard",shard,num,"/",len(chosen),"verified",len(ok),"rejected",len(rejections),flush=True)
 pd.DataFrame(ok).to_csv(root/"FY2024_original_NSE_three_FY_strict_numeric_2022_to_2024.csv",index=False)
 pd.DataFrame(rejections).to_csv(root/"FY2024_original_NSE_three_FY_source_rejected.csv",index=False)
 report={
  "scope":"V11_4_2024_FOLD_FY2022_2023_2024_NSE_NUMERIC_PIT",
  "asof_date":FOLD,"all_2024_source_candidates":len(groups),
  "shard":shard,"shards":shards,"attempted_companies":len(chosen),
  "three_fiscal_years_numeric_verified":len(ok),"original_source_rejected":len(rejections),
  "same_statement_mode":True,"original_source_filing_SHA256":True,
  "only_source_dates_before_original_decision":True,
  "two_year_sales_CAGR_calculable":len(ok),
  "two_year_profit_CAGR_calculable":sum(pd.notna(v["FY2022_to_FY2024_PAT_two_year_CAGR"]) for v in ok),
  "three_financial_years_is_two_growth_intervals":True,
  "no_forward_return_labels_read":True,"original_frozen_V11_4_model_not_changed":True
 }
 (root/"FY2024_original_three_FY_numeric_source_report.json").write_text(json.dumps(report,indent=2))
 print(json.dumps(report,indent=2),flush=True)
 return report
def main():
 p=argparse.ArgumentParser()
 p.add_argument("--index",required=True);p.add_argument("--out",required=True)
 p.add_argument("--shard",type=int,default=0);p.add_argument("--shards",type=int,default=1);p.add_argument("--limit",type=int,default=60)
 a=p.parse_args()
 extract_original_financials(pd.read_csv(a.index,dtype=str).fillna(""),a.out,a.shard,a.shards,a.limit)
if __name__=="__main__":main()
