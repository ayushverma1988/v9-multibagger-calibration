"""Strict original FY2021 FY2022 FY2023 NSE three-annual-year numeric pilot.

No current Screener figures, no lookahead, no arbitrary 3Y CAGR from 3
observed points. As-of 2023-12-29 15:30 IST; same original financial
statement mode and independently hashed NSE source for every fiscal year.
"""
from __future__ import annotations
import argparse,hashlib,json,math
from pathlib import Path
import pandas as pd
from v11_4_longterm_fundamentals import Client
from v11_4_recover_legacy_FY2022_FourD import extract_2022_legacy_fourd
from v11_4_strict_annual_numeric_features import strict_annual_facts,fold_close
FOLD="2023-12-29"
FY=(2021,2022,2023)
def strict_three_annual(source,limit,out):
 z=source.copy()
 required={"symbol","mode","fy","pub","xbrl_url"}
 if not required.issubset(z):raise ValueError("Original 3-year source index missing fields")
 z["fy"]=pd.to_numeric(z["fy"],errors="raise").astype(int)
 if z.duplicated(["symbol","fy"]).any():raise ValueError("Duplicate fiscal observation")
 group=list(z.groupby("symbol",sort=True))
 if limit<1 or limit>256:raise ValueError("Bound study sample to 1..256 companies")
 selected=group[:limit]
 if not selected:raise ValueError("Empty three-year source index")
 out=Path(out);out.mkdir(parents=True,exist_ok=True)
 client=Client()
 approved=[];rejected=[]
 cutoff=fold_close(FOLD)
 for i,(sym,t) in enumerate(selected,1):
  record={"symbol":sym,"asof_IST":FOLD,"three_fiscal_years":"2021|2022|2023",
          "source_filing_mode":str(t["mode"].iloc[0]),"never_overwrite_frozen_rank":True}
  try:
   if set(t["fy"])!=set(FY) or len(t)!=3:raise ValueError("Missing complete original 3FY group")
   if t["mode"].nunique()!=1:raise ValueError("Mixed standalone/consolidated fiscal mode")
   vals={}
   for r in t.itertuples(index=False):
    yr=int(r.fy);mode=str(r.mode)
    pub=pd.to_datetime(r.pub,utc=True,errors="coerce")
    if pd.isna(pub) or pub>cutoff:raise ValueError("Future source not eligible at 2023 close")
    if not str(r.xbrl_url).startswith("https://nsearchives.nseindia.com/") or not str(r.xbrl_url).lower().endswith(".xml"):
     raise ValueError("Non-original NSE financial document host")
    raw=client.get(str(r.xbrl_url)).content
    digest=hashlib.sha256(raw).hexdigest()
    if yr in (2021,2022):
     numeric,reason=extract_2022_legacy_fourd(raw,f"{yr}-03-31",mode_expected=mode,source_is_annual=True)
    elif yr==2023:
     numeric,reason=strict_annual_facts(raw,pd.Timestamp(f"{yr}-03-31").date(),source_is_annual=True)
     if any(numeric.get(c) is None for c in ("revenue","pat")):raise ValueError("Missing verified current fiscal revenue/PAT")
    else:raise ValueError("Unexpected fiscal year")
    values=[numeric.get("revenue"),numeric.get("pat")]
    if any(v is None or not math.isfinite(float(v)) for v in values):
     raise ValueError("Non-finite annual financial fact")
    vals[yr]=numeric
    record[f"FY{yr}_revenue_INR"]=float(numeric["revenue"])
    record[f"FY{yr}_PAT_INR"]=float(numeric["pat"])
    record[f"FY{yr}_sha256"]=digest
    record[f"FY{yr}_published_utc"]=pub.isoformat()
    record[f"FY{yr}_original_nse_xbrl_url"]=str(r.xbrl_url)
    record[f"FY{yr}_strict_fiscal_audit"]=json.dumps(reason,sort_keys=True)
   rev=[vals[yr]["revenue"] for yr in FY]
   profit=[vals[yr]["pat"] for yr in FY]
   if min(rev)<=0:raise ValueError("All fiscal revenues must be positive for historical growth")
   record["FY2022_vs_2021_sales_yoy_fraction"]=rev[1]/rev[0]-1
   record["FY2023_vs_2022_sales_yoy_fraction"]=rev[2]/rev[1]-1
   record["FY2021_to_FY2023_sales_2yr_CAGR_fraction"]=(rev[2]/rev[0])**.5-1
   record["FY2022_vs_2021_profit_yoy_fraction"]=profit[1]/profit[0]-1 if profit[0]>0 else None
   record["FY2023_vs_2022_profit_yoy_fraction"]=profit[2]/profit[1]-1 if profit[1]>0 else None
   record["FY2021_to_FY2023_profit_2yr_CAGR_fraction"]=(profit[2]/profit[0])**.5-1 if profit[0]>0 and profit[2]>0 else None
   record["verified_3_consecutive_annual_revenue_PAT"]=True
   record["actual_3yr_CAGR_calculated"]=False
   record["record_is_training_approved"]=False
   approved.append(record)
  except Exception as e:
   record["rejection_reason"]=str(e)[:240]
   rejected.append(record)
  if i%8==0:
   print("strict-three-year-pilot",i,"/",len(selected),"fully verified",len(approved),"rejected",len(rejected),flush=True)
 pd.DataFrame(approved).to_csv(out/"three_FY_FY2021_2023_strict_annual_numeric_RESEARCH_ONLY.csv",index=False)
 pd.DataFrame(rejected).to_csv(out/"three_FY_FY2021_2023_original_source_rejections.csv",index=False)
 summary={"scope":"ORIGINAL_NSE_FY2021_FY2022_FY2023_STRICT_NUMERIC_RESEARCH_NOT_TRAINING",
   "three_FY_companies_requested":len(selected),
   "three_FY_source_complete_company_count":len(approved),
   "three_FY_source_rejected_company_count":len(rejected),
   "verified_3_FY_revenue_and_PAT_coverage":len(approved)/len(selected),
   "FY2021_2023_two_year_revenue_CAGR_calculable":sum(pd.notna(x["FY2021_to_FY2023_sales_2yr_CAGR_fraction"]) for x in approved),
   "FY2021_2023_two_year_profit_CAGR_calculable":sum(pd.notna(x["FY2021_to_FY2023_profit_2yr_CAGR_fraction"]) for x in approved),
   "original_asof_date_IST":FOLD,
   "three_fiscal_reports_are_two_growth_intervals_not_three":True,
   "original_files_hashed_and_url_preserved":True,
   "published_after_2023_decision_rejected":True,
   "three_FY_numeric_source_verified_for_18_historical_folds":False,
   "no_backward_revised_filing_values":True,
   "production_model_predictions_changed":False,
   "financial_feature_training_approved":False}
 (out/"three_FY_original_financial_numeric_pilot.json").write_text(json.dumps(summary,indent=2))
 print(json.dumps(summary,indent=2),flush=True)
 return summary

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--inventory",required=True);p.add_argument("--limit",type=int,default=24);p.add_argument("--out",required=True)
 a=p.parse_args()
 strict_three_annual(pd.read_csv(a.inventory,dtype=str).fillna(""),a.limit,a.out)
if __name__=="__main__":main()
