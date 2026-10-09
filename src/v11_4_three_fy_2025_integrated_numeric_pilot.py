"""Strict as-of Dec 2025 FY2023/24/25 source audit with integrated FY2025 YTD.

FY2025 integrated March Q4 OneD (quarter-only) is forbidden. Only original
FourD annual YTD numerics and FY2023/24 12-month annual fiscal values qualify.
Each three-year company is source-hashed and matched by consolidated mode.
"""
from __future__ import annotations
import argparse,hashlib,json,math
from pathlib import Path
import pandas as pd
from v11_4_longterm_fundamentals import Client
from v11_4_strict_annual_numeric_features import strict_annual_facts,fold_close
CLOSE="2025-12-31"
YEARS=(2023,2024,2025)
def strict_three_fy_2025(index,limit,out,shard=0,shards=1):
 z=index.copy()
 required={"symbol","mode","fy_end_ts","published_at","xbrl_url","source"}
 if not required.issubset(z):raise ValueError("Modern original integrated source bridge fields missing")
 z["year"]=pd.to_datetime(z["fy_end_ts"],utc=True,errors="raise").dt.year
 if z.duplicated(["symbol","year"]).any():raise ValueError("Duplicate issuer FY record in 2025")
 groups=sorted(z.groupby("symbol"),key=lambda x:x[0])
 if not 0<=shard<shards or not 1<=limit<=256:raise ValueError("Invalid modern fiscal shard")
 selected=groups[shard::shards][:limit]
 if not selected:raise ValueError("No eligible original 2025 source triples")
 root=Path(out);root.mkdir(parents=True,exist_ok=True)
 cutoff=fold_close(CLOSE);client=Client()
 results=[];rejected=[]
 for i,(sym,g) in enumerate(selected,1):
  rec={"symbol":sym,"original_historical_fold":CLOSE,"source_mode":str(g.iloc[0]["mode"]),"original_model_unchanged":True}
  try:
   if len(g)!=3 or set(g["year"])!=set(YEARS):raise ValueError("Incomplete 3-FY original source")
   if g["mode"].nunique()!=1:raise ValueError("Consolidated mode mixing")
   vals={}
   for r in g.itertuples(index=False):
    fy=int(r.year);period=f"{fy}-03-31";pub=pd.to_datetime(r.published_at,utc=True)
    if pub>cutoff or pub<pd.Timestamp(period,tz="UTC"):raise ValueError("FY original published after decision or before period")
    if not str(r.xbrl_url).startswith("https://nsearchives.nseindia.com/") or not str(r.xbrl_url).endswith(".xml"):
     raise ValueError("Invalid original NSE financial source")
    if fy==2025 and str(r.source)!="NSE_Integrated_2025Q4_FourD_YTD_INDEX_CANDIDATE_ONLY":
     raise ValueError("FY2025 record not from independently audited NSE integrated bridge")
    raw=client.get(str(r.xbrl_url)).content
    values,audit=strict_annual_facts(raw,period,source_is_annual=True)
    if values["revenue"] is None or values["pat"] is None:
     raise ValueError("Missing original fiscal revenue or PAT")
    if not all(math.isfinite(float(values[k])) for k in ("revenue","pat")):
     raise ValueError("Non-finite original FY revenue/PAT")
    if fy==2025:
     if audit.get("revenue",{}).get("context")!="FourD" or audit.get("pat",{}).get("context")!="FourD":
      raise ValueError("FY2025 quarterly OneD is not accepted as annual")
    rec[f"FY{fy}_revenue_INR"]=float(values["revenue"])
    rec[f"FY{fy}_PAT_INR"]=float(values["pat"])
    rec[f"FY{fy}_source_SHA256"]=hashlib.sha256(raw).hexdigest()
    rec[f"FY{fy}_published_utc"]=pub.isoformat()
    rec[f"FY{fy}_annual_fact_context"]=json.dumps(audit,sort_keys=True)
    rec[f"FY{fy}_source_URL"]=str(r.xbrl_url)
    vals[fy]=values
   rev=[vals[y]["revenue"] for y in YEARS];pat=[vals[y]["pat"] for y in YEARS]
   if min(rev)<=0:raise ValueError("Annual source revenue baseline non-positive")
   rec["revenue_2023_to_2025_2yr_CAGR"]=math.sqrt(rev[2]/rev[0])-1
   rec["revenue_FY2024_yoy"]=rev[1]/rev[0]-1
   rec["revenue_FY2025_yoy"]=rev[2]/rev[1]-1
   rec["profit_2023_to_2025_2yr_CAGR"]=math.sqrt(pat[2]/pat[0])-1 if pat[0]>0 and pat[2]>0 else None
   rec["profit_FY2025_yoy"]=pat[2]/pat[1]-1 if pat[1]>0 else None
   rec["fully_verified_source_three_years"]=True
   rec["valid_for_3year_sales_CAGR"]=False
   rec["financial_training_approved"]=False
   results.append(rec)
  except Exception as e:
   rec["rejection_reason"]=str(e)[:240];rejected.append(rec)
  if i%8==0:
   print("2025 integrated full FY2023/24/25",i,"/",len(selected),"strict",len(results),"rejected",len(rejected),flush=True)
 pd.DataFrame(results).to_csv(root/"three_FY2023_2025_original_NSE_integrated_STRICT_NUMERICS.csv",index=False)
 pd.DataFrame(rejected).to_csv(root/"three_FY2023_2025_original_NSE_strict_rejections.csv",index=False)
 report={"scope":"2025_DEC_2023_24_25_ORIGINAL_NSE_FINANCIAL_THREE_FY_SOURCE_ONLY",
  "sampled":len(selected),"three_year_annual_numeric_verified":len(results),
  "three_year_original_source_rejected":len(rejected),
  "current_integrated_FY2025_annual_FourD_Q4_verified":len(results),
  "two_year_revenue_CAGR_available":len(results),
  "two_year_profit_CAGR_valid":sum(pd.notna(r["profit_2023_to_2025_2yr_CAGR"]) for r in results),
  "all_company_2025_three_FY_link_candidates":len(groups),
  "shard":shard,"shards":shards,
  "year_end_2025_historical_investment_decision_cutoff_IST":CLOSE,
  "no_later_original_restatements":True,"no_quarter_only_YTD_fiscal_confusion":True,
  "is_model_training_or_calibrated_2x_result":False}
 (root/"three_FY_2025_strict_numeric_quality_summary.json").write_text(json.dumps(report,indent=2))
 print(json.dumps(report,indent=2),flush=True)
 return report
def main():
 a=argparse.ArgumentParser()
 a.add_argument("--index",required=True);a.add_argument("--limit",type=int,default=48)
 a.add_argument("--shard",type=int,default=0);a.add_argument("--shards",type=int,default=1)
 a.add_argument("--out",required=True);opts=a.parse_args()
 strict_three_fy_2025(pd.read_csv(opts.index,dtype=str).fillna(""),opts.limit,opts.out,opts.shard,opts.shards)
if __name__=="__main__":main()
