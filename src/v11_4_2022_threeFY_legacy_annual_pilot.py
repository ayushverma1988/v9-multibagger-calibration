"""Bounded FY2020/21/22 original NSE fiscal numeric backfill for the 2022 fold.

The 2022 earlier fold is a NEW historical training source, not a rerun of
the already evaluated 2025 holdout. Requires original NSE March-yearly
FourD annual metadata and full-year INR revenue and PAT in all 3 years.
Do not mix standalone/consolidated. Never inject any forward outcome label.
"""
from __future__ import annotations
import argparse,hashlib,json,math
from pathlib import Path
import pandas as pd
from v11_4_threeFY_all18fold_source_feasibility import eligible_three_fy
from v11_4_recover_legacy_FY2022_FourD import extract_2022_legacy_fourd
from v11_4_longterm_fundamentals import Client
from v11_4_strict_annual_numeric_features import fold_close
FOLD="2022-12-30"
FYS=(2020,2021,2022)
def original_2022_pilot(annual,snapshot,fold_dates,limit,out,shard=0,shards=1):
 if not 0<=shard<shards<=24 or not 1<=limit<=128:
  raise ValueError("Bounded original 2022 fiscal source cohort invalid")
 _,links=eligible_three_fy(annual,snapshot,fold_dates)
 z=links.loc[links["fold"].eq(FOLD)].copy()
 z["symbol"]=z["symbol"].astype(str).str.upper().str.strip()
 candidates=sorted(z["symbol"].unique(),key=lambda name:hashlib.sha256(name.encode()).hexdigest())
 if not candidates:raise ValueError("Original 2022 market fiscal annual source index missing")
 sample=candidates[shard::shards][:limit]
 client=Client();cutoff=fold_close(FOLD)
 good=[];bad=[]
 for i,sym in enumerate(sample,1):
  g=z[z["symbol"].eq(sym)].copy()
  meta={"symbol":sym,"original_market_fold_IST":FOLD,
        "original_full_FY_end_years":"2020|2021|2022",
        "source_mode":g["mode"].iloc[0],"training_approved":False}
  try:
   if len(g)!=3 or set(g["fiscal_year"])!=set(FYS) or g["mode"].nunique()!=1:
    raise ValueError("Missing exact original same-mode three fiscal years")
   annuals={}
   for r in g.itertuples(index=False):
    fy=int(r.fiscal_year);doc=str(r.xbrl_url)
    pub=pd.to_datetime(r.pub,utc=True,errors="coerce")
    if not doc.startswith("https://nsearchives.nseindia.com/") or not doc.lower().endswith(".xml"):
     raise ValueError("Non-original 2022 NSE fiscal source")
    if pd.isna(pub) or pub>cutoff:raise ValueError("Financial source published after original 2022 decision")
    raw=client.get(doc).content
    facts,audit=extract_2022_legacy_fourd(raw,fy=f"{fy}-03-31",mode_expected=r.mode,source_is_annual=True)
    if any(facts.get(k) is None or not math.isfinite(float(facts[k])) for k in ("revenue","pat")):
     raise ValueError("Missing exact original legacy FY revenue/PAT")
    annuals[fy]=facts
    meta[f"FY{fy}_revenue_INR"]=float(facts["revenue"])
    meta[f"FY{fy}_PAT_INR"]=float(facts["pat"])
    meta[f"FY{fy}_source_sha256"]=hashlib.sha256(raw).hexdigest()
    meta[f"FY{fy}_published_utc"]=pub.isoformat()
    meta[f"FY{fy}_source_url"]=doc
    meta[f"FY{fy}_source_context_audit"]=json.dumps(audit,sort_keys=True)
   vals=[annuals[y]["revenue"] for y in FYS]
   pat=[annuals[y]["pat"] for y in FYS]
   if min(vals)<=0:raise ValueError("Nonpositive original fiscal annual revenue")
   meta["FY2020_to_FY2022_revenue_two_year_CAGR"]=(vals[2]/vals[0])**.5-1
   meta["FY2022_vs_FY2021_revenue_yoy"]=vals[2]/vals[1]-1
   meta["FY2021_vs_FY2020_revenue_yoy"]=vals[1]/vals[0]-1
   meta["FY2020_to_FY2022_PAT_two_year_CAGR"]=(pat[2]/pat[0])**.5-1 if pat[0]>0 and pat[2]>0 else None
   meta["actual_three_year_CAGR_calculated"]=False
   good.append(meta)
  except Exception as exc:
   meta["rejection_reason"]=str(exc)[:230]
   bad.append(meta)
  if i%8==0:
   print("FY2022 legacy annual original cohort",i,"/",len(sample),"verified",len(good),"rejected",len(bad),flush=True)
 out=Path(out);out.mkdir(parents=True,exist_ok=True)
 pd.DataFrame(good).to_csv(out/"original_2022_FY2020_2021_2022_strict_annual_numerics_RESEARCH.csv",index=False)
 pd.DataFrame(bad).to_csv(out/"original_2022_FY2020_2021_2022_rejections.csv",index=False)
 summary={
  "scope":"NSE_DEC_2022_ORIGINAL_FY2020_FY2021_FY2022_STRICT_ANNUAL_NUMERIC_PILOT",
  "original_2022_threeFY_link_eligible_companies":len(candidates),
  "requested_company_numerics":len(sample),
  "fully_verified_original_2020_2021_2022_annual_revenue_PAT_companies":len(good),
  "original_FY2020_source_rejections":len(bad),
  "numeric_source_pass_fraction":len(good)/len(sample),
  "all_2022_source_filing_years_strict_before_2022_12_30_1530_IST":True,
  "original_Source_sha256_by_FY_retained":True,
  "three_fiscal_years_only_two_growth_intervals":True,
  "original_model_and_2025_holdout_unmodified":True,
  "historical_stock_double_target_labels_not_loaded":True,
  "new_2022_pilot_not_permitted_as_trained_predictor":True,
  "shard":shard,"shards":shards}
 (out/"original_FY2022_three_FY_numeric_source_pilot_summary.json").write_text(json.dumps(summary,indent=2))
 print(json.dumps(summary,indent=2),flush=True)
 return summary
def main():
 p=argparse.ArgumentParser()
 p.add_argument("--annual-index",required=True);p.add_argument("--snapshot",required=True)
 p.add_argument("--folds",required=True);p.add_argument("--limit",type=int,default=48)
 p.add_argument("--shard",type=int,default=0);p.add_argument("--shards",type=int,default=1)
 p.add_argument("--out",required=True);a=p.parse_args()
 original_2022_pilot(pd.read_csv(a.annual_index,dtype=str).fillna(""),
  pd.read_parquet(a.snapshot,columns=["date","symbol"]),pd.read_csv(a.folds),
  a.limit,a.out,a.shard,a.shards)
if __name__=="__main__":main()
