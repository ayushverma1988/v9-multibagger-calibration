"""Independent full Dec2022 original NSE FY2020-22 numeric reconciliation.

Reconstruct the frozen 1123-stock 2022 original market snapshot; no outcome
labels, no modern revisions; verify every 12-shard issuer disjointness and
match all actual original source URLs, timestamps, annual modes and SHA256.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd
import numpy as np
from v11_4_threeFY_all18fold_source_feasibility import eligible_three_fy
from v11_4_strict_annual_numeric_features import fold_close
FOLD="2022-12-30"
YEARS=(2020,2021,2022)
def reconcile(raw_shards,index,snapshot,folds):
 _,allurls=eligible_three_fy(index,snapshot,folds)
 original=allurls.loc[allurls["fold"].eq(FOLD)].copy()
 symbols=sorted(set(original["symbol"]))
 if len(symbols)!=836 or len(original)!=836*3:
  raise ValueError("Original NSE 2022 3FY PIT eligible source company count changed")
 root=Path(raw_shards)
 metas=list(root.rglob("original_FY2022_three_FY_numeric_source_pilot_summary.json"))
 if len(metas)!=12:raise ValueError("Expected twelve original issuer partitions")
 approved=[];rejected=[];seen=set()
 for path in metas:
  status=json.loads(path.read_text())
  shard=int(status["shard"])
  if shard in seen:raise ValueError("Duplicate fiscal partition")
  seen.add(shard)
  if status["shards"]!=12 or status["original_2022_threeFY_link_eligible_companies"]!=836:
   raise ValueError("Annual fiscal source cohort changed")
  v=path.parent/"original_2022_FY2020_2021_2022_strict_annual_numerics_RESEARCH.csv"
  e=path.parent/"original_2022_FY2020_2021_2022_rejections.csv"
  if not v.is_file() or not e.is_file():raise ValueError("Missing original audit artifacts")
  good=pd.read_csv(v) if status["fully_verified_original_2020_2021_2022_annual_revenue_PAT_companies"] else pd.DataFrame()
  bad=pd.read_csv(e) if status["original_FY2020_source_rejections"] else pd.DataFrame()
  expected=set(sorted(symbols,key=lambda n:__import__("hashlib").sha256(n.encode()).hexdigest())[shard::12])
  a=set(good["symbol"]) if len(good) else set()
  b=set(bad["symbol"]) if len(bad) else set()
  if a&b or a|b!=expected or len(a)!=len(good) or len(b)!=len(bad):
   raise ValueError("Original NSE 2022 source expected issuer shard identities mismatch")
  if len(a)!=status["fully_verified_original_2020_2021_2022_annual_revenue_PAT_companies"] or len(b)!=status["original_FY2020_source_rejections"]:
   raise ValueError("Original source/strict rejection ledger totals mismatch")
  if len(good):approved.append(good)
  if len(bad):rejected.append(bad)
 if seen!=set(range(12)):raise ValueError("Original 2022 financial source partition missing")
 verified=pd.concat(approved,ignore_index=True)
 errors=pd.concat(rejected,ignore_index=True) if rejected else pd.DataFrame()
 if len(verified)+len(errors)!=836 or verified["symbol"].duplicated().any():
  raise ValueError("Duplicate or omitted 2022 source company")
 source=original.set_index(["symbol","fiscal_year"])
 cutoff=fold_close(FOLD)
 for y in YEARS:
  vrev=pd.to_numeric(verified[f"FY{y}_revenue_INR"],errors="coerce")
  vpat=pd.to_numeric(verified[f"FY{y}_PAT_INR"],errors="coerce")
  if not vrev.gt(0).all() or not np.isfinite(vpat).all():
   raise ValueError("Original NSE 2022 3FY nonnumeric/bad annual fact")
  if not verified[f"FY{y}_source_sha256"].str.fullmatch("[0-9a-f]{64}").all():
   raise ValueError("Original yearly file SHA256 absent")
  pubs=pd.to_datetime(verified[f"FY{y}_published_utc"],errors="coerce",utc=True)
  if pubs.isna().any() or (pubs>cutoff).any():raise ValueError("Future original FY source used")
  for r in verified.itertuples(index=False):
   orig=source.loc[(r.symbol,y)]
   if getattr(r,f"FY{y}_source_url")!=orig["xbrl_url"]:
    raise ValueError("Unverified 2022 original fiscal URL substitution")
  # Original publication dates are preserved and cross-verified as timestamps.
  src_pubs=pd.to_datetime(source.loc[pd.MultiIndex.from_arrays([verified["symbol"],[y]*len(verified)]),"pub"],utc=True)
  if not pubs.reset_index(drop=True).eq(src_pubs.reset_index(drop=True)).all():
   raise ValueError("Original 2022 fiscal publication provenance changed")
 calc=(verified["FY2022_revenue_INR"]/verified["FY2020_revenue_INR"])**.5-1
 if not np.allclose(calc,verified["FY2020_to_FY2022_revenue_two_year_CAGR"],atol=1e-10,rtol=1e-10):
  raise ValueError("Actual historical annual 2022 2-year CAGR algebra changed")
 if verified["actual_three_year_CAGR_calculated"].astype(str).str.lower().eq("true").any():
  raise ValueError("Only two growth intervals cannot be called true three year CAGR")
 market=snapshot.copy()
 market["date"]=pd.to_datetime(market["date"],errors="raise").dt.normalize()
 market["symbol"]=market["symbol"].astype(str).str.upper().str.strip()
 original_stocks=sorted(set(market.loc[market["date"].eq(pd.Timestamp(FOLD)),"symbol"]))
 if len(original_stocks)!=1123:raise ValueError("Original 2022 market stock identities unexpectedly changed")
 needed=["symbol","source_mode"]+[f"FY{yr}_{key}_INR" for yr in YEARS for key in ("revenue","PAT")]+[
  f"FY{yr}_source_sha256" for yr in YEARS]+[f"FY{yr}_published_utc" for yr in YEARS]
 matched=pd.DataFrame({"date":FOLD,"symbol":original_stocks}).merge(
    verified[needed],on="symbol",how="left",validate="1:1")
 if len(matched)!=1123:raise ValueError("Original 2022 market universe altered")
 matched["has_exact_3FY_FY2020_FY2022_original_NSE"]=matched["FY2020_source_sha256"].notna()
 ratio=len(verified)/len(matched)
 audit={
  "scope":"NSE_DEC_2022_FULL_FY2020_FY2021_FY2022_STRICT_RECONCILED_RESEARCH_ONLY",
  "original_frozen_2022_stock_universe":len(original_stocks),
  "strict_source_url_eligible_original_companies":836,
  "verified_complete_three_FY_annual_revenue_PAT_companies":len(verified),
  "original_NSE_annual_source_rejections":len(errors),
  "original_market_numeric_three_FY_coverage_percent":round(100*ratio,2),
  "single_2022_fold_70pct_source_numeric_gate_passed":bool(ratio>=.70),
  "no_original_2022_stock_dropped":True,
  "all_2022_original_three_year_filing_times_and_original_urls_verified":True,
  "no_3year_CAGR_from_three_annual_points":True,
  "2025_holdout_or_forward_model_predictions_modified":False,
  "ML_training_or_new_prospective_recommendations_created":False
 }
 return verified,errors,matched,audit
def main():
 p=argparse.ArgumentParser()
 p.add_argument("--shards",required=True);p.add_argument("--annual-index",required=True)
 p.add_argument("--snapshot",required=True);p.add_argument("--folds",required=True)
 p.add_argument("--out",required=True)
 a=p.parse_args()
 stock=pd.read_parquet(a.snapshot,columns=["date","symbol"])
 v,b,m,s=reconcile(a.shards,pd.read_csv(a.annual_index,dtype=str).fillna(""),
   stock,pd.read_csv(a.folds))
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 v.to_csv(out/"original_FY2022_fold_verified_FY2020_2021_2022_annual_facts.csv",index=False)
 b.to_csv(out/"original_FY2022_fold_rejected_FY2020_2021_2022_facts.csv",index=False)
 m.to_parquet(out/"original_1123_stocks_Dec2022_fiscal3FY_source_quality.parquet",index=False)
 (out/"original_Dec2022_verified_three_FY_numeric_reconciliation_summary.json").write_text(json.dumps(s,indent=2))
 print(json.dumps(s,indent=2),flush=True)
if __name__=="__main__":main()
