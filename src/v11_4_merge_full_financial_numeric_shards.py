"""Reconcile eight parallel NSE FY2024/FY2025 annual numeric source shards.

Materializes a strictly point-in-time, original stock-universe financial
research matrix; never replaces missing observations with zeros or trains
predictive weights. Coverage gates apply only to sample source readiness,
not to the production performance threshold.
"""
from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path
import pandas as pd

def close_utc(value):
 return (pd.Timestamp(value).normalize().tz_localize("Asia/Kolkata")
         +pd.Timedelta(hours=15,minutes=30)).tz_convert("UTC")

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--artifacts",required=True)
 p.add_argument("--snapshot",required=True)
 p.add_argument("--output",required=True)
 p.add_argument("--expected-shards",type=int,default=4)
 a=p.parse_args();root=Path(a.artifacts);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 snapshots=pd.read_parquet(a.snapshot,columns=["date","symbol"])
 snapshots["date"]=pd.to_datetime(snapshots["date"],errors="coerce").dt.normalize()
 snapshots["symbol"]=snapshots["symbol"].astype(str).str.upper().str.strip()
 merged=[];perfold=[];source_errors=[];filings_total=0
 for fy in (2024,2025):
  manifest_name="strict_2024_fold_annual_numeric_coverage.json" if fy==2024 else "FY2025_annual_PIT_pairing_summary.json"
  manifests=list(root.rglob(manifest_name))
  if len(manifests)!=a.expected_shards:
   raise SystemExit(f"Year {fy} expected {a.expected_shards} complete shards, found {len(manifests)}")
  ids=set();seen=set();selected=0;eligible_n=set();fold=f"{fy}-12-31"
  all_features=[]
  for filename in manifests:
   data=json.loads(filename.read_text());shard=int(data["shard_index"])
   if shard in ids:raise SystemExit(f"Duplicate {fy} shard {shard}")
   ids.add(shard)
   if data["shard_count"]!=a.expected_shards:raise SystemExit("Shard configuration mismatch")
   requested=set(data["requested_symbols"])
   if seen&requested:raise SystemExit(f"Cross-shard company overlap year={fy}")
   seen.update(requested)
   n=data["sample_requested"] if fy==2024 else data["sampled_companies"]
   if n!=len(requested):raise SystemExit("Requested-symbol manifest mismatch")
   selected+=n
   eligible_n.add(data["candidate_symbol_total"] if fy==2024 else data["eligible_companies_with_2024_legacy_and_2025_integrated_same_mode"])
   src_errors=data["source_download_errors"] if fy==2024 else data["download_errors"]
   if src_errors:source_errors.append({"year":fy,"shard":shard,"failed_files":src_errors})
   filings_total+=data["source_documents_downloaded"] if fy==2024 else data["attempted_docs"]
   feature_file=filename.parent/"features"/"short_horizon_financial_PIT_research_features.csv"
   if not feature_file.exists():raise SystemExit(f"Missing financial features from {filename}")
   frame=pd.read_csv(feature_file)
   if len(frame) and not set(frame["symbol"].astype(str)).issubset(requested):
    raise SystemExit(f"Feature ticker not belonging to shard year={fy}, shard={shard}")
   all_features.append(frame)
  if ids!=set(range(a.expected_shards)) or len(eligible_n)!=1:
   raise SystemExit("Numeric shard indexes or historic eligible counts inconsistent")
  if selected!=next(iter(eligible_n)):
   raise SystemExit(f"Insufficient historical company request coverage {fy}: {selected} of {eligible_n}")
  allx=pd.concat(all_features,ignore_index=True)
  if len(allx) and allx.duplicated(["fold_date","symbol"]).any():
   raise SystemExit(f"Duplicate financial stock row across {fy} shards")
  if not allx["fold_date"].eq(fold).all():raise SystemExit("Wrong fold/year financial source contamination")
  if not allx["qa_status"].eq("CANDIDATE_SHORT_HORIZON_ONLY").all():
   raise SystemExit("Unapproved numerical source record")
  pub=pd.to_datetime(allx["source_available_utc"],utc=True,errors="coerce",format="mixed")
  if pub.isna().any() or (pub>close_utc(fold)).any():
   raise SystemExit("Historic after-market published financial facts leaked")
  target=snapshots.loc[snapshots["date"].eq(pd.Timestamp(fold)),["symbol"]].drop_duplicates()
  if (~allx["symbol"].isin(set(target["symbol"]))).any():
   raise SystemExit("Not members of original historic fold")
  cols=["symbol","reporting_mode","latest_fy_end","prior_fy_end",
        "revenue_yoy_pct","pat_yoy_pct","pat_margin_latest_pct",
        "pat_margin_delta_pp","profit_turnaround","earlier_profit_negative",
        "latest_fiscal_age_days","source_available_utc","qa_status"]
  if not set(cols).issubset(allx):raise SystemExit("Numeric source features missing fields")
  mergedfold=target.merge(allx[cols],on="symbol",how="left",validate="1:1")
  mergedfold["date"]=fold
  mergedfold["has_verified_annual_yoy_source"]=mergedfold["qa_status"].notna()
  mergedfold["fold_close_UTC"]=close_utc(fold).isoformat()
  if mergedfold["symbol"].isna().any() or len(mergedfold)!=len(target):
   raise SystemExit("Historic universe row count changed")
  merged.append(mergedfold)
  row={"fold":fold,"historical_stock_universe":len(target),
       "eligible_source_pair_companies":next(iter(eligible_n)),
       "attempted_companies":selected,
       "verified_annual_yoy_companies":len(allx),
       "revenue_yoy_nonnull":int(allx["revenue_yoy_pct"].notna().sum()),
       "pat_yoy_nonnull":int(allx["pat_yoy_pct"].notna().sum()),
       "revenue_coverage":float(allx["revenue_yoy_pct"].notna().sum()/len(target)),
       "pat_coverage":float(allx["pat_yoy_pct"].notna().sum()/len(target))}
  perfold.append(row)
  print("FOLD",json.dumps(row),flush=True)
 allmatrix=pd.concat(merged,ignore_index=True)
 if allmatrix.duplicated(["date","symbol"]).any():
  raise SystemExit("Duplicate historic stock-date in source matrix")
 allmatrix.to_parquet(out/"full_2024_2025_annual_XBRL_PIT_RESEARCH.parquet",index=False,compression="zstd")
 pd.DataFrame(perfold).to_csv(out/"full_annual_numeric_by_fold.csv",index=False)
 summary={
  "scope":"FULL_TWOFOLD_NUMERIC_SOURCE_READY_NOT_MODEL_PREDICTION",
  "years":[2024,2025],"shards_per_year":a.expected_shards,
  "verified_source_matrix_rows":len(allmatrix),
  "original_companies_in_folds":int(sum(x["historical_stock_universe"] for x in perfold)),
  "original_filings_attempted":filings_total,
  "source_download_errors":source_errors,
  "revenue_coverage":[x["revenue_coverage"] for x in perfold],
  "pat_coverage":[x["pat_coverage"] for x in perfold],
  "zero_fill_in_missings":False,
  "v10_production_changed":False,
  "training_done":False,"18fold_financial_source_complete":False,
  "source_years_summary":perfold,
 }
 (out/"full_2fold_numeric_research_summary.json").write_text(json.dumps(summary,indent=2))
 print(json.dumps({k:v for k,v in summary.items() if k not in ("source_years_summary","source_download_errors")},indent=2),flush=True)
 if any(z["revenue_coverage"]<0.65 for z in perfold):
  raise SystemExit("One source year still below 65% historical revenue coverage")
if __name__=="__main__":main()
