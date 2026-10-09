"""Reconcile independent FY2022/FY2023 original NSE financial source research.

Never join future outcome or model-ranking columns. Verify independent shards
are disjoint from initial pilot; check fiscal-year growth, source hashes,
historical publication cutoffs and original 2023 stock-universe identity.
Keep all unrecovered 2023 stocks with NaNs rather than survivorship filtering.
"""
from __future__ import annotations
import argparse,hashlib,json,re
from pathlib import Path
import numpy as np
import pandas as pd
from v11_4_strict_annual_numeric_features import fold_close
from v11_4_fy2023_historical_annual_pilot import select_2023_fold

FIELDS=("FY2022_revenue_INR","FY2023_revenue_INR","FY2022_PAT_INR","FY2023_PAT_INR")
FOLDDATE="2023-12-29"
def normalize_pilot(pilot,filings):
    x=pilot.rename(columns={
        "reporting_mode":"mode",
        "FY2022_revenue_raw_INR":"FY2022_revenue_INR",
        "FY2023_revenue_raw_INR":"FY2023_revenue_INR",
        "FY2022_pat_raw_INR":"FY2022_PAT_INR",
        "FY2023_pat_raw_INR":"FY2023_PAT_INR",
        "FY2022_original_SHA256":"FY2022_sha256",
        "FY2023_original_SHA256":"FY2023_sha256",
        "annual_YOY_revenue_fraction":"revenue_yoy_fraction",
        "annual_YOY_pat_fraction":"pat_yoy_fraction"}).copy()
    z=filings[filings["status"].eq("VERIFIED_CORE_REVENUE_PAT")|
              filings["fy_end"].eq("2022-03-31")].copy()
    if z.duplicated(["symbol","mode","fy_end"]).any():
        raise ValueError("Ambiguous original pilot fiscal filing")
    for year in (2022,2023):
        sub=z[z["fy_end"].eq(f"{year}-03-31")][[
            "symbol","mode","source_available_utc","original_xml_sha256"]].rename(
            columns={"source_available_utc":f"FY{year}_publication_utc",
                     "original_xml_sha256":f"FY{year}_original_sha256_catalog"})
        x=x.merge(sub,on=["symbol","mode"],how="left",validate="1:1")
        if not x[f"FY{year}_sha256"].eq(x[f"FY{year}_original_sha256_catalog"]).all():
            raise ValueError("Original FY pilot source hash mismatch")
    x["source_recovery_cohort"]="first_24_parser_development"
    return x
def normalize_independent(independent):
    x=independent.rename(columns={
      "fy2022_revenue_INR":"FY2022_revenue_INR",
      "fy2023_revenue_INR":"FY2023_revenue_INR",
      "fy2022_PAT_INR":"FY2022_PAT_INR",
      "fy2023_PAT_INR":"FY2023_PAT_INR",
      "fy2022_source_sha256":"FY2022_sha256",
      "fy2023_source_sha256":"FY2023_sha256",
      "fy2022_publication_utc":"FY2022_publication_utc",
      "fy2023_publication_utc":"FY2023_publication_utc",
      "annual_revenue_yoy_fraction":"revenue_yoy_fraction",
      "annual_PAT_yoy_fraction":"pat_yoy_fraction"}).copy()
    x["source_recovery_cohort"]="independent_unseen_128"
    return x
def reconcile(pilot,prior_filings,root,snapshot):
    root=Path(root)
    summary_paths=list(root.rglob("independent_FY2023_annual_numeric_shard_summary.json"))
    if len(summary_paths)!=4:raise ValueError(f"Expected four independent shards; found {len(summary_paths)}")
    checks=[];frames=[];failures=[];cohorts=set()
    for path in summary_paths:
        audit=json.loads(path.read_text())
        shard=int(audit["shard"])
        if shard in cohorts:raise ValueError("Duplicate independent numeric shard")
        cohorts.add(shard)
        if audit["shard_count"]!=4 or audit["new_companies_requested"]!=32:
            raise ValueError("Unexpected independent company-shard count")
        if audit["both_original_fiscal_facts_verified"]<24:
            raise ValueError("Independent heldout original NSE numeric coverage <75%")
        if audit["original_standalone_model_modified"] or not audit["no_imputation"]:
            raise ValueError("Source trial altered model/missing data")
        good=path.parent/"independent_original_FY2022_FY2023_PIT_numeric_RESEARCH.csv"
        bad=path.parent/"independent_FY2022_FY2023_PIT_rejected.csv"
        if not good.exists() or not bad.exists():raise ValueError("Absent original research or rejection manifests")
        ok=pd.read_csv(good)
        # The independent source pipeline writes a zero-byte CSV for a shard
        # with no rejected filings; never treat that as an unreported error.
        reject=(pd.read_csv(bad) if audit["FY2022_FY2023_source_rejected"]>0
                else pd.DataFrame())
        if len(ok)!=audit["both_original_fiscal_facts_verified"]:
            raise ValueError("Original independent artifact counts inconsistent")
        if len(reject)!=audit["FY2022_FY2023_source_rejected"]:
            raise ValueError("Independent rejected filing manifest count inconsistent")
        checks.append(audit)
        frames.append(normalize_independent(ok))
        if len(reject):failures.append(reject.assign(source_shard=shard))
    if cohorts!={0,1,2,3}:raise ValueError("Missing independent shards")
    first=normalize_pilot(pilot,prior_filings)
    indie=pd.concat(frames,ignore_index=True)
    if set(first["symbol"])&set(indie["symbol"]):raise ValueError("Independent cohort overlaps first parser pilot")
    x=pd.concat([first,indie],ignore_index=True)
    if x["symbol"].duplicated().any():raise ValueError("Duplicate 2023 stock for recovered original XBRL")
    hist_fold,stocks=select_2023_fold(snapshot)
    if hist_fold!=FOLDDATE:raise ValueError("Unexpected original historical fold date")
    if not set(x["symbol"]).issubset(stocks):raise ValueError("Survivorship substitution not in original fold")
    for col in FIELDS+("revenue_yoy_fraction","pat_yoy_fraction"):
        x[col]=pd.to_numeric(x[col],errors="coerce")
    if x[list(FIELDS)].isna().any(axis=1).any():raise ValueError("Recovered source missing one core FY revenue/PAT")
    if (x["FY2022_revenue_INR"]<=0).any():raise ValueError("Invalid baseline")
    compare=x["FY2023_revenue_INR"]/x["FY2022_revenue_INR"]-1
    if not np.allclose(compare,x["revenue_yoy_fraction"],atol=1e-10,rtol=1e-9):
        raise ValueError("Reconstructed source fiscal revenue YoY mismatch")
    pat=x["FY2022_PAT_INR"]>0
    if not np.allclose((x.loc[pat,"FY2023_PAT_INR"]/x.loc[pat,"FY2022_PAT_INR"]-1),
                       x.loc[pat,"pat_yoy_fraction"],atol=1e-10,rtol=1e-9):
        raise ValueError("Source PAT year-on-year inconsistent")
    if x.loc[~pat,"pat_yoy_fraction"].notna().any():raise ValueError("Non-positive PAT incorrectly divided")
    close=fold_close(FOLDDATE)
    for yr in (2022,2023):
        stamps=pd.to_datetime(x[f"FY{yr}_publication_utc"],utc=True,errors="coerce")
        if stamps.isna().any() or (stamps>close).any():raise ValueError("Future original NSE financial publication")
        if not x[f"FY{yr}_sha256"].astype(str).str.fullmatch("[0-9a-f]{64}").all():
            raise ValueError("Missing original fiscal annual source SHA256")
    base=pd.DataFrame({"date":FOLDDATE,"symbol":sorted(stocks)})
    keep=["symbol","mode","revenue_yoy_fraction","pat_yoy_fraction","FY2022_sha256","FY2023_sha256",
          "FY2022_publication_utc","FY2023_publication_utc","source_recovery_cohort"]
    joined=base.merge(x[keep],on="symbol",how="left",validate="1:1")
    joined["has_verifiable_2022_2023_original_annual_yoy"]=joined["FY2022_sha256"].notna()
    if len(joined)!=len(stocks):raise ValueError("Original 2023 frozen universe changed")
    report={"scope":"V11_4_FY2023_ORIGINAL_NSE_HISTORIC_ANNUAL_RESEARCH_RECONCILIATION",
      "original_historical_2023_universe":len(stocks),
      "first_parser_development_sample":len(first),
      "independent_unseen_cohort_requested":sum(z["new_companies_requested"] for z in checks),
      "independent_unseen_annual_pairs_verified":len(indie),
      "source_pairs_verified_combined":len(x),
      "original_stock_universe_annual_pair_coverage":float(len(x)/len(stocks)),
      "revenue_yoy_calculable":int(joined["revenue_yoy_fraction"].notna().sum()),
      "pat_yoy_calculable":int(joined["pat_yoy_fraction"].notna().sum()),
      "source_rejections_still_excluded":sum(z["FY2022_FY2023_source_rejected"] for z in checks),
      "full_2023_financial_70_percent_coverage_gate_passed":bool(len(x)/len(stocks)>=.70),
      "verified_3y_5y_7y_growth_extracted":False,
      "no_frozen_2023_company_dropped":True,
      "missing_older_financial_values_left_unknown":True,
      "already_frozen_standalone_predictions_modified":False,
      "financial_rule_training_activated":False}
    return joined, (pd.concat(failures,ignore_index=True) if failures else pd.DataFrame()), report

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--first-24",required=True);p.add_argument("--pilot-filings",required=True)
 p.add_argument("--shards",required=True);p.add_argument("--snapshot",required=True);p.add_argument("--out",required=True)
 a=p.parse_args()
 f,errors,r=reconcile(pd.read_csv(a.first_24),pd.read_csv(a.pilot_filings),
                      a.shards,pd.read_parquet(a.snapshot,columns=["date","symbol"]))
 dest=Path(a.out);dest.mkdir(parents=True,exist_ok=True)
 f.to_parquet(dest/"FY2023_historical_original_stock_universe_annual_RESEARCH_ONLY.parquet",index=False)
 f.to_csv(dest/"FY2023_original_annual_PIT_feature_coverage.csv",index=False)
 errors.to_csv(dest/"FY2023_rejected_original_fiscal_source_research.csv",index=False)
 (dest/"FY2023_original_verified_annual_source_coverage.json").write_text(json.dumps(r,indent=2))
 print(json.dumps(r,indent=2),flush=True)
 if r["independent_unseen_annual_pairs_verified"]<96:raise SystemExit("Independent historical source expansion did not meet coverage integrity")
if __name__=="__main__":main()
