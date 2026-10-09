"""Immutable private Hugging Face persistence for NSE FY2023/24/25 PIT records."""
from __future__ import annotations
import argparse,hashlib,json,os
from pathlib import Path
from huggingface_hub import HfApi
REPO="ayushverma1988/v10-multibagger-archive"
ROOT="v11_4/historical_source_original_NSE/FY2023_FY2024_FY2025_asof_2025_12_31"
FILES=("FY2025_original_NSE_full_3FY_2023_2025_annual_financial_facts.csv",
 "FY2025_original_NSE_source_rejected.csv",
 "FY2025_original_1314_companies_3FY_verified_research.parquet",
 "FY2025_original_three_FY_annual_source_reconciliation.json")
def save(out,token,runid,api=None):
 if not token:raise ValueError("Private archival access missing")
 root=Path(out)
 for f in FILES:
  if not (root/f).is_file():raise ValueError("Original 2025 source file missing: "+f)
 meta=json.loads((root/FILES[-1]).read_text())
 if meta.get("original_FY2025_historical_stock_universe")!=1314 or
    meta.get("original_V11_4_frozen_model_unchanged") is not True:
  raise ValueError("Unverified private fiscal source integrity")
 if meta.get("all_original_company_rows_retained") is not True or
    meta.get("integrated_2025_FourD_only_not_quarter_OneD") is not True:
  raise ValueError("Source publication/quarter integrity missing")
 client=api or HfApi(token=token)
 info=client.repo_info(repo_id=REPO,repo_type="dataset")
 if info.private is not True:raise ValueError("Do not publish proprietary V11.4 research data")
 prefix=f"{ROOT}/verified_run_{runid}"
 if any(p.startswith(prefix+"/") for p in client.list_repo_files(repo_id=REPO,repo_type="dataset")):
  raise ValueError("Private three-FY historical source snapshot must not be overwritten")
 hashes={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in FILES}
 (root/"FY2025_three_fy_original_source_SHA256_manifest.json").write_text(json.dumps({
  "scope":"PRIVATE_IMMUTABLE_FINANCIAL_SOURCE_ONLY",
  "historical_asof_IST":"2025-12-31",
  "source_company_count":meta["FY2023_FY2024_FY2025_three_year_company_numerics_verified"],
  "files_SHA256":hashes,"source_only_no_predictive_model_fit":True
 },indent=2))
 client.upload_folder(folder_path=str(root),repo_id=REPO,repo_type="dataset",
                      path_in_repo=prefix,parent_commit=info.sha,
                      commit_message=f"Archive original NSE FY2023-2025 3FY strictly verified {runid}")
 print(json.dumps({"saved_verified_2025_three_FY_source_private":True,
   "numeric_company_count":meta["FY2023_FY2024_FY2025_three_year_company_numerics_verified"],
   "original_stock_universe":1314,
   "three_fy_numeric_coverage_percent":meta["three_fy_original_2025_universe_coverage_pct"],
   "owner_private_archive":prefix,"model_retrained":False},indent=2),flush=True)
def main():
 p=argparse.ArgumentParser();p.add_argument("--directory",required=True);p.add_argument("--run-id",required=True)
 a=p.parse_args();save(a.directory,os.getenv("HF_ARCHIVE_TOKEN","").strip(),a.run_id)
if __name__=="__main__":main()
