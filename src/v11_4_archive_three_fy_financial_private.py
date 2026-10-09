"""Archive verified V11.4 three fiscal-year historic data privately, no overwrite.

GitHub Actions artifacts are temporary: persist original NSE FY2021-23 source
hashes, retained rejection records, and complete original 2023 universe to
owner's PRIVATE Hugging Face research archive. Never publish stock picks.
"""
from __future__ import annotations
import argparse,hashlib,json,os
from pathlib import Path
from huggingface_hub import HfApi
REPO="ayushverma1988/v10-multibagger-archive"
PREFIX="v11_4/historical_source_original_NSE/FY2021_FY2022_FY2023_asof_2023_12_29"
REQUIRED=(
 "original_NSE_2021_2023_three_FY_STRICT_FINANCIAL_FACTS.csv",
 "original_NSE_2021_2023_strict_source_rejected_companies.csv",
 "original_NSE_2023_frozen_universe_3FY_POINT_IN_TIME_RESEARCH.parquet",
 "three_FY_full_numeric_recovery_2023_PIT_source_gate.json")
def archive(directory,token,run_id,api=None):
 if not token:raise ValueError("Missing private HF archive access")
 base=Path(directory)
 for name in REQUIRED:
  if not (base/name).is_file():raise ValueError("Three fiscal-year source report missing: "+name)
 s=json.loads((base/REQUIRED[-1]).read_text())
 if s.get("original_3_fy_link_company_count")!=965 or not s.get("no_missing_company_dropped_from_original_fold"):
  raise ValueError("Three-year original source integrity not proven")
 if s.get("frozen_model_ranking_and_training_unchanged") is not True:
  raise ValueError("Historical financial source illegally altered frozen stock selection")
 destination=f"{PREFIX}/verified_run_{run_id}"
 client=api or HfApi(token=token)
 info=client.repo_info(repo_id=REPO,repo_type="dataset")
 if info.private is not True:raise ValueError("Refusing public research archive")
 existing=client.list_repo_files(repo_id=REPO,repo_type="dataset")
 if any(x.startswith(destination+"/") for x in existing):
  raise ValueError("Immutable research financial source archive already exists")
 checksums={name:hashlib.sha256((base/name).read_bytes()).hexdigest() for name in REQUIRED}
 (base/"archived_original_three_fy_SHA256_manifest.json").write_text(json.dumps({
  "scope":"IMMUTABLE_OWNER_PRIVATE_3FY_NSE_HISTORICAL_SOURCE",
  "github_run_id":str(run_id),"historical_decision_date_IST":"2023-12-29",
  "file_SHA256":checksums,"financial_numeric_source_rejected_not_imputed":True,
  "not_predictive_model_training":True},indent=2))
 client.upload_folder(folder_path=str(base),repo_id=REPO,repo_type="dataset",
                      path_in_repo=destination,parent_commit=info.sha,
                      commit_message=f"Archive audited original NSE 3FY historical research {run_id}")
 print(json.dumps({"stored_original_three_FY_private":True,
    "historical_verified_company_pairs":s.get("original_3_fy_fiscal_numeric_verified"),
    "historical_original_stock_universe":s.get("historical_stock_universe_count"),
    "original_3_fy_source_coverage_percent":s.get("annual_3_FY_full_numeric_coverage_percent"),
    "private_prefix":destination,"model_predictions_modified":False},indent=2),flush=True)
def main():
 a=argparse.ArgumentParser();a.add_argument("--directory",required=True);a.add_argument("--run-id",required=True)
 arg=a.parse_args();archive(arg.directory,os.getenv("HF_ARCHIVE_TOKEN","").strip(),arg.run_id)
if __name__=="__main__":main()
