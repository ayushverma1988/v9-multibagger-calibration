"""Archive original as-of-2024 NSE three fiscal-year source in owner-private HF."""
import argparse,json,os,hashlib
from pathlib import Path
from huggingface_hub import HfApi
REPO="ayushverma1988/v10-multibagger-archive"
ROOT="v11_4/historical_source_original_NSE/FY2022_FY2023_FY2024_asof_2024_12_31"
FILES=("strict_original_FY2022_2024_annual_facts.csv",
       "strict_rejected_original_2024_company_financial_sources.csv",
       "original_dec2024_stock_universe_3FY_financial_sidecar.parquet",
       "original_2024_full_3FY_reconciliation.json")
def save(folder,run_id,token):
 p=Path(folder)
 for f in FILES:
  if not (p/f).is_file():raise ValueError("Missing full historical 2024 source: "+f)
 q=json.loads((p/FILES[-1]).read_text())
 if (q["original_dec2024_equities"]!=1345 or
     q["exact_historic_stock_universe_preserved"] is not True or
     q["no_imputation"] is not True or
     q["frozen_V11_4_scores_changed"] is not False):
  raise ValueError("Original 2024 historic company membership or source integrity failed")
 if not token:raise ValueError("HF_ARCHIVE_TOKEN missing")
 api=HfApi(token=token)
 info=api.repo_info(repo_id=REPO,repo_type="dataset")
 if info.private is not True:raise ValueError("Financial research archive must remain private")
 dest=f"{ROOT}/verified_run_{run_id}"
 if any(x.startswith(dest+"/") for x in api.list_repo_files(repo_id=REPO,repo_type="dataset")):
  raise ValueError("Historical owner-private filing archive cannot be overwritten")
 hashes={f:hashlib.sha256((p/f).read_bytes()).hexdigest() for f in FILES}
 (p/"original_fiscal_FY2022_2024_file_hash_manifest.json").write_text(json.dumps({
  "scope":"2024_3FY_PRIVATE_SOURCE_NO_PREDICTION_TRAINING",
  "historical_decision_date":"2024-12-31",
  "source_files_SHA256":hashes,
  "numeric_verified_companies":q["source_verified_FY2022_FY2023_FY2024_companies"]},indent=2))
 api.upload_folder(folder_path=str(p),repo_id=REPO,repo_type="dataset",
     path_in_repo=dest,parent_commit=info.sha,commit_message="Original NSE as-of2024 3FY strict numerical research source")
 print(json.dumps({"owner_private_NSE_2024_threeFY_saved":True,
  "complete_source_companies":q["source_verified_FY2022_FY2023_FY2024_companies"],
  "original_stock_universe":1345,
  "numeric_source_coverage_pct":q["three_FY_numeric_coverage_percent"],
  "private_path":dest,
  "model_unchanged":True},indent=2))
def main():
 p=argparse.ArgumentParser();p.add_argument("--directory",required=True)
 p.add_argument("--run-id",required=True)
 a=p.parse_args()
 save(a.directory,a.run_id,os.getenv("HF_ARCHIVE_TOKEN",""))
if __name__=="__main__":main()
