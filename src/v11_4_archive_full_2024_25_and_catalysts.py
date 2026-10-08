"""Archive SHA256-provenanced V11.4 point-in-time source research privately.

This operation does not write V10 production, publish top picks, or claim
successful multibagger backtesting. No private HF data goes to public repos.
"""
from __future__ import annotations
import argparse,hashlib,json,os,shutil
from pathlib import Path
from huggingface_hub import HfApi

def digest(file):
 h=hashlib.sha256()
 with open(file,"rb") as f:
  for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
 return h.hexdigest()

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--financial",required=True)
 p.add_argument("--catalysts",required=True)
 p.add_argument("--destination",required=True)
 p.add_argument("--repo-id",required=True)
 p.add_argument("--prefix",required=True)
 a=p.parse_args()
 fin=Path(a.financial);cat=Path(a.catalysts);out=Path(a.destination)
 f=json.loads((fin/"full_2fold_numeric_research_summary.json").read_text())
 c=json.loads((cat/"historical_catalyst_18fold_summary.json").read_text())
 if f.get("verified_source_matrix_rows")!=2659 or f.get("scope")!="FULL_TWOFOLD_NUMERIC_SOURCE_READY_NOT_MODEL_PREDICTION":
  raise SystemExit("Unverified or incomplete annual numeric reconciliation")
 if c.get("original_fold_count")!=18 or c.get("matrix_rows")!=18569:
  raise SystemExit("Incomplete original 18-fold NSE event source")
 if f.get("training_done") or c.get("six_month_market_backtest_completed"):
  raise SystemExit("Reject unexpected predictive backtest status")
 token=os.environ.get("HF_ARCHIVE_TOKEN","")
 if not token:raise SystemExit("No approved private Hugging Face archive credentials")
 wanted={
  "annual_numeric":(fin,[
   "full_2024_2025_annual_XBRL_PIT_RESEARCH.parquet",
   "full_annual_numeric_by_fold.csv",
   "full_2fold_numeric_research_summary.json"]),
  "historical_catalyst":(cat,[
   "canonical_NSE_18fold_catalyst_metadata_PIT_RESEARCH.parquet",
   "18fold_catalyst_coverage.csv",
   "historical_catalyst_18fold_summary.json",
   "sample_source_provenance_by_fold.csv"])
 }
 out.mkdir(parents=True,exist_ok=True)
 manifest={"scope":"V11_4_RESEARCH_ONLY_NO_V10_CHANGE",
           "financial_fold_count":2,"catalyst_fold_count":18,
           "numeric_2024_revenue_coverage":f["revenue_coverage"][0],
           "numeric_2025_revenue_coverage":f["revenue_coverage"][1],
           "source_files":{}}
 for folder,(source,files) in wanted.items():
  dest=out/folder;dest.mkdir(exist_ok=True)
  for name in files:
   path=source/name
   if not path.is_file():raise SystemExit("Missing required provenance "+str(path))
   shutil.copy2(path,dest/name)
   manifest["source_files"][f"{folder}/{name}"]={
    "sha256":digest(dest/name),"bytes":(dest/name).stat().st_size
   }
 (out/"source_provenance_manifest.json").write_text(json.dumps(manifest,indent=2))
 api=HfApi(token=token)
 info=api.repo_info(repo_id=a.repo_id,repo_type="dataset")
 if not getattr(info,"private",False):
  raise SystemExit("Refusing data upload: target dataset not private")
 api.upload_folder(repo_id=a.repo_id,repo_type="dataset",
                   folder_path=str(out),path_in_repo=a.prefix.strip("/"),
                   commit_message="Archive V11.4 verified 2fold financial and 18fold NSE catalyst PIT research")
 print(json.dumps({"saved_to_private_HF":True,"repo":a.repo_id,
                   "path":a.prefix,"files_saved":len(manifest["source_files"])+1,
                   "source_manifest_sha256":digest(out/"source_provenance_manifest.json"),
                   "original_v10_unchanged":True},indent=2))
if __name__=="__main__":main()
