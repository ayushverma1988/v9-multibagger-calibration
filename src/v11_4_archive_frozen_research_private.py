"""Archive strictly frozen V11.4 standalone *research* release to private HF.

Only this frozen research job's original artifact is accepted. No other
model, no current ranks without verified live NSE inputs, no public dataset.
"""
import argparse,hashlib,json,os
from pathlib import Path
from huggingface_hub import HfApi

EXPECTED_ID="V11.4-standalone-research-frozen-20261008"
FILES=["v11_4_frozen_research_model.joblib",
       "v11_4_frozen_research_manifest.json",
       "prospective_selection_registry_EMPTY.csv"]
REPO="ayushverma1988/v10-multibagger-archive"
PREFIX="v11_4/frozen_research_candidate/20261008"

def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
 return h.hexdigest()

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--folder",required=True)
 a=p.parse_args()
 src=Path(a.folder)
 for n in FILES:
  if not (src/n).is_file():raise SystemExit("Missing frozen artifact "+n)
 manifest=json.loads((src/FILES[1]).read_text())
 if manifest.get("model_id")!=EXPECTED_ID or manifest.get("production_approved") is not False:
  raise SystemExit("Not a valid V11.4 research-only release")
 if manifest.get("genuinely_unseen_future_holdout_completed") is not False:
  raise SystemExit("Do not re-label nonconfirmatory research")
 if manifest.get("source_SHA256",{}).get("frozen_model_joblib")!=sha(src/FILES[0]):
  raise SystemExit("Frozen model SHA256 changed")
 if len((src/FILES[2]).read_text().splitlines())!=1:
  raise SystemExit("Prospective outcomes ledger pre-populated using historical selections")
 token=os.getenv("HF_ARCHIVE_TOKEN","").strip()
 if not token:raise SystemExit("HF_ARCHIVE_TOKEN absent")
 api=HfApi(token=token)
 if not api.repo_info(repo_id=REPO,repo_type="dataset").private:
  raise SystemExit("Refuse to publish private model in public repo")
 api.upload_folder(repo_id=REPO,repo_type="dataset",folder_path=str(src),
    path_in_repo=PREFIX,commit_message="V11.4 frozen standalone six-month research model and immutable forward-only registry")
 print(json.dumps({"private_archive_success":True,"repo":REPO,"prefix":PREFIX,
       "archive_files":len(FILES),"model_SHA256":sha(src/FILES[0]),
       "manifest_SHA256":sha(src/FILES[1]),
       "not_production_approved":True},indent=2))
if __name__=="__main__":main()
