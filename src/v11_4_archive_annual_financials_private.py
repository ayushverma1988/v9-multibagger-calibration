"""Archive immutable, source-audited V11.4 financial research inputs privately.

Do NOT overwrite V10 production. Preserve file digests and source provenance.
Only source index and 2-fold research feature matrix are uploaded. The index
does not imply valid 5y/7y financial numeric histories.
"""
import argparse,hashlib,json,os,shutil
from pathlib import Path
from huggingface_hub import HfApi

def filehash(p):
    h=hashlib.sha256()
    with p.open("rb") as fh:
        for piece in iter(lambda:fh.read(1024*1024),b""):h.update(piece)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--annual",required=True)
    p.add_argument("--matrix",required=True)
    p.add_argument("--repo-id",required=True)
    p.add_argument("--prefix",required=True)
    p.add_argument("--stage",required=True)
    a=p.parse_args()
    src=Path(a.annual);matrix=Path(a.matrix);stage=Path(a.stage)
    source=json.loads((src/"full_historical_annual_index_summary.json").read_text())
    research=json.loads((matrix/"short_horizon_2fold_matrix_summary.json").read_text())
    if source.get("original_fold_union_companies")!=2037 or source.get("companies_queried")!=2037:
        raise SystemExit("Annual full-market historical universe differs from expected 2037")
    if source.get("source_errors") or source.get("point_in_time_violations")!=0:
        raise SystemExit("Refuse to archive unverified or incomplete source")
    if research.get("folds")!=2 or research.get("rows")!=2659:
        raise SystemExit("Unexpected research feature matrix size or version")
    if research.get("final_18fold_model_backtest_done") is not False:
        raise SystemExit("Model test status unexpectedly marked complete")
    token=os.environ.get("HF_ARCHIVE_TOKEN","").strip()
    if not token:raise SystemExit("HF_ARCHIVE_TOKEN secret absent")
    for filename in ("full_historical_annual_index_PIT_CANDIDATES.parquet",
                     "full_historical_annual_index_summary.json"):
        if not (src/filename).is_file():raise SystemExit("Missing annual source "+filename)
    for filename in ("v11_4_research_2024_2025_annual_yoy_PIT.parquet",
                     "research_feature_coverage_by_fold.csv",
                     "short_horizon_2fold_matrix_summary.json"):
        if not (matrix/filename).is_file():raise SystemExit("Missing feature matrix "+filename)
    stage.mkdir(parents=True,exist_ok=True)
    files={}
    for label,root,names in [
        ("source",src,("full_historical_annual_index_PIT_CANDIDATES.parquet",
                       "full_historical_annual_index_summary.json")),
        ("features",matrix,("v11_4_research_2024_2025_annual_yoy_PIT.parquet",
                            "research_feature_coverage_by_fold.csv",
                            "short_horizon_2fold_matrix_summary.json"))]:
        (stage/label).mkdir(exist_ok=True)
        for name in names:
            shutil.copy2(root/name,stage/label/name)
            files[f"{label}/{name}"]={
                "sha256":filehash(stage/label/name),
                "bytes":(stage/label/name).stat().st_size
            }
    manifest={
        "scope":"V11_4_RESEARCH_ONLY_NOT_PRODUCTION",
        "source_companies":source["companies_queried"],
        "source_index_rows":source["total_annual_index_rows"],
        "research_two_fold_rows":research["rows"],
        "missing_numeric_fields_preserved":True,
        "full_18_fold_model_accuracy_accepted":False,
        "files":files,
    }
    (stage/"provenance.json").write_text(json.dumps(manifest,indent=2))
    api=HfApi(token=token)
    info=api.repo_info(repo_id=a.repo_id,repo_type="dataset")
    if not getattr(info,"private",False):
        raise SystemExit("Refuse to redistribute source to public HF repository")
    api.upload_folder(
        repo_id=a.repo_id,repo_type="dataset",
        folder_path=str(stage),path_in_repo=a.prefix.strip("/"),
        commit_message="Archive verified historical NSE annual source and 2fold annual research features"
    )
    print(json.dumps({"private_archive_saved":True,"dataset":a.repo_id,
                      "path":a.prefix,"files_uploaded":len(files)+1,
                      "provenance_sha256":filehash(stage/"provenance.json")},indent=2))
if __name__=="__main__":main()
