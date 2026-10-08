"""Durable private HF archive for fully reconciled research PIT feature inputs.

Preserves exact SHA-256 file manifests. Refuses public HF and rejects
incomplete original 18-fold or erroneous marketwide annual numeric outputs.
Does not modify V10 production or call any scoring model.
"""
from __future__ import annotations
import argparse,hashlib,json,os,shutil
from pathlib import Path
from huggingface_hub import HfApi

def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
    return h.hexdigest()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--numeric",required=True)
    ap.add_argument("--catalyst",required=True)
    ap.add_argument("--output",required=True)
    ap.add_argument("--repo-id",required=True)
    ap.add_argument("--prefix",required=True)
    a=ap.parse_args()
    numeric=Path(a.numeric);catalyst=Path(a.catalyst);out=Path(a.output)
    summary=json.loads((numeric/"eight_shard_full_annual_research_summary.json").read_text())
    combined=json.loads((numeric/"full_18fold_research_matrix"/"18fold_research_feature_summary.json").read_text())
    past=json.loads((catalyst/"historical_catalyst_18fold_summary.json").read_text())
    if not (summary["FY2024_verified_companies"]>=1100 and summary["FY2025_verified_companies"]>=1100
            and summary["source_shards_2024"]==4 and summary["source_shards_2025"]==4):
        raise SystemExit("Refuse to archive incomplete full-market annual source backfill")
    if combined.get("folds")!=18 or combined.get("historical_rows")!=18569:
        raise SystemExit("Refuse altered historical fold matrix")
    if combined.get("forward_target_label_columns_in_matrix") is not False:
        raise SystemExit("Forward outcomes contaminate PIT research source")
    if past.get("original_fold_count")!=18 or past.get("matrix_rows")!=18569:
        raise SystemExit("Original 18fold NSE catalyst source inconsistent")
    sourcefiles={
        "annual/verified_FY2024_annual_yoy_PIT_candidates.csv":numeric/"verified_FY2024_annual_yoy_PIT_candidates.csv",
        "annual/verified_FY2025_annual_yoy_PIT_candidates.csv":numeric/"verified_FY2025_annual_yoy_PIT_candidates.csv",
        "annual/eight_shard_full_annual_research_summary.json":numeric/"eight_shard_full_annual_research_summary.json",
        "annual/all_eight_annual_source_shard_audit.csv":numeric/"all_eight_annual_source_shard_audit.csv",
        "financial_matrix/v11_4_research_2024_2025_annual_yoy_PIT.parquet":numeric/"large_twofold_matrix"/"v11_4_research_2024_2025_annual_yoy_PIT.parquet",
        "financial_matrix/short_horizon_2fold_matrix_summary.json":numeric/"large_twofold_matrix"/"short_horizon_2fold_matrix_summary.json",
        "combined/v11_4_18fold_source_PIT_FEATURES_RESEARCH.parquet":numeric/"full_18fold_research_matrix"/"v11_4_18fold_source_PIT_FEATURES_RESEARCH.parquet",
        "combined/18fold_source_feature_coverage.csv":numeric/"full_18fold_research_matrix"/"18fold_source_feature_coverage.csv",
        "combined/18fold_research_feature_summary.json":numeric/"full_18fold_research_matrix"/"18fold_research_feature_summary.json",
        "catalysts/historical_catalyst_18fold_summary.json":catalyst/"historical_catalyst_18fold_summary.json",
        "catalysts/18fold_catalyst_coverage.csv":catalyst/"18fold_catalyst_coverage.csv",
        "catalysts/sample_source_provenance_by_fold.csv":catalyst/"sample_source_provenance_by_fold.csv",
    }
    out.mkdir(parents=True,exist_ok=True)
    manifests={}
    for target,src in sourcefiles.items():
        if not src.is_file():raise SystemExit("Source absent: "+str(src))
        dest=out/target
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(src,dest)
        manifests[target]={"sha256":sha(dest),"bytes":dest.stat().st_size}
    manifest={
        "scope":"RESEARCH_UNTRAINED_SOURCE_ONLY_NEVER_V10_PRODUCTION",
        "date":"2026-10-08",
        "historic_folds":18,
        "historic_stock_dates":18569,
        "FY2024_verified_annual_companies":summary["FY2024_verified_companies"],
        "FY2025_verified_annual_companies":summary["FY2025_verified_companies"],
        "files":manifests,
        "next_gate":"Original frozen V10 selection-matched 6mo alpha/lift/dd30 and 12-of-18 numeric coverage",
        "V11_4_prediction_alpha_verified":False,
    }
    (out/"provenance.json").write_text(json.dumps(manifest,indent=2))
    token=os.environ.get("HF_ARCHIVE_TOKEN","").strip()
    if not token:raise SystemExit("HF_ARCHIVE_TOKEN is missing")
    api=HfApi(token=token)
    if not api.repo_info(repo_id=a.repo_id,repo_type="dataset").private:
        raise SystemExit("Refuse archival to public dataset")
    api.upload_folder(repo_id=a.repo_id,repo_type="dataset",folder_path=str(out),
        path_in_repo=a.prefix.strip("/"),commit_message="Freeze original NSE 18fold research events and verified 2024/25 annual financials")
    print(json.dumps({"private_archive_saved":True,"files":len(sourcefiles)+1,
        "repo_id":a.repo_id,"prefix":a.prefix,
        "sha256_manifest":sha(out/"provenance.json")},indent=2))
if __name__=="__main__":main()
