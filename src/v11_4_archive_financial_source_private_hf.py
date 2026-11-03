"""Persist a validated V11.4 research data artifact to the existing PRIVATE HF dataset.

Uses a dedicated immutable prefix; does not replace V10 production snapshots.
"""
from __future__ import annotations
import argparse,hashlib,json,os
from pathlib import Path
from huggingface_hub import HfApi

def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):h.update(chunk)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source-dir",required=True)
    ap.add_argument("--repo-id",required=True)
    ap.add_argument("--prefix",required=True)
    args=ap.parse_args()
    token=os.environ.get("HF_ARCHIVE_TOKEN","").strip()
    if not token:raise SystemExit("HF_ARCHIVE_TOKEN not available; archive not committed")
    folder=Path(args.source_dir)
    summary_path=folder/"full_integrated_source_summary.json"
    if not summary_path.exists():raise SystemExit("Source completion report missing")
    status=json.loads(summary_path.read_text())
    if status.get("status")!="FULL_HISTORICAL_UNIVERSE_INDEX_RECOVERY_REHEARSAL":
        raise SystemExit("Refuse to archive unrecognized unvalidated dataset")
    if status.get("completed_shards")!=4 or status.get("filing_metadata_timestamp_violations"):
        raise SystemExit("Refuse incomplete or PIT-invalid source")
    files=sorted(p for p in folder.iterdir() if p.is_file())
    if not files:raise SystemExit("No files to archive")
    manifest={"source_summary":status,"files":{p.name:{"bytes":p.stat().st_size,"sha256":digest(p)} for p in files},
              "purpose":"Research-only point-in-time integrated financial source; not V10 production"}
    (folder/"archive_provenance_manifest.json").write_text(json.dumps(manifest,indent=2))
    api=HfApi(token=token)
    info=api.repo_info(repo_id=args.repo_id,repo_type="dataset")
    if not getattr(info,"private",False):
        raise SystemExit("Refuse to publish source archive to a non-private HF dataset")
    api.upload_folder(repo_id=args.repo_id,repo_type="dataset",folder_path=str(folder),
                      path_in_repo=args.prefix.strip("/"),
                      commit_message="Archive V11.4 2025 NSE integrated financial source research (PIT)")
    print(json.dumps({"archived":True,"repo_id":args.repo_id,"path_in_repo":args.prefix,
                      "files":len(files)+1,"sha256_manifest":digest(folder/"archive_provenance_manifest.json")},indent=2))
if __name__=="__main__":main()
