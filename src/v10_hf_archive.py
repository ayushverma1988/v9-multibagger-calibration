from __future__ import annotations

import argparse, hashlib, json, os, shutil
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

SOURCE_DATASET="tejhq/indian-markets"
DEFAULT_ARCHIVE_REPO="ayushverma1988/v10-multibagger-archive"


def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def copy_if_exists(src: Path, dst: Path, files: list[dict]):
    if not src.exists():
        return
    dst.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(src,dst)
    files.append({
        "path":str(dst),
        "bytes":int(dst.stat().st_size),
        "sha256":sha256_file(dst),
    })


def stage_market_day(data_end: pd.Timestamp, stage: Path, files: list[dict]):
    year=int(data_end.year)
    src=hf_hub_download(
        repo_id=SOURCE_DATASET,
        filename=f"nse/year={year}/nse_{year}.parquet",
        repo_type="dataset",
    )
    df=pd.read_parquet(src)
    if "date" not in df:
        raise RuntimeError("Source NSE parquet missing date column")
    df["date"]=pd.to_datetime(df["date"])
    day=df[df["date"].dt.normalize()==data_end.normalize()].copy()
    if day.empty:
        raise RuntimeError(f"No raw NSE rows found for production data_end={data_end.date()}")
    out=stage/f"market_data/nse_daily/{year}/{data_end.date()}.parquet"
    out.parent.mkdir(parents=True,exist_ok=True)
    day.to_parquet(out,index=False)
    files.append({"path":str(out),"bytes":int(out.stat().st_size),"sha256":sha256_file(out),"rows":int(len(day))})


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--production-dir",required=True)
    ap.add_argument("--events-current",required=True)
    ap.add_argument("--events-recent",required=True)
    ap.add_argument("--event-summary",required=True)
    ap.add_argument("--repo-id",default=DEFAULT_ARCHIVE_REPO)
    ap.add_argument("--stage-dir",default="hf_archive_stage")
    ap.add_argument("--market-ingest-dir",default=None)
    args=ap.parse_args()

    token=os.environ.get("HF_ARCHIVE_TOKEN")
    if not token:
        raise RuntimeError("HF_ARCHIVE_TOKEN is not set")

    prod=Path(args.production_dir)
    manifest_path=prod/"production_manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"Missing production manifest: {manifest_path}")
    manifest=json.load(open(manifest_path))
    data_end=pd.Timestamp(manifest["data_end"]).normalize()
    run_date=pd.Timestamp.now(tz="Asia/Kolkata").date()

    stage=Path(args.stage_dir)
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    files=[]

    # 1) Raw daily market source used by the production date.
    # Prefer the exact official NSE files when this production run used them.
    if args.market_ingest_dir:
        mid=Path(args.market_ingest_dir)
        sm=mid/"market_ingest_summary.json"
        if sm.exists():
            ingest=json.load(open(sm))
            td=pd.Timestamp(ingest["target_date"])
            root=stage/f"market_data/nse_official_raw/{td.year}/{td.month:02d}/{td.date()}"
            for p in (mid/"raw").glob("*"):
                if p.is_file():
                    copy_if_exists(p,root/p.name,files)
            copy_if_exists(sm,root/"market_ingest_summary.json",files)
            overlay=Path(ingest.get("overlay_parquet",""))
            if overlay.exists():
                copy_if_exists(overlay,stage/f"market_data/current_year_overlays/{td.year}/nse_{td.year}_official_overlay_{td.date()}.parquet",files)
        else:
            stage_market_day(data_end,stage,files)
    else:
        stage_market_day(data_end,stage,files)

    # 2) Exact recent NSE announcement fetch for this run.
    recent_src=Path(args.events_recent)
    recent_dst=stage/f"corporate_announcements/scrape_windows/{run_date.year}/{run_date}_recent_fetch.parquet"
    copy_if_exists(recent_src,recent_dst,files)

    # 3) Production outputs and audit evidence, immutable per production date.
    run_root=stage/f"production_runs/{data_end.date()}"
    for name in [
        "latest_top10.csv",
        "latest_top10_compact.csv",
        "current_universe_frozen_selector.csv",
        "production_manifest.json",
        "integrity_summary.json",
        "drift_report.json",
        "latest_status.json",
        "event_refresh_summary.json",
        "prospective_performance_by_batch.csv",
    ]:
        copy_if_exists(prod/name,run_root/name,files)

    # 4) Moving long-term ledger copies; Hub commit history preserves prior versions.
    copy_if_exists(prod/"prediction_ledger.csv",stage/"prediction_ledger/prediction_ledger.csv",files)
    copy_if_exists(prod/"latest_status.json",stage/"manifests/latest_status.json",files)
    copy_if_exists(prod/"production_manifest.json",stage/"manifests/latest_production_manifest.json",files)

    # 5) Archive the event refresh summary separately by scrape date.
    copy_if_exists(Path(args.event_summary),stage/f"corporate_announcements/refresh_summaries/{run_date.year}/{run_date}.json",files)

    api=HfApi(token=token)
    info=api.repo_info(repo_id=args.repo_id,repo_type="dataset")
    if not getattr(info,"private",False):
        raise RuntimeError("Archive dataset must be private")

    remote_files=set(api.list_repo_files(repo_id=args.repo_id,repo_type="dataset"))
    canonical_remote="corporate_announcements/canonical/events_all_current.parquet"
    if canonical_remote not in remote_files:
        canonical_src=Path(args.events_current)
        canonical_dst=stage/canonical_remote
        copy_if_exists(canonical_src,canonical_dst,files)

    archive_manifest={
        "archive_repo":args.repo_id,
        "archive_private":True,
        "production_model":manifest.get("production_model"),
        "data_end":str(data_end.date()),
        "archive_run_date_ist":str(run_date),
        "github_run_id":os.environ.get("GITHUB_RUN_ID"),
        "github_sha":os.environ.get("GITHUB_SHA"),
        "source_market_dataset":SOURCE_DATASET,
        "canonical_event_baseline_uploaded":canonical_remote not in remote_files,
        "files":files,
    }
    dated_manifest=stage/f"manifests/archive_runs/{run_date.year}/{run_date}_{os.environ.get('GITHUB_RUN_ID','manual')}.json"
    dated_manifest.parent.mkdir(parents=True,exist_ok=True)
    json.dump(archive_manifest,open(dated_manifest,"w"),indent=2,default=str)
    files.append({"path":str(dated_manifest),"bytes":int(dated_manifest.stat().st_size),"sha256":sha256_file(dated_manifest)})

    latest_manifest=stage/"manifests/latest_archive_manifest.json"
    latest_manifest.parent.mkdir(parents=True,exist_ok=True)
    json.dump(archive_manifest,open(latest_manifest,"w"),indent=2,default=str)

    res=api.upload_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=str(stage),
        path_in_repo="",
        commit_message=f"Archive V10.2 production run {data_end.date()} (GitHub {os.environ.get('GITHUB_RUN_ID','manual')})",
    )

    print(json.dumps({
        "archive_repo":args.repo_id,
        "private":True,
        "data_end":str(data_end.date()),
        "staged_files":len(files)+1,
        "commit_url":str(res),
        "canonical_baseline_uploaded":canonical_remote not in remote_files,
    },indent=2))


if __name__=="__main__":
    main()
