from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import zipfile
from pathlib import Path

import pandas as pd
import requests
from huggingface_hub import HfApi, hf_hub_download

DEFAULT_REPO="ayushverma1988/v10-multibagger-archive"
GITHUB_REPO="ayushverma1988/v9-multibagger-calibration"

# Exact legacy artifacts used by the accepted V10.2 integrity validation run.
LEGACY_ARTIFACTS={
    2003:10959427979,
    2004:10959721175,
    2005:10959706367,
    2006:10959373233,
    2007:10959004485,
    2008:10959328276,
    2009:10959097834,
}
REQUIRED={"date","symbol","series","close","volume","turnover"}


def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def validate_file(path:Path,year:int,expected:dict|None=None)->dict:
    x=pd.read_parquet(path)
    missing=sorted(REQUIRED-set(x.columns))
    if missing:
        raise RuntimeError(f"{path} missing columns {missing}")
    x["date"]=pd.to_datetime(x["date"],errors="coerce")
    if x["date"].isna().any():
        raise RuntimeError(f"{path} contains invalid dates")
    if not x["date"].dt.year.eq(year).all():
        raise RuntimeError(f"Out-of-year rows in {year}")
    if len(x)<10000 or x["date"].nunique()<150:
        raise RuntimeError(f"Legacy file {year} unexpectedly sparse: rows={len(x)} dates={x['date'].nunique()}")
    digest=sha256_file(path)
    if expected is not None:
        if digest != expected["sha256"]:
            raise RuntimeError(f"SHA256 mismatch for {year}")
        if len(x) != int(expected["rows"]):
            raise RuntimeError(f"Row-count mismatch for {year}")
        if x["date"].nunique() != int(expected["trading_dates"]):
            raise RuntimeError(f"Trading-date mismatch for {year}")
    return {
        "year":year,
        "rows":int(len(x)),
        "symbols":int(x["symbol"].nunique()),
        "trading_dates":int(x["date"].nunique()),
        "date_min":str(x["date"].min().date()),
        "date_max":str(x["date"].max().date()),
        "sha256":digest,
    }


def restore_hf(repo_id:str,out:Path,token:str)->list[dict]:
    manifest_src=hf_hub_download(
        repo_id=repo_id,
        filename="market_data/legacy/manifest.json",
        repo_type="dataset",
        token=token,
    )
    manifest=json.load(open(manifest_src))
    expected={int(x["year"]):x for x in manifest["files"]}
    verified=[]
    for y in range(2003,2010):
        if y not in expected:
            raise RuntimeError(f"Legacy archive manifest missing year {y}")
        src=Path(hf_hub_download(
            repo_id=repo_id,
            filename=f"market_data/legacy/nse_{y}.parquet",
            repo_type="dataset",
            token=token,
        ))
        dst=out/f"nse_{y}.parquet"
        shutil.copy2(src,dst)
        verified.append(validate_file(dst,y,expected[y]))
    return verified


def bootstrap_from_github(out:Path,repo_id:str,hf_token:str,gh_token:str)->list[dict]:
    headers={
        "Authorization":f"Bearer {gh_token}",
        "Accept":"application/vnd.github+json",
        "X-GitHub-Api-Version":"2022-11-28",
    }
    verified=[]
    for y,aid in LEGACY_ARTIFACTS.items():
        url=f"https://api.github.com/repos/{GITHUB_REPO}/actions/artifacts/{aid}/zip"
        r=requests.get(url,headers=headers,timeout=120)
        r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            matches=[n for n in z.namelist() if Path(n).name==f"nse_{y}.parquet"]
            if len(matches)!=1:
                raise RuntimeError(f"Artifact {aid} for {y} contains {matches}")
            dst=out/f"nse_{y}.parquet"
            with z.open(matches[0]) as src,dst.open("wb") as d:
                shutil.copyfileobj(src,d)
        verified.append(validate_file(dst,y))

    manifest={
        "source":"Pinned GitHub Actions artifacts used by accepted V10.2 integrity validation",
        "source_repository":GITHUB_REPO,
        "artifact_ids":LEGACY_ARTIFACTS,
        "years":"2003-2009",
        "files":verified,
    }
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2))

    api=HfApi(token=hf_token)
    info=api.repo_info(repo_id,repo_type="dataset")
    if not info.private:
        raise RuntimeError("Legacy archive destination must be private")
    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(out),
        path_in_repo="market_data/legacy",
        commit_message="Persist accepted V10.2 NSE 2003-2009 legacy history",
    )
    return verified


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--repo-id",default=DEFAULT_REPO)
    ap.add_argument("--output-dir",default="legacy")
    ap.add_argument("--bootstrap-github-artifacts",action="store_true")
    args=ap.parse_args()

    hf_token=os.environ.get("HF_ARCHIVE_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_ARCHIVE_TOKEN is not set")

    out=Path(args.output_dir)
    out.mkdir(parents=True,exist_ok=True)

    source="private_hf_archive"
    try:
        verified=restore_hf(args.repo_id,out,hf_token)
    except Exception as exc:
        if not args.bootstrap_github_artifacts:
            raise
        print(f"HF legacy restore unavailable, bootstrapping pinned accepted artifacts: {exc}",flush=True)
        gh_token=os.environ.get("GITHUB_TOKEN")
        if not gh_token:
            raise RuntimeError("GITHUB_TOKEN required for legacy bootstrap")
        verified=bootstrap_from_github(out,args.repo_id,hf_token,gh_token)
        # Re-read from HF and verify the persisted copy before production uses it.
        for p in out.glob("nse_*.parquet"):
            p.unlink()
        verified=restore_hf(args.repo_id,out,hf_token)
        source="pinned_artifacts_seeded_and_verified_in_private_hf"

    summary={
        "status":"verified_private_hf_legacy_archive",
        "source":source,
        "repo_id":args.repo_id,
        "years":"2003-2009",
        "files":verified,
    }
    (out/"legacy_fetch_summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
