from __future__ import annotations

import argparse, hashlib, json, os, shutil
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

DEFAULT_REPO="ayushverma1988/v10-multibagger-archive"


def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--repo-id",default=DEFAULT_REPO)
    ap.add_argument("--output-dir",default="legacy")
    args=ap.parse_args()

    token=os.environ.get("HF_ARCHIVE_TOKEN")
    if not token:
        raise RuntimeError("HF_ARCHIVE_TOKEN is not set")

    manifest_src=hf_hub_download(
        repo_id=args.repo_id,
        filename="market_data/legacy/manifest.json",
        repo_type="dataset",
        token=token,
    )
    manifest=json.load(open(manifest_src))
    expected={int(x["year"]):x for x in manifest["files"]}

    out=Path(args.output_dir)
    out.mkdir(parents=True,exist_ok=True)
    verified=[]
    for y in range(2003,2010):
        if y not in expected:
            raise RuntimeError(f"Legacy archive manifest missing year {y}")
        src=Path(hf_hub_download(
            repo_id=args.repo_id,
            filename=f"market_data/legacy/nse_{y}.parquet",
            repo_type="dataset",
            token=token,
        ))
        dst=out/f"nse_{y}.parquet"
        shutil.copy2(src,dst)

        digest=sha256_file(dst)
        if digest != expected[y]["sha256"]:
            raise RuntimeError(f"SHA256 mismatch for {y}: {digest} != {expected[y]['sha256']}")

        x=pd.read_parquet(dst)
        x["date"]=pd.to_datetime(x["date"])
        if len(x) != int(expected[y]["rows"]):
            raise RuntimeError(f"Row-count mismatch for {y}")
        if not x["date"].dt.year.eq(y).all():
            raise RuntimeError(f"Out-of-year rows in {y}")
        if x["date"].nunique() != int(expected[y]["trading_dates"]):
            raise RuntimeError(f"Trading-date count mismatch for {y}")
        verified.append({
            "year":y,
            "rows":int(len(x)),
            "symbols":int(x["symbol"].nunique()),
            "trading_dates":int(x["date"].nunique()),
            "sha256":digest,
        })

    summary={
        "status":"verified_private_hf_legacy_archive",
        "repo_id":args.repo_id,
        "years":"2003-2009",
        "files":verified,
    }
    (out/"legacy_fetch_summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
