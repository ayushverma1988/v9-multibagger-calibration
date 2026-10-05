from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests
from huggingface_hub import hf_hub_download

HF_REPO = "tejhq/indian-markets"
UDIFF_BASE = "https://archives.nseindia.com/content/cm"
FULL_BASE = "https://archives.nseindia.com/products/content"
HEADERS = {
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/129 Safari/537.36",
    "accept": "*/*",
    "accept-language": "en-US,en;q=0.9",
}
SERIES = {"EQ", "BE", "BZ"}


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def get(url: str, timeout: int = 90) -> bytes:
    r = requests.get(url, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    if not r.content:
        raise RuntimeError(f"empty response: {url}")
    return r.content


def first_col(df: pd.DataFrame, names: list[str], required: bool = True):
    cmap = {str(c).strip().upper(): c for c in df.columns}
    for n in names:
        c = cmap.get(n.upper())
        if c is not None:
            return c
    if required:
        raise KeyError(f"missing columns {names}; got {list(df.columns)}")
    return None


def normalize_udiff(csv_bytes: bytes, target: pd.Timestamp) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(csv_bytes), low_memory=False)
    c = {
        "date": first_col(df, ["TradDt", "TRADE_DATE"]),
        "symbol": first_col(df, ["TckrSymb", "SYMBOL"]),
        "series": first_col(df, ["SctySrs", "SERIES"]),
        "isin": first_col(df, ["ISIN"], required=False),
        "open": first_col(df, ["OpnPric", "OPEN_PRICE"]),
        "high": first_col(df, ["HghPric", "HIGH_PRICE"]),
        "low": first_col(df, ["LwPric", "LOW_PRICE"]),
        "close": first_col(df, ["ClsPric", "CLOSE_PRICE"]),
        "volume": first_col(df, ["TtlTradgVol", "TOTTRDQTY"]),
        "turnover": first_col(df, ["TtlTrfVal", "TOTTRDVAL"]),
    }
    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df[c["date"]], errors="coerce").dt.normalize()
    out["symbol"] = df[c["symbol"]].astype(str).str.strip().str.upper()
    out["series"] = df[c["series"]].astype(str).str.strip().str.upper()
    out["isin"] = df[c["isin"]].astype(str).str.strip().str.upper() if c["isin"] else pd.NA
    for x in ["open", "high", "low", "close", "volume", "turnover"]:
        out[x] = pd.to_numeric(df[c[x]], errors="coerce")
    out = out[out["series"].isin(SERIES)].copy()
    out = out[out["date"].eq(target.normalize())].copy()
    out = out.dropna(subset=["symbol", "close", "volume", "turnover"])
    out = out.drop_duplicates(["date", "symbol", "series"], keep="last")
    return out.reset_index(drop=True)


def normalize_full(csv_bytes: bytes, target: pd.Timestamp) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(csv_bytes), skipinitialspace=True, low_memory=False)
    df.columns = [str(x).strip() for x in df.columns]
    c = {
        "symbol": first_col(df, ["SYMBOL"]),
        "series": first_col(df, ["SERIES"]),
        "close": first_col(df, ["CLOSE_PRICE", "LAST_PRICE"]),
        "volume": first_col(df, ["TTL_TRD_QNTY", "TOTTRDQTY"], required=False),
        "turnover": first_col(df, ["TURNOVER_LACS", "TOTTRDVAL"], required=False),
    }
    out = pd.DataFrame({
        "date": target.normalize(),
        "symbol": df[c["symbol"]].astype(str).str.strip().str.upper(),
        "series": df[c["series"]].astype(str).str.strip().str.upper(),
        "close_check": pd.to_numeric(df[c["close"]], errors="coerce"),
    })
    if c["volume"]:
        out["volume_check"] = pd.to_numeric(df[c["volume"]], errors="coerce")
    if c["turnover"]:
        out["turnover_check"] = pd.to_numeric(df[c["turnover"]], errors="coerce")
    out = out[out["series"].isin(SERIES)].dropna(subset=["symbol", "close_check"])
    return out.drop_duplicates(["symbol", "series"], keep="last").reset_index(drop=True)


def validate(primary: pd.DataFrame, secondary: pd.DataFrame) -> dict:
    if len(primary) < 500:
        raise RuntimeError(f"UDiFF equity rows too low: {len(primary)}")
    m = primary.merge(secondary, on=["symbol", "series"], how="inner")
    if len(m) < 500:
        raise RuntimeError(f"official-source overlap too low: {len(m)}")
    overlap = len(m) / max(1, len(primary))
    diff = (m["close"] - m["close_check"]).abs()
    p99 = float(diff.quantile(0.99))
    med = float(diff.median())
    if overlap < 0.90:
        raise RuntimeError(f"official-source overlap ratio {overlap:.3f} < 0.90")
    if med > 0.01 or p99 > 0.10:
        raise RuntimeError(f"official close validation failed median={med:.4f}, p99={p99:.4f}")
    return {
        "primary_rows": int(len(primary)),
        "secondary_rows": int(len(secondary)),
        "overlap_rows": int(len(m)),
        "overlap_ratio": float(overlap),
        "median_abs_close_diff": med,
        "p99_abs_close_diff": p99,
        "validation_pass": True,
    }


def merge_with_hf_year(primary: pd.DataFrame, target: pd.Timestamp, output: Path) -> dict:
    year = int(target.year)
    src = hf_hub_download(HF_REPO, f"nse/year={year}/nse_{year}.parquet", repo_type="dataset")
    base = pd.read_parquet(src)
    required = ["date", "symbol", "series", "isin", "close", "volume", "turnover"]
    for col in required:
        if col not in base:
            raise RuntimeError(f"HF baseline missing {col}")
    base = base[required].copy()
    base["date"] = pd.to_datetime(base["date"]).dt.normalize()
    source_max = base["date"].max()
    official = primary[required].copy()
    # Replace any existing target-date rows with official exchange data.
    base = base[~base["date"].eq(target.normalize())]
    combined = pd.concat([base, official], ignore_index=True)
    combined = combined.sort_values(["symbol", "date", "series"]).drop_duplicates(
        ["date", "symbol", "series"], keep="last"
    ).reset_index(drop=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output, index=False)
    return {
        "hf_baseline_max_date": str(pd.Timestamp(source_max).date()),
        "combined_max_date": str(pd.Timestamp(combined["date"].max()).date()),
        "combined_rows": int(len(combined)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--output-dir", default="nse_direct")
    ap.add_argument("--poll-attempts", type=int, default=1)
    ap.add_argument("--poll-seconds", type=int, default=1200)
    args = ap.parse_args()

    target = pd.Timestamp(args.date).normalize()
    outdir = Path(args.output_dir)
    rawdir = outdir / "raw"
    rawdir.mkdir(parents=True, exist_ok=True)

    ymd = target.strftime("%Y%m%d")
    dmy = target.strftime("%d%m%Y")
    zip_name = f"BhavCopy_NSE_CM_0_0_0_{ymd}_F_0000.csv.zip"
    primary_url = f"{UDIFF_BASE}/{zip_name}"
    secondary_name = f"sec_bhavdata_full_{dmy}.csv"
    secondary_url = f"{FULL_BASE}/{secondary_name}"

    last = None
    for attempt in range(1, args.poll_attempts + 1):
        try:
            zip_bytes = get(primary_url)
            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
                names = [n for n in z.namelist() if n.lower().endswith(".csv")]
                if not names:
                    raise RuntimeError("UDiFF zip contains no CSV")
                csv_bytes = z.read(names[0])
            full_bytes = get(secondary_url)

            primary = normalize_udiff(csv_bytes, target)
            secondary = normalize_full(full_bytes, target)
            v = validate(primary, secondary)

            (rawdir / zip_name).write_bytes(zip_bytes)
            (rawdir / Path(names[0]).name).write_bytes(csv_bytes)
            (rawdir / secondary_name).write_bytes(full_bytes)

            merged_path = outdir / f"nse_{target.year}_official_overlay.parquet"
            m = merge_with_hf_year(primary, target, merged_path)

            summary = {
                "status": "official_nse_validated",
                "target_date": str(target.date()),
                "primary_url": primary_url,
                "secondary_url": secondary_url,
                "udiff_zip_sha256": sha256_bytes(zip_bytes),
                "udiff_csv_sha256": sha256_bytes(csv_bytes),
                "secondary_sha256": sha256_bytes(full_bytes),
                "primary_csv_name": Path(names[0]).name,
                "overlay_parquet": str(merged_path),
                **v,
                **m,
            }
            (outdir / "market_ingest_summary.json").write_text(json.dumps(summary, indent=2))
            print(json.dumps(summary, indent=2))
            return
        except Exception as exc:
            last = repr(exc)
            print(f"attempt {attempt}/{args.poll_attempts} failed: {last}", flush=True)
            if attempt < args.poll_attempts:
                time.sleep(args.poll_seconds)

    raise RuntimeError(f"Official NSE market data unavailable/invalid for {target.date()}: {last}")


if __name__ == "__main__":
    main()
