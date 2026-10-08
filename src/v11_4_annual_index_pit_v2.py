"""Source-only PIT annual 5y/7y index audit v2: mode-consistent runs.

Unlike the exploratory v1, v2 evaluates consolidated and standalone separately.
It never mixes reporting modes and never uses after-close filings. Index
availability is NOT proof of correctly extracted numeric values.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd

def fold_close(d):
    return (pd.Timestamp(d).normalize().tz_localize("Asia/Kolkata")
            +pd.Timedelta(hours=15,minutes=30)).tz_convert("UTC")

def canonical_mode(value):
    label=str(value or "").strip().lower()
    if "non-consolidated" in label or "standalone" in label:return "standalone"
    if "consolidated" in label:return "consolidated"
    return "unknown"

def find_run(fiscal_ends,cutoff,years):
    fiscal_ends=sorted(set(fiscal_ends),reverse=True)
    for i in range(max(len(fiscal_ends)-years,0)):
        selected=fiscal_ends[i:i+years+1]
        if cutoff-selected[0]>pd.Timedelta(days=570):continue
        if all(335 <= (a-b).days <= 395 for a,b in zip(selected,selected[1:])):
            return selected
    return []

def best_mode_run(records,cutoff,years):
    choices=[]
    for mode,dates in records.items():
        if mode not in ("consolidated","standalone"):continue
        seq=find_run(dates,cutoff,years)
        if seq:choices.append((len(seq), seq[0], mode=="consolidated",mode,seq))
    if not choices:return "none",[]
    best=max(choices)
    return best[3],best[4]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--annual-index",required=True)
    ap.add_argument("--snapshot",required=True)
    ap.add_argument("--folds",required=True)
    ap.add_argument("--output",required=True)
    args=ap.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    idx=pd.read_csv(args.annual_index,dtype=str).fillna("")
    required={"symbol","fy_end","available_at_utc","consolidated","xbrl_url"}
    if not required.issubset(idx):
        raise SystemExit(f"Missing source fields: {required-set(idx)}")
    idx["symbol"]=idx["symbol"].str.upper().str.strip()
    idx["end"]=pd.to_datetime(idx["fy_end"],errors="coerce",utc=True,format="mixed")
    idx["available"]=pd.to_datetime(idx["available_at_utc"],errors="coerce",utc=True,format="mixed")
    idx["mode"]=idx["consolidated"].map(canonical_mode)
    src=idx["xbrl_url"].str.match(r"^https://nsearchives\.nseindia\.com/.*\.xml$",case=False)
    raw_rows=len(idx)
    raw_2025=int((idx["fy_end"]=="2025-03-31").sum())
    after_date_rows=int((idx["end"].notna()&idx["available"].notna()).sum())
    idx=idx[idx["end"].notna()&idx["available"].notna()&
            (idx["available"]>=idx["end"])&src&idx["symbol"].ne("")].copy()
    snap=pd.read_parquet(args.snapshot,columns=["date","symbol"])
    snap["date"]=pd.to_datetime(snap["date"],errors="coerce").dt.normalize()
    snap["symbol"]=snap["symbol"].astype(str).str.upper().str.strip()
    folds=pd.read_csv(args.folds)
    fdates=pd.to_datetime(folds["date"],errors="coerce").dropna().sort_values().unique()
    result=[];detail=[]
    for rawdate in fdates:
        day=pd.Timestamp(rawdate).normalize()
        cutoff=fold_close(day)
        syms=set(snap.loc[snap["date"].eq(day),"symbol"])
        data=idx[idx["symbol"].isin(syms)&(idx["available"]<=cutoff)]
        for symbol,part in data.groupby("symbol"):
            # Each fiscal period is valid only after the original public filing date.
            modes={k:list(v["end"]) for k,v in part.groupby("mode")}
            mode5,run5=best_mode_run(modes,cutoff,5)
            mode7,run7=best_mode_run(modes,cutoff,7)
            year2025={
                k:pd.Timestamp("2025-03-31T00:00:00Z") in set(dates)
                for k,dates in modes.items()
            }
            detail.append({
                "date":str(day.date()),"symbol":symbol,
                "index_5y_candidate":bool(run5),"index_7y_candidate":bool(run7),
                "5y_reporting_mode":mode5,"7y_reporting_mode":mode7,
                "5y_latest_fy":str(run5[0].date()) if run5 else "",
                "7y_latest_fy":str(run7[0].date()) if run7 else "",
                "FY2025_consolidated_in_index":bool(year2025.get("consolidated",False)),
                "FY2025_standalone_in_index":bool(year2025.get("standalone",False)),
                "latest_consolidated_fy":str(max(modes["consolidated"]).date()) if modes.get("consolidated") else "",
                "latest_standalone_fy":str(max(modes["standalone"]).date()) if modes.get("standalone") else "",
            })
        now=[r for r in detail if r["date"]==str(day.date())]
        row={
            "date":str(day.date()),"universe_n":len(syms),
            "annual_index_symbols_asof":len(now),
            "index_5y_candidate_symbols":sum(x["index_5y_candidate"] for x in now),
            "index_7y_candidate_symbols":sum(x["index_7y_candidate"] for x in now),
            "FY2025_any_index_symbols":sum(x["FY2025_consolidated_in_index"] or x["FY2025_standalone_in_index"] for x in now),
            "FY2025_consolidated_index_symbols":sum(x["FY2025_consolidated_in_index"] for x in now),
            "FY2025_standalone_index_symbols":sum(x["FY2025_standalone_in_index"] for x in now),
        }
        row["index_5y_coverage"]=row["index_5y_candidate_symbols"]/max(len(syms),1)
        row["index_7y_coverage"]=row["index_7y_candidate_symbols"]/max(len(syms),1)
        result.append(row)
    if len(result)!=18:raise SystemExit(f"Expected 18 frozen folds, saw {len(result)}")
    if raw_2025>=1000 and int((idx["fy_end"]=="2025-03-31").sum())<800:
        raise SystemExit("FY2025 source rows were silently lost after parsing; abort as-of audit")
    results=pd.DataFrame(result)
    results.to_csv(out/"annual_index_v2_by_fold.csv",index=False)
    pd.DataFrame(detail).to_csv(out/"annual_index_v2_by_symbol.csv",index=False)
    summary={
        "scope":"EXPERIMENTAL_PIT_INDEX_V2_NOT_NUMERIC_FEATURES",
        "folds":len(result),"valid_index_rows":len(idx),
        "unfiltered_source_rows":raw_rows,
        "raw_FY2025_rows":raw_2025,
        "FY2025_rows_after_valid_timestamp_and_url_check":int((idx["fy_end"]=="2025-03-31").sum()),
        "rows_with_parseable_publication_and_fy":after_date_rows,
        "covered_symbols":idx["symbol"].nunique(),
        "max_5y_candidates":int(results["index_5y_candidate_symbols"].max()),
        "max_7y_candidates":int(results["index_7y_candidate_symbols"].max()),
        "FY2025_dec_5y_candidates":int(results.set_index("date").loc["2025-12-31","index_5y_candidate_symbols"]),
        "FY2025_dec_7y_candidates":int(results.set_index("date").loc["2025-12-31","index_7y_candidate_symbols"]),
        "mode_mix_allowed":False,"market_close":"15:30 Asia/Kolkata",
        "reconstruction_validated":False,"prediction_accuracy_evaluated":False,
        "source_numeric_values_validated":False,
        "v10_production_changed":False,
    }
    (out/"annual_index_v2_summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)
    print(results.tail(5).to_string(index=False),flush=True)
    if raw_2025>=1000 and summary["FY2025_dec_5y_candidates"]<100:
        raise SystemExit("FY2025 annual source merge still not reflected in historical continuity; flag for investigation")

if __name__=="__main__":main()
