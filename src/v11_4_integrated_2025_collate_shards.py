"""Collate disjoint 2025 NSE integrated financial filing source shards.

Files remain unpromoted source evidence. No XBRL values are inferred here.
"""
import argparse,json
from pathlib import Path
import pandas as pd

CUTOFF=pd.Timestamp("2025-12-31T10:00:00Z")
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--artifacts",required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--expected-shards",type=int,default=4)
    a=p.parse_args();root=Path(a.artifacts);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    manifests=list(root.rglob("integrated_filing_pilot_summary.json"))
    if len(manifests)!=a.expected_shards:
        raise SystemExit(f"Expected {a.expected_shards} complete shard summaries; saw {len(manifests)}")
    shards={};frames=[];expected=set()
    for file in manifests:
        s=json.loads(file.read_text())
        key=s.get("shard_index")
        if key in shards:raise SystemExit("Duplicate shard "+str(key))
        if s.get("shard_count")!=a.expected_shards:raise SystemExit("Shard count inconsistent")
        if s.get("sample_companies")!=s.get("attempted_companies"):raise SystemExit(f"Shard {key} did not attempt all requested companies")
        requested=set(s.get("requested_symbols",[]))
        if expected&requested:raise SystemExit("Cross-shard symbol overlap")
        expected.update(requested)
        shards[key]=s
        csvpath=file.parent/"integrated_filing_index_PIT_CANDIDATES.csv"
        if not csvpath.exists():raise SystemExit("Missing index "+str(csvpath))
        frames.append(pd.read_csv(csvpath,dtype=str).fillna(""))
    if set(shards)!=set(range(a.expected_shards)):raise SystemExit("Some shard indexes missing")
    full_universe={s["full_universe_size"] for s in shards.values()}
    if len(full_universe)!=1:raise SystemExit("Inconsistent historical stock-universe size across shards")
    full_size=next(iter(full_universe))
    if len(expected)!=full_size:raise SystemExit(f"Shards cover {len(expected)} symbols, but historical universe contains {full_size}")
    all_rows=pd.concat(frames,ignore_index=True)
    needed=["symbol","period_end","available_at_utc","seq_id","xbrl_url"]
    if not set(needed).issubset(all_rows.columns):raise SystemExit("Missing source fields")
    all_rows["symbol"]=all_rows["symbol"].str.upper().str.strip()
    pd_end=pd.to_datetime(all_rows["period_end"],errors="coerce",utc=True)
    ts=pd.to_datetime(all_rows["available_at_utc"],errors="coerce",utc=True)
    time_ok=(ts.notna())&(pd_end.notna())&(ts>=pd_end)
    if not time_ok.all():
        raise SystemExit(f"{int((~time_ok).sum())} integrated filing rows have invalid PIT timestamps")
    if (~all_rows["symbol"].isin(expected)).any():
        raise SystemExit("Unexpected symbol outside the historical 2025 universe")
    all_rows=all_rows.drop_duplicates(["symbol","period_end","available_at_utc","seq_id"]).copy()
    all_rows["period_end"]=pd.to_datetime(all_rows["period_end"],utc=True)
    all_rows["available_at_utc"]=pd.to_datetime(all_rows["available_at_utc"],utc=True)
    all_rows.to_parquet(out/"integrated_financial_index_2025_REHEARSAL_ONLY.parquet",index=False,compression="zstd")
    all_rows.to_csv(out/"integrated_financial_index_2025_REHEARSAL_ONLY.csv",index=False)
    preclose=all_rows[(all_rows["period_end"]>=pd.Timestamp("2025-03-01",tz="UTC"))&
                      (all_rows["available_at_utc"]<=CUTOFF)]
    covered=set(preclose["symbol"])
    missing=sorted(expected-covered)
    (out/"symbols_without_preclose_2025_filings.json").write_text(json.dumps(missing,indent=2))
    totalerrs=sum(len(s.get("errors",[])) for s in shards.values())
    summary={
        "status":"FULL_HISTORICAL_UNIVERSE_INDEX_RECOVERY_REHEARSAL",
        "full_2025_dec_historical_universe":full_size,
        "requested_companies":len(expected),
        "completed_shards":len(shards),
        "index_rows":len(all_rows),
        "preclose_2025_filings":len(preclose),
        "companies_with_preclose_2025_filings":len(covered),
        "coverage":len(covered)/max(full_size,1),
        "missing_companies":len(missing),
        "source_errors":totalerrs,
        "filing_metadata_timestamp_violations":int((~time_ok).sum()),
        "ready_for_marketwide_numeric_xbrl_parse":len(covered)>=0.80*full_size and totalerrs==0,
        "production_eligible":False,
        "numerical_fundamentals_ready":False,
        "original_cutoff":"2025-12-31 15:30 Asia/Kolkata",
        "source":"Original NSE Integrated Filing - Financials index across archived December-2025 stock universe",
    }
    (out/"full_integrated_source_summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)
    if totalerrs or len(covered)<0.50*full_size:
        raise SystemExit("Insufficient accessible source history for full-universe financial backfill")
if __name__=="__main__":main()
