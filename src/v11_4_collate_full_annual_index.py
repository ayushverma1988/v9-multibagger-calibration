"""Merge NSE annual filing index shards with disjoint-symbol and PIT integrity QA."""
import argparse,json
from pathlib import Path
import pandas as pd

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--artifacts",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--expected-shards",type=int,default=8)
    a=p.parse_args();root=Path(a.artifacts);out=Path(a.output);out.mkdir(exist_ok=True,parents=True)
    manifests=list(root.rglob("annual_shard_summary.json"))
    if len(manifests)!=a.expected_shards:raise SystemExit(f"Expected {a.expected_shards} shard summaries; got {len(manifests)}")
    allframes=[];seen=set();allerrors=[];counts=[];ids=set()
    for f in manifests:
        summary=json.loads(f.read_text())
        ix=int(summary["shard_index"])
        if ix in ids:raise SystemExit("Duplicate shard "+str(ix))
        ids.add(ix)
        if summary["shard_count"]!=a.expected_shards:raise SystemExit("Shard config mismatch")
        statuses=json.loads((f.parent/"annual_shard_status.json").read_text())
        for st in statuses:
            sym=st["symbol"]
            if sym in seen:raise SystemExit("Duplicate ticker across shards: "+sym)
            seen.add(sym)
        if summary["attempted"]!=len(statuses):raise SystemExit("Attempted mismatch")
        if summary["requested"]!=summary["attempted"]:
            allerrors.append(f"Shard {ix}: unattempted={summary['requested']-summary['attempted']}")
        allerrors.extend(summary.get("errors",[]))
        data=f.parent/"historical_annual_filing_index_shard.parquet"
        if not data.exists():raise SystemExit("Missing source parquet "+str(data))
        allframes.append(pd.read_parquet(data))
        counts.append(summary["historical_fold_union_n"])
    if len(ids)!=a.expected_shards or len(set(counts))!=1:raise SystemExit("Incomplete or inconsistent shard set")
    total=counts[0]
    if len(seen)!=total:allerrors.append(f"Attempted companies {len(seen)} of {total}")
    x=pd.concat(allframes,ignore_index=True)
    x["end"]=pd.to_datetime(x["fy_end"],utc=True,errors="coerce")
    x["available"]=pd.to_datetime(x["available_at_utc"],utc=True,errors="coerce")
    invalid=x["end"].isna()|x["available"].isna()|(x["available"]<x["end"])
    if invalid.any():raise SystemExit(f"Invalid PIT timestamps in {int(invalid.sum())} annual filing rows")
    if (~x["symbol"].isin(seen)).any():raise SystemExit("Unknown symbols in source files")
    x=x.drop(columns=["end","available"]).drop_duplicates(["symbol","fy_end","available_at_utc","xbrl_url"])
    x.to_parquet(out/"full_historical_annual_index_PIT_CANDIDATES.parquet",index=False,compression="zstd")
    x.to_csv(out/"full_historical_annual_index_PIT_CANDIDATES.csv",index=False)
    summary={
        "scope":"FULL_MARKET_ANNUAL_FILING_INDEX_REHEARSAL",
        "original_fold_union_companies":total,
        "original_folds_preserved":18,
        "shards_expected":a.expected_shards,"shards_validated":len(ids),
        "companies_queried":len(seen),
        "companies_with_annual_index":x["symbol"].nunique(),
        "total_annual_index_rows":len(x),
        "earliest_fiscal_period":str(x["fy_end"].min()) if len(x) else None,
        "earliest_publication":str(x["available_at_utc"].min()) if len(x) else None,
        "point_in_time_violations":int(invalid.sum()),
        "source_errors":allerrors,
        "production_ready":False,
        "numerical_facts_ready":False,
        "frozen_fold_gate_changed":False,
    }
    (out/"full_historical_annual_index_summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)
    if allerrors:raise SystemExit(f"Annual source catalog has {len(allerrors)} unresolved retrieval errors")
if __name__=="__main__":main()
