"""Measure point-in-time 5y/7y annual filing availability in the frozen 18 folds.

An INDEX record establishes only that a filing existed; not that comparable
financial values or their units have been verified. No backtest training occurs.
"""
import argparse
import json
from pathlib import Path
import pandas as pd

def fold_close(v):
    return (pd.Timestamp(v).normalize().tz_localize("Asia/Kolkata")
            +pd.Timedelta(hours=15, minutes=30)).tz_convert("UTC")

def normalize_symbols(x):
    return x.astype(str).str.upper().str.strip()

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--annual-index",required=True)
    p.add_argument("--snapshot",required=True)
    p.add_argument("--folds",required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    out=Path(a.output);out.mkdir(exist_ok=True,parents=True)
    idx=pd.read_csv(a.annual_index,dtype=str).fillna("")
    required={"symbol","fy_end","available_at_utc","xbrl_url","consolidated"}
    if not required.issubset(idx.columns):
        raise SystemExit("Annual index missing "+str(required-set(idx.columns)))
    idx["symbol"]=normalize_symbols(idx["symbol"])
    idx["end"]=pd.to_datetime(idx["fy_end"],errors="coerce",utc=True)
    idx["available"]=pd.to_datetime(idx["available_at_utc"],errors="coerce",utc=True)
    idx["mode"]=idx["consolidated"].str.lower().map(
        lambda x:"standalone" if "standalone" in x or "non-consolidated" in x
        else "consolidated" if "consolidated" in x else "unknown")
    # Never substitute quarter-end for the publication timestamp.
    idx=idx[idx["end"].notna()&idx["available"].notna()
            &(idx["available"]>=idx["end"])
            &idx["xbrl_url"].str.match(r"^https://nsearchives\.nseindia\.com/.*\.xml$",case=False)]
    snap=pd.read_parquet(a.snapshot,columns=["date","symbol"])
    snap["date"]=pd.to_datetime(snap["date"],errors="coerce").dt.normalize()
    snap["symbol"]=normalize_symbols(snap["symbol"])
    folds=pd.read_csv(a.folds)
    results=[];records=[]
    for date in pd.to_datetime(folds["date"],errors="coerce").dropna().sort_values().unique():
        day=pd.Timestamp(date).normalize()
        cutoff=fold_close(day)
        universe=set(snap.loc[snap["date"].eq(day),"symbol"])
        rows=idx[idx["symbol"].isin(universe)&(idx["available"]<=cutoff)]
        for sym,sub in rows.groupby("symbol",sort=True):
            # Pick a consistent financial reporting mode as-of the historical fold.
            bymode={mode:set(part["end"].dt.normalize()) for mode,part in sub.groupby("mode")}
            preferred="consolidated" if len(bymode.get("consolidated",set()))>=len(bymode.get("standalone",set())) else "standalone"
            years=bymode.get(preferred,set())
            # This is not a valid history if observations are scattered by years.
            def fiscal_run(n):
                if len(years)<n+1:return False
                yrs=sorted(years,reverse=True)
                for i in range(len(yrs)-n):
                    run=yrs[i:i+n+1]
                    gaps=[(a-b).days for a,b in zip(run,run[1:])]
                    if all(335<=g<=395 for g in gaps) and (cutoff-yrs[i])<=pd.Timedelta(days=570):
                        return True
                return False
            has5=fiscal_run(5)
            has7=fiscal_run(7)
            records.append({"date":str(day.date()),"symbol":sym,
                            "mode":preferred,"financial_years_asof":len(years),
                            "index_5y_candidate":bool(has5),"index_7y_candidate":bool(has7)})
        subset=[r for r in records if r["date"]==str(day.date())]
        by5=sum(z["index_5y_candidate"] for z in subset)
        by7=sum(z["index_7y_candidate"] for z in subset)
        results.append({"date":str(day.date()),"universe_n":len(universe),
                        "annual_index_symbols_asof":len(subset),
                        "index_5y_candidate_symbols":by5,"index_7y_candidate_symbols":by7,
                        "index_5y_coverage":by5/max(len(universe),1),
                        "index_7y_coverage":by7/max(len(universe),1)})
    summary={
        "scope":"PIT_INDEX_FEASIBILITY_ONLY",
        "original_folds":len(results),
        "annual_source_symbols":int(idx["symbol"].nunique()),
        "index_filings_with_valid_source_timestamp":int(len(idx)),
        "max_5y_candidates_in_fold":int(max((r["index_5y_candidate_symbols"] for r in results),default=0)),
        "max_7y_candidates_in_fold":int(max((r["index_7y_candidate_symbols"] for r in results),default=0)),
        "numeric_revenue_PAT_validated":False,
        "historical_fold_acceptance_threshold_changed":False,
        "future_lookahead_allowed":False,
        "note":"A missing-index or incomplete 5y/7y run is missing, never filled with current fundamental values.",
    }
    pd.DataFrame(results).to_csv(out/"annual_filing_index_5y7y_by_fold.csv",index=False)
    pd.DataFrame(records).to_csv(out/"annual_index_symbols_PIT_5y7y.csv",index=False)
    (out/"annual_index_5y7y_summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)
    print(pd.DataFrame(results).to_string(index=False),flush=True)
    if len(results)!=18:raise SystemExit("Expected exactly 18 frozen validation folds")
    if summary["index_filings_with_valid_source_timestamp"]==0:
        raise SystemExit("No PIT annual index records")
if __name__=="__main__":main()
