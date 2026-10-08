"""Isolated 2025 NSE integrated-financial metadata merge and point-in-time freshness QC.

No production overwrite and no fabricated financial amounts.
"""
import argparse,json
from pathlib import Path
import pandas as pd

IST="Asia/Kolkata"
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--official",required=True)
    p.add_argument("--integrated",required=True)
    p.add_argument("--snapshot",required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    old=pd.read_parquet(a.official)
    x=pd.read_csv(a.integrated,dtype=str).fillna("")
    needed=["symbol","period_end","available_at_utc","seq_id"]
    if not all(k in x for k in needed):raise SystemExit("Integrated filing index lacks required fields")
    end=pd.to_datetime(x["period_end"],utc=True,errors="coerce")
    available=pd.to_datetime(x["available_at_utc"],utc=True,errors="coerce")
    okay=end.notna()&available.notna()&(available>=end)&(end>=pd.Timestamp("2025-03-01",tz="UTC"))&(x["symbol"].str.len()>0)
    x=x.loc[okay].copy()
    integrated=pd.DataFrame({
        "symbol":x["symbol"].str.upper().str.strip(),
        "symbol_norm":x["symbol"].str.upper().str.strip(),
        "toDate":x["period_end"],
        "fromDate":pd.NA,
        "broadCastDate":x["available_at_utc"],
        "filingDate":x["available_at_utc"],
        "period":"Quarterly",
        "consolidated":x["consolidated"],
        "xbrl":x["xbrl_url"],
        "integrated_seq_id":x["seq_id"],
        "origin":"nse_integrated_filing_financial",
    })
    old=old.copy();old["origin"]="official_quarterly_archive"
    # Normalize both filing generations to one minimal, typed PIT schema.
    # Do not try to serialize every legacy object-valued API field.
    columns=["symbol","symbol_norm","toDate","fromDate","broadCastDate",
             "filingDate","period","consolidated","xbrl","origin","integrated_seq_id"]
    old=old.reindex(columns=columns)
    integrated=integrated.reindex(columns=columns)
    for col in ("toDate","fromDate","broadCastDate","filingDate"):
        old[col]=pd.to_datetime(old[col],errors="coerce",utc=True)
        integrated[col]=pd.to_datetime(integrated[col],errors="coerce",utc=True)
    for col in ("symbol","symbol_norm","period","consolidated","xbrl","origin","integrated_seq_id"):
        old[col]=old[col].astype("string")
        integrated[col]=integrated[col].astype("string")
    merged=pd.concat([old,integrated],ignore_index=True,sort=False)
    merged.to_parquet(out/"combined_financial_index_REHEARSAL_ONLY.parquet",index=False)
    sn=pd.read_parquet(a.snapshot,columns=["date","symbol"])
    sn["date"]=pd.to_datetime(sn["date"],errors="coerce").dt.normalize()
    sn["symbol"]=sn["symbol"].astype(str).str.upper().str.strip()
    diagnostics=[]
    for fold in ("2024-12-31","2025-06-30","2025-12-31"):
        day=pd.Timestamp(fold)
        cutoff=(day.tz_localize(IST)+pd.Timedelta(hours=15,minutes=30)).tz_convert("UTC")
        universe=set(sn.loc[sn["date"].eq(day),"symbol"])
        row={"date":fold,"universe_size":len(universe)}
        for name,frame in (("archive",old),("integrated",integrated),("combined",merged)):
            symbol="symbol_norm" if "symbol_norm" in frame else "symbol"
            ss=frame[symbol].astype(str).str.upper().str.strip()
            ts=pd.to_datetime(frame["broadCastDate"],errors="coerce",utc=True)
            periods=pd.to_datetime(frame["toDate"],errors="coerce",utc=True)
            valid=(ss.isin(universe))&(ts<=cutoff)&(ts>=periods)&(periods<=cutoff)&(periods>=cutoff-pd.Timedelta(days=365))
            row[f"{name}_fresh_365d"]=int(ss[valid].nunique())
            row[f"{name}_coverage"]=round(row[f"{name}_fresh_365d"]/max(len(universe),1),4)
        diagnostics.append(row)
    d=pd.DataFrame(diagnostics)
    d.to_csv(out/"integrated_2025_freshness_comparison.csv",index=False)
    summary={
        "status":"PIT_METADATA_MERGE_ONLY",
        "original_archive_rows":len(old),
        "integrated_rows_verified_timestamps":len(integrated),
        "integrated_symbols":int(integrated["symbol"].nunique()),
        "merged_rows":len(merged),
        "production_ready":False,
        "numeric_facts_parsed":False,
        "market_close":"15:30 Asia/Kolkata",
        "note":"No after-cutoff filings allowed. Older original quarterly archive retained. Not a model accuracy test."
    }
    (out/"integrated_merge_summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)
    print(d.to_string(index=False),flush=True)
    if d.loc[d["date"].eq("2025-12-31"),"combined_fresh_365d"].iloc[0] <= d.loc[d["date"].eq("2025-12-31"),"archive_fresh_365d"].iloc[0]:
        raise SystemExit("Integrated filing index failed to improve 2025 PIT financial freshness")
if __name__=="__main__":main()
