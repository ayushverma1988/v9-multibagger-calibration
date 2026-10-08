"""Scale strict 2024-close 1y annual financial PIT feature validation.

Only 2023 & 2024 FY NSE Annual-source documents, published before the original
fold close. No old-format context guesses. The sample is independently hashed,
deterministic and drawn from the frozen historical stock universe.
"""
from __future__ import annotations
import argparse,hashlib,json,time
from pathlib import Path
import pandas as pd
from v11_4_longterm_fundamentals import Client
from v11_4_strict_annual_numeric_features import strict_annual_facts,fold_close

def mode(s):
    t=str(s).lower()
    if "non-consolidated" in t or "standalone" in t:return "standalone"
    if "consolidated" in t:return "consolidated"
    return "unknown"

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--annual-index",required=True)
    p.add_argument("--snapshot",required=True)
    p.add_argument("--fold-date",default="2024-12-31")
    p.add_argument("--max-companies",type=int,default=128)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    cutoff=fold_close(a.fold_date)
    s=pd.read_parquet(a.snapshot,columns=["date","symbol"])
    s["date"]=pd.to_datetime(s["date"],errors="coerce").dt.normalize()
    universe=set(s.loc[s["date"].eq(pd.Timestamp(a.fold_date)),"symbol"].astype(str).str.upper().str.strip())
    if len(universe)<500:raise SystemExit("Expected frozen large-cap/mid-cap 2024 fold original universe")
    z=pd.read_csv(a.annual_index,dtype=str).fillna("")
    z["symbol"]=z["symbol"].str.upper().str.strip()
    z["fy"]=pd.to_datetime(z["fy_end"],utc=True,format="mixed",errors="coerce")
    z["pub"]=pd.to_datetime(z["available_at_utc"],utc=True,format="mixed",errors="coerce")
    z["mode"]=z["consolidated"].map(mode)
    z=z[z["symbol"].isin(universe)&z["fy"].isin([
        pd.Timestamp("2024-03-31T00:00:00Z"),pd.Timestamp("2023-03-31T00:00:00Z")
    ])&z["pub"].notna()&(z["pub"]<=cutoff)&(z["pub"]>=z["fy"])&
        z["mode"].isin(["consolidated","standalone"])&
        z["xbrl_url"].str.match(r"^https://nsearchives\.nseindia\.com/.*\.xml$",case=False)&
        ~z["xbrl_url"].str.contains(r"/(?:INTEGRATED_FILING_)?BANKING_",case=False,regex=True)].copy()
    # Preserve only the earliest as-of publication for each symbol/mode/year;
    # never replace it with any subsequently filed correction.
    z=z.sort_values("pub").drop_duplicates(["symbol","mode","fy"],keep="first")
    candidates=[]
    for sym,g in z.groupby("symbol",sort=False):
        sizes=g.groupby("mode")["fy"].nunique()
        viable=[m for m in ("consolidated","standalone") if sizes.get(m,0)==2]
        if viable:
            selected=viable[0]
            candidates.append((sym,selected,g[g["mode"].eq(selected)].sort_values("fy",ascending=False)))
    candidates.sort(key=lambda x:hashlib.sha256(x[0].encode()).hexdigest())
    selected=candidates[:a.max_companies]
    client=Client()
    records=[];errors=[]
    for i,(sym,report_mode,rows) in enumerate(selected,1):
        for r in rows.itertuples(index=False):
            entry={
                "symbol":sym,"mode":report_mode,"fy_end":str(r.fy.date()),
                "available_at_utc":r.available_at_utc,
                "xbrl_url":r.xbrl_url,
                "source":"official_NSE_annual_filings_frozen_fold",
                "status":"pending",
            }
            try:
                raw=client.get(r.xbrl_url).content
                entry["original_xml_sha256"]=hashlib.sha256(raw).hexdigest()
                nums,audit=strict_annual_facts(raw,r.fy.date(),source_is_annual=True)
                entry.update(nums)
                entry["fact_audit"]=json.dumps(audit,sort_keys=True)
                entry["status"]="complete" if nums.get("revenue") is not None and nums.get("pat") is not None else "missing_core"
                records.append(entry)
            except Exception as e:
                errors.append({**entry,"error":repr(e)[:200]})
        if i%16==0:
            print("PROGRESS",i,"/",len(selected),"parsed_rows",len(records),"complete",sum(x["status"]=="complete" for x in records),
                  "errors",len(errors),flush=True)
    allfacts=pd.DataFrame(records)
    allfacts.to_csv(out/"strict_annual_source_numeric_2023_2024.csv",index=False)
    pd.DataFrame(errors).to_csv(out/"strict_annual_source_parse_errors.csv",index=False)
    complete=allfacts[allfacts["status"].eq("complete")] if len(allfacts) else pd.DataFrame()
    two=complete.groupby(["symbol","mode"])["fy_end"].nunique() if len(complete) else pd.Series(dtype=int)
    metrics={
        "scope":"HISTORICAL_FOLD_RESEARCH_DATA_ONLY",
        "fold":a.fold_date,"original_universe_symbols":len(universe),
        "eligible_companies_with_both_fiscal_index_records":len(candidates),
        "sample_requested":len(selected),"source_documents_expected":2*len(selected),
        "source_documents_downloaded":len(records),
        "strict_annual_revenue_and_PAT_docs":len(complete),
        "source_download_errors":len(errors),
        "companies_with_both_2023_2024_verified_numeric_documents":int((two>=2).sum()),
        "asof_utc":cutoff.isoformat(),
        "reporting_modes_mixed":False,
        "older_XBRL_unbound_contexts_imputed":False,
        "training_or_prediction_executed":False,
        "production_approved":False,
    }
    (out/"strict_2024_fold_annual_numeric_coverage.json").write_text(json.dumps(metrics,indent=2))
    print(json.dumps(metrics,indent=2),flush=True)
    if int((two>=2).sum())<max(12,a.max_companies//3):
        raise SystemExit("Strict annual 2023/2024 paired numeric extraction below minimum research coverage")
if __name__=="__main__":main()
