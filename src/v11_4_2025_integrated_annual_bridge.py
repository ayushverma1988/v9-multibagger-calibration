"""Verify FY2025 annual year-to-date facts inside Q4 NSE Integrated Filings.

NSE moved FY2025 financial data to Integrated Filing endpoint. A March
2025 quarter can have FourD full-year YTD facts. This independent pilot
never labels ordinary OneD (Q4 only) as an annual result.
"""
import argparse,hashlib,json
from pathlib import Path
import pandas as pd
from v11_4_longterm_fundamentals import Client
from v11_4_strict_annual_numeric_features import strict_annual_facts

FY="2025-03-31"
CUTOFF=pd.Timestamp("2025-12-31T10:00:00Z")
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--integrated-index",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--max-companies",type=int,default=24)
    a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    x=pd.read_parquet(a.integrated_index)
    x["end"]=pd.to_datetime(x["period_end"],utc=True,errors="coerce")
    x["pub"]=pd.to_datetime(x["available_at_utc"],utc=True,errors="coerce")
    x["symbol"]=x["symbol"].astype(str).str.upper().str.strip()
    eligible=x[
        (x["end"]==pd.Timestamp(FY,tz="UTC"))&
        x["pub"].notna()&(x["pub"]<=CUTOFF)&(x["pub"]>=x["end"])&
        x["xbrl_url"].astype(str).str.match(r"^https://nsearchives\.nseindia\.com/.*\.xml$",case=False)&
        ~x["xbrl_url"].astype(str).str.contains(r"/INTEGRATED_FILING_BANKING_",case=False)
    ].copy()
    eligible["prefer_consolidated"]=eligible["consolidated"].astype(str).str.lower().eq("consolidated")
    eligible=eligible.sort_values(["symbol","prefer_consolidated","pub"],ascending=[True,False,True])
    eligible=eligible.drop_duplicates(["symbol"],keep="first")
    selected=eligible.sort_values("symbol",key=lambda z:z.map(lambda x:hashlib.sha256(x.encode()).hexdigest())).head(a.max_companies)
    client=Client();rows=[];errors=[]
    for r in selected.itertuples():
        row={"symbol":r.symbol,"fy_end":FY,"available_at_utc":r.available_at_utc,
             "xbrl_url":r.xbrl_url,"original_filing_seq_id":str(r.seq_id),
             "consolidated":r.consolidated}
        try:
            res=client.get(r.xbrl_url)
            row["sha256"]=hashlib.sha256(res.content).hexdigest()
            facts,audit=strict_annual_facts(res.content,FY,source_is_annual=True)
            row.update(facts)
            row["revenue_context"]=audit.get("revenue",{}).get("context")
            row["pat_context"]=audit.get("pat",{}).get("context")
            row["status"]="verified_FourD_YTD" if (
                facts.get("revenue") is not None and facts.get("pat") is not None
                and row["revenue_context"]=="FourD" and row["pat_context"]=="FourD"
            ) else "missing_or_unmapped_FourD"
            rows.append(row)
        except Exception as err:
            errors.append({**row,"error":str(err)[:170]})
        print(r.symbol,row.get("status","download_error"),flush=True)
    pd.DataFrame(rows).to_csv(out/"FY2025_integrated_annual_YTD_research_candidates.csv",index=False)
    pd.DataFrame(errors).to_csv(out/"FY2025_integrated_annual_YTD_errors.csv",index=False)
    verified=[z for z in rows if z.get("status")=="verified_FourD_YTD"]
    summary={
        "scope":"INTEGRATED_FY2025_YTD_ANNUAL_FACT_PILOT_ONLY",
        "FY_end":FY,
        "asof_cutoff_UTC":CUTOFF.isoformat(),
        "Q4_2025_integrated_index_symbols":int(eligible["symbol"].nunique()),
        "sampled":len(selected),"verified_FourD_YTD":len(verified),
        "retrieval_errors":len(errors),"unmapped":len(rows)-len(verified),
        "original_document_sha256_unique":len(set(z["sha256"] for z in rows if z.get("sha256"))),
        "no_OneD_quarter_as_annual":True,
        "full_2025_annual_numeric_rebuild_done":False,
        "production_approved":False,
        "note":"Q4 March 2025 FourD YTD candidate requires independent published financial-statement reconciliation before broad 5y/7y backtest use.",
    }
    (out/"FY2025_integrated_annual_bridge_summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)
    if len(verified)<max(8,len(selected)//2):
        raise SystemExit("Insufficient independently verified FY2025 integrated annual YTD facts")
if __name__=="__main__":main()
