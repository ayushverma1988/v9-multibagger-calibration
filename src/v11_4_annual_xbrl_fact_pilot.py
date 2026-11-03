"""PIT annual XBRL numeric fact extraction pilot from NSE's original archived filings.

Uses v11_4_longterm_fundamentals.extract_annual from frozen v10-fresh-current.
Does NOT convert missing fields to zero or use filings published after historical folds.
"""
from __future__ import annotations
import argparse
import json
from datetime import date
from pathlib import Path
import pandas as pd
from v11_4_longterm_fundamentals import Client, extract_annual

NUMERIC=("revenue","pat","pbt","finance_cost","depreciation","equity_total",
         "borrowings_total","cfo","ppe","cash")
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--catalog",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--max-companies",type=int,default=8)
    p.add_argument("--max-filings-per-company",type=int,default=4)
    a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    frame=pd.read_csv(a.catalog,dtype=str).fillna("")
    if not {"symbol","fy_end","available_at_utc","xbrl_url","consolidated"}.issubset(frame.columns):
        raise SystemExit("Annual filing source catalog missing core PIT columns")
    frame["_fy"]=pd.to_datetime(frame["fy_end"],errors="coerce")
    frame["_pub"]=pd.to_datetime(frame["available_at_utc"],utc=True,errors="coerce")
    frame=frame[(frame["_fy"].notna())&(frame["_pub"].notna())&
      (frame["_pub"]<=pd.Timestamp("2021-12-31T10:00:00Z"))&
      (frame["_pub"]>=frame["_fy"].dt.tz_localize("UTC"))&
      frame["xbrl_url"].str.match(r"^https://nsearchives\.nseindia\.com/.*\.xml$",case=False)
      ].copy()
    if frame.empty:raise SystemExit("No historical annual XBRL with original PIT date")
    frame["_consolidated"]=frame["consolidated"].str.contains("consolidated",case=False)&~frame["consolidated"].str.contains("non",case=False)
    frame=frame.sort_values(["symbol","_fy","_consolidated","_pub"],ascending=[True,False,False,True])
    frame=frame.drop_duplicates(["symbol","_fy"],keep="first")
    companies=sorted(frame["symbol"].unique())[:a.max_companies]
    wanted=pd.concat([frame.loc[frame["symbol"].eq(s)].head(a.max_filings_per_company) for s in companies])
    client=Client()
    rows=[];failures=[]
    for row in wanted.itertuples():
        period=pd.Timestamp(row._fy).date() if hasattr(row,"_fy") else date.fromisoformat(str(row.fy_end))
        facts=extract_annual(client,row.xbrl_url,period)
        entry={"symbol":row.symbol,"fy_end":row.fy_end,
               "available_at_utc":row.available_at_utc,
               "xbrl_url":row.xbrl_url,
               "consolidated":row.consolidated}
        if not facts or facts.get("_error"):
            failures.append({**entry,"error":str((facts or {}).get("_error") or "no facts")[:190]})
        else:
            entry.update({k:facts.get(k) for k in NUMERIC})
            entry["core_fields_available"]=sum(facts.get(k) is not None for k in ("revenue","pat","pbt","equity_total"))
            rows.append(entry)
        print(row.symbol,row.fy_end,"facts",len(rows),"errors",len(failures),flush=True)
    pd.DataFrame(rows).to_csv(out/"annual_XBRL_PIT_fact_candidates.csv",index=False)
    pd.DataFrame(failures).to_csv(out/"annual_XBRL_parse_errors.csv",index=False)
    summary={
        "scope":"SOURCE_NUMERIC_PILOT_ONLY_NOT_PRODUCTION",
        "requested_filings":len(wanted),
        "parsed_filings":len(rows),
        "errors":len(failures),
        "rows_with_revenue_and_PAT":sum(x.get("revenue") is not None and x.get("pat") is not None for x in rows),
        "source":"NSE-hosted original historic annual XBRL documents",
        "PIT":"Each numeric fact must carry its original broadcast time, no invented annual financials",
        "note":"Further unit normalization, period/context, revisions, consolidations and identity verification required before backtest.",
    }
    (out/"annual_XBRL_fact_probe_summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)
    if not rows:raise SystemExit("No historic numeric annual facts parsed")
if __name__=="__main__":main()
