"""Original NSE Integrated Financial XBRL numeric sample, source-locked and PIT checked.

A downloaded record is not accepted as a financial fact until its own XBRL
period end matches the index. Values remain in original document units.
"""
import argparse,json,time,hashlib
from pathlib import Path
import pandas as pd,requests
from nse_xbrl import FilingResult

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--catalog",required=True);p.add_argument("--output",required=True)
    p.add_argument("--max-companies",type=int,default=8)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    src=pd.read_csv(a.catalog,dtype=str).fillna("")
    cutoff=pd.Timestamp("2025-12-31T10:00:00Z")
    src["_end"]=pd.to_datetime(src["period_end"],errors="coerce")
    src["_available"]=pd.to_datetime(src["available_at_utc"],utc=True,errors="coerce")
    eligible=src[(src["_end"]>=pd.Timestamp("2025-03-31"))&(src["_end"]<=pd.Timestamp("2025-12-31"))&
                 (src["_available"]<=cutoff)&src["xbrl_url"].str.startswith("https://nsearchives.nseindia.com/")&
                 src["xbrl_url"].str.lower().str.endswith(".xml")].copy()
    # Stable one filing per symbol and quarter, prioritize original and consolidated.
    eligible["_priority"]=eligible["consolidated"].str.lower().eq("consolidated").astype(int)
    eligible=eligible.sort_values(["symbol","_end","_priority","_available"],ascending=[True,True,False,True])
    eligible=eligible.drop_duplicates(["symbol","_end"])
    syms=sorted(eligible["symbol"].unique(),key=lambda z:hashlib.sha256(z.encode()).hexdigest())[:a.max_companies]
    selected=eligible.loc[eligible["symbol"].isin(syms)].sort_values(["symbol","_end"]).groupby("symbol").tail(2)
    session=requests.Session();session.headers.update({
        "User-Agent":"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124 Safari/537.36",
        "Referer":"https://www.nseindia.com/","Accept":"application/xml,text/xml,*/*",
    })
    rows=[];errors=[]
    for record in selected.itertuples():
        context={"symbol":record.symbol,"quarter_end_index":record.period_end,
                 "available_at_utc":record.available_at_utc,"source_url":record.xbrl_url,
                 "source_sha256":""}
        try:
            res=session.get(record.xbrl_url,timeout=22);res.raise_for_status()
            context["source_sha256"]=hashlib.sha256(res.content).hexdigest()
            parsed=FilingResult.from_xbrl(
                res.content.decode("utf-8",errors="replace"),symbol=record.symbol,
                is_consolidated=(str(record.consolidated).lower()=="consolidated")
            )
            fy=getattr(parsed,"period_end",None)
            context["quarter_end_xbrl"]=str(fy) if fy else ""
            if str(fy)!=str(record.period_end):
                raise ValueError("XBRL quarter end does not match exchange filing index")
            context["q_revenue_raw_units"]=getattr(parsed,"q_revenue",None)
            context["q_pat_raw_units"]=getattr(parsed,"q_pat",None)
            context["q_ebitda_raw_units"]=getattr(parsed,"q_ebitda",None)
            context["revenue_present"]=context["q_revenue_raw_units"] is not None
            context["pat_present"]=context["q_pat_raw_units"] is not None
            context["status"]="parsed"
            rows.append(context)
        except (requests.RequestException,ValueError,Exception) as exc:
            context["status"]="failed"
            context["error"]=f"{type(exc).__name__}: {str(exc)[:160]}"
            errors.append(context)
        print(record.symbol,record.period_end,context["status"],flush=True)
        time.sleep(0.3)
    pd.DataFrame(rows).to_csv(out/"integrated_XBRL_2025_numeric_candidates.csv",index=False)
    pd.DataFrame(errors).to_csv(out/"integrated_XBRL_2025_errors.csv",index=False)
    summary={
        "requested_documents":len(selected),"parsed_documents":len(rows),"errors":len(errors),
        "valid_revenue_and_PAT":sum(r["revenue_present"] and r["pat_present"] for r in rows),
        "original_document_hashes":len(set(r["source_sha256"] for r in rows)),
        "production_eligible":False,
        "requires":"Confirm monetary units and context semantics against official rendered tables before numeric backtesting",
        "point_in_time":"Quarter end and broadcast timestamp both checked; historical as-of cutoff 15:30 IST",
    }
    (out/"integrated_XBRL_2025_numeric_summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)
    if len(rows)<4:raise SystemExit("Integrated numeric XBRL pilot insufficient")
if __name__=="__main__":main()
