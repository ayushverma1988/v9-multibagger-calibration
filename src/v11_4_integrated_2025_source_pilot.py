"""Read-only NSE Integrated Filing - Financials 2025 PIT source pilot.

The NSE moved the post-March-2025 results to /api/integrated-filing-results.
This script is *source discovery*, not approved numerical fundamentals.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
import pandas as pd
import requests

API="https://www.nseindia.com/api/integrated-filing-results"
HOMEPAGE="https://www.nseindia.com/companies-listing/corporate-integrated-filing"
FILING_TYPE="Integrated Filing- Financials"

def ts(v):
    if not v:return pd.NaT
    x=pd.to_datetime(v,dayfirst=True,errors="coerce")
    if pd.isna(x):return x
    return x.tz_localize("Asia/Kolkata").tz_convert("UTC") if x.tzinfo is None else x.tz_convert("UTC")

def date(v):
    if not v:return pd.NaT
    return pd.to_datetime(v,dayfirst=True,errors="coerce").normalize()

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--snapshot",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--max-symbols",type=int,default=16)
    a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    snap=pd.read_parquet(a.snapshot,columns=["date","symbol"])
    snap["date"]=pd.to_datetime(snap["date"],errors="coerce").dt.normalize()
    universe=set(snap.loc[snap["date"].eq(pd.Timestamp("2025-12-31")),"symbol"].astype(str).str.upper().str.strip())
    chosen=[z for z in ("RELIANCE","INFY","TCS","SBIN","20MICRONS","HDFCBANK") if z in universe]
    extras=sorted(universe-set(chosen),key=lambda z:hashlib.sha256(z.encode()).hexdigest())
    chosen=(chosen+extras)[:a.max_symbols]
    session=requests.Session()
    session.headers.update({
        "User-Agent":"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Referer":HOMEPAGE,"Accept":"application/json, text/plain, */*",
        "X-Requested-With":"XMLHttpRequest",
    })
    try:session.get(HOMEPAGE,timeout=15)
    except requests.RequestException:pass
    rows=[];errors=[];schemas={}
    for sym in chosen:
        total=0
        try:
            for page in range(1,4):
                r=session.get(API,params={"index":"equities","symbol":sym,"type":FILING_TYPE,"page":page,"size":100},timeout=30)
                if r.status_code==429:raise RuntimeError("NSE returned 429 rate-limit; stop")
                r.raise_for_status()
                data=r.json()
                if not isinstance(data,dict) or not isinstance(data.get("data"),list):
                    raise ValueError(f"Unexpected integrated JSON top type={type(data).__name__} fields={list(data)[:12] if isinstance(data,dict) else None}")
                info=data.get("data",[])
                if info and sym not in schemas:
                    schemas[sym]={"response_keys":list(data),"filing_keys":list(info[0])}
                for z in info:
                    if not isinstance(z,dict):continue
                    filing_sym=str(z.get("symbol") or sym).strip().upper()
                    if filing_sym!=sym:continue
                    period=date(z.get("qe_Date"))
                    broadcast=ts(z.get("broadcast_Date"))
                    if pd.isna(period) or pd.isna(broadcast) or broadcast<period.tz_localize("UTC"):continue
                    rows.append({
                        "symbol":sym,"period_end":period.date().isoformat(),
                        "available_at_utc":broadcast.isoformat(),
                        "seq_id":str(z.get("seq_Id") or ""),
                        "filing_type":str(z.get("type_Sub") or ""),
                        "audit":str(z.get("audited") or ""),
                        "consolidated":str(z.get("consolidated") or ""),
                        "xbrl_url":str(z.get("xbrl") or ""),
                        "ixbrl_url":str(z.get("ixbrl") or ""),
                        "origin":"official_nse_integrated_financial_index",
                    })
                    total+=1
                if len(info)<100:break
                time.sleep(0.2)
            print(f"{sym} integrated_filings={total}",flush=True)
        except (ValueError,requests.RequestException,RuntimeError) as exc:
            errors.append({"symbol":sym,"error":f"{type(exc).__name__}: {str(exc)[:190]}"})
            print(f"{sym} error={errors[-1]['error']}",flush=True)
            if "429" in str(exc):break
        time.sleep(0.4)
    columns=["symbol","period_end","available_at_utc","seq_id","filing_type","audit",
             "consolidated","xbrl_url","ixbrl_url","origin"]
    df=pd.DataFrame(rows,columns=columns).drop_duplicates(["symbol","period_end","available_at_utc","seq_id"])
    df.to_csv(out/"integrated_filing_index_PIT_CANDIDATES.csv",index=False)
    pd.DataFrame(errors,columns=["symbol","error"]).to_csv(out/"integrated_source_errors.csv",index=False)
    cutoff=pd.Timestamp("2025-12-31T10:00:00Z")
    end=pd.to_datetime(df["period_end"],errors="coerce",utc=True)
    filed=pd.to_datetime(df["available_at_utc"],errors="coerce",utc=True)
    eligible=df[(end>=pd.Timestamp("2025-03-01T00:00:00Z"))&(filed<=cutoff)]
    summary={
        "source":API,"source_type":"NSE Integrated Filing - Financials",
        "sample_companies":len(chosen),"successful_companies":len(chosen)-len(errors),
        "rows":len(df),"2025_plus_rows_filed_before_2025_dec_close":len(eligible),
        "covered_companies_by_2025_dec_close":int(eligible["symbol"].nunique()),
        "sample_symbols":chosen,"api_schema_by_sample":schemas,
        "errors":errors,
        "production_eligible":False,
        "message":"Historical 2025 filing availability, not XBRL numeric accuracy or final 18-fold readiness.",
    }
    (out/"integrated_filing_pilot_summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k not in ("api_schema_by_sample","sample_symbols")},indent=2),flush=True)
    if len(eligible)==0:raise SystemExit("No pre-cutoff 2025 integrated filing records; source not validated")
if __name__=="__main__":main()
