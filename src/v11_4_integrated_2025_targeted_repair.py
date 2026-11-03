"""Repair only failed NSE integrated-financial source symbols from archived full-index run.

Do not regenerate 1,314 successful requests; preserve existing original broadcast times.
"""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import pandas as pd,requests

API="https://www.nseindia.com/api/integrated-filing-results"
HOME="https://www.nseindia.com/companies-listing/corporate-integrated-filing"
CUT=pd.Timestamp("2025-12-31T10:00:00Z")

def parse_utc(v):
    if not v:return pd.NaT
    d=pd.to_datetime(v,errors="coerce",dayfirst=True)
    if pd.isna(d):return d
    return d.tz_localize("Asia/Kolkata").tz_convert("UTC") if d.tzinfo is None else d.tz_convert("UTC")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--seed",required=True);p.add_argument("--summary",required=True)
    p.add_argument("--missing",required=True);p.add_argument("--output",required=True)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    prior=json.loads(Path(a.summary).read_text())
    if prior.get("full_2025_dec_historical_universe")!=1314 or prior.get("completed_shards")!=4:
        raise SystemExit("Refuse repair of incomplete or unexpected source universe")
    df=pd.read_parquet(a.seed)
    absent=json.loads(Path(a.missing).read_text())
    if not isinstance(absent,list) or len(absent)>20:raise SystemExit("Unexpected missing-list size")
    sess=requests.Session();sess.headers.update({
        "User-Agent":"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124 Safari/537.36",
        "Accept":"application/json,text/plain,*/*","Referer":HOME,"X-Requested-With":"XMLHttpRequest",
    })
    try:sess.get(HOME,timeout=18)
    except requests.RequestException:pass
    new=[];statuses=[]
    for sym in sorted(absent):
        status={"symbol":sym,"attempts":0,"retrieved":0,"status":"not_queried"}
        for attempt in range(1,4):
            status["attempts"]=attempt
            try:
                r=sess.get(API,params={"symbol":sym,"index":"equities",
                         "type":"Integrated Filing- Financials","page":1,"size":100},timeout=35)
                if r.status_code==429:raise requests.HTTPError("NSE rate-limited HTTP 429")
                r.raise_for_status();data=r.json()
                filings=data.get("data") if isinstance(data,dict) else None
                if not isinstance(filings,list):raise ValueError("Unexpected NSE schema")
                for z in filings:
                    if not isinstance(z,dict):continue
                    period=pd.to_datetime(z.get("qe_Date"),errors="coerce",dayfirst=True)
                    available=parse_utc(z.get("broadcast_Date"))
                    if pd.isna(period) or pd.isna(available) or available<period.tz_localize("UTC"):continue
                    new.append({
                        "symbol":sym,"period_end":period.tz_localize("UTC"),
                        "available_at_utc":available,
                        "seq_id":str(z.get("seq_Id") or ""),
                        "filing_type":str(z.get("type_Sub") or ""),
                        "audit":str(z.get("audited") or ""),
                        "consolidated":str(z.get("consolidated") or ""),
                        "xbrl_url":str(z.get("xbrl") or ""),
                        "ixbrl_url":str(z.get("ixbrl") or ""),
                        "origin":"official_nse_integrated_financial_index_retry",
                    })
                    status["retrieved"]+=1
                status["status"]="retrieved" if status["retrieved"]>0 else "official_api_empty"
                break
            except (requests.RequestException,ValueError) as exc:
                status["status"]="network_or_schema_error"
                status["error"]=type(exc).__name__+": "+str(exc)[:150]
                time.sleep(attempt*3)
        print(json.dumps(status),flush=True);statuses.append(status)
        time.sleep(.6)
    if new:
        df=pd.concat([df,pd.DataFrame(new)],ignore_index=True)
    df["period_end"]=pd.to_datetime(df["period_end"],utc=True,errors="coerce")
    df["available_at_utc"]=pd.to_datetime(df["available_at_utc"],utc=True,errors="coerce")
    if (df["available_at_utc"]<df["period_end"]).any():
        raise SystemExit("Invalid source publication time earlier than quarter end")
    df=df.drop_duplicates(["symbol","period_end","available_at_utc","seq_id"])
    df.to_parquet(out/"integrated_financial_index_2025_REPAIRED_REHEARSAL_ONLY.parquet",index=False,compression="zstd")
    df.to_csv(out/"integrated_financial_index_2025_REPAIRED_REHEARSAL_ONLY.csv",index=False)
    eligible=df[(df["period_end"]>=pd.Timestamp("2025-03-01",tz="UTC"))&(df["available_at_utc"]<=CUT)]
    covered=set(eligible["symbol"])
    missing=sorted(set(absent)-covered)
    (out/"unresolved_source_symbols.json").write_text(json.dumps(missing,indent=2))
    network_errors=[z for z in statuses if z["status"]=="network_or_schema_error"]
    output={
        "status":"FULL_HISTORICAL_UNIVERSE_INDEX_RECOVERY_REHEARSAL",
        "original_shards":prior["completed_shards"],
        "completed_shards":prior["completed_shards"],
        "full_2025_dec_historical_universe":prior["full_2025_dec_historical_universe"],
        "requested_companies":prior["requested_companies"],
        "index_rows":len(df),
        "preclose_2025_filings":len(eligible),
        "companies_with_preclose_2025_filings":len(covered),
        "coverage":len(covered)/prior["full_2025_dec_historical_universe"],
        "missing_companies":len(missing),
        "retried_symbols":statuses,
        "unresolved_retrieval_errors":len(network_errors),
        "source_errors":len(network_errors),
        "filing_metadata_timestamp_violations":0,
        "ready_for_marketwide_numeric_xbrl_parse":len(covered)>=0.8*prior["full_2025_dec_historical_universe"] and not network_errors,
        "production_eligible":False,
        "numerical_fundamentals_ready":False,
        "source":"NSE Integrated Filing Financials original 2025 history, with targeted source repair",
    }
    (out/"full_integrated_source_summary.json").write_text(json.dumps(output,indent=2))
    print(json.dumps(output,indent=2),flush=True)
    if network_errors:raise SystemExit("Some original NSE source errors remain after retry")
    if len(covered)<0.80*prior["full_2025_dec_historical_universe"]:
        raise SystemExit("Insufficient 2025 historical financial filing coverage")
if __name__=="__main__":main()
