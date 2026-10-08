"""2025-12 fold: source-verified FY2024 legacy + FY2025 Integrated YTD pair.

Only matched reporting modes and original historical publication timestamps.
Integrated March 2025 Q4 FourD is accepted as fiscal-year YTD only when
the annual-vs-quarterly distinction has independently passed source QA.
"""
import argparse,hashlib,json
from pathlib import Path
import pandas as pd
from v11_4_longterm_fundamentals import Client
from v11_4_strict_annual_numeric_features import fold_close,strict_annual_facts

def reportmode(s):
    s=str(s).lower()
    if "non-consolidated" in s or "standalone" in s:return "standalone"
    if "consolidated" in s:return "consolidated"
    return "unknown"

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--annual",required=True)
    p.add_argument("--integrated",required=True)
    p.add_argument("--snapshot",required=True)
    p.add_argument("--qa",required=True)
    p.add_argument("--max-companies",type=int,default=96)
    p.add_argument("--output",required=True)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    qa=json.loads(Path(a.qa).read_text())
    if qa.get("verified_FourD_YTD",0)<20:raise SystemExit("Independent FY2025 FourD source mapping QA insufficient")
    cutoff=fold_close("2025-12-31")
    sn=pd.read_parquet(a.snapshot,columns=["date","symbol"])
    sn["date"]=pd.to_datetime(sn["date"],errors="coerce").dt.normalize()
    syms=set(sn.loc[sn["date"].eq(pd.Timestamp("2025-12-31")),"symbol"].astype(str).str.upper().str.strip())
    old=pd.read_csv(a.annual,dtype=str).fillna("")
    old["period_end"]=pd.to_datetime(old["fy_end"],utc=True,format="mixed",errors="coerce")
    old["pub"]=pd.to_datetime(old["available_at_utc"],utc=True,format="mixed",errors="coerce")
    old["mode"]=old["consolidated"].map(reportmode)
    old["symbol"]=old["symbol"].str.upper().str.strip()
    old=old[old["symbol"].isin(syms)&old["period_end"].eq(pd.Timestamp("2024-03-31",tz="UTC"))&
        old["pub"].notna()&(old["pub"]<=cutoff)&(old["pub"]>=old["period_end"])&
        old["xbrl_url"].str.match(r"^https://nsearchives\.nseindia\.com/.*\.xml$",case=False)&
        ~old["xbrl_url"].str.contains(r"/BANKING_",case=False)].copy()
    new=pd.read_parquet(a.integrated)
    new["period_end"]=pd.to_datetime(new["period_end"],utc=True,format="mixed",errors="coerce")
    new["pub"]=pd.to_datetime(new["available_at_utc"],utc=True,format="mixed",errors="coerce")
    new["mode"]=new["consolidated"].map(reportmode)
    new["symbol"]=new["symbol"].astype(str).str.upper().str.strip()
    new=new[new["symbol"].isin(syms)&new["period_end"].eq(pd.Timestamp("2025-03-31",tz="UTC"))&
        new["pub"].notna()&(new["pub"]<=cutoff)&(new["pub"]>=new["period_end"])&
        new["xbrl_url"].astype(str).str.match(r"^https://nsearchives\.nseindia\.com/.*\.xml$",case=False)&
        ~new["xbrl_url"].astype(str).str.contains(r"/INTEGRATED_FILING_BANKING_",case=False)].copy()
    old=old.sort_values("pub").drop_duplicates(["symbol","mode"])
    new=new.sort_values("pub").drop_duplicates(["symbol","mode"])
    older={(r.symbol,r.mode):r for r in old.itertuples(index=False)}
    newer={(r.symbol,r.mode):r for r in new.itertuples(index=False)}
    keys=set(older)&set(newer)
    companies={x[0] for x in keys}
    chosen=[]
    for symbol in sorted(companies,key=lambda z:hashlib.sha256(z.encode()).hexdigest())[:a.max_companies]:
        mode="consolidated" if (symbol,"consolidated") in keys else "standalone"
        chosen.append((symbol,mode))
    client=Client();facts=[];errors=[];fully_verified=0
    for i,(symbol,mode) in enumerate(chosen,1):
        complete=True
        for period,source_row in (("2024-03-31",older[(symbol,mode)]),("2025-03-31",newer[(symbol,mode)])):
            item={"symbol":symbol,"mode":mode,"fy_end":period,
                "available_at_utc":source_row.available_at_utc,
                "xbrl_url":source_row.xbrl_url,
                "source_type":"NSE_legacy_Annual" if period=="2024-03-31" else "NSE_integrated_2025Q4_annual_FourD",
                "status":"pending"}
            try:
                doc=client.get(source_row.xbrl_url).content
                item["original_xml_sha256"]=hashlib.sha256(doc).hexdigest()
                numerics,audit=strict_annual_facts(doc,period,source_is_annual=True)
                item.update(numerics)
                item["fact_audit"]=json.dumps(audit,sort_keys=True)
                item["status"]="complete" if all(numerics.get(x) is not None for x in ("revenue","pat")) else "missing_core"
                if period=="2025-03-31" and item["status"]=="complete":
                    if not all(audit.get(x,{}).get("context")=="FourD" for x in ("revenue","pat")):
                        item["status"]="rejected_nonFourD_2025"
                facts.append(item)
                complete &=item["status"]=="complete"
            except Exception as e:
                errors.append({**item,"error":str(e)[:180]});complete=False
        fully_verified+=int(complete)
        if i%16==0:print("paired",i,"/",len(chosen),"verified",fully_verified,"errors",len(errors),flush=True)
    pd.DataFrame(facts).to_csv(out/"strict_2025_fold_annual_numeric_pairs.csv",index=False)
    pd.DataFrame(errors).to_csv(out/"strict_2025_fold_source_errors.csv",index=False)
    summary={
        "scope":"FY2025_PAIRED_ANNUAL_PIT_NUMERIC_RESEARCH",
        "original_historical_universe":len(syms),
        "eligible_companies_with_2024_legacy_and_2025_integrated_same_mode":len(companies),
        "sampled_companies":len(chosen),"paired_revenue_PAT_docs":fully_verified*2,
        "fully_verified_2024_2025_pairs":fully_verified,
        "attempted_docs":len(facts)+len(errors),"download_errors":len(errors),
        "mode_mixing":False,"asof_close":cutoff.isoformat(),
        "production_approved":False,"full_market_numeric_coverage_validated":False,
    }
    (out/"FY2025_annual_PIT_pairing_summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)
    if fully_verified<max(16,a.max_companies//3):
        raise SystemExit("Paired verified FY2024 vs FY2025 annual source financials insufficient")
if __name__=="__main__":main()
