"""Recover FY2022/FY2023 original NSE annual source pairs as-of 2023 fold.

A bounded RESEARCH-ONLY historical backfill pilot. Requires original filing
publications before that fold's 15:30 IST close, consistent reporting mode,
exact FY periods, XBRL currency and context. Does NOT create 5y/7y numbers,
use outcome labels, or append values to the standalone prediction model.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import pandas as pd
from v11_4_strict_annual_numeric_features import strict_annual_facts,fold_close

def mode(s):
    a=str(s).lower()
    if "standalone" in a or "non-consolidated" in a:return "standalone"
    if "consolidated" in a:return "consolidated"
    return "unknown"

def choose_pairs(catalog, universe, fold, years=(2022,2023)):
    """Strictly paired original annual URLs, not today's reconstructed facts."""
    need={"symbol","fy_end","available_at_utc","consolidated","xbrl_url"}
    if not need.issubset(catalog):
        raise ValueError("Annual source index omitted immutable source fields")
    if len(years)!=2 or years[1]-years[0]!=1:
        raise ValueError("Expected two consecutive fiscal years")
    cut=fold_close(fold)
    z=catalog.copy()
    z["symbol"]=z["symbol"].astype(str).str.upper().str.strip()
    z["fy_end"]=pd.to_datetime(z["fy_end"],errors="coerce",utc=True)
    z["pub"]=pd.to_datetime(z["available_at_utc"],errors="coerce",utc=True,format="mixed")
    z["mode"]=z["consolidated"].map(mode)
    good_host=z["xbrl_url"].str.match(r"^https://nsearchives\.nseindia\.com/[^?#]+\.xml$",case=False,na=False)
    nonbank=~z["xbrl_url"].str.contains(r"/(?:INTEGRATED_FILING_)?BANKING_",case=False,regex=True,na=False)
    target_years=[pd.Timestamp(f"{yr}-03-31T00:00:00Z") for yr in years]
    z=z[z["symbol"].isin(universe)&z["fy_end"].isin(target_years)&
        z["pub"].notna()&(z["pub"]>=z["fy_end"])&(z["pub"]<=cut)&
        z["mode"].isin(["consolidated","standalone"])&good_host&nonbank].copy()
    z=z.sort_values(["pub","xbrl_url"]).drop_duplicates(["symbol","mode","fy_end"],keep="first")
    pairs=[]
    for sym,records in z.groupby("symbol",sort=False):
        by_mode=records.groupby("mode")["fy_end"].nunique()
        candidates=[x for x in ("consolidated","standalone") if by_mode.get(x,0)==2]
        if not candidates:continue
        chosen=candidates[0]
        paired=records[records["mode"].eq(chosen)].sort_values("fy_end")
        if len(paired)!=2:raise ValueError("Duplicate financial fiscal pair")
        pairs.append((sym,chosen,paired))
    return sorted(pairs,key=lambda t:hashlib.sha256(t[0].encode()).hexdigest())

def select_2023_fold(snapshot):
    x=snapshot.copy()
    x["date"]=pd.to_datetime(x["date"],errors="raise").dt.normalize()
    x["symbol"]=x["symbol"].astype(str).str.upper().str.strip()
    dates=sorted(x.loc[(x["date"].dt.year==2023)&
                       (x["date"].dt.month==12),"date"].unique())
    if len(dates)!=1:raise ValueError("Must find exactly one frozen Dec 2023 fold")
    day=pd.Timestamp(dates[0]).normalize()
    universe=set(x.loc[x["date"].eq(day),"symbol"])
    if len(universe)<500:raise ValueError("Missing original 2023 fold company universe")
    return str(day.date()),universe

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--annual-index",required=True)
    p.add_argument("--snapshot",required=True)
    p.add_argument("--max-companies",type=int,default=24)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    if not 1<=a.max_companies<=96:raise SystemExit("Bound sample size to 1-96 companies")
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    history=pd.read_parquet(a.snapshot,columns=["date","symbol"])
    fold,universe=select_2023_fold(history)
    catalog=pd.read_csv(a.annual_index,dtype=str).fillna("")
    candidates=choose_pairs(catalog,universe,fold)
    chosen=candidates[:a.max_companies]
    from v11_4_longterm_fundamentals import Client
    client=Client()
    docs=[];errors=[];complete={}
    for i,(symbol,statement_mode,pair) in enumerate(chosen,1):
        for rec in pair.itertuples(index=False):
            data={"symbol":symbol,"mode":statement_mode,
                  "fy_end":str(rec.fy_end.date()),
                  "source_available_utc":rec.pub.isoformat(),
                  "original_nse_xbrl_url":rec.xbrl_url}
            try:
                raw=client.get(rec.xbrl_url).content
                numeric,audit=strict_annual_facts(raw,rec.fy_end.date(),source_is_annual=True)
                data.update(numeric)
                data["original_xml_sha256"]=hashlib.sha256(raw).hexdigest()
                data["fact_context_audit"]=json.dumps(audit,sort_keys=True)
                data["status"]="VERIFIED_CORE_REVENUE_PAT" if all(numeric.get(k) is not None for k in ("revenue","pat")) else "INCOMPLETE_OR_UNSUPPORTED_CONTEXT"
                docs.append(data)
                if data["status"]=="VERIFIED_CORE_REVENUE_PAT":
                    complete.setdefault(symbol,{})[rec.fy_end.year]=numeric
            except Exception as exc:
                errors.append({**data,"error":str(exc)[:200]})
        if i%8==0:print("FY2023 historical original NSE annual pairs",i,"/",len(chosen),
                       "strict complete docs",sum(x["status"]=="VERIFIED_CORE_REVENUE_PAT" for x in docs),
                       "errors",len(errors),flush=True)
    rows=[]
    for sym,statement_mode,_ in chosen:
        values=complete.get(sym,{})
        r={"symbol":sym,"mode":statement_mode,"historical_fold":fold,
           "source_years_verified":len(values),"fiscal_years":"2022|2023",
           "fy2023_vs_fy2022_revenue_yoy_fraction":None,
           "fy2023_vs_fy2022_pat_yoy_fraction":None}
        if all(y in values for y in (2022,2023)):
            for key,col in (("revenue","fy2023_vs_fy2022_revenue_yoy_fraction"),
                            ("pat","fy2023_vs_fy2022_pat_yoy_fraction")):
                old=values[2022][key];new=values[2023][key]
                if old is not None and new is not None and old>0:
                    r[col]=new/old-1
        rows.append(r)
    pd.DataFrame(docs).to_csv(out/"original_NSE_FY2022_FY2023_XBRL_documents_RESEARCH.csv",index=False)
    pd.DataFrame(errors).to_csv(out/"FY2023_historical_parse_failures.csv",index=False)
    pd.DataFrame(rows).to_csv(out/"FY2023_paired_annual_research_only.csv",index=False)
    valid=sum(set(d)=={2022,2023} for d in complete.values())
    report={
      "scope":"HISTORICAL_FY2023_ORIGINAL_NSE_ANNUAL_NUMERIC_RECOVERY_PILOT",
      "historical_fold":fold,"cutoff_UTC":fold_close(fold).isoformat(),
      "original_universe_stocks":len(universe),
      "eligible_FY2022_FY2023_source_pairs":len(candidates),
      "deterministically_sampled_pairs":len(chosen),
      "original_XML_documents_parsed":len(docs),
      "original_XML_fetch_errors":len(errors),
      "strict_annual_core_revenue_PAT_verified_documents":sum(z["status"]=="VERIFIED_CORE_REVENUE_PAT" for z in docs),
      "companies_with_two_strict_annual_revenue_PAT_documents":valid,
      "companies_with_one_year_revenue_growth":sum(pd.notna(z["fy2023_vs_fy2022_revenue_yoy_fraction"]) for z in rows),
      "companies_with_one_year_profit_growth":sum(pd.notna(z["fy2023_vs_fy2022_pat_yoy_fraction"]) for z in rows),
      "no_same_day_or_future_source":True,"no_mixed_reporting_modes":True,
      "no_five_seven_year_inferences":True,"no_missing_values_imputed":True,
      "historical_fold_model_training_or_predictions_changed":False,
      "source_coverage_qualifies_for_four_family_training":False,
      "next":"Expand with source-quality controlled historical shards only if pilot finds verified pairs"
    }
    (out/"FY2023_original_annual_source_pilot_summary.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)
    if len(candidates)==0:
        raise SystemExit("No 2023 point-in-time historical original NSE annual pairs indexed; no fallback to current figures")
if __name__=="__main__":main()
