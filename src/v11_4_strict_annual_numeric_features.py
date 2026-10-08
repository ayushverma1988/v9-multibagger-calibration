"""Strict, point-in-time NSE historical annual revenue/PAT growth pilot.

Numerics are read ONLY from source XBRL annual-duration contexts ending on
the requested FY, and only when published by the fold's 15:30 IST close.
No future revisions, mismatched contexts, mixed reporting modes or imputation.
Banks are excluded pending a separate banking metrics methodology.
"""
from __future__ import annotations
import argparse,hashlib,json,math,xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd
from v11_4_longterm_fundamentals import Client, TAG_GROUPS, context_map, local, parse_num

CURRENCY="INR"
def fold_close(value):
    return (pd.Timestamp(value).normalize().tz_localize("Asia/Kolkata")
            +pd.Timedelta(hours=15,minutes=30)).tz_convert("UTC")

def parse_units(root):
    result={}
    for el in root.iter():
        if local(el.tag)!="unit":continue
        vals=[(c.text or "").strip().rsplit(":",1)[-1].upper()
              for c in el.iter() if local(c.tag)=="measure"]
        result[el.attrib.get("id","")]=vals
    return result

def dimensional_contexts(root):
    out={}
    for el in root.iter():
        if local(el.tag)!="context":continue
        out[el.attrib.get("id")]=sum(local(c.tag) in ("explicitMember","typedMember") for c in el.iter())
    return out

def strict_annual_facts(xml,fy):
    root=ET.fromstring(xml)
    ctx=context_map(root);units=parse_units(root);dimensions=dimensional_contexts(root)
    target=pd.Timestamp(fy).date()
    ans={};audit={}
    for key in ("revenue","pat"):
        candidates=[]
        for el in root.iter():
            tag=local(el.tag)
            if tag not in TAG_GROUPS[key]:continue
            ref=el.get("contextRef")
            context=ctx.get(ref,{})
            if context.get("end")!=target:continue
            days=context.get("duration")
            if days is None or not 330<=days<=400:continue
            if dimensions.get(ref,0)!=0:continue
            unit=el.get("unitRef","")
            # Financial values must be documented in INR; no synthetic scaling.
            if CURRENCY not in units.get(unit,[]):continue
            number=parse_num(el.text)
            if number is None or not math.isfinite(number):continue
            candidates.append((TAG_GROUPS[key].index(tag),tag,ref,unit,number))
        if not candidates:
            ans[key]=None
        else:
            v=min(candidates,key=lambda z:z[0])
            ans[key]=v[4]
            audit[key]={"tag":v[1],"context":v[2],"unit":v[3],"value":v[4]}
    return ans,audit

def coherent_run(sub,maxyears=8):
    # One reporting mode throughout each company. Never mix consolidated
    # and standalone when calculating a time-series CAGR.
    x=sub.copy()
    x["mode"]=x["consolidated"].str.lower().map(
        lambda s:"standalone" if "standalone" in s or "non-consolidated" in s
        else "consolidated" if "consolidated" in s else "unknown")
    opts=[]
    for mode,rows in x.groupby("mode"):
        if mode=="unknown":continue
        rows=rows.sort_values(["end","publication"],ascending=[False,False]).drop_duplicates("end")
        years=list(rows.itertuples(index=False))
        if not years:continue
        for start in range(len(years)):
            chain=[years[start]]
            for a in years[start+1:]:
                gap=(chain[-1].end-a.end).days
                if not 335<=gap<=395:break
                chain.append(a)
                if len(chain)>=maxyears:break
            opts.append((len(chain),mode,chain))
    if not opts:return [],"unknown"
    best=max(opts,key=lambda z:(z[0],z[1]=="consolidated"))
    return best[2],best[1]

def growth(a,b,years):
    if a is None or b is None or a<=0 or b<=0:return None
    return 100*((a/b)**(1/years)-1)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--catalog",required=True)
    p.add_argument("--fold-date",default="2024-12-31")
    p.add_argument("--max-companies",type=int,default=32)
    p.add_argument("--max-years",type=int,default=8)
    p.add_argument("--output",required=True)
    a=p.parse_args();out=Path(a.output);out.mkdir(exist_ok=True,parents=True)
    catalog=pd.read_csv(a.catalog,dtype=str).fillna("")
    assert {"symbol","fy_end","available_at_utc","consolidated","xbrl_url"}.issubset(catalog.columns)
    catalog["end"]=pd.to_datetime(catalog["fy_end"],utc=True,errors="coerce")
    catalog["publication"]=pd.to_datetime(catalog["available_at_utc"],utc=True,errors="coerce")
    cutoff=fold_close(a.fold_date)
    x=catalog[
        catalog["end"].notna()&catalog["publication"].notna()&
        (catalog["publication"]<=cutoff)&(catalog["publication"]>=catalog["end"])&
        catalog["xbrl_url"].str.match(r"^https://nsearchives\.nseindia\.com/.*\.xml$",case=False)
    ].copy()
    # Exclude banking filing taxonomy; profit/turnover concepts differ.
    x=x[~x["xbrl_url"].str.contains(r"/(?:INTEGRATED_FILING_)?BANKING_",case=False,regex=True)].copy()
    candidates=[]
    for sym,part in x.groupby("symbol",sort=True):
        seq,mode=coherent_run(part,maxyears=a.max_years)
        if len(seq)>=6:
            candidates.append((sym,mode,seq))
    # Stable deterministic sample: choose companies with deepest historical
    # runs first, then hashed company symbol; this is coverage QA, not a
    # statistically random estimate of live model performance.
    candidates.sort(key=lambda t:(-len(t[2]),hashlib.sha256(t[0].encode()).hexdigest()))
    selected=candidates[:a.max_companies]
    client=Client();facts=[];errors=[];metrics=[]
    for i,(sym,mode,records) in enumerate(selected,1):
        values=[];seenyears=set()
        for rec in records:
            fy=rec.end.date()
            if fy in seenyears:continue
            seenyears.add(fy)
            url=rec.xbrl_url
            ctx={"symbol":sym,"mode":mode,"fy_end":str(fy),
                 "available_at_utc":rec.available_at_utc,"xbrl_url":url}
            try:
                response=client.get(url)
                numeric,audit=strict_annual_facts(response.content,fy)
                ctx.update(numeric)
                ctx["fact_audit"]=json.dumps(audit,sort_keys=True)
                ctx["status"]="complete" if all(numeric.get(k) is not None for k in ("revenue","pat")) else "missing_core"
                facts.append(ctx)
                values.append({"fy":fy,"facts":numeric})
            except Exception as err:
                errors.append({**ctx,"error":str(err)[:180]})
        values.sort(key=lambda z:z["fy"],reverse=True)
        row={"symbol":sym,"mode":mode,"fold_date":a.fold_date,
             "index_years_available":len(records),
             "numeric_complete_years":sum(v["facts"]["revenue"] is not None and v["facts"]["pat"] is not None for v in values),
             "sales_growth_5y_pct":None,"profit_growth_5y_pct":None,
             "sales_growth_7y_pct":None,"profit_growth_7y_pct":None}
        if len(values)>=6 and all(335<=(values[j]["fy"]-values[j+1]["fy"]).days<=395 for j in range(5)):
            if all(z["facts"]["revenue"] is not None for z in (values[0],values[5])):
                row["sales_growth_5y_pct"]=growth(values[0]["facts"]["revenue"],values[5]["facts"]["revenue"],5)
            if all(z["facts"]["pat"] is not None for z in (values[0],values[5])):
                row["profit_growth_5y_pct"]=growth(values[0]["facts"]["pat"],values[5]["facts"]["pat"],5)
        if len(values)>=8 and all(335<=(values[j]["fy"]-values[j+1]["fy"]).days<=395 for j in range(7)):
            if all(z["facts"]["revenue"] is not None for z in (values[0],values[7])):
                row["sales_growth_7y_pct"]=growth(values[0]["facts"]["revenue"],values[7]["facts"]["revenue"],7)
            if all(z["facts"]["pat"] is not None for z in (values[0],values[7])):
                row["profit_growth_7y_pct"]=growth(values[0]["facts"]["pat"],values[7]["facts"]["pat"],7)
        metrics.append(row)
        print(f"candidate {i}/{len(selected)} {sym} asof={a.fold_date} {row['numeric_complete_years']}/{len(records)} strict numeric years",flush=True)
    pd.DataFrame(facts).drop(columns=["fact_audit"],errors="ignore").to_parquet(out/"annual_exact_context_PIT_numeric_candidates.parquet",index=False)
    pd.DataFrame(facts).to_csv(out/"annual_exact_context_PIT_numeric_candidates.csv",index=False)
    pd.DataFrame(errors).to_csv(out/"annual_numeric_parse_errors.csv",index=False)
    m=pd.DataFrame(metrics)
    m.to_csv(out/"annual_5y7y_numeric_features_PIT_CANDIDATES.csv",index=False)
    summary={
        "scope":"2024_FOLD_RESEARCH_NUMERICS_ONLY",
        "fold_cutoff_utc":cutoff.isoformat(),
        "original_files_queried":len(facts)+len(errors),"parsed":len(facts),"errors":len(errors),
        "companies_index_5y_potential":len(candidates),
        "companies_sampled":len(selected),
        "strict_revenue_and_pat_document_rows":sum(z.get("status")=="complete" for z in facts),
        "companies_sales_growth_5y_calculable":int(m["sales_growth_5y_pct"].notna().sum()) if len(m) else 0,
        "companies_profit_growth_5y_calculable":int(m["profit_growth_5y_pct"].notna().sum()) if len(m) else 0,
        "companies_sales_growth_7y_calculable":int(m["sales_growth_7y_pct"].notna().sum()) if len(m) else 0,
        "companies_profit_growth_7y_calculable":int(m["profit_growth_7y_pct"].notna().sum()) if len(m) else 0,
        "banking_facts":"intentionally excluded",
        "PIT":"No filings after historical 15:30 IST close. Strict FY period end/duration and INR unit checks.",
        "not_ready_for_production":True,
        "caution":"Still requires annual-metric numerical reconciliation, company-ID stitching, sector-specific annual tags and stable universe-wide coverage.",
    }
    (out/"annual_5y7y_numeric_pilot_summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)
    if len(selected)<5:raise SystemExit("Insufficient candidate company fiscal-year histories")
    if len(facts)<6:raise SystemExit("Insufficient downloaded numerical data")
if __name__=="__main__":main()
