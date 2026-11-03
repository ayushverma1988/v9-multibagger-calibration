"""Original bank-sector NSE XBRL economic facts, with period-context matching.

Bank total income != non-bank sales. Never substitute industrial OPM or ROCE
for banking balance-sheet/credit-quality measures.
"""
from __future__ import annotations
import json,hashlib,time
from datetime import date
from pathlib import Path
import xml.etree.ElementTree as ET
import requests,pandas as pd

OUT=Path("outputs_v11_4_bank_fact_probe")
DOCS=[
("BANKBARODA","2025-06-30","quarter","https://nsearchives.nseindia.com/corporate/xbrl/INTEGRATED_FILING_BANKING_1493754_25072025060911_WEB.xml"),
("BANKBARODA","2025-09-30","quarter","https://nsearchives.nseindia.com/corporate/xbrl/INTEGRATED_FILING_BANKING_1562872_31102025070853_WEB.xml"),
("CANBK","2025-06-30","quarter","https://nsearchives.nseindia.com/corporate/xbrl/INTEGRATED_FILING_BANKING_1492265_24072025034753_WEB.xml"),
("CANBK","2025-09-30","quarter","https://nsearchives.nseindia.com/corporate/xbrl/INTEGRATED_FILING_BANKING_1561739_30102025060404_WEB.xml"),
("INDIANB","2025-06-30","quarter","https://nsearchives.nseindia.com/corporate/xbrl/INTEGRATED_FILING_BANKING_1492860_24072025082715_WEB.xml"),
("INDIANB","2025-09-30","quarter","https://nsearchives.nseindia.com/corporate/xbrl/INTEGRATED_FILING_BANKING_1550927_16102025073931_WEB.xml"),
("INDIANB","2021-03-31","annual","https://nsearchives.nseindia.com/corporate/xbrl/BANKING_70725_453443_30052021045500_WEB.xml"),
("INDIANB","2020-03-31","annual","https://nsearchives.nseindia.com/corporate/xbrl/BANKING_56325_267952_24062020123420_WEB.xml"),
("INDIANB","2019-03-31","annual","https://nsearchives.nseindia.com/corporate/xbrl/BANKING_48501_136145_14092019030859_WEB.xml"),
]
TAGSETS={
    "bank_total_income":["Income"],
    "bank_interest_earned":["InterestEarned"],
    "bank_other_income":["OtherIncome"],
    "bank_pat":["ProfitLossForThePeriod","ProfitLossFromOrdinaryActivitiesAfterTax"],
    "bank_operating_profit_pre_provisions":["OperatingProfitBeforeProvisionAndContingencies"],
    "bank_gross_npa":["GrossNonPerformingAssets"],
    "bank_net_npa":["NonPerformingAssets"],
}
NS="http://www.xbrl.org/2003/instance"
def local(tag):return str(tag).rsplit("}",1)[-1]
def contexts(root):
    out={}
    for el in root.iter():
        if local(el.tag)!="context":continue
        id=el.get("id");start=end=None;dimensions=0
        for sub in el.iter():
            nm=local(sub.tag)
            if nm=="startDate":
                try:start=date.fromisoformat(str(sub.text or "").strip())
                except ValueError:pass
            if nm=="endDate":
                try:end=date.fromisoformat(str(sub.text or "").strip())
                except ValueError:pass
            if nm=="instant" and end is None:
                try:end=date.fromisoformat(str(sub.text or "").strip())
                except ValueError:pass
            if nm in ("explicitMember","typedMember"):dimensions+=1
        out[id]={"start":start,"end":end,"dimensions":dimensions}
    return out

def parse(xml,fy_end,mode):
    root=ET.fromstring(xml);ctx=contexts(root);results={}
    target=date.fromisoformat(fy_end)
    for k,tagset in TAGSETS.items():
        candidates=[]
        for el in root.iter():
            tag=local(el.tag)
            if tag not in tagset or not el.text or not el.get("contextRef"):continue
            context=ctx.get(el.get("contextRef"),{})
            if context.get("end")!=target:continue
            start=context.get("start");end=context.get("end")
            duration=((end-start).days+1) if start and end else None
            if mode=="quarter" and not(duration is not None and 60<=duration<=120):
                if k not in ("bank_gross_npa","bank_net_npa") or duration is not None:continue
            if mode=="annual" and not(duration is not None and 300<=duration<=430):
                if k not in ("bank_gross_npa","bank_net_npa") or duration is not None:continue
            try:v=float(el.text.strip().replace(",",""))
            except ValueError:continue
            score=1000-10*context.get("dimensions",0)
            if duration is not None:score-=abs(duration-(91 if mode=="quarter" else 365))/10
            score+=3*(len(tagset)-tagset.index(tag))
            candidates.append((score,v,tag,el.get("unitRef",""),el.get("decimals",""),el.get("contextRef")))
        if candidates:
            candidates.sort(reverse=True)
            z=candidates[0]
            results[k+"_raw"]=z[1]
            results[k+"_tag"]=z[2]
            results[k+"_unit"]=z[3]
            results[k+"_context"]=z[5]
        else:results[k+"_raw"]=None
    if all(results.get(k+"_raw") is not None for k in ("bank_total_income","bank_interest_earned","bank_other_income")):
        total=results["bank_total_income_raw"]
        calc=results["bank_interest_earned_raw"]+results["bank_other_income_raw"]
        results["income_components_relative_diff"]=abs(total-calc)/max(abs(total),1)
    return results

def main():
    OUT.mkdir(exist_ok=True,parents=True)
    ses=requests.Session();ses.headers.update({
        "User-Agent":"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124 Safari/537.36",
        "Referer":"https://www.nseindia.com/","Accept":"application/xml,text/xml,*/*",
    })
    rows=[];errors=[]
    for sym,period,mode,url in DOCS:
        row={"symbol":sym,"period_end":period,"statement_type":mode,"source_url":url}
        try:
            r=ses.get(url,timeout=25);r.raise_for_status()
            row["document_sha256"]=hashlib.sha256(r.content).hexdigest()
            row.update(parse(r.content,period,mode))
            row["status"]="parsed"
            rows.append(row)
        except (requests.RequestException,ET.ParseError,ValueError) as exc:
            errors.append({**row,"error":f"{type(exc).__name__}: {str(exc)[:140]}"})
        print(sym,period,mode,"parsed" if row.get("status")=="parsed" else "ERROR",flush=True)
        time.sleep(.25)
    pd.DataFrame(rows).to_csv(OUT/"banking_fact_candidate_values.csv",index=False)
    pd.DataFrame(errors).to_csv(OUT/"banking_fact_parse_errors.csv",index=False)
    summary={
        "filings_requested":len(DOCS),"filings_parsed":len(rows),"source_errors":len(errors),
        "bank_total_income_and_pat_both_available":sum(r.get("bank_total_income_raw") is not None and r.get("bank_pat_raw") is not None for r in rows),
        "bank_interest_and_other_income_reconciled_within_10_percent":sum(r.get("income_components_relative_diff",1)<=0.10 for r in rows),
        "production_ready":False,
        "note":"Banking-sector candidates only. Monetary units, annual/quarter concept mappings and credit-risk ratios require independent signoff.",
    }
    (OUT/"banking_xbrl_fact_summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)
    if len(rows)<7 or summary["bank_total_income_and_pat_both_available"]<5:
        raise SystemExit("Banking fact extraction pilot did not meet source extraction gate")
if __name__=="__main__":main()
