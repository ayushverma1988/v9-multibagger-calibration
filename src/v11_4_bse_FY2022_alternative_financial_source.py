"""Read-only BSE FY2022 historical financial-results archive discovery.

BSE public financial-results listing is an *alternate candidate source*,
not a validated NSE-to-BSE identity map or a recovered 2022 financial fact.
Capture publication timestamps, year/type, original response fingerprints
and possible XBRL links. Do NOT use a scrip's displayed name to join stocks.
"""
from __future__ import annotations
import argparse, hashlib, json, re, time
from pathlib import Path
from urllib.parse import urljoin, urlparse, parse_qs
from datetime import datetime, timezone, timedelta
import requests
from bs4 import BeautifulSoup

IST=timezone(timedelta(hours=5,minutes=30))
ROOT="https://www.bseindia.com"
API="https://api.bseindia.com/BseIndiaAPI/api"
PAGE=ROOT+"/corporates/comp_results.aspx"
SAMPLES=(
    ("JBCHEPHARM","506943"),
    ("TEXMOPIPES","533164"),
    ("BLBLIMITED","532290"),
    ("HDFCBANK","500180"),
)
UA="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36"
TIME=re.compile(r"\b(\d{2}-\d{2}-\d{4}\s+\d{2}:\d{2}:\d{2})\b")
FY=re.compile(r"\b20\d{2}\s*[-–]\s*20\d{2}\b")
FY2022=re.compile(r"\b2021\s*[-–]\s*2022\b")
HIST_URLS=("/corporates/comp_results.aspx", "/corporates/Comp_Resultsnew.aspx")

def clean_bse_url(url):
    p=urlparse(url)
    if p.scheme!="https" or p.hostname not in ("www.bseindia.com","api.bseindia.com"):
        raise ValueError("BSE source must be HTTPS on official domain")
    return url

def candidate_row(cells, links, fetch_time, expected_code):
    combined=" | ".join(" ".join(c.split()) for c in cells)
    year=FY.search(combined)
    if not year:return None
    timestamps=TIME.findall(combined)
    parsed=[]
    for t in timestamps:
        try: parsed.append(datetime.strptime(t,"%d-%m-%Y %H:%M:%S").replace(tzinfo=IST))
        except ValueError: pass
    text=combined.lower()
    # BSE may have Year, Quarter, Half Year. Never equate Q4 with annual.
    is_fy2022=bool(FY2022.search(combined))
    has_annual=bool(re.search(r"\byear\b|\bannual\b",text))
    has_quarter=bool(re.search(r"\bquarter\b",text))
    # Date-year label alone is NOT fiscal-period proof.
    urls=[]
    for link in links:
        try:
            target=clean_bse_url(urljoin(ROOT,link))
            urls.append(target)
        except ValueError:continue
    before_cutoff=bool(parsed and min(parsed)<=datetime(2023,12,29,15,30,tzinfo=IST))
    return {
        "bse_scrip_code":expected_code,
        "reported_fiscal_year":year.group().replace(" ",""),
        "is_FY2021_2022":is_fy2022,
        "row_type_year":has_annual,
        "row_type_quarter":has_quarter,
        "reported_filing_timestamps_IST":[v.isoformat() for v in parsed],
        "date_precision_verified":bool(parsed),
        "filing_before_2023_Dec_29_close":before_cutoff,
        "source_row_plaintext_snippet":combined[:420],
        "official_links":urls[:10],
        "XBRL_exact_context_and_numeric_units_verified":False,
        "NSE_ISIN_BSE_ISIN_identity_verified":False,
    }

def parse_listing(html,code,fetch_time):
    soup=BeautifulSoup(html,"html.parser")
    title=soup.title.get_text(" ",strip=True) if soup.title else ""
    candidates=[]
    for row in soup.select("tr"):
        cells=[c.get_text(" ",strip=True) for c in row.find_all(["th","td"],recursive=False)]
        if not cells: continue
        links=[a.get("href","") for a in row.find_all("a",href=True)]
        x=candidate_row(cells,links,fetch_time,code)
        if x: candidates.append(x)
    # Literal original HTML is useful to distinguish inaccessible page vs
    # no FY2022 table, but it is NOT evidence of a financial value.
    return {
       "title":title,
       "rows_with_fiscal_year":len(candidates),
       "FY2021_2022_candidate_rows":sum(x["is_FY2021_2022"] for x in candidates),
       "FY2021_2022_annual_rows":sum(x["is_FY2021_2022"] and x["row_type_year"] for x in candidates),
       "FY2021_2022_annual_date_verified":sum(
           x["is_FY2021_2022"] and x["row_type_year"] and x["filing_before_2023_Dec_29_close"] for x in candidates),
       "rows":candidates,
       "HTML_has_FY2021_2022_text":bool(FY2022.search(soup.get_text(" ",strip=True))),
    }

def checked_get(session,url,params=None,timeout=25):
    clean_bse_url(url)
    resp=session.get(url,params=params,timeout=timeout)
    result={
      "requested_url":resp.url if clean_bse_url(resp.url) else "",
      "status":resp.status_code,
      "type":resp.headers.get("Content-Type",""),
      "bytes":len(resp.content),
      "sha256":hashlib.sha256(resp.content).hexdigest(),
    }
    return resp,result

def run(out,max_codes=4):
    out=Path(out);out.mkdir(exist_ok=True,parents=True)
    sess=requests.Session()
    sess.headers.update({"User-Agent":UA,"Referer":PAGE,"Accept":"text/html,application/json,text/plain,*/*"})
    diagnostics=[];rows=[]
    for symbol,code in SAMPLES[:max_codes]:
        entry={"probe_expected_NSE_symbol_UNVERIFIED":symbol,"BSE_scrip_code":code,
               "source_code_mapping_is_not_independent_ISIN_match":True}
        try:
            resp,proof=checked_get(sess,PAGE,{"Code":code,"PID":1})
            entry["source_http"]=proof
            resp.raise_for_status()
            ct=resp.headers.get("Content-Type","").lower()
            if "html" not in ct: raise ValueError("BSE returned non-HTML results page")
            if len(resp.content)<5000: raise ValueError("Too small for original BSE results page")
            page=parse_listing(resp.text,code,datetime.now(timezone.utc).isoformat())
            entry.update({k:v for k,v in page.items() if k!="rows"})
            rows.extend({"expected_symbol_unverified":symbol,**z} for z in page["rows"])
            entry["source_readable"]=True
        except (requests.RequestException,ValueError) as e:
            entry["source_readable"]=False
            entry["source_error"]=type(e).__name__+": "+str(e)[:250]
        # Separate official BSE API probe; response may be provider-blocked
        for key,route,params in (
            ("header", "/getScripHeaderData/w",{"scripcode":code}),
            ("scrip_master","/ListofScripData/w",{"scripcode":code,"Group":"","industry":"","segment":"Equity","status":"Active"}),
        ):
            try:
                resp,proof=checked_get(sess,API+route,params)
                entry["bse_"+key+"_transport"]=proof
                if resp.ok and "json" in resp.headers.get("Content-Type","").lower():
                    obj=resp.json()
                    entry["bse_"+key+"_json_schema_top_level"]=(
                        list(obj)[:25] if isinstance(obj,dict) else "JSON_LIST"
                    )
                    # Output only candidate metadata; never infer ISIN if missing.
                    if isinstance(obj,dict):
                        text=json.dumps(obj)[:100000]
                        ids=sorted(set(re.findall(r"\bINE[A-Z0-9]{9}\b",text)))
                        entry["bse_"+key+"_ISIN_candidates"]=ids[:10]
                else:entry["bse_"+key+"_json_readable"]=False
            except (requests.RequestException,ValueError) as e:
                entry["bse_"+key+"_error"]=type(e).__name__+": "+str(e)[:180]
        diagnostics.append(entry)
        print("BSE archived-financials",code,
              "HTTP",entry.get("source_http",{}).get("status"),
              "FY2022 rows",entry.get("FY2021_2022_candidate_rows",0),
              "FY2022 annual",entry.get("FY2021_2022_annual_rows",0),flush=True)
        time.sleep(0.4)
    (out/"bse_FY2022_candidates_not_PIT_validated.json").write_text(json.dumps(rows,indent=2))
    report={
      "scope":"BSE_OFFICIAL_FY2022_HISTORICAL_ALTERNATE_SOURCE_DISCOVERY_ONLY",
      "sample_scrip_codes":len(diagnostics),
      "official_BSE_source_pages_readable":sum(z.get("source_readable",False) for z in diagnostics),
      "FY2021_2022_annual_rows_found":sum(z.get("FY2021_2022_annual_rows",0) for z in diagnostics),
      "FY2021_2022_annual_rows_with_timestamp_prior_to_frozen_2023_close":
         sum(z.get("FY2021_2022_annual_date_verified",0) for z in diagnostics),
      "source_ISIN_cross_exchange_verified":False,
      "annual_revenue_PAT_recovered":False,
      "original_frozen_V11_4_model_changed":False,
      "scraped_company_name_join_to_historical_NSE_prohibited":True,
      "no_2022_annual_growth_promoted":True,
      "probes":diagnostics,
    }
    (out/"bse_original_2022_source_discovery_summary.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)
    if not report["official_BSE_source_pages_readable"]:
       raise SystemExit("BSE source blocked/unavailable from runner: retain transport status")
    return report

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",required=True)
    p.add_argument("--max-codes",type=int,default=4)
    args=p.parse_args()
    if args.max_codes<1 or args.max_codes>4:raise ValueError("bounded sample 1-4")
    run(args.out,args.max_codes)
if __name__=="__main__":main()
