"""Strict historical NSE legacy annual FY2022 FourD extraction, research only.

Source: original NSE yearly financial-result XBRL, which in some older files
contains named fiscal-period facts attached to FourD but no FourD xbrli:context.
Require ALL corroborating yearly metadata, explicit FY dates, reporting mode,
INR unit, and original source hash. This does NOT change the V11.4 predictor.
"""
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd
from v11_4_longterm_fundamentals import TAG_GROUPS,local,parse_num,context_map,Client
from v11_4_strict_annual_numeric_features import parse_units, fold_close

def one_value(root,tag,ref):
 vals=[(e.text or "").strip() for e in root.iter() if local(e.tag)==tag and e.get("contextRef")==ref]
 if len(vals)!=1 or not vals[0]:raise ValueError("Missing or ambiguous explicit "+tag+" "+ref)
 return vals[0]

def statement_mode(value):
 x=str(value).strip().lower()
 if x in ("consolidated","c"):return "consolidated"
 if x in ("standalone","non-consolidated","non consolidated","s"):return "standalone"
 raise ValueError("Unrecognized historical standalone/consolidated mode")

def extract_2022_legacy_fourd(raw,fy="2022-03-31",mode_expected=None,source_is_annual=False):
 if not source_is_annual:
  raise ValueError("Quarterly XBRL cannot be promoted to annual fiscal numerics")
 target=pd.Timestamp(fy).date()
 if target.month!=3 or target.day!=31:raise ValueError("Only fiscal year ended March 31 supported")
 start=pd.Timestamp(year=target.year-1,month=4,day=1).date()
 root=ET.fromstring(raw)
 if local(root.tag)!="xbrl":raise ValueError("Not original NSE XBRL instance")
 y=one_value(root,"ReportingQuarter","OneD")
 if y.casefold() not in ("yearly","annual","year"):
  raise ValueError("Source never explicitly identified annual Yearly financial result")
 from_s=one_value(root,"DateOfStartOfReportingPeriod","FourD")
 to_s=one_value(root,"DateOfEndOfReportingPeriod","FourD")
 try:source_start=pd.Timestamp(from_s).date(); source_end=pd.Timestamp(to_s).date()
 except (ValueError,TypeError):raise ValueError("Unparseable original explicit fiscal period") from None
 if source_start!=start or source_end!=target:
  raise ValueError("Original FourD fiscal period does not match the expected FY")
 mode=statement_mode(one_value(root,"NatureOfReportStandaloneConsolidated","FourD"))
 if mode_expected is not None and mode!=statement_mode(mode_expected):
  raise ValueError("Historical original NSE FY mode changed or mismatched")
 # If a genuine FourD xbrli context exists, it must agree, never override it.
 contexts=context_map(root)
 if "FourD" in contexts:
  c=contexts["FourD"]
  if c.get("end")!=target or c.get("start")!=start:
   raise ValueError("Real FourD XBRL context contradicts labelled FY metadata")
 units=parse_units(root)
 results={};provenance={}
 for feature in ("revenue","pat"):
  selected=None
  for tag in TAG_GROUPS[feature]:
   choices=[e for e in root.iter() if local(e.tag)==tag and e.get("contextRef")=="FourD"]
   if not choices:continue
   if len(choices)>1:raise ValueError("Ambiguous repeated FourD "+feature+" tag "+tag)
   item=choices[0]
   u=item.get("unitRef","")
   if "INR" not in units.get(u,[]):raise ValueError("Unsupported currency unit for FourD "+feature)
   val=parse_num(item.text)
   if val is None or not math.isfinite(val):raise ValueError("Non-numeric FourD "+feature)
   selected=(float(val),tag,u)
   break
  if selected is None:raise ValueError("Missing exact FourD "+feature+" value")
  results[feature]=selected[0]
  provenance[feature]={"tag":selected[1],"unit_id":selected[2],"context":"FourD",
                       "explicit_fiscal_start":str(start),"explicit_fiscal_end":str(target)}
 if results["revenue"]<0:raise ValueError("Implausible negative FY revenue")
 return results,{"report_period_label":y,"statement_mode":mode,
                 "source_is_annual":True,"explicit_fiscal_start":str(start),
                 "explicit_fiscal_end":str(target),"numeric_concept_provenance":provenance,
                 "legacy_yearly_FourD_without_guessed_context":True}

def run(pilot,out):
 docs=pd.read_csv(pilot,dtype=str)
 required={"symbol","mode","fy_end","source_available_utc","original_nse_xbrl_url",
           "original_xml_sha256","status","revenue","pat"}
 if not required.issubset(docs):raise ValueError("Historical independent XBRL pilot missing strict source records")
 # Keep both years and originally verified matching mode to avoid cherry-picking.
 cutoff=fold_close("2023-12-29")
 results=[];errors=[];client=Client()
 for symbol,part in docs.groupby("symbol",sort=True):
  curr=part[(part["fy_end"]=="2022-03-31")]
  prev=part[(part["fy_end"]=="2023-03-31") & (part["status"]=="VERIFIED_CORE_REVENUE_PAT")]
  if len(curr)!=1 or len(prev)!=1:continue
  old=curr.iloc[0];new=prev.iloc[0]
  record={"symbol":symbol,"historical_fold":"2023-12-29",
          "reporting_mode":old["mode"],"FY2022_source":old["original_nse_xbrl_url"],
          "FY2023_source":new["original_nse_xbrl_url"],
          "FY2022_original_SHA256":old["original_xml_sha256"],
          "FY2023_original_SHA256":new["original_xml_sha256"],
          "PIT_annual_data_not_promoted":True}
  try:
   if old["mode"]!=new["mode"]:raise ValueError("Source mode mixture")
   for r in (old,new):
    date=pd.to_datetime(r["source_available_utc"],utc=True,errors="coerce")
    if pd.isna(date) or date>cutoff:raise ValueError("Source arrived after historical model 15:30 decision")
    if not str(r["original_nse_xbrl_url"]).startswith("https://nsearchives.nseindia.com/"):
     raise ValueError("Untrusted NSE original document host")
   source=client.get(old["original_nse_xbrl_url"]).content
   sha=hashlib.sha256(source).hexdigest()
   if sha!=old["original_xml_sha256"]:raise ValueError("Changed original FY2022 XML bytes")
   original22,audit=extract_2022_legacy_fourd(source,"2022-03-31",old["mode"],source_is_annual=True)
   original23={"revenue":float(new["revenue"]),"pat":float(new["pat"])}
   if not all(math.isfinite(v) for v in original23.values()):
    raise ValueError("Missing originally verified 2023 core numerics")
   record.update({"FY2022_revenue_raw_INR":original22["revenue"],
                  "FY2022_pat_raw_INR":original22["pat"],
                  "FY2023_revenue_raw_INR":original23["revenue"],
                  "FY2023_pat_raw_INR":original23["pat"],
                  "annual_YOY_revenue_fraction":(
                      original23["revenue"]/original22["revenue"]-1
                      if original22["revenue"]>0 else None),
                  "annual_YOY_pat_fraction":(
                      original23["pat"]/original22["pat"]-1
                      if original22["pat"]>0 else None),
                  "strict_FY2022_provenance":json.dumps(audit,sort_keys=True),
                  "status":"PAIRED_ORIGINAL_2022_2023_NUMERIC_RESEARCH_ONLY"})
   results.append(record)
  except (ValueError,ET.ParseError,TypeError,OverflowError) as err:
   record["status"]="REJECTED_UNVERIFIED_FY2022_SOURCE"
   record["error"]=str(err)[:200]
   errors.append(record)
  print("FY2022 FourD",symbol,record["status"],flush=True)
 output=Path(out);output.mkdir(parents=True,exist_ok=True)
 pd.DataFrame(results).to_csv(output/"strict_FY2022_FY2023_recovered_filing_pairs_RESEARCH_ONLY.csv",index=False)
 pd.DataFrame(errors).to_csv(output/"rejected_FY2022_legacy_XBRL_pairs.csv",index=False)
 report={"scope":"SOURCE_VERIFIED_NSE_LEGACY_FY2022_FOURD_ANNUAL_RESEARCH",
         "original_company_pairs_considered":len(results)+len(errors),
         "FY2022_annual_numeric_legacy_parser_pairs_recovered":len(results),
         "FY2022_source_errors_or_unmatched_contexts":len(errors),
         "one_year_revenue_growth_calculable":sum(pd.notna(z["annual_YOY_revenue_fraction"]) for z in results),
         "one_year_PAT_growth_calculable":sum(pd.notna(z["annual_YOY_pat_fraction"]) for z in results),
         "verified_actual_three_five_seven_year_growth_ready":False,
         "added_prediction_training_folds":0,
         "old_model_predictions_modified":False,
         "all_source_published_before_2023_12_29_close":True,
         "original_FY2022_file_sha256_verified":True,
         "no_missing_numeric_imputation":True,
         "sample_not_full_market_coverage":True}
 (output/"NSE_FY2022_original_FourD_numeric_recovery_summary.json").write_text(json.dumps(report,indent=2))
 print(json.dumps(report,indent=2),flush=True)
 # A sample that fails complete pairs is not evidence of a source improvement.
 if len(results)<4:raise SystemExit("Strict original 2022-2023 fiscal source reconstruction inadequate")
 return report

def main():
 p=argparse.ArgumentParser();p.add_argument("--pilot",required=True);p.add_argument("--out",required=True)
 a=p.parse_args();run(a.pilot,a.out)
if __name__=="__main__":main()
