"""Diagnose why early NSE annual XBRL fails strict FY pair comparability.

Never infer numeric financials from ambiguous contexts; preserve source hash,
period end, unit and dimensional context. Research only, no model changes.
"""
import argparse,json,collections,hashlib,xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd
from v11_4_longterm_fundamentals import Client,TAG_GROUPS,context_map,local,parse_num
from v11_4_strict_annual_numeric_features import parse_units,dimensional_contexts

def context_reasons(raw,fy):
 root=ET.fromstring(raw)
 ctx=context_map(root);units=parse_units(root);dims=dimensional_contexts(root)
 year_end=pd.Timestamp(fy).date()
 out={}
 for metric in ("revenue","pat"):
  found=[]
  for el in root.iter():
   tag=local(el.tag)
   if tag not in TAG_GROUPS[metric]:continue
   ref=el.get("contextRef","")
   c=ctx.get(ref,{})
   n=parse_num(el.text)
   measures=units.get(el.get("unitRef",""),[])
   record={"tag":tag,"context":ref,"end_matches":c.get("end")==year_end,
           "end":str(c.get("end")),"duration":c.get("duration"),
           "dimensions":dims.get(ref),"units":measures,
           "number_parseable":n is not None}
   found.append(record)
  counts=collections.Counter()
  for r in found:
   if not r["end_matches"]:counts["wrong_fiscal_end"]+=1
   elif r["dimensions"]!=0:counts["dimensioned_context"]+=1
   elif "INR" not in r["units"]:counts["non_INR_unit"]+=1
   elif not r["number_parseable"]:counts["non_numeric"]+=1
   elif r["duration"] is None:counts["no_duration"]+=1
   elif 330<=r["duration"]<=400:counts["explicit_annual_duration"]+=1
   elif 60<=r["duration"]<=120:counts["quarter_duration_requires_proven_annual_report_FourD"]+=1
   else:counts["unsupported_duration"]+=1
  out[metric]={"matched_tag_count":len(found),"reasons":dict(counts),
    "candidate_fiscal_end_distribution":dict(collections.Counter(r["end"] for r in found)),
    "examples":[{"tag":r["tag"],"context":r["context"],
                 "duration":r["duration"],"end":r["end"],
                 "end_matches":r["end_matches"],
                 "units":r["units"]} for r in found][:8]}
 return out

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--pilot",required=True)
 p.add_argument("--out",required=True)
 p.add_argument("--max-probes",type=int,default=4)
 a=p.parse_args()
 df=pd.read_csv(a.pilot)
 req={"symbol","fy_end","status","original_nse_xbrl_url","original_xml_sha256"}
 if not req.issubset(df):raise ValueError("Historic pilot source evidence missing")
 tab=(df.groupby(["fy_end","status"],dropna=False).size().reset_index(name="documents"))
 issues=df[df["status"].ne("VERIFIED_CORE_REVENUE_PAT")].copy()
 sample=issues.sort_values(["fy_end","symbol"]).head(a.max_probes)
 client=Client();diagnostics=[]
 for r in sample.itertuples(index=False):
  src=str(r.original_nse_xbrl_url)
  if not src.startswith("https://nsearchives.nseindia.com/") or not src.lower().endswith(".xml"):
   raise ValueError("Invalid original NSE historical XBRL host")
  raw=client.get(src).content
  if hashlib.sha256(raw).hexdigest()!=r.original_xml_sha256:
   raise ValueError("Historic original NSE XBRL document source bytes differ; cannot diagnose revised file")
  obj={"fy_end":str(r.fy_end),"source_sha256":r.original_xml_sha256,
       "context":context_reasons(raw,r.fy_end)}
  # A genuine later annual filing can carry originally filed prior-year
  # comparatives, known by the later file's broadcast date. Analyze only,
  # never promote until comparative concepts/units/contexts are verified.
  later=df[(df["symbol"].eq(r.symbol))&
           (df["fy_end"].eq("2023-03-31"))&
           (df["status"].eq("VERIFIED_CORE_REVENUE_PAT"))]
  if len(later)==1:
   comparison=later.iloc[0]
   cmp_url=str(comparison["original_nse_xbrl_url"])
   if not cmp_url.startswith("https://nsearchives.nseindia.com/"):
    raise ValueError("Comparison filing not original NSE document")
   cmp_raw=client.get(cmp_url).content
   if hashlib.sha256(cmp_raw).hexdigest()!=comparison["original_xml_sha256"]:
    raise ValueError("Later annual comparison source hash mismatch")
   obj["FY2023_filing_comparatives_for_FY2022"]=context_reasons(cmp_raw,"2022-03-31")
  diagnostics.append(obj)
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 tab.to_csv(out/"FY2022_2023_strict_context_coverage.csv",index=False)
 report={"scope":"HISTORICAL_NSE_2022_2023_XBRL_CONTEXT_DIAGNOSTIC_NOT_PREDICTIVE",
         "status_counts":tab.to_dict("records"),
         "selected_incomplete_documents_for_diagnosis":len(diagnostics),
         "sample_context_findings":diagnostics,
         "requires_fiscal_tag_unit_context_review_before_any_promotion":True,
         "original_NSE_source_hash_verified":True,
         "no_derived_missing_values_or_five_seven_year_guesses":True,
         "model_training_and_frozen_predictions_changed":False}
 (out/"older_annual_XBRL_context_diagnostic.json").write_text(json.dumps(report,indent=2))
 print(json.dumps(report,indent=2),flush=True)
if __name__=="__main__":main()
