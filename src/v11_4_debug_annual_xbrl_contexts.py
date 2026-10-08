"""Explain why strict historical annual XBRL extractor rejected source monetary facts."""
import argparse,json,xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd
from v11_4_longterm_fundamentals import Client,TAG_GROUPS,context_map,local
from v11_4_strict_annual_numeric_features import parse_units,dimensional_contexts

def main():
 p=argparse.ArgumentParser();p.add_argument("--catalog",required=True);p.add_argument("--output",required=True);a=p.parse_args()
 out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 src=pd.read_csv(a.catalog,dtype=str).fillna("")
 cutoff=pd.Timestamp("2024-12-31T10:00:00Z")
 src["_fy"]=pd.to_datetime(src["fy_end"],utc=True,errors="coerce")
 src["_pub"]=pd.to_datetime(src["available_at_utc"],utc=True,errors="coerce")
 selected=[]
 for symbol in ("UGARSUGAR","RKFORGE","TPLPLASTEH","KAMDHENU"):
  z=src[(src["symbol"]==symbol)&(src["_pub"]<=cutoff)&(src["xbrl_url"].str.startswith("https://nsearchives.nseindia.com/"))].sort_values("_fy",ascending=False)
  if len(z): selected.append(z.iloc[0])
 client=Client();reports=[]
 for row in selected:
  root=ET.fromstring(client.get(row["xbrl_url"]).content)
  ctx=context_map(root);dims=dimensional_contexts(root);units=parse_units(root)
  target=row["_fy"].date()
  matching=set(TAG_GROUPS["revenue"]+TAG_GROUPS["pat"])
  candidate=[]
  for e in root.iter():
   tag=local(e.tag)
   if tag not in matching:continue
   ref=e.get("contextRef","")
   c=ctx.get(ref,{})
   candidate.append({"tag":tag,"value":str(e.text or "")[:40],
      "fy_requested":str(target),"ctx_ref":ref,
      "end":str(c.get("end")),"duration":c.get("duration"),
      "dims":dims.get(ref),"unit_ref":e.get("unitRef"),
      "unit_measures":units.get(e.get("unitRef",""),[])})
  exact=[r for r in candidate if r["end"]==str(target)]
  report={"symbol":row["symbol"],"fy_end":str(target),"source_url":row["xbrl_url"],
          "total_matching_tag_facts":len(candidate),
          "exact_period_end_matches":len(exact),
          "units_defined":list(units.items())[:20],
          "candidate_facts_top":candidate[:20],
          "exact_period_facts_top":exact[:20],
          "top_context_ids":list(ctx.items())[:8]}
  reports.append(report)
  print(json.dumps({k:v for k,v in report.items() if k not in ("candidate_facts_top","exact_period_facts_top","top_context_ids")},default=str),flush=True)
  print("SAMPLES",json.dumps(report["exact_period_facts_top"][:5],default=str),flush=True)
 (out/"annual_xbrl_strict_rejection_diagnostic.json").write_text(json.dumps(reports,indent=2,default=str))
if __name__=="__main__":main()
