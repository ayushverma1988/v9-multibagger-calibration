"""Read-only original FY2022 NSE XML structural source audit. Never infer financial values."""
import argparse,hashlib,json,re,collections,xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd
from v11_4_longterm_fundamentals import Client,local
DATE_TOKEN=re.compile(r"^(?:\d{2}[-/]\d{2}[-/]\d{4}|\d{4}[-/]\d{2}[-/]\d{2}|\d{2}[- ][A-Za-z]{3}[- ]\d{4})$")
INTEREST=re.compile(r"date|period|year|quarter|end|start|duration|context|scenario|natureofreport|consolidat|standalone|financialresult|reporttype",re.I)
def structural_provenance(xml):
 root=ET.fromstring(xml)
 names=collections.Counter(local(e.tag) for e in root.iter())
 dates=[];related=[]
 for e in root.iter():
  tag=local(e.tag);val=(e.text or "").strip()
  if not val or len(val)>140:continue
  if DATE_TOKEN.fullmatch(val) or (INTEREST.search(tag) and len(val)<90):
   record={"element":tag,"value":val[:100],"contextRef":e.attrib.get("contextRef","")}
   if DATE_TOKEN.fullmatch(val):dates.append(record)
   else:related.append(record)
 return {"xml_root_element":local(root.tag),"XBRL_context_element_count":names.get("context",0),
  "numeric_legacy_OneD_mention":sum(e.attrib.get("contextRef")=="OneD" for e in root.iter()),
  "numeric_legacy_FourD_mention":sum(e.attrib.get("contextRef")=="FourD" for e in root.iter()),
  "distinct_XML_element_names":len(names),"top_45_element_names":names.most_common(45),
  "exact_date_labeled_metadata":dates[:70],"other_period_report_metadata":related[:70]}
def main():
 p=argparse.ArgumentParser();p.add_argument("--pilot",required=True);p.add_argument("--out",required=True)
 p.add_argument("--sample",type=int,default=4);a=p.parse_args()
 if not 1<=a.sample<=10:raise ValueError("Bound original FY2022 sample 1-10")
 facts=pd.read_csv(a.pilot,dtype=str)
 needed={"symbol","fy_end","status","original_nse_xbrl_url","original_xml_sha256"}
 if not needed.issubset(facts):raise ValueError("Original source fields missing")
 subset=facts[(facts["fy_end"]=="2022-03-31")&(facts["status"]=="INCOMPLETE_OR_UNSUPPORTED_CONTEXT")].sort_values("symbol").head(a.sample)
 if subset.empty:raise ValueError("Missing previously verified FY2022 file sample")
 cli=Client();documents=[]
 for rec in subset.itertuples(index=False):
  url=str(rec.original_nse_xbrl_url)
  if not re.fullmatch(r"https://nsearchives\.nseindia\.com/[a-zA-Z0-9/_\-.]+\.xml",url,re.I):
   raise ValueError("Untrusted original financial URL")
  raw=cli.get(url).content
  dig=hashlib.sha256(raw).hexdigest()
  if dig!=rec.original_xml_sha256:raise ValueError("Original XML bytes/hash mismatch")
  inspected=structural_provenance(raw)
  documents.append({"symbol":rec.symbol,"expected_fiscal_end":"2022-03-31",
   "original_xml_sha256":dig,"original_bytes":len(raw),"structure":inspected,
   "document_confirms_exact_FY2022_annual_numeric_context":False})
  print("FY2022",rec.symbol,"root",inspected["xml_root_element"],"contexts",inspected["XBRL_context_element_count"],"dates",len(inspected["exact_date_labeled_metadata"]),flush=True)
 r={"scope":"LEGACY_NSE_FY2022_XML_STRUCTURE_AND_METADATA_ONLY",
  "documents_checked":len(documents),"all_raw_source_SHA256_matches":True,
  "FY2022_financial_source_values_recovered_and_promoted":False,
  "original_model_predictions_untouched":True,"legacy_file_details":documents}
 dest=Path(a.out);dest.mkdir(parents=True,exist_ok=True)
 (dest/"FY2022_original_xml_structural_evidence.json").write_text(json.dumps(r,indent=2))
 print(json.dumps(r,indent=2),flush=True)
if __name__=="__main__":main()
