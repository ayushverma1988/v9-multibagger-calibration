"""Inspect official old/new banking XBRL taxonomies without guessing line-item meanings."""
import hashlib,json,re,time
from collections import Counter
from pathlib import Path
import xml.etree.ElementTree as ET
import requests

FILES={
    "BANKBARODA_2025Q2":"https://nsearchives.nseindia.com/corporate/xbrl/INTEGRATED_FILING_BANKING_1493754_25072025060911_WEB.xml",
    "CANBK_2025Q2":"https://nsearchives.nseindia.com/corporate/xbrl/INTEGRATED_FILING_BANKING_1492265_24072025034753_WEB.xml",
    "INDIANB_2021FY":"https://nsearchives.nseindia.com/corporate/xbrl/BANKING_70725_453443_30052021045500_WEB.xml",
}
OUT=Path("outputs_v11_4_banking_taxonomy")
PATTERN=re.compile(r"income|profit|interest|earning|cashflow|turnover|revenue|advance|deposit|npa|nonperform",re.I)
def local(name):return str(name).split("}")[-1]
def main():
    OUT.mkdir(exist_ok=True)
    ses=requests.Session();ses.headers.update({
        "User-Agent":"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124 Safari/537.36",
        "Referer":"https://www.nseindia.com/","Accept":"application/xml,text/xml,*/*",
    })
    reports={}
    for name,url in FILES.items():
        r=ses.get(url,timeout=25);r.raise_for_status()
        root=ET.fromstring(r.content)
        units=Counter();contexts=Counter();tags=Counter();facts=[]
        for e in root.iter():
            tag=local(e.tag)
            if "contextRef" not in e.attrib:continue
            tags[tag]+=1
            units[e.attrib.get("unitRef","")]+=1
            contexts[e.attrib.get("contextRef","")]+=1
            val=str(e.text or "").strip()
            if PATTERN.search(tag) and re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)",val):
                facts.append({
                    "tag":tag,"context":e.attrib.get("contextRef"),"unit":e.attrib.get("unitRef"),
                    "decimals":e.attrib.get("decimals"),"value":val[:75]
                })
        report={
            "document_sha256":hashlib.sha256(r.content).hexdigest(),
            "url":url,"xml_bytes":len(r.content),"root_tag":local(root.tag),
            "matching_fact_tags":sorted(set(x["tag"] for x in facts)),
            "frequent_contexts":contexts.most_common(12),
            "units":units.most_common(10),
            "matching_facts_first_100":facts[:100],
        }
        reports[name]=report
        print(name,"matching_tags",len(report["matching_fact_tags"]),
              "fact_rows",len(facts),"bytes",len(r.content),flush=True)
        time.sleep(0.3)
    (OUT/"banking_xbrl_taxonomy_examples.json").write_text(json.dumps(reports,indent=2))
    for name,report in reports.items():
        print(name,report["matching_fact_tags"][:90],flush=True)
if __name__=="__main__":main()
