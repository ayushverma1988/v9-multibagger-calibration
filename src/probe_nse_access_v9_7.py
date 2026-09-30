from __future__ import annotations
import json, requests

URLS = [
    ("homepage","https://www.nseindia.com/"),
    ("announcements_page","https://www.nseindia.com/companies-listing/corporate-filings-announcements"),
    ("api_direct","https://www.nseindia.com/api/corporate-announcements?index=equities&from_date=01-01-2025&to_date=31-01-2025&reqXbrl=false"),
]
UAS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/129.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/128.0 Safari/537.36",
]
rows=[]
for ua in UAS:
    s=requests.Session()
    h={
        "user-agent":ua,
        "accept-language":"en-US,en;q=0.9",
        "accept":"application/json,text/plain,*/*",
        "referer":"https://www.nseindia.com/companies-listing/corporate-filings-announcements",
    }
    for name,url in URLS:
        try:
            r=s.get(url,headers=h,timeout=30,allow_redirects=True)
            rows.append({
                "ua":ua[:30],
                "name":name,
                "status":r.status_code,
                "content_type":r.headers.get("content-type"),
                "bytes":len(r.content),
                "server":r.headers.get("server"),
                "final_url":r.url,
            })
        except Exception as e:
            rows.append({"ua":ua[:30],"name":name,"error":repr(e)})
print(json.dumps(rows,indent=2))
