"""Fetch original current classification files with first-observation clocks."""
import argparse,concurrent.futures,hashlib,json,urllib.request
from pathlib import Path
from datetime import datetime,timezone
SOURCES={'amfi.xlsx':'https://portal.amfiindia.com/spages/AverageMarketCapitalization30Jun2026.xlsx',
 'bank.csv':'https://www.niftyindices.com//IndexConstituent/ind_niftybanklist.csv',
 'insurance.csv':'https://www.niftyindices.com/IndexConstituent/ind_niftyInsurance_list.csv'}
def run(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    def get(item):
        name,url=item
        raw=urllib.request.urlopen(url,timeout=30).read();(out/name).write_bytes(raw)
        return dict(name=name,url=url,sha256=hashlib.sha256(raw).hexdigest(),first_observed_utc=datetime.now(timezone.utc).isoformat())
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:r=list(ex.map(get,SOURCES.items()))
    (out/'receipt.json').write_text(json.dumps(r,indent=2));return r
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();run(a.output)
