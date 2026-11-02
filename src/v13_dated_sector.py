"""One-source, dated sector interpretation; never infer future memberships."""
import json,re
from pathlib import Path
import pandas as pd
from v13_information_clock import cutoff
from v13_alternate_sources import sector_kind

def attach(frame,root):
    d=frame.copy();d['available_sector_kind']='UNKNOWN'
    d['available_sector_source_sha256']=None;d['available_sector_observed_utc']=None
    path=Path(root)/'master_data/sector_master.json'
    if not path.exists():return d
    items={r['isin']:r for r in json.loads(path.read_text())}
    for i,row in d.iterrows():
        item=items.get(row['isin'])
        if not item:continue
        sources=[s for s in item.get('sources',[]) if s.get('provider_family')=='NSE'
          and s.get('isin')==row['isin'] and s.get('symbol')==row.symbol
          and s.get('identity_proof') and re.fullmatch('[0-9a-f]{64}',str(s.get('source_sha256','')))]
        valid=[]
        for s in sources:
            seen=pd.Timestamp(s['first_observed_utc'])
            if seen.tzinfo is not None and 0<=(cutoff(row)-seen).total_seconds()<=366*86400:valid.append(s)
        if not valid:continue
        s=max(valid,key=lambda s:s['first_observed_utc']);kind=sector_kind(s.get('taxonomy',[]))
        if kind not in {'BANK','INSURER','OTHER_FINANCIAL','NONFINANCIAL'}:continue
        d.at[i,'available_sector_kind']=kind
        d.at[i,'available_sector_source_sha256']=s['source_sha256']
        d.at[i,'available_sector_observed_utc']=s['first_observed_utc']
    return d
