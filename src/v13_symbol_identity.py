"""Dated official rename paths, requiring the original and master equity ISIN.

A rename proves identity only, not economic comparability or adjusted returns.
Newly observed mapping evidence is never backdated to its effective date.
"""
import io,json,re
from pathlib import Path
from datetime import datetime,timezone
from xml.etree import ElementTree as ET
import pandas as pd
from v12_dated_source_collection import Store,sha
from v11_4_longterm_fundamentals import local

URL='https://nsearchives.nseindia.com/content/equities/symbolchange.csv'

def parse_changes(raw,proof,previous=()):
    d=pd.read_csv(io.BytesIO(raw),dtype=str).fillna('')
    d.columns=d.columns.str.strip()
    required={'NAME OF THE COMPANY','OLD SYMBOL','NEW SYMBOL','EFFECTIVE DATE'}
    if set(d.columns)!=required:raise ValueError('Official rename schema changed')
    prior={(r['old'],r['new'],r['effective_date']):r for r in previous};rows=[]
    for r in d.to_dict('records'):
        a=r['OLD SYMBOL'].strip().upper();b=r['NEW SYMBOL'].strip().upper()
        date=str(pd.to_datetime(r['EFFECTIVE DATE'].strip(),format='%d-%b-%Y').date())
        if not a or not b:raise ValueError('Invalid official rename')
        if a==b:continue  # Official CSV also contains unchanged-symbol notices.
        old=prior.get((a,b,date))
        rows.append(old or {'old':a,'new':b,'effective_date':date,'source_url':URL,
          'source_sha256':proof['sha256'],'first_seen_utc':proof['first_retrieved_utc']})
    return list({(r['old'],r['new'],r['effective_date']):r for r in rows}.values())

def collect(root):
    folder=Path(root)/'identity';folder.mkdir(exist_ok=True)
    path=folder/'symbol_changes.json';previous=json.loads(path.read_text()) if path.exists() else []
    vintage=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    raw,proof=Store(folder/'raw'/vintage).get(URL)
    rows=parse_changes(raw,proof,previous)
    # Preserve previously observed edges if a current response is thinner.
    merged={(r['old'],r['new'],r['effective_date']):r for r in previous}
    merged.update({(r['old'],r['new'],r['effective_date']):r for r in rows})
    result=list(merged.values());path.write_text(json.dumps(result,indent=2));return result

def rename_path(old,new,rows,asof):
    if old==new:return []
    edges={}
    for r in rows:
        if r['effective_date']<=asof:edges.setdefault(r['old'],[]).append(r)
    paths=[]
    def walk(node,path,seen):
        if len(path)>20:return
        for r in edges.get(node,[]):
            if r['new'] in seen:continue
            if path and r['effective_date']<path[-1]['effective_date']:continue
            p=path+[r]
            if r['new']==new:paths.append(p)
            else:walk(r['new'],p,seen|{r['new']})
    walk(old,[],{old})
    if len(paths)!=1:raise ValueError('No unique dated official rename path')
    return paths[0]

def parse_mapped(raw,record,asof,master_isin,changes):
    from v12_5_expanded_facts import parse_extra
    root=ET.fromstring(raw)
    def scalar(tags):
        values={(e.text or '').strip().upper() for e in root.iter() if local(e.tag) in tags and (e.text or '').strip()}
        if len(values)!=1:raise ValueError('Missing or ambiguous original identity')
        return next(iter(values))
    symbol=scalar({'Symbol','NSESymbol'})
    if symbol==record['symbol']:
        facts=parse_extra(raw,record,asof,include_ytd=True)
        if any(f['security_isin']!=master_isin for f in facts):raise ValueError('Document/current master ISIN conflict')
        return facts
    isin=scalar({'ISIN'})
    if not re.fullmatch(r'INE[A-Z0-9]{8}[0-9]',isin) or isin!=master_isin:raise ValueError('Rename original/master ISIN conflict')
    path=rename_path(symbol,record['symbol'],changes,asof)
    facts=parse_extra(raw,{**record,'symbol':symbol},asof,include_ytd=True)
    available=max([pd.Timestamp(record['available_at_utc'])]+[pd.Timestamp(r['first_seen_utc']) for r in path])
    meta={'original_document_symbol':symbol,'original_publication_utc':record['available_at_utc'],
      'identity_mapping_source_sha256':json.dumps([r['source_sha256'] for r in path]),
      'identity_mapping_first_seen_utc':max(r['first_seen_utc'] for r in path),
      'available_at_utc':available.isoformat(),'identity_source':'DOCUMENT_AND_CURRENT_MASTER_ISIN_OFFICIAL_RENAME'}
    return [{**f,**meta} for f in facts]
