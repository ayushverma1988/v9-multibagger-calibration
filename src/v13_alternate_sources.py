"""Public exchange metadata and Accord/Finology statements, with receipts.

Use normal HTTPS requests only. Stop each host after access denial or rate
limiting. Company financials cannot be substituted for adjusted price history.
"""
from datetime import datetime, timezone
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import threading
import time
from urllib.parse import quote

from bs4 import BeautifulSoup
import pandas as pd
import requests

from v13_master_data import dump, display_number, parse_screener_inputs
from v13_parallel_collection import bounded_map

def normal_name(value):
    value=re.sub(r'\s+share price\s*$','',value,flags=re.I)
    value=re.sub(r'\b(?:limited|ltd)\.?\s*$','',value,flags=re.I)
    return re.sub(r'[^A-Z0-9]','',value.upper().replace('&','AND'))

def sector_kind(labels):
    words=' '.join(labels).lower()
    if re.search(r'\bbanks?\b|banking',words):return 'BANK'
    if re.search(r'insurance|insurer',words):return 'INSURER'
    if re.search(r'financial|finance|capital markets|stock.?brok|investment|asset management|leasing',words):return 'OTHER_FINANCIAL'
    macros={'consumer discretionary','consumer durables','industrials','capital goods',
        'information technology','technology','healthcare','health care',
        'fast moving consumer goods','fmcg','energy','oil gas','oil, gas',
        'commodities','chemicals','utilities','telecommunication','communication services',
        'realty','real estate','services','diversified','auto','automobile'}
    return 'NONFINANCIAL' if any(m in words for m in macros) else 'UNKNOWN'

def parse_quote(raw,issuer,receipt):
    data=json.loads(raw)
    rows=[r for r in data.get('equityResponse',[]) if
        r.get('metaData',{}).get('symbol')==issuer['nse_current_symbol'] and
        r.get('metaData',{}).get('isinCode')==issuer['isin'] and
        r.get('metaData',{}).get('series')==issuer['nse_current_series']]
    if len(rows)!=1:raise ValueError('Exchange quote identity/series mismatch')
    row=rows[0];meta=row['metaData'];sec=row.get('secInfo',{});trade=row.get('tradeInfo',{})
    labels=[sec.get(k,'') or '' for k in ('macro','sector','industryInfo','basicIndustry')]
    if not labels[-1]:raise ValueError('Exchange basic industry missing')
    return {'symbol':issuer['nse_current_symbol'],'isin':issuer['isin'],
        'company_name':meta.get('companyName'),'series':meta['series'],
        'taxonomy':labels,'sector_kind':sector_kind(labels),'provider_family':'NSE',
        'basic_industry':sec.get('basicIndustry'),'last_source_update':row.get('lastUpdateTime'),
        'issued_shares':trade.get('issuedSize'),'face_value':trade.get('faceValue'),
        'market_cap_INR':trade.get('totalMarketCap'),
        'source_close':meta.get('closePrice'),'source_volume':trade.get('totalTradedVolume'),
        'source_url':receipt['requested_url'],'source_sha256':receipt['sha256'],
        'first_observed_utc':receipt['first_retrieved_utc'],
        'identity_proof':'EXACT_EXCHANGE_ISIN_SYMBOL_AND_SERIES',
        'price_history_or_split_adjustment_inferred':False}

def parse_finology(raw,issuer,receipt):
    soup=BeautifulSoup(raw,'html.parser')
    h1=soup.find('h1')
    if h1 is None or normal_name(h1.get_text(' ',strip=True))!=normal_name(issuer['nse_current_name']):
        raise ValueError('Finology exact normalized public issuer-name mismatch')
    mode=soup.select_one('#mainContent_hfMode')
    if mode is None or mode.get('value') not in {'S','C'}:raise ValueError('Financial page reporting mode missing')
    basis={'S':'standalone','C':'consolidated'}[mode['value']]
    if 'Data Feed provided by Accord Fintech' not in soup.get_text(' ',strip=True):
        raise ValueError('Financial provider family not identified')
    canonical=soup.select_one('link[rel="canonical"]')
    sector=[a.get_text(' ',strip=True) for a in soup.select('a[href^="/sector/"]')]
    common={'symbol':issuer['nse_current_symbol'],'isin':issuer['isin'],
        'reporting_mode':basis,'publisher':'FINOLOGY','provider_family':'ACCORD_FINTECH',
        'source_url':receipt['requested_url'],'source_sha256':receipt['sha256'],
        'first_observed_utc':receipt['first_retrieved_utc'],
        'identity_proof':'REQUESTED_CURRENT_NSE_SYMBOL_AND_EXACT_UNIQUE_PUBLIC_ISSUER_NAME_FINANCIAL_ONLY',
        'vendor_canonical_url':canonical.get('href') if canonical else None}
    metrics={'Net Sales':'revenue','Profit Before Tax':'pbt','Interest':'finance_cost',
        'Depreciation':'depreciation','Other Income':'other_income','Exceptional Items':'exceptional_items',
        'Operating Cash Flow':'cfo','Investing Cash Flow':'cfi','Financing Cash Flow':'cff',
        'Share Capital':'paidup_equity_capital','Total Reserves':'reserves','Total Assets':'assets',
        'Current liabilities':'current_liabilities','Borrowings':'provider_borrowings_unspecified',
        'Net Profit':'provider_net_profit','Profit After Tax':'provider_net_profit',
        'Adjusted EPS (Rs)':'provider_eps_unspecified','Adjusted EPS (Rs.)':'provider_eps_unspecified'}
    sections=[('mainContent_quarterly','quarter'),('profit','annual'),
        ('balance','instant'),('mainContent_cashflows','annual')]
    result=[]
    for ident,kind in sections:
        section=soup.find(id=ident)
        if section is None:continue
        if not re.search(r'All Figures (?:are )?in (?:Cr\.|Crores)',section.get_text(' ',strip=True)):
            raise ValueError('Statement unit is not explicitly INR crore')
        table=section.find('table')
        if table is None:continue
        trs=table.find_all('tr');headers=[c.get_text(' ',strip=True) for c in trs[0].find_all(['th','td'])][1:]
        for tr in trs[1:]:
            cells=[c.get_text(' ',strip=True) for c in tr.find_all(['th','td'])]
            if not cells or cells[0] not in metrics:continue
            metric=metrics[cells[0]]
            for label,value in zip(headers,cells[1:]):
                if not re.fullmatch(r'[A-Z][a-z]{2} 20\d\d',label):continue
                scale=1 if metric=='provider_eps_unspecified' else 10_000_000
                number=display_number(value,scale)
                if number is None:continue
                end=pd.to_datetime(label,format='%b %Y')+pd.offsets.MonthEnd(0)
                result.append({**common,'period_kind':kind,'period_end':str(end.date()),
                    'metric':metric,'value_INR':number[0],'rounding_half_width_INR':number[1],
                    'display_value':value,'display_unit':'INR_PER_SHARE' if scale==1 else 'INR_CRORE',
                    'definition_status':'UNRESOLVED' if metric.startswith('provider_') else 'NAMED_FINANCIAL_LINE'})
    if not result:raise ValueError('No dated financial statement values on public page')
    metadata={**common,'taxonomy':sector,'sector_kind':sector_kind(sector)}
    return result,metadata

def parse_screener_sector(raw,issuer,receipt):
    # Verify page identity even when a consolidated statement table is empty.
    from v13_financial_crosscheck import parse_screener
    parse_screener(raw,issuer['nse_current_symbol'],issuer['isin'],receipt)
    soup=BeautifulSoup(raw,'html.parser');peers=soup.find(id='peers')
    labels=[a.get_text(' ',strip=True) for a in peers.select('a[href^="/market/IN"]')] if peers else []
    if len(labels)!=4:raise ValueError('Four-level public sector taxonomy missing')
    return {'symbol':issuer['nse_current_symbol'],'isin':issuer['isin'],'taxonomy':labels,
        'sector_kind':sector_kind(labels),'provider_family':'SCREENER',
        'source_url':receipt['requested_url'],'source_sha256':receipt['sha256'],
        'first_observed_utc':receipt['first_retrieved_utc'],'identity_proof':'EXACT_NSE_SYMBOL_JOINED_PUBLIC_ISIN'}

def collect(root,universe_path=None,budget=3200,seconds=1800,sources=('NSE','FINOLOGY','SCREENER')):
    root=Path(root);folder=root/'master_data';folder.mkdir(parents=True,exist_ok=True)
    universe=pd.read_parquet(universe_path or root/'universe.parquet')
    # Reject ambiguous normalized company names for the issuer-level provider.
    names=universe.nse_current_name.map(normal_name);unique_names=set(names[~names.duplicated(keep=False)])
    public_rows=universe.to_dict('records')
    attempts_path=folder/'alternate_attempts.json'
    attempts=json.loads(attempts_path.read_text()) if attempts_path.exists() else {}
    lock=threading.Lock();last={};local=threading.local()
    def get(url,source):
        with lock:
            slot=max(time.monotonic(),last.get(source,0)+.3);last[source]=slot
        if slot>time.monotonic():time.sleep(slot-time.monotonic())
        if not hasattr(local,'session'):local.session=requests.Session()
        response=local.session.get(url,headers={'User-Agent':'Mozilla/5.0'},timeout=(8,25));response.raise_for_status()
        raw=response.content;proof={'requested_url':url,'final_url':response.url,
            'sha256':hashlib.sha256(raw).hexdigest(),'first_retrieved_utc':datetime.now(timezone.utc).isoformat(),
            'status':response.status_code,'bytes':len(raw)}
        target=folder/'raw'/proof['sha256'];target.mkdir(parents=True,exist_ok=True)
        (target/(source+'.gz')).write_bytes(gzip.compress(raw,mtime=0));dump(target/'receipt.json',proof)
        return raw,proof
    summary={}
    for source in sources:
        rows=sorted(public_rows,key=lambda r:(attempts.get(source+'|'+r['isin'],'0000'),hashlib.sha256(r['isin'].encode()).hexdigest()))[:budget]
        accepted=0;errors=[];done=0
        def worker(r):
            isin=r['isin'];symbol=r['nse_current_symbol']
            output=folder/('sector_'+source.lower())/(isin+'.json')
            values=folder/'provider_inputs'/(isin+'_'+source.lower()+'.json')
            # Successfully parsed source vintages are reusable on the same UTC
            # day; empty statement pages are never cached as completed data.
            if output.exists() and (source=='NSE' or values.exists()):
                old=json.loads(output.read_text())
                if old['first_observed_utc'][:10]==datetime.now(timezone.utc).date().isoformat():return {'cached':True}
            try:
                if source=='NSE':
                    url='https://www.nseindia.com/api/NextApi/apiClient/GetQuoteApi?functionName=getSymbolData&marketType=N&series='+quote(r['nse_current_series'])+'&symbol='+quote(symbol,safe='')
                    raw,proof=get(url,source);metadata=parse_quote(raw,r,proof)
                elif source=='FINOLOGY':
                    if normal_name(r['nse_current_name']) not in unique_names:raise ValueError('Ambiguous public issuer name')
                    url='https://ticker.finology.in/company/'+quote(symbol,safe='')
                    raw,proof=get(url,source);observations,metadata=parse_finology(raw,r,proof)
                    dump(values,observations)
                else:
                    url='https://www.screener.in/company/'+quote(symbol,safe='')+'/consolidated/'
                    raw,proof=get(url,source);metadata=parse_screener_sector(raw,r,proof)
                    observations=parse_screener_inputs(raw,symbol,isin,proof)
                    if not observations:
                        url='https://www.screener.in/company/'+quote(symbol,safe='')+'/'
                        raw,proof=get(url,source);observations=parse_screener_inputs(raw,symbol,isin,proof)
                    if not observations:raise ValueError('No dated financial values in either public reporting mode')
                    dump(values,observations)
                dump(output,metadata);return {'accepted':True}
            except Exception as exc:return {'error':str(exc)[:180],'symbol':symbol}
        for r,result in bounded_map(rows,worker,workers=4,seconds=seconds):
            done+=1;accepted+=int(result.get('accepted',False))
            attempts[source+'|'+r['isin']]=datetime.now(timezone.utc).isoformat()
            if result.get('error'):errors.append({'symbol':r['nse_current_symbol'],'error':result['error']})
            if done%100==0:
                dump(attempts_path,attempts)
                print('Alternate master source',source,'processed',done,'new accepted',accepted,flush=True)
        summary[source]={'processed':done,'new_accepted':accepted,'errors':errors,'public_universe':len(universe)}
        dump(attempts_path,attempts)
        dump(folder/'alternate_collection.json',summary)
    return summary

def sector_master(root):
    folder=Path(root)/'master_data';by_isin={}
    for source in ('nse','finology','screener'):
        for p in (folder/('sector_'+source)).glob('*.json'):
            r=json.loads(p.read_text());by_isin.setdefault(r['isin'],[]).append(r)
    rows=[]
    for isin,sources in by_isin.items():
        by_family={r['provider_family']:r for r in sources}
        kinds={r['sector_kind'] for r in sources if r['sector_kind']!='UNKNOWN'}
        primary=by_family.get('NSE');agreed=bool(primary and len(by_family)>=3 and len(kinds)==1 and all(r['sector_kind']!='UNKNOWN' for r in sources))
        rows.append({'isin':isin,'symbol':sources[0]['symbol'],
            'sector_kind':next(iter(kinds)) if len(kinds)==1 else 'UNKNOWN',
            'industry':('Banks' if kinds=={'BANK'} else 'Insurance' if kinds=={'INSURER'} else
                primary['basic_industry'] if primary and kinds=={'NONFINANCIAL'} else 'Financial Services' if kinds=={'OTHER_FINANCIAL'} else None),
            'primary_plus_two_additional_sources_agree':agreed,
            'status':'THREE_SOURCE_CLASSIFICATION_AGREEMENT' if agreed else 'CLASSIFICATION_REVIEW_PENDING',
            'available_at_utc':max(r['first_observed_utc'] for r in sources),'sources':sources})
    dump(folder/'sector_master.json',rows)
    return {'observed_issuers':len(rows),'three_source_agreement':sum(r['primary_plus_two_additional_sources_agree'] for r in rows)}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--state',required=True);p.add_argument('--universe');p.add_argument('--budget',type=int,default=3200);p.add_argument('--seconds',type=int,default=1800)
    a=p.parse_args();collect(a.state,a.universe,a.budget,a.seconds);print('SECTOR_MASTER',sector_master(a.state),flush=True)
