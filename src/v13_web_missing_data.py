"""Find statement pages for API gaps and retain single-source dated numbers.

Google access denial stops Google requests. Direct public statement sites are
separate sources and remain usable. Search snippets never become financial
facts. Downloaded pages need exchange identity, stated units and dated cells.
"""
import argparse
from datetime import datetime,timezone
import gzip,hashlib,json,re,threading,time
from pathlib import Path
from urllib.parse import quote,urlparse,parse_qs,urlencode
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from v13_master_data import dump,display_number
from v13_parallel_collection import bounded_map

def parse_stockanalysis(raw,issuer,proof,section,quarterly=False):
    soup=BeautifulSoup(raw,'html.parser');text=soup.get_text(' ',strip=True)
    symbol=issuer['nse_current_symbol']
    if not re.search(r'\(NSE:\s*'+re.escape(symbol)+r'\)',text):raise ValueError('Exact public NSE ticker identity absent')
    if not re.search(r'(?:[Mm]illions INR|[Ff]inancials in millions INR)',text):raise ValueError('Explicit INR million statement units absent')
    table=soup.find('table')
    if not table:raise ValueError('No public financial table')
    trs=table.find_all('tr');headers=trs[0].find_all(['th','td'])[1:]
    first_label=trs[0].find(['th','td']).get_text(' ',strip=True).lower()
    if quarterly and 'quarter' not in first_label:raise ValueError('Requested quarterly page did not provide quarterly statements')
    if not quarterly and 'year' not in first_label:raise ValueError('Requested annual page did not provide annual statements')
    periods=[h.get('id') if re.fullmatch(r'20\d\d-\d\d-\d\d',h.get('id','')) else None for h in headers]
    if section=='balance-sheet':
        kind='instant';metrics={'Cash & Equivalents':'cash','Accounts Receivable':'receivables_current',
            'Long-Term Receivables':'receivables_noncurrent','Non-Current Receivables':'receivables_noncurrent',
            'Property, Plant & Equipment':'ppe','Total Assets':'assets',
            'Total Current Liabilities':'current_liabilities','Long-Term Debt':'borrowings_noncurrent',
            "Shareholders' Equity":'equity','Common Stock':'paidup_equity_capital'}
    elif section=='cash-flow-statement':
        kind='annual' if not quarterly else 'quarter';metrics={'Operating Cash Flow':'cfo'}
    else:
        kind='quarter' if quarterly else 'annual';metrics={'Revenue':'revenue','Pretax Income':'pbt',
            'Earnings Before Tax':'pbt','Depreciation & Amortization':'depreciation',
            'Net Income':'provider_net_profit','EPS (Diluted)':'provider_diluted_eps'}
    common={'symbol':symbol,'isin':issuer['isin'],'reporting_mode':'consolidated',
        'publisher':'STOCKANALYSIS','provider_family':'SP_GLOBAL' if 'S&P Global' in text else 'STOCKANALYSIS','source_url':proof['requested_url'],
        'source_sha256':proof['sha256'],'first_observed_utc':proof['first_retrieved_utc'],
        'identity_proof':'EXACT_CURRENT_PUBLIC_NSE_TICKER_JOINED_TO_ISIN',
        'reporting_basis_proof':'PROVIDER_GROUP_STATEMENTS_CONVENTION_NOT_INDEPENDENTLY_CONFIRMED',
        'reporting_basis_independently_confirmed':False,'cross_source_validation_required':False}
    rows=[];debt={}
    for tr in trs[2:]:
        cells=tr.find_all(['th','td'],recursive=False)
        if len(cells)<2:continue
        label=' '.join(cells[0].get_text(' ',strip=True).split())
        metric=metrics.get(label)
        if metric is None and label not in {'Short-Term Debt','Current Portion of Long-Term Debt'}:continue
        for end,cell in zip(periods,cells[1:]):
            if end is None:continue
            display=cell.get_text(' ',strip=True);scale=1 if metric=='provider_diluted_eps' else 1_000_000
            number=display_number(display,scale)
            if number is None:continue
            if metric is None:
                debt.setdefault(end,{})[label]=(number,display);continue
            rows.append({**common,'period_kind':kind,'period_end':end,'metric':metric,
                'value_INR':number[0],'rounding_half_width_INR':number[1],
                'display_value':display,'display_unit':'INR_PER_SHARE' if scale==1 else 'INR_MILLION',
                'definition_status':'UNRESOLVED' if metric.startswith('provider_') else 'NAMED_FINANCIAL_LINE',
                'source_row_label':label})
    for end,pair in debt.items():
        if set(pair)=={'Short-Term Debt','Current Portion of Long-Term Debt'}:
            rows.append({**common,'period_kind':'instant','period_end':end,'metric':'borrowings_current',
                'value_INR':sum(x[0][0] for x in pair.values()),'rounding_half_width_INR':sum(x[0][1] for x in pair.values()),
                'display_unit':'INR_MILLION','definition_status':'DERIVED_FROM_SAME_WEB_STATEMENT_LINES',
                'calculation':'Short-Term Debt + Current Portion of Long-Term Debt'})
    if not rows:raise ValueError('No supported dated financial lines on public page')
    return rows

def missing_issuers(root):
    path=Path(root)/'master_data/master_input_sources.parquet'
    if not path.exists():return {}
    cells=pd.read_parquet(path);missing=cells[cells.resolved_value.isna()]
    return {isin:sorted(set(g.column)) for isin,g in missing.groupby('isin')}

def collect(root,budget=3200,seconds=1200,google_budget=10):
    root=Path(root);folder=root/'master_data';needs=missing_issuers(root)
    u=pd.read_parquet(root/'universe.parquet');rows=[r for r in u.to_dict('records') if r['isin'] in needs]
    journal_path=folder/'web_missing_attempts.json';attempts=json.loads(journal_path.read_text()) if journal_path.exists() else {}
    rows=sorted(rows,key=lambda r:(attempts.get(r['isin'],'0000'),-len(needs[r['isin']]),hashlib.sha256(r['isin'].encode()).hexdigest()))[:budget]
    year=str(datetime.now(timezone.utc).year)
    queries=[{'symbol':r['nse_current_symbol'],'isin':r['isin'],'missing_columns':needs[r['isin']],
        'query':r['nse_current_name']+' NSE '+r['nse_current_symbol']+' financial results annual report '+year+' '+', '.join(needs[r['isin']][:4])} for r in rows]
    dump(folder/'missing_web_queries.json',queries)
    google=[]
    for item in queries[:google_budget]:
        url='https://www.google.com/search?'+urlencode({'q':item['query']})
        try:
            response=requests.get(url,headers={'User-Agent':'Mozilla/5.0'},timeout=(8,20))
            if response.status_code in {403,429}:
                google.append({'symbol':item['symbol'],'status':'AUTOMATED_GOOGLE_ACCESS_LIMIT','http_status':response.status_code});break
            response.raise_for_status();soup=BeautifulSoup(response.content,'html.parser');links=[]
            for anchor in soup.select('a[href]'):
                link=anchor['href']
                if link.startswith('/url?'):link=(parse_qs(urlparse(link).query).get('q') or [''])[0]
                if link.startswith('https://') and 'google.' not in urlparse(link).netloc:links.append(link)
            google.append({'symbol':item['symbol'],'status':'SOURCE_LINKS_DISCOVERED','source_urls':sorted(set(links))})
        except Exception as exc:google.append({'symbol':item['symbol'],'status':'SEARCH_UNAVAILABLE','error':str(exc)[:160]})
    dump(folder/'google_missing_discovery.json',google)
    # Source pages discovered through web search have stable financial URLs;
    # the same pages can be collected when a search endpoint is rate limited.
    from v13_alternate_sources import collect as collect_named_web
    named=collect_named_web(root,budget=budget,seconds=min(seconds,900),
        sources=('SCREENER','FINOLOGY'),only_isins=set(needs))
    local=threading.local();lock=threading.Lock();last=[0.]
    def fetch(url):
        with lock:slot=max(time.monotonic(),last[0]+.4);last[0]=slot
        if slot>time.monotonic():time.sleep(slot-time.monotonic())
        if not hasattr(local,'session'):local.session=requests.Session()
        response=local.session.get(url,headers={'User-Agent':'Mozilla/5.0'},timeout=(8,25));response.raise_for_status()
        raw=response.content;proof={'requested_url':url,'final_url':response.url,'sha256':hashlib.sha256(raw).hexdigest(),
            'first_retrieved_utc':datetime.now(timezone.utc).isoformat(),'status':response.status_code}
        dest=folder/'raw'/proof['sha256'];dest.mkdir(parents=True,exist_ok=True)
        (dest/'STOCKANALYSIS.gz').write_bytes(gzip.compress(raw,mtime=0));dump(dest/'receipt.json',proof)
        return raw,proof
    def worker(issuer):
        symbol=issuer['nse_current_symbol'];base='https://stockanalysis.com/quote/nse/'+quote(symbol,safe='')+'/financials/'
        output=[];errors=[]
        for section,quarterly in [('income-statement',True),('income-statement',False),('balance-sheet',False),('cash-flow-statement',False)]:
            url=base+section+'/'+('?p=quarterly' if quarterly else '')
            try:
                raw,proof=fetch(url);output.extend(parse_stockanalysis(raw,issuer,proof,section,quarterly))
            except Exception as exc:
                error=str(exc)[:160];errors.append(error)
                if '403' in error or '429' in error:
                    if output:dump(folder/'provider_inputs'/(issuer['isin']+'_stockanalysis.json'),output)
                    return {'error':error,'rows':len(output)}
        if output:dump(folder/'provider_inputs'/(issuer['isin']+'_stockanalysis.json'),output)
        return {'rows':len(output),'errors':errors}
    accepted=0;facts=0;processed=0;errors=[]
    def safe(issuer):
        try:return worker(issuer)
        except Exception as exc:return {'error':str(exc)[:160]}
    for issuer,result in bounded_map(rows,safe,workers=4,seconds=seconds):
        processed+=1;attempts[issuer['isin']]=datetime.now(timezone.utc).isoformat()
        if result.get('error'):errors.append({'symbol':issuer['nse_current_symbol'],'error':result['error']})
        if result.get('rows'):accepted+=1;facts+=result['rows']
        if processed%100==0:dump(journal_path,attempts);print('Missing web inputs',processed,'accepted',accepted,'facts',facts,flush=True)
    dump(journal_path,attempts)
    summary={'issuers_with_remaining_gaps':len(needs),'queued':len(rows),'processed':processed,
        'accepted_issuers':accepted,'financial_observations':facts,'errors':errors,
        'named_financial_websites':named,'google_discovery':google,'cross_source_validation_required':False}
    dump(folder/'web_missing_collection.json',summary);return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--state',required=True);p.add_argument('--budget',type=int,default=3200);p.add_argument('--seconds',type=int,default=1200);p.add_argument('--google-budget',type=int,default=10);a=p.parse_args();print(json.dumps(collect(a.state,a.budget,a.seconds,a.google_budget)))
