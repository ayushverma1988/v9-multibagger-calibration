"""Public Yahoo statements, the upstream used by 0xramm's MIT-licensed API.

No paid API, credential, model rank or guessed reporting basis is used. The
wrapper and Yahoo are one provider family. An exact public name/ticker match
binds identity; at least two nonzero original facts bind the statement basis.
"""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
from urllib.parse import quote, urlencode

import numpy as np
import pandas as pd
import requests

from v13_alternate_sources import normal_name
from v13_master_data import cached_primary_index, dump
from v13_parallel_collection import bounded_map

METRICS={
    'TotalRevenue':('revenue','flow'),
    'PretaxIncome':('pbt','flow'),
    'NetIncomeIncludingNoncontrollingInterests':('pat','flow'),
    'DepreciationAndAmortizationInIncomeStatement':('depreciation','flow'),
    'OperatingCashFlow':('cfo','annual'),
    'TotalAssets':('assets','instant'),
    'StockholdersEquity':('equity','instant'),
    'CurrentDebt':('borrowings_current','instant'),
    'LongTermDebt':('borrowings_noncurrent','instant'),
    'CashAndCashEquivalents':('cash','instant'),
    'CurrentLiabilities':('current_liabilities','instant'),
    'NetPPE':('ppe','instant'),
}

def collect_api_quotes(root,budget=80):
    checkout=os.environ.get('INDIAN_STOCK_API_CHECKOUT')
    if not checkout:return {'status':'UPSTREAM_API_CHECKOUT_NOT_CONFIGURED'}
    # Use the reviewed upstream implementation at exactly the audited version.
    actual=subprocess.check_output(['git','-C',checkout,'rev-parse','HEAD'],text=True).strip()
    if actual!='147cd5c92a7888104e0be63eef7ea72a5b61ad3e':raise ValueError('Unreviewed upstream API version')
    root=Path(root).resolve();u=pd.read_parquet(root/'universe.parquet')
    u.to_json(root/'master_data/api_public_universe.json',orient='records',date_format='iso')
    node=shutil.which('node') or os.environ.get('CODEX_PRIMARY_RUNTIME_NODE')
    if not node:raise RuntimeError('Node runtime unavailable for upstream API')
    script=Path(__file__).with_name('v13_indian_stock_api.mjs')
    subprocess.run([node,str(script),checkout,str(root),str(budget)],check=True,timeout=210)
    return json.loads((root/'master_data/indian_api_collection.json').read_text())

def parse(raw, chart_raw, issuer, proof, chart_proof, primary):
    ticker=issuer['nse_current_symbol']+'.NS'
    chart=json.loads(chart_raw);meta=chart['chart']['result'][0]['meta']
    if (meta.get('symbol')!=ticker or meta.get('exchangeName') not in {'NSI','NSE'} or
        meta.get('instrumentType')!='EQUITY' or meta.get('currency')!='INR' or
        normal_name(meta.get('longName',''))!=normal_name(issuer['nse_current_name'])):
        raise ValueError('Yahoo exact current exchange ticker/name/currency mismatch')
    data=json.loads(raw)
    if data.get('timeseries',{}).get('error'):raise ValueError('Yahoo statement response error')
    rows=[]
    for block in data.get('timeseries',{}).get('result') or []:
        if block.get('meta',{}).get('symbol')!=[ticker]:raise ValueError('Statement ticker mismatch')
        types=block.get('meta',{}).get('type') or []
        if len(types)!=1:raise ValueError('Ambiguous Yahoo statement type')
        typ=types[0];prefix='annual' if typ.startswith('annual') else 'quarterly' if typ.startswith('quarterly') else None
        if prefix is None or typ[len(prefix):] not in METRICS:continue
        metric,kind=METRICS[typ[len(prefix):]]
        if kind=='annual' and prefix!='annual':continue
        if kind=='flow':kind='annual' if prefix=='annual' else 'quarter'
        for item in block.get(typ,[]):
            value=(item.get('reportedValue') or {}).get('raw')
            if item.get('currencyCode')!='INR' or value is None or not np.isfinite(value):continue
            if item.get('periodType')!=('12M' if prefix=='annual' else '3M'):continue
            end=item.get('asOfDate')
            if not end or pd.isna(pd.to_datetime(end,errors='coerce')):continue
            # Yahoo raw numbers have no declared display-rounding uncertainty.
            rows.append({'symbol':issuer['nse_current_symbol'],'isin':issuer['isin'],
                'period_kind':kind,'period_end':end,'metric':metric,'value_INR':float(value),
                'rounding_half_width_INR':0.,'display_value':str(value),'display_unit':'INR',
                'provider_family':'YAHOO_FINANCE','publisher':'YAHOO_FINANCE',
                'source_url':proof['requested_url'],'source_sha256':proof['sha256'],
                'first_observed_utc':proof['first_retrieved_utc'],
                'identity_proof':'EXACT_CURRENT_NSE_TICKER_AND_PUBLIC_LEGAL_NAME',
                'identity_source_url':chart_proof['requested_url'],
                'identity_source_sha256':chart_proof['sha256'],
                'definition_status':'NAMED_FINANCIAL_LINE','upstream_field':typ})
    # Yahoo does not declare standalone/consolidated in this endpoint. Require
    # two distinct nonzero named facts matching originals. Amount conflicts
    # remain subject to per-cell reconciliation; they never fill a gap.
    candidates=[]
    for basis in ('standalone','consolidated'):
        matches=[];conflicts=[]
        for r in rows:
            key=(r['isin'],basis,r['period_kind'],r['period_end'],r['metric'])
            p=primary.get(key)
            if p is None or p['value_INR']==0:continue
            if abs(p['value_INR']-r['value_INR'])<=1:
                matches.append({'metric':r['metric'],'period_end':r['period_end'],
                    'original_source_url':p['source_url'],'original_sha256':p['source_sha256']})
            else:conflicts.append(key)
        if len({(r['metric'],r['period_end']) for r in matches})>=2:
            candidates.append((basis,matches))
    if len(candidates)!=1:raise ValueError('Yahoo statement reporting basis not uniquely corroborated')
    basis,anchors=candidates[0]
    for r in rows:r.update(reporting_mode=basis,reporting_basis_proof='UNIQUE_BASIS_MULTIPLE_ORIGINAL_FACT_ANCHORS',basis_anchor_sources=anchors)
    if not rows:raise ValueError('No dated Yahoo financial facts')
    return rows

def collect(root,budget=3200,seconds=1500,asof=None):
    root=Path(root);folder=root/'master_data';folder.mkdir(parents=True,exist_ok=True)
    if asof is None:asof=json.loads((root/'latest_run.json').read_text())['asof']
    wide=pd.read_parquet(root/'financial_wide.parquet')
    clock=datetime.now(timezone.utc).isoformat();primary=cached_primary_index(root,wide,asof,clock)
    attempt_path=folder/'yahoo_attempts.json';attempts=json.loads(attempt_path.read_text()) if attempt_path.exists() else {}
    u=pd.read_parquet(root/'universe.parquet')
    queue=sorted(u.to_dict('records'),key=lambda r:(attempts.get(r['isin'],'0000'),hashlib.sha256(r['isin'].encode()).hexdigest()))[:budget]
    last=[0.];lock=threading.Lock();local=threading.local()
    def fetch(url):
        with lock:slot=max(time.monotonic(),last[0]+.35);last[0]=slot
        if slot>time.monotonic():time.sleep(slot-time.monotonic())
        if not hasattr(local,'session'):local.session=requests.Session()
        response=local.session.get(url,headers={'User-Agent':'Mozilla/5.0'},timeout=(8,20));response.raise_for_status()
        raw=response.content;proof={'requested_url':url,'final_url':response.url,'sha256':hashlib.sha256(raw).hexdigest(),
            'first_retrieved_utc':datetime.now(timezone.utc).isoformat(),'status':response.status_code}
        dest=folder/'raw'/proof['sha256'];dest.mkdir(parents=True,exist_ok=True)
        (dest/'YAHOO_FINANCE.gz').write_bytes(gzip.compress(raw,mtime=0));dump(dest/'receipt.json',proof)
        return raw,proof
    fields=[prefix+m for m,(_,kind) in METRICS.items() for prefix in ('annual','quarterly') if kind!='annual' or prefix=='annual']
    low=int((pd.Timestamp(asof,tz='UTC')-pd.DateOffset(years=3)).timestamp())
    high=int((pd.Timestamp(asof,tz='UTC')+pd.Timedelta(days=1)).timestamp())
    def worker(issuer):
        symbol=issuer['nse_current_symbol'];ticker=symbol+'.NS';target=folder/'provider_inputs'/(issuer['isin']+'_yahoo.json')
        if target.exists():
            old=json.loads(target.read_text())
            if old and old[0]['first_observed_utc'][:10]==clock[:10]:return {'rows':len(old),'cached':True}
        chart,cp=fetch('https://query1.finance.yahoo.com/v8/finance/chart/'+quote(ticker,safe='')+'?interval=1d&range=5d')
        raw,proof=fetch('https://query1.finance.yahoo.com/ws/fundamentals-timeseries/v1/finance/timeseries/'+quote(ticker,safe='')+'?'+urlencode({'symbol':ticker,'type':','.join(fields),'period1':low,'period2':high}))
        rows=parse(raw,chart,issuer,proof,cp,primary);dump(target,rows)
        return {'rows':len(rows),'cached':False}
    def safe_worker(issuer):
        try:return worker(issuer)
        except Exception as exc:return {'error':str(exc)[:180]}
    errors=[];accepted=0;facts=0
    processed=0
    for issuer,value in bounded_map(queue,safe_worker,workers=4,seconds=seconds):
        processed+=1;error=value.get('error')
        attempts[issuer['isin']]=clock
        if error:errors.append({'symbol':issuer['nse_current_symbol'],'error':str(error)[:180]})
        else:accepted+=1;facts+=value['rows']
        if processed%100==0:
            dump(attempt_path,attempts);print('Yahoo processed',processed,'accepted',accepted,'facts',facts,flush=True)
    dump(attempt_path,attempts)
    summary={'public_universe':len(u),'processed':processed,'accepted_issuers':accepted,
        'numeric_observations':facts,'errors':errors,'required_additional_sources':1,
        'wrapper_and_upstream_count_as_one_family':True,'private_model_ranks_used_in_requests':False,
        'stopped_on_host_denial':any('403' in e['error'] or '429' in e['error'] for e in errors)}
    if not summary['stopped_on_host_denial']:
        try:summary['indian_stock_api_quotes']=collect_api_quotes(root)
        except Exception as exc:summary['indian_stock_api_quotes']={'status':'API_COLLECTION_INCOMPLETE','error':str(exc)[:180]}
    dump(folder/'yahoo_collection.json',summary);return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--state',default='v13_state');p.add_argument('--budget',type=int,default=3200);p.add_argument('--seconds',type=int,default=1500);p.add_argument('--asof')
    a=p.parse_args();print(json.dumps(collect(a.state,a.budget,a.seconds,a.asof),default=str))
