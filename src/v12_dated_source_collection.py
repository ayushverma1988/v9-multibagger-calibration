"""Public exchange catalogs and BSE histories; preserve original dated bytes.

Outbound cohorts derive from public catalogs/references, never model picks.
Empty/error feeds do not establish absence of events. No model fitting here.
"""
from __future__ import annotations
import argparse, hashlib, json, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode, urlparse, quote
import pandas as pd
import requests
from v12_public_history import bounds

def sha(raw): return hashlib.sha256(raw).hexdigest()

class Store:
    def __init__(self, folder):
        self.folder=Path(folder); self.folder.mkdir(parents=True,exist_ok=True)
        self.lock=threading.Lock(); self.clocks={}; self.local=threading.local()
    def get(self,url,params=None):
        requested=requests.Request('GET',url,params=params).prepare().url
        host=urlparse(requested).hostname
        if urlparse(requested).scheme!='https' or host not in {'www.nseindia.com','nsearchives.nseindia.com','query1.finance.yahoo.com'}:
            raise ValueError('Unapproved source')
        key=sha(requested.encode()); p=self.folder/(key+'.json'); b=self.folder/(key+'.bin')
        if p.exists() and b.exists():
            meta=json.loads(p.read_text());raw=b.read_bytes()
            if meta['requested_url']!=requested or sha(raw)!=meta['sha256']:raise ValueError('Cache integrity failure')
            if meta['status']!=200:raise ValueError('Cached source error')
            return raw,meta
        with self.lock:
            wait=max(0,self.clocks.get(host,0)+.25-time.monotonic())
            if wait:time.sleep(wait)
            self.clocks[host]=time.monotonic()
        if not hasattr(self.local,'session'):
            self.local.session=requests.Session()
            self.local.session.headers.update({'User-Agent':'Mozilla/5.0','Accept':'application/json,text/html,application/xml,*/*','Referer':'https://www.nseindia.com/'})
        r=self.local.session.get(requested,timeout=(8,25));raw=r.content
        if urlparse(r.url).hostname!=host:raise ValueError('Source redirected off host')
        m={'requested_url':requested,'status':r.status_code,'first_retrieved_utc':datetime.now(timezone.utc).isoformat(),'sha256':sha(raw),'bytes':len(raw),'body_file':b.name}
        b.write_bytes(raw);p.write_text(json.dumps(m,indent=2));r.raise_for_status()
        return raw,m

def catalogs(output,asof):
    out=Path(output);out.mkdir(parents=True,exist_ok=True);store=Store(out/'raw')
    low,high=bounds(asof);financial=[];events=[];errors=[];coverage=[]
    for venue in ['equities','sme']:
        for period in ['Annual','Quarterly']:
            try:
                raw,proof=store.get('https://www.nseindia.com/api/corporates-financial-results',{'index':venue,'period':period,'from_date':low.strftime('%d-%m-%Y'),'to_date':high.strftime('%d-%m-%Y')})
                j=json.loads(raw)
                if not isinstance(j,list):raise ValueError('Legacy catalog schema')
                financial.extend({'schema':'legacy','venue_index':venue,'catalog_sha256':proof['sha256'],'record':r} for r in j)
                print('legacy',venue,period,len(j),flush=True)
            except Exception as e:errors.append({'kind':'financial_catalog','venue':venue,'period':period,'error':str(e)[:200]})
        seen=set(); page=1;total=None
        while True:
            try:
                raw,proof=store.get('https://www.nseindia.com/api/integrated-filing-results',{'index':venue,'type':'Integrated Filing- Financials','page':page,'size':1000,'from_date':low.strftime('%d-%m-%Y'),'to_date':high.strftime('%d-%m-%Y')})
                j=json.loads(raw)
                if not isinstance(j,dict) or not isinstance(j.get('data'),list):raise ValueError('Integrated catalog schema')
                rows=j['data'];keys={str(r.get('seq_Id')) for r in rows}
                if keys and keys.issubset(seen):raise ValueError('Pagination repeated previous records')
                if int(j.get('page',-1))!=page-1:raise ValueError('Unexpected zero-based response page')
                financial.extend({'schema':'integrated','venue_index':venue,'catalog_sha256':proof['sha256'],'record':r} for r in rows)
                seen.update(keys);total=int(j['totalCount']);print('integrated',venue,page,len(rows),total,flush=True)
                if page*1000>=total or not rows:break
                page+=1
                if page>100:raise ValueError('Unexpected catalog exceeds pagination bound')
            except Exception as e:errors.append({'kind':'integrated_catalog','venue':venue,'page':page,'error':str(e)[:200]});break
    # At most 29 calendar days per request; public archive includes all issuers.
    jobs=[];d=low
    while d<=high:
        end=min(d+pd.Timedelta(days=28),high)
        for venue in ['equities','sme']:jobs.append((venue,d,end))
        d=end+pd.Timedelta(days=1)
    def read_event(job):
        venue,start,end=job
        try:
            raw,proof=store.get('https://www.nseindia.com/api/corporate-announcements',{'index':venue,'from_date':start.strftime('%d-%m-%Y'),'to_date':end.strftime('%d-%m-%Y')})
            j=json.loads(raw)
            if not isinstance(j,list):raise ValueError('Announcement schema')
            return job,j,proof,None
        except Exception as e:return job,[],None,str(e)[:200]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for job,rows,proof,error in pool.map(read_event,jobs):
            venue,start,end=job
            coverage.append({'venue_index':venue,'start':str(start.date()),'end':str(end.date()),'rows':len(rows),'status':'ERROR' if error else 'CATALOG_RESPONSE','catalog_sha256':proof['sha256'] if proof else None})
            if error:errors.append({'kind':'announcements','venue':venue,'start':str(start.date()),'error':error})
            else:events.extend({'venue_index':venue,'catalog_sha256':proof['sha256'],'record':r} for r in rows)
            print('announcements',venue,str(start.date()),len(rows),error or '',flush=True)
    (out/'financial_catalog.json').write_text(json.dumps(financial))
    (out/'announcement_catalog.json').write_text(json.dumps(events))
    (out/'coverage.json').write_text(json.dumps(coverage,indent=2))
    (out/'errors.json').write_text(json.dumps(errors,indent=2))
    (out/'collection_summary.json').write_text(json.dumps({'financial_catalog_rows':len(financial),'announcement_catalog_rows':len(events),'errors':len(errors),'window':[str(low.date()),str(high.date())],'catalog_completeness_independently_proven':False},indent=2))

def bse_histories(reference,output,asof):
    from v12_bse_transfer import parse_bse_chart
    out=Path(output);out.mkdir(parents=True,exist_ok=True);(out/'securities').mkdir(exist_ok=True)
    ref=pd.read_parquet(reference)
    ref=ref[ref.bse_exclusive_official_interop_reference.eq(True)&ref.bse_quote_verified_current_identity_and_name.eq(True)&ref.same_asof_trade_day.eq(True)].copy()
    if ref.duplicated('isin').any():raise ValueError('Ambiguous public BSE identity')
    ref.to_parquet(out/'public_BSE_exclusive_request_universe.parquet',index=False)
    store=Store(out/'raw');low,high=bounds(asof)
    a=int(low.tz_localize('Asia/Kolkata').timestamp());b=int((high+pd.Timedelta(days=1)).tz_localize('Asia/Kolkata').timestamp())
    errors=[];audit=[]
    def fetch(r):
        sym=str(r['provider_symbol']);url='https://query1.finance.yahoo.com/v8/finance/chart/'+quote(sym,safe='')+'?'+urlencode({'interval':'1d','period1':a,'period2':b,'events':'div,splits'})
        try:
            raw,proof=store.get(url);bars,info=parse_bse_chart(raw,r,asof,proof)
            bars.to_parquet(out/'securities'/(sha(r['isin'].encode())+'.parquet'),index=False)
            return info,None
        except Exception as e:return None,{'isin':r['isin'],'provider_symbol':sym,'error':str(e)[:200]}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i,(info,error) in enumerate(pool.map(fetch,ref.to_dict('records')),1):
            if info:audit.append(info)
            if error:errors.append(error)
            if i%100==0:print('BSE',i,'usable',len(audit),'errors',len(errors),flush=True)
    pd.DataFrame(audit).to_parquet(out/'history_audit.parquet',index=False)
    (out/'history_errors.json').write_text(json.dumps(errors,indent=2))
    (out/'history_summary.json').write_text(json.dumps({'public_requests':len(ref),'usable_histories':len(audit),'errors':len(errors),'valid_bars':sum(r['valid_bars'] for r in audit),'window':[str(low.date()),str(high.date())],'full_current_BSE_master_validated':False,'survivor_bias':True,'independent_corporate_actions_verified':False},indent=2))

def main():
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['catalogs','bse'],required=True);p.add_argument('--output',required=True);p.add_argument('--asof',default='2026-10-09');p.add_argument('--reference');a=p.parse_args()
    if a.kind=='catalogs':catalogs(a.output,a.asof)
    else:bse_histories(a.reference,a.output,a.asof)
if __name__=='__main__':main()
