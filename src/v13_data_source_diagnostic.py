"""Public source diagnostic and user-requested earlier-name comparison.

No forecasts, outcomes, or trained heads are changed. Raw replies and clocks
are retained; a readable provider page is not itself confirmation of a fact.
"""
import argparse, base64, hashlib, io, json, os, time, tarfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from v12_public_history import bounds, parse_chart

EXAMPLES = ['MASTEK','SHREEJISPG','TANLA','CUPID','HEXT','THELEELA',
            'LTTS','PERSISTENT','ZENSARTECH','SUNTV','MPHASIS']


def get(url, out, name):
    out.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, headers={'User-Agent':'Mozilla/5.0',
        'Accept':'text/html,application/json,application/xml,*/*'}, timeout=(8,30))
    raw = r.content
    p = out / name
    p.write_bytes(raw)
    proof = {'requested_url':url,'final_url':r.url,'status':r.status_code,
        'sha256':hashlib.sha256(raw).hexdigest(), 'bytes':len(raw),
        'first_retrieved_utc':datetime.now(timezone.utc).isoformat()}
    p.with_suffix(p.suffix+'.receipt.json').write_text(json.dumps(proof,indent=2))
    r.raise_for_status()
    time.sleep(.3)
    return raw, proof


def main(artifact, output):
    import zipfile
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(artifact) as z:
        report=json.loads(z.read('v13_state/latest_run.json'));asof=report['asof']
        scored=json.loads(z.read('v13_state/redevelopment/latest.json'))['records']
        queue=json.loads(z.read('v13_state/early_potential/coverage_queue.json'))['records']
        public={r['symbol']:r['isin'] for r in [*queue,*scored]}
        pd.DataFrame([{'symbol':s,'isin':i} for s,i in sorted(public.items())]).to_parquet(out/'public_universe.parquet',index=False)
        (out/'financial_wide.parquet').write_bytes(z.read('v13_state/financial_wide.parquet'))
        archive=tarfile.open(fileobj=io.BytesIO(z.read('v13_state.tar.gz')))
        manifest=json.load(archive.extractfile('v13_checkpoint_manifest.json'))
    # The sample comes from the unranked public coverage backlog.
    stale=[r for r in queue if r['history_status']=='STORED_BUT_NOT_SCORED']
    stale=sorted(stale,key=lambda r:hashlib.sha256(r['symbol'].encode()).hexdigest())[:16]
    low,high=bounds(asof)
    p1=int(low.tz_localize('Asia/Kolkata').timestamp())
    p2=int((high+pd.Timedelta(days=1)).tz_localize('Asia/Kolkata').timestamp())
    price=[]
    for row in stale:
        s=row['symbol'];base='https://query1.finance.yahoo.com/v8/finance/chart/'+quote(s+'.NS',safe='')
        for variant,query in [('dated',f'interval=1d&period1={p1}&period2={p2}&events=div%2Csplits'),
            ('extended',f'interval=1d&period1={p1}&period2={p2+86400}&events=div%2Csplits'),
            ('range','interval=1d&range=5y&events=div%2Csplits')]:
            item={'symbol':s,'isin':row['isin'],'variant':variant}
            try:
                raw,proof=get(base+'?'+query,out/'price_raw',s+'_'+variant+'.json')
                bars,info=parse_chart(raw,s,row['isin'],asof,proof)
                item.update(latest_valid_bar=info['latest_bar'],fresh=info['latest_bar']==asof,
                    invalid_bars=info['invalid_bars_excluded'],source_sha256=proof['sha256'])
            except Exception as exc:item['error']=str(exc)[:200]
            price.append(item);print(json.dumps(item),flush=True)
    (out/'price_probe.json').write_text(json.dumps(price,indent=2))
    controls=sorted(public,key=lambda s:hashlib.sha256(s.encode()).hexdigest())[:8]
    pages=[]
    for symbol in dict.fromkeys([*EXAMPLES,*controls]):
        if symbol not in public:continue
        targets=[('screener',f'https://www.screener.in/company/{quote(symbol,safe="")}/consolidated/'),
            ('stockanalysis_annual',f'https://stockanalysis.com/quote/nse/{quote(symbol,safe="")}/financials/'),
            ('stockanalysis_quarterly',f'https://stockanalysis.com/quote/nse/{quote(symbol,safe="")}/financials/?p=quarterly')]
        for source,url in targets:
            item={'symbol':symbol,'isin':public[symbol],'source':source}
            try:
                raw,proof=get(url,out/'financial_raw',symbol+'_'+source+'.html')
                soup=BeautifulSoup(raw,'html.parser')
                item.update(proof,title=soup.title.get_text(' ',strip=True) if soup.title else '',
                    tables=len(soup.select('table')))
            except Exception as exc:item['error']=str(exc)[:200]
            pages.append(item);print(json.dumps(item),flush=True)
    (out/'financial_pages.json').write_text(json.dumps(pages,indent=2))
    # Read only the needed immutable checkpoint objects, not the whole archive.
    try:
        from huggingface_hub import hf_hub_download
        from v13_checkpoint import validate_object, ARCHIVE
        subset=['catalog.parquet','universe.parquet']
        items={r['path']:r for r in manifest['files']}
        for name in subset:
            item=items[name];validate_object(item)
            p=hf_hub_download(ARCHIVE,item['object_path'],repo_type='dataset',token=os.environ.get('HF_ARCHIVE_TOKEN'))
            raw=Path(p).read_bytes()
            if hashlib.sha256(raw).hexdigest()!=item['sha256']:raise ValueError('Checkpoint hash mismatch')
            (out/name).write_bytes(raw)
        cat=pd.read_parquet(out/'catalog.parquet');cat=cat[cat.symbol.isin(EXAMPLES)]
        cat.to_parquet(out/'earlier_name_catalog.parquet',index=False)
        print('Earlier-name catalog columns',list(cat),flush=True)
    except Exception as exc:
        (out/'checkpoint_error.json').write_text(json.dumps({'error':str(exc)[:300]}))
    primary='https://www.bseindia.com/xml-data/corpfiling/AttachLive/5d16bb1b-52a4-4933-bc41-b643be9fa83c.pdf'
    try:get(primary,out/'primary','MASTEK_JUN2026.pdf')
    except Exception as exc:(out/'primary_error.json').write_text(json.dumps({'error':str(exc)}))
    summary={'source_asof':asof,'public_universe':len(public),'price_cases':len(price),
        'fresh_by_variant':{v:sum(r.get('fresh',False) for r in price if r['variant']==v) for v in ['dated','extended','range']},
        'financial_pages':len(pages),'financial_pages_readable':sum(r.get('tables',0)>0 for r in pages),
        'outcomes_or_predictions_changed':False}
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--artifact',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();main(a.artifact,a.out)
