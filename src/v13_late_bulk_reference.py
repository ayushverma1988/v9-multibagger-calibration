"""Capture the whole late exchange file and match one immutable adjusted source. No forecast."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,io,json,re,zipfile
import pandas as pd,numpy as np
from v13_breadth_capture import API,BASE,MAX_BYTES,get
from v13_breadth_primary import rows_in_original,url

def collect(folder,fetch=get,clock=lambda:datetime.now(timezone.utc)):
    folder=Path(folder);src=json.loads((folder/'primary.json').read_text());day=src['source_asof'];raw=(folder/'nse.zip').read_bytes()
    if (src['scope']!='NSE_LATE_REFERENCE_ONLY' or src['source_url']!=url(day) or hashlib.sha256(raw).hexdigest()!=src['sha256'] or
        len(raw)!=src['bytes'] or rows_in_original(raw,day)!=src['nse_rows']):raise ValueError('Mismatched late original')
    with zipfile.ZipFile(io.BytesIO(raw)) as z:o=pd.read_csv(z.open(z.namelist()[0]),low_memory=False)
    rename={'ISIN':'isin','TckrSymb':'symbol','SctySrs':'series','OpnPric':'open','HghPric':'high','LwPric':'low','ClsPric':'close','TtlTradgVol':'volume'}
    f=o[list(rename)].rename(columns=rename)
    f=f[f.series.isin(['EQ','BE','BZ','SM','ST'])&f['isin'].astype(str).str.fullmatch(r'INE[A-Z0-9]{8}[0-9]')].copy()
    keys=['isin','symbol','series'];nums=['open','high','low','close','volume']
    for c in nums:f[c]=pd.to_numeric(f[c],errors='coerce')
    f['valid_primary']=~f.duplicated(keys,keep=False)&np.isfinite(f[nums]).all(axis=1)&f[nums].gt(0).all(axis=1)&f.low.le(f[['open','close']].min(axis=1))&f.high.ge(f[['open','close']].max(axis=1))
    r={'source_asof':day,'scope':'LATE_PUBLIC_REFERENCE_ONLY','official_sha256':src['sha256'],'official_received_utc':src['first_retrieved_utc'],'equity_series_rows':len(f),'valid_primary':int(f.valid_primary.sum()),'formal_scored_rows_added':0,'verified_entries_or_returns_added':0,'corroborated':0}
    try:
        file='prices_adjusted/nse_'+day[:4]+'.parquet';items=[x for x in json.loads(fetch(API,2_000_000)) if x.get('path')==file]
        if len(items)!=1:raise ValueError('No unique source year')
        meta=items[0];rev=meta['lastCommit']['id']
        if not re.fullmatch(r'[0-9a-f]{40}',rev):raise ValueError('Invalid public revision')
        link=BASE+rev+'/'+file+'?download=true';data=fetch(link,MAX_BYTES);seen=clock().isoformat();h=hashlib.sha256(data).hexdigest()
        if h!=meta['lfs']['oid'] or len(data)!=meta['size']:raise ValueError('Publisher hash/length mismatch')
        d=pd.read_parquet(io.BytesIO(data),columns=['date',*keys,*nums,'adj_close','adj_factor_cumulative'])
        d.date=pd.to_datetime(d.date);r['provider']={'url':link,'sha256':h,'bytes':len(data),'received_utc':seen,'publisher_commit_utc':meta['lastCommit']['date'],'through':str(d.date.max().date())}
        d=d[d.date.eq(pd.Timestamp(day))];r['status']='SAME_SESSION_COMPARED' if not d.empty else 'SOURCE_DAY_NOT_IN_ALTERNATE'
        d=d[~d.duplicated(keys,keep=False)]
        f=f.merge(d.drop(columns='date'),on=keys,how='left',suffixes=('','_provider'),validate='m:1')
        ok=f.valid_primary & f.volume.eq(f.volume_provider)
        for c in nums[:-1]:ok&=(f[c]-f[c+'_provider']).abs().le(.011)
        fac=f.adj_factor_cumulative;ok&=np.isfinite(fac)&fac.gt(0)&np.isclose(f.adj_close,f.close_provider*fac,rtol=1e-5,atol=.001)
        f['raw_and_reported_factor_agree']=ok;r['corroborated']=int(ok.sum())
    except (OSError,KeyError,TypeError,ValueError) as e:
        r.update(status='ALTERNATE_UNAVAILABLE',reason=type(e).__name__+': '+str(e)[:150])
    table=folder/'market_reference.parquet';f.to_parquet(table,index=False,compression='zstd');r['reviewed_utc']=clock().isoformat();r['table_sha256']=hashlib.sha256(table.read_bytes()).hexdigest()
    (folder/'market_reference.json').write_text(json.dumps(r,indent=2)+'\n');return r
if __name__=='__main__':print(json.dumps(collect(Path('v13_late_nse_reference'))))
