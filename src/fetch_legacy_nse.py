from __future__ import annotations
import argparse, io, zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import requests
import pandas as pd

HEADERS={
    'User-Agent':'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/134 Safari/537.36',
    'Accept':'*/*',
    'Referer':'https://www.nseindia.com/'
}

COLMAP={
    'SYMBOL':'symbol','SERIES':'series','OPEN':'open','HIGH':'high','LOW':'low','CLOSE':'close',
    'LAST':'last','PREVCLOSE':'prev_close','TOTTRDQTY':'volume','TOTTRDVAL':'turnover',
    'TIMESTAMP':'date','TOTALTRADES':'trades','ISIN':'isin'
}

def url_for(d:pd.Timestamp):
    mon=d.strftime('%b').upper(); yyyy=d.strftime('%Y'); dd=d.strftime('%d')
    return f'https://nsearchives.nseindia.com/content/historical/EQUITIES/{yyyy}/{mon}/cm{dd}{mon}{yyyy}bhav.csv.zip'

def one_day(d):
    u=url_for(d)
    try:
        r=requests.get(u,headers=HEADERS,timeout=20)
        if r.status_code!=200 or len(r.content)<100: return None
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            names=[n for n in z.namelist() if n.lower().endswith('.csv')]
            if not names: return None
            x=pd.read_csv(z.open(names[0]))
        x.columns=[str(c).strip().upper() for c in x.columns]
        keep=[c for c in COLMAP if c in x.columns]
        x=x[keep].rename(columns=COLMAP)
        if 'date' not in x: x['date']=d
        else: x['date']=pd.to_datetime(x['date'],errors='coerce',dayfirst=True).fillna(d)
        if 'isin' not in x: x['isin']=None
        if 'trades' not in x: x['trades']=None
        for c in ['open','high','low','close','last','prev_close','volume','turnover','trades']:
            if c in x: x[c]=pd.to_numeric(x[c],errors='coerce')
        return x
    except Exception:
        return None

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--year',type=int,required=True); ap.add_argument('--out',required=True); ap.add_argument('--workers',type=int,default=8)
    a=ap.parse_args(); year=a.year
    dates=pd.date_range(f'{year}-01-01',f'{year}-12-31',freq='B')
    frames=[]
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        fut={ex.submit(one_day,d):d for d in dates}
        for k,f in enumerate(as_completed(fut),1):
            x=f.result()
            if x is not None and len(x): frames.append(x)
            if k%50==0: print(year,'checked',k,'days','valid',len(frames),flush=True)
    if not frames: raise RuntimeError(f'No NSE bhavcopy rows fetched for {year}')
    df=pd.concat(frames,ignore_index=True)
    df=df[df['series'].isin(['EQ','BE','BZ'])].copy() if 'series' in df else df
    df=df.sort_values(['date','symbol']).drop_duplicates(['date','symbol','series'])
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); df.to_parquet(out,index=False)
    print('saved',out,'rows',len(df),'symbols',df.symbol.nunique(),'dates',df.date.nunique())

if __name__=='__main__': main()
