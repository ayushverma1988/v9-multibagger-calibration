"""Compare free BSE history sources using fixed public source controls.

Provider code/name agreement is not an official code/ISIN certification.
StockAnalysis displays delayed S&P-sourced prices, not guaranteed live ticks.
"""
from __future__ import annotations
import argparse,json,re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd
from bs4 import BeautifulSoup
from v11_4_free_source_fallbacks import PublicEvidenceStore,finite_number

def collect(mapping_path,history_path,output):
    out=Path(output);out.mkdir(parents=True,exist_ok=True);store=PublicEvidenceStore(out/'raw')
    mapping=json.loads(Path(mapping_path).read_text());history=Path(history_path)/'securities';hist={}
    for p in history.glob('*.parquet'):
        d=pd.read_parquet(p);hist[d['isin'].iloc[0]]=d
    rows=[];errors=[];audits=[]
    def fetch(ref):
        code=ref['bse_provider_code'];url=f'https://stockanalysis.com/quote/bom/{code}/history/'
        try:
            raw,proof=store.get(url);soup=BeautifulSoup(raw,'html.parser');text=soup.get_text(' ',strip=True)
            norm=lambda s:re.sub('[^a-z0-9]','',s.lower())
            if ref['provider_directory_name'].lower() not in text.lower() or 'INR' not in text or f'BOM:{code}' not in text:raise ValueError('Secondary page issuer/venue/currency not matched')
            found=[]
            for table in soup.select('table'):
                heads=[c.get_text(' ',strip=True) for c in table.select('thead th')]
                if not all(c in heads for c in ['Date','Open','High','Low','Close']):continue
                for tr in table.select('tbody tr'):
                    cells=[c.get_text(' ',strip=True) for c in tr.select('td')]
                    if len(cells)!=len(heads):continue
                    r=dict(zip(heads,cells));date=pd.to_datetime(r['Date'],format='%b %d, %Y',errors='coerce');close=finite_number(r['Close'])
                    if pd.notna(date) and close is not None and close>0:found.append({'date':date,'secondary_close':close})
            if not found:raise ValueError('No visible finite historical rows')
            d=pd.DataFrame(found).drop_duplicates('date')
            if ref['isin'] not in hist:raise ValueError('No four-year Yahoo control source')
            y=hist[ref['isin']];merged=d.merge(y[['date','close']],on='date',how='inner');merged=merged[merged.date.le(pd.Timestamp('2026-10-09'))]
            for _,r in merged.iterrows():
                difference=abs(r.secondary_close-r.close)
                rows.append({'isin':ref['isin'],'provider_symbol':ref['provider_symbol'],'bse_provider_code':code,'date':str(r.date.date()),'Yahoo_close':r.close,'secondary_close':r.secondary_close,'absolute_delta_INR':difference,'within_0_011_INR':bool(difference<=.011),'source_sha256':proof['sha256'],'first_retrieved_utc':proof['first_retrieved_utc'],'official_code_ISIN_binding_verified':False})
            return {'isin':ref['isin'],'visible_history_rows':len(d),'matched_dates':len(merged),'source_url':url,'source_sha256':proof['sha256'],'provider_directory_name_matched':True,'official_code_ISIN_binding_verified':False},None
        except Exception as e:return None,{'isin':ref['isin'],'code':code,'source_url':url,'error':str(e)[:200]}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for audit,error in pool.map(fetch,mapping):
            if audit:audits.append(audit)
            if error:errors.append(error)
    pd.DataFrame(rows).to_parquet(out/'BSE_secondary_price_comparisons.parquet',index=False)
    (out/'source_audits.json').write_text(json.dumps(audits,indent=2));(out/'errors.json').write_text(json.dumps(errors,indent=2))
    summary={'public_controls_requested':len(mapping),'pages_usable':len(audits),'errors':len(errors),'matched_stock_dates':len(rows),'agreements_within_0_011_INR':sum(r['within_0_011_INR'] for r in rows),'disagreements':sum(not r['within_0_011_INR'] for r in rows),'official_code_ISIN_binding_verified':False,'realtime_latency_guaranteed':False,'production_approved':False}
    (out/'summary.json').write_text(json.dumps(summary,indent=2));return summary

def main():
    p=argparse.ArgumentParser();p.add_argument('--mapping',required=True);p.add_argument('--history',required=True);p.add_argument('--output',required=True);a=p.parse_args();print(json.dumps(collect(a.mapping,a.history,a.output),indent=2))
if __name__=='__main__':main()
