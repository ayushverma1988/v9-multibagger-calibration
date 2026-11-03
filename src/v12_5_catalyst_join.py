"""Join reviewed, source-bound disclosures; missing evidence stays unknown.

Contract scale is neither recognised revenue nor forecast profit. Sampled
document counts are observed evidence, never a complete event inventory.
"""
import argparse, hashlib, json, re
from pathlib import Path
import numpy as np
import pandas as pd
from v12_5_features import CATALYST, source_wide, pick, number, ratio

def eligible(records, isin, cutoff):
    return [r for r in records if r.get('issuer_identity_read') is True
            and re.fullmatch(r'IN[A-Z0-9]{10}',str(r.get('isin','')))
            and r['isin']==isin and pd.Timestamp(r['available_at_utc'])<=cutoff]

def observed(records, isin, date, annual_revenue=np.nan):
    cutoff=(pd.Timestamp(date).tz_localize('Asia/Kolkata')+pd.Timedelta(hours=15,minutes=30)).tz_convert('UTC')
    known=eligible(records,isin,cutoff)
    recent=[r for r in known if cutoff-pd.Timestamp(r['available_at_utc'])<=pd.Timedelta(days=90)]
    f={c:np.nan for c in CATALYST}; used=[]
    commissioned=[r for r in recent if r.get('commissioned') is True]
    if commissioned:
        f['commissioned_document_count90']=len({r['source_sha256'] for r in commissioned});used+=commissioned
    utilisation=[r for r in recent if r.get('utilisation_pct') is not None]
    if utilisation:
        r=max(utilisation,key=lambda r:r['available_at_utc'])
        f['utilisation_disclosed_pct']=r['utilisation_pct'];used.append(r)
    orders={}
    for r in recent:
        if r.get('firm_order') is not True or not r.get('order_due'):continue
        due=pd.Timestamp(r['order_due']); now=pd.Timestamp(date)
        if now<=due<=now+pd.DateOffset(months=12):orders[r['order_id']]=r
    if orders:
        f['disclosed_order_value_due12_to_sales']=ratio(sum(r['order_value_INR_as_disclosed'] for r in orders.values()),annual_revenue)
        used+=list(orders.values())
    return f,used

def run(matrix,ledger_path,documents,base,extra,output):
    records=json.loads(Path(ledger_path).read_text());root=Path(documents)
    evidence={r['source_sha256']:r for r in json.loads((root/'document_evidence.json').read_text())}
    # A reviewed record cannot silently change identity, URL or source clock.
    for r in records:
        e=evidence[r['source_sha256']]
        for k in ['isin','symbol','available_at_utc','attachment_url','issuer_identity_read']:
            if r[k]!=e[k]:raise ValueError('Review/source mismatch: '+k)
    raw_hashes={hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'raw').glob('*.bin')}
    if not all(r['source_sha256'] in raw_hashes for r in records):raise ValueError('Missing original document bytes')
    d=pd.read_parquet(matrix);w=source_wide(pd.read_parquet(base),pd.read_parquet(extra));links=[]
    d=d.drop(columns=['verified_order_revenue12_to_sales'],errors='ignore')
    for c in CATALYST:d[c]=np.nan
    for i,r in d[d['isin'].isin([x['isin'] for x in records if x.get('isin')])].iterrows():
        date=pd.Timestamp(r.date);cutoff=(date.tz_localize('Asia/Kolkata')+pd.Timedelta(hours=15,minutes=30)).tz_convert('UTC')
        known=w[w.security_isin.eq(r['isin'])&w.available.le(cutoff)&w.period.le(date)]
        basis='consolidated' if known.reporting_mode.eq('consolidated').any() else 'standalone'
        annual=pick(known[known.reporting_mode.eq(basis)],'annual')
        rev=number(annual,'revenue') if annual and (date-annual['period']).days<=450 else np.nan
        f,used=observed(records,r['isin'],date,rev)
        for c,v in f.items():d.at[i,c]=v
        for z in used:links.append({'date':str(date.date()),'isin':r['isin'],'source_sha256':z['source_sha256'],'available_at_utc':z['available_at_utc']})
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    d.to_parquet(out/'expanded_training_inputs.parquet',index=False)
    (out/'source_links.json').write_text(json.dumps(links,indent=2))
    summary={'reviewed_documents':len(records),'features_nonmissing':d[list(CATALYST)].notna().sum().to_dict(),'sampling_is_complete':False,'verified_future_profit_rows':0}
    (out/'coverage.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ['matrix','ledger','documents','base','extra','output']:p.add_argument('--'+k,required=True)
    a=p.parse_args();run(a.matrix,a.ledger,a.documents,a.base,a.extra,a.output)
