"""Reviewed same-statement diluted EPS cells, bound to original documents.

Bounding boxes are reviewed extraction instructions, not guessed OCR columns.
Reported EPS never establishes recurring earnings or forward profitability.
"""
import io,json,re,hashlib
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from v12_dated_source_collection import Store

def compact(text):return re.sub(r'\s+','',text).upper()

def validate(review,statement,identity,anchor):
    import pdfplumber
    from v12_5_expanded_facts import parse_extra
    for name,raw in [('statement',statement),('identity',identity),('anchor',anchor)]:
        if hashlib.sha256(raw).hexdigest()!=review[name]['sha256']:raise ValueError('Statement review original hash mismatch')
    r=review
    if r['reporting_mode']!='consolidated' or r.get('metric')!='REPORTED_DILUTED_EPS_CONTINUING':raise ValueError('Unapproved EPS basis')
    record={'symbol':r['symbol'],'catalog_isin':r['isin'],'period_end':r['current_period_end'],'reporting_mode':r['reporting_mode']}
    facts=parse_extra(anchor,record,r['source_asof'],include_ytd=True)
    current=[f['value'] for f in facts if f['metric']=='eps_diluted_continuing' and f['period_kind']=='quarter' and f['period_end']==r['current_period_end'] and f['security_isin']==r['isin']]
    if len(current)!=1:raise ValueError('Original current quarter EPS anchor unavailable')
    with pdfplumber.open(io.BytesIO(identity)) as pdf:
        idtext=' '.join(pdf.pages[i].extract_text() or '' for i in r.get('identity_page_indices',range(len(pdf.pages))))
    with pdfplumber.open(io.BytesIO(statement)) as pdf:
        page=pdf.pages[r['page_index']];full=' '.join(p.extract_text() or '' for p in pdf.pages)
        for text in (idtext,full):
            if compact(r['cin']) not in compact(text) or compact(r['symbol']) not in compact(text):raise ValueError('Original statement/company binding failed')
        if r['isin'] not in compact(idtext):raise ValueError('Original current ISIN identity proof missing')
        if 'CONSOLIDATED' not in compact(page.extract_text() or ''):raise ValueError('Statement page basis missing')
        label=page.crop(r['label_box']).extract_text() or ''
        if 'DILUTED' not in label.upper():raise ValueError('EPS row label mismatch')
        values=[]
        for name,date in [('current',r['current_period_end']),('comparative',r['comparative_period_end'])]:
            cell=r[name+'_cell_box'];header=r[name+'_header_box']
            if not header[0]<=cell[0]<cell[2]<=header[2] or header[3]>=cell[1]:raise ValueError('EPS column/header geometry mismatch')
            head=compact(page.crop(header).extract_text() or '')
            stamp=pd.Timestamp(date)
            if 'THREEMONTHS' not in head or compact(stamp.strftime('%d %B %Y')) not in head:raise ValueError('EPS quarter header mismatch')
            text=(page.crop(cell).extract_text() or '').strip().replace(',','')
            if not re.fullmatch(r'\(?-?\d+(?:\.\d+)?\)?',text):raise ValueError('Ambiguous EPS cell')
            value=float(text.strip('()'))*(-1 if text.startswith('(') else 1)
            if not np.isclose(value,r[name+'_eps'],rtol=0,atol=.00001):raise ValueError('Reviewed EPS cell value mismatch')
            values.append(value)
    if pd.Timestamp(r['current_period_end'])-pd.DateOffset(years=1)!=pd.Timestamp(r['comparative_period_end']):raise ValueError('Not year-ago same quarter')
    if not np.isclose(current[0],values[0],rtol=0,atol=.005):raise ValueError('PDF/XBRL EPS anchor conflict')
    return {'isin':r['isin'],'symbol':r['symbol'],'current_period_end':r['current_period_end'],
      'comparative_period_end':r['comparative_period_end'],'current_reported_diluted_EPS':values[0],
      'comparative_reported_diluted_EPS':values[1],
      'reported_diluted_EPS_yoy_same_document':values[0]/values[1]-1 if values[1]>0 else None,
      'statement_sha256':r['statement']['sha256'],'identity_sha256':r['identity']['sha256'],
      'anchor_sha256':r['anchor']['sha256'],'recurring_EPS_verified':False,
      'source_publication_utc':r['statement']['published_at_utc'],'page_index':r['page_index']}

def collect(root,seed):
    root=Path(root);seed=Path(seed);out=root/'statement_eps';out.mkdir(exist_ok=True)
    if not seed.exists():return {'reviewed_statements':0,'accepted':0,'errors':[]}
    existing_path=out/'accepted.json';existing=json.loads(existing_path.read_text()) if existing_path.exists() else []
    prior={r.get('review_sha256',r['statement_sha256']):r for r in existing};accepted=[];errors=[]
    for r in json.loads(seed.read_text()):
        try:
            data={};proofs={}
            for name in ('statement','identity','anchor'):
                url=r[name]['url']
                if not re.fullmatch(r'https://nsearchives\.nseindia\.com/corporate/[^?]+\.(?:pdf|xml)',url,re.I):raise ValueError('Unapproved statement source')
                data[name],proofs[name]=Store(out/'raw').get(url)
            fact=validate(r,**data)
            review_sha=hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            fact['review_sha256']=review_sha
            now=datetime.now(timezone.utc).isoformat()
            fact['first_reviewed_utc']=prior.get(review_sha,{}).get('first_reviewed_utc',now)
            clocks=[pd.Timestamp(fact['first_reviewed_utc'])]
            for name in data:
                publication=pd.Timestamp(r[name].get('published_at_utc'))
                if pd.notna(publication):
                    if publication.tzinfo is None:raise ValueError('Publication clock lacks timezone')
                    clocks.append(publication)
                clocks.append(pd.Timestamp(proofs[name]['first_retrieved_utc']))
            fact['available_at_utc']=max(clocks).isoformat();accepted.append(fact)
        except Exception as exc:errors.append({'symbol':r.get('symbol'),'error':str(exc)[:200]})
    for f in accepted:prior[f['review_sha256']]=f
    existing_path.write_text(json.dumps(list(prior.values()),indent=2))
    result={'reviewed_statements':len(json.loads(seed.read_text())),'accepted':len(accepted),'errors':errors,
      'reported_EPS_not_recurring':True,'new_review_never_backdated':True}
    (out/'audit.json').write_text(json.dumps(result,indent=2));return result

def enrich(frame,root):
    d=frame.copy();path=Path(root)/'statement_eps/accepted.json'
    if not path.exists():return d
    from v13_information_clock import cutoff
    for i,row in d.iterrows():
        rows=[r for r in json.loads(path.read_text()) if r['isin']==row['isin'] and
          pd.Timestamp(r['available_at_utc'])<=cutoff(row) and 0<=(pd.Timestamp(row.date)-pd.Timestamp(r['current_period_end'])).days<=190]
        if not rows:continue
        r=max(rows,key=lambda x:(x['current_period_end'],x['available_at_utc']))
        d.at[i,'reported_diluted_EPS_yoy_same_document']=r['reported_diluted_EPS_yoy_same_document']
        d.at[i,'statement_eps_source_sha256']=r['statement_sha256']
    return d
