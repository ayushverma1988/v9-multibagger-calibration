"""Original-source execution evidence, review backlog and conservative joins.

Automatic text extraction produces review candidates, never certified future
revenue/profit. Reviewed facts require matching bytes, issuer, excerpt and clock.
"""
from pathlib import Path
import json,re,time
import numpy as np
import pandas as pd
from v12_dated_source_collection import Store,sha
from v11_4_primary_document_catalysts import extract_pdf,classify_document
from v12_3_catalyst_evidence import enrich
from datetime import datetime,timezone
from urllib.parse import urlparse

FIELDS=('commissioned','capacity_units_per_year','utilisation_pct','firm_order','order_id',
 'order_value_INR','order_due','expected_margin_pct','funding_sufficient','approval_completed','material_dilution_pct')
FIELDS=FIELDS+('capacity_quantity','capacity_unit','execution_stage','order_duration_months','order_start_date')

def seed_primary_reviews(root,seed_path):
    """Install reviewed public original documents, hash checked on each source.
    First observation is recorded at retrieval; printed dates are not clocks.
    """
    import requests
    root=Path(root);seed_path=Path(seed_path)
    if not seed_path.exists():return {'installed':0,'errors':[]}
    out=root/'catalyst_evidence';out.mkdir(exist_ok=True);rawdir=out/'raw';rawdir.mkdir(exist_ok=True)
    dp=out/'document_evidence.json';rp=out/'reviewed_execution.json'
    docs=json.loads(dp.read_text()) if dp.exists() else [];reviews=json.loads(rp.read_text()) if rp.exists() else []
    count=0;errors=[]
    for r in json.loads(seed_path.read_text()):
        if r['source_sha256'] in {z['source_sha256'] for z in reviews}:continue
        try:
            u=urlparse(r['attachment_url'])
            if u.scheme!='https' or u.hostname not in {'www.aeroflexindia.com','nsearchives.nseindia.com'}:raise ValueError('Unapproved primary company source')
            response=requests.get(r['attachment_url'],timeout=(8,25));response.raise_for_status()
            if urlparse(response.url).hostname!=u.hostname or sha(response.content)!=r['source_sha256']:raise ValueError('Reviewed original source changed')
            text=extract_pdf(response.content)
            if r['issuer_name'].casefold() not in re.sub(r'\s+',' ',text).casefold() or not re.search(r'(?:Trading\s+|NSE\s+)?Symbol\s*:\s*'+re.escape(r['symbol'])+r'\b',text,re.I):raise ValueError('Original issuer identity mismatch')
            now=datetime.now(timezone.utc).isoformat()
            existing=next((z for z in docs if z['source_sha256']==r['source_sha256'] and z.get('attachment_url')==r['attachment_url'] and z.get('isin')==r['isin']),None)
            published=r.get('published_at_utc')
            if published and (not existing or pd.Timestamp(existing['available_at_utc'])!=pd.Timestamp(published)):
                raise ValueError('Publication clock must match the saved original exchange catalogue')
            e={**(existing or {}),**{k:r[k] for k in ('isin','symbol','attachment_url','source_sha256')}}
            e.update(issuer_identity_read=True,text_sha256=sha(text.encode()),first_retrieved_utc=existing.get('first_retrieved_utc',now) if existing else now,
              first_reviewed_utc=now,available_at_utc=pd.Timestamp(published).isoformat() if published else now,
              document_date=r['document_date'],quantified_evidence=existing.get('quantified_evidence',[]) if existing else [],review_status='REVIEWED_ORIGINAL_SOURCE_FIELDS')
            (rawdir/(r['source_sha256']+'.bin')).write_bytes(response.content);(out/(r['source_sha256']+'.txt')).write_text(text)
            reviewed(r,e,text)
            if existing:docs=[z for z in docs if z is not existing]
            docs.append(e);reviews.append(r);count+=1
        except Exception as exc:errors.append({'source_url':r['attachment_url'],'error':str(exc)[:200]})
    dp.write_text(json.dumps(docs,indent=2));rp.write_text(json.dumps(reviews,indent=2));return {'installed':count,'errors':errors}

def reviewed(record,evidence,text):
    for k in ('isin','symbol','source_sha256','attachment_url'):
        if record.get(k)!=evidence.get(k):raise ValueError('Catalyst source mismatch: '+k)
    if evidence.get('issuer_identity_read') is not True:raise ValueError('Issuer not matched in original document')
    if sha(text.encode())!=evidence['text_sha256']:raise ValueError('Extracted text changed')
    for field in FIELDS:
        if record.get(field) is not None:
            excerpt=record.get('field_excerpts',{}).get(field)
            if not excerpt or re.sub(r'\s+',' ',excerpt).strip() not in re.sub(r'\s+',' ',text):raise ValueError('Unbound quantified field: '+field)
            if field=='commissioned' and record[field] is True:
                if not re.search(r'commissioned|commenced|started|operational|commercial production',excerpt,re.I) or re.search(r'\b(will|plan|plans|planned|proposed|expected|target|intend)\b',excerpt,re.I):
                    raise ValueError('Announced intent is not completed commissioning')
            if field=='approval_completed' and record[field] is True and (not re.search(r'granted|received|obtained|approved',excerpt,re.I) or re.search(r'pending|subject to|expected',excerpt,re.I)):
                raise ValueError('Approval has not been completed')
            if field=='funding_sufficient' and record[field] is True and (not re.search(r'funded|secured|sanctioned|available|financial closure|arranged',excerpt,re.I) or re.search(r'\b(will|plan|planned|proposed|expected|pending)\b|subject to',excerpt,re.I)):
                raise ValueError('Funding intent is not secured funding')
            if field=='firm_order' and record[field] is True and (not re.search(r'received|awarded|signed|executed',excerpt,re.I) or re.search(r'\b(will|plan|planned|proposed|expected)\b',excerpt,re.I)):
                raise ValueError('Order intent is not a received firm contract')
    published=record.get('published_at_utc')
    # Unknown publication times use first retrieval conservatively. Never
    # assign midnight on a document's printed date as a broadcast timestamp.
    publication=pd.Timestamp(published or evidence['first_retrieved_utc'])
    available=max(publication,pd.Timestamp(evidence.get('first_reviewed_utc',publication)))
    if available.tzinfo is None:raise ValueError('Missing catalyst timezone')
    if published and evidence.get('available_at_utc')!=pd.Timestamp(published).isoformat():raise ValueError('Unverified publication time')
    for k in ('capacity_units_per_year','capacity_quantity','order_duration_months','utilisation_pct','order_value_INR','material_dilution_pct'):
        if record.get(k) is not None and (not np.isfinite(record[k]) or record[k]<0):raise ValueError('Invalid quantified catalyst')
    if record.get('utilisation_pct') is not None and record['utilisation_pct']>100:raise ValueError('Utilisation unit conflict')
    if record.get('expected_margin_pct') is not None and not -100<=record['expected_margin_pct']<=100:raise ValueError('Invalid margin units')
    if record.get('capacity_quantity') is not None and not record.get('capacity_unit'):raise ValueError('Capacity quantity needs explicit physical unit')
    return {**record,'available_at_utc':available.isoformat(),'publication_time_status':'VERIFIED_BROADCAST' if published else 'UNKNOWN_USING_FIRST_RETRIEVAL',
      'unknown_fields':[k for k in FIELDS if record.get(k) is None],'future_profit_verified':False}

def collect(root,asof,budget=40):
    root=Path(root);out=root/'catalyst_evidence';out.mkdir(exist_ok=True)
    paths=list((root/'sources').glob('*/events.parquet'))
    if not paths:return {'documents':0,'backlog':0,'errors':[],'catalogue_complete':False}
    e=pd.concat([pd.read_parquet(p) for p in paths],ignore_index=True)
    low=pd.Timestamp(asof)-pd.DateOffset(years=4)
    e=e[pd.to_datetime(e.available_at_utc,utc=True,format='mixed').ge(low.tz_localize('Asia/Kolkata'))]
    # Public whole-venue queue includes approvals/funding/actions, not just
    # headline order counts. No private prediction list is sent to a provider.
    keep=e.text.str.contains(r'capacity|commission|utilisation|utilization|order|contract|approval|fund|allotment|rights|split|bonus|merger|demerger',case=False,na=False)
    e=e[keep&e.attachment_url.str.match(r'https://nsearchives\.nseindia\.com/.*\.pdf$',case=False)].drop_duplicates(['attachment_url','available_at_utc'])
    docs_path=out/'document_evidence.json';docs=json.loads(docs_path.read_text()) if docs_path.exists() else []
    known={(r['attachment_url'],r.get('available_at_utc')) for r in docs}
    attempts_path=out/'attempts.json';attempts=json.loads(attempts_path.read_text()) if attempts_path.exists() else {}
    queue=e[[ (r.attachment_url,r.available_at_utc) not in known for r in e.itertuples() ]].copy()
    queue['last_attempt']=queue.attachment_url.map(lambda s:attempts.get(s,'0000'))
    queue=queue.sort_values(['last_attempt','available_at_utc','symbol'],ascending=[True,False,True]).head(budget)
    store=Store(out/'raw');errors=[];start=time.monotonic()
    for r in queue.to_dict('records'):
        if time.monotonic()-start>300:break
        attempts[r['attachment_url']]=asof
        try:
            raw,proof=store.get(r['attachment_url']);text=extract_pdf(raw)
            info=enrich(text,classify_document(text,r['symbol'],r['issuer_name']))
            (out/(proof['sha256']+'.txt')).write_text(text)
            docs.append({**r,**info,'source_sha256':proof['sha256'],'text_sha256':sha(text.encode()),'first_retrieved_utc':proof['first_retrieved_utc'],
              'publication_time_status':'CATALOG_BROADCAST','review_status':'NEEDS_QUANTIFIED_EXECUTION_REVIEW','unknown_fields':list(FIELDS)})
        except Exception as exc:errors.append({'source_url':r['attachment_url'],'error':str(exc)[:200]})
    docs_path.write_text(json.dumps(docs,indent=2));attempts_path.write_text(json.dumps(attempts,indent=2))
    summary={'documents':len(docs),'issuer_matched':sum(r.get('issuer_identity_read') is True for r in docs),
      'quantified_review_candidates':sum(len(r.get('quantified_evidence',[])) for r in docs),'backlog':len(e)-len(known),
      'errors':errors,'catalogue_complete':False,'automatic_future_profit_claims':0}
    (out/'coverage.json').write_text(json.dumps(summary,indent=2));return summary

def join(frame,root):
    root=Path(root);out=root/'catalyst_evidence';docs_path=out/'document_evidence.json'
    docs=json.loads(docs_path.read_text()) if docs_path.exists() else []
    indexed={r['source_sha256']:r for r in docs};records=[];rejected=[]
    review=out/'reviewed_execution.json'
    if review.exists():
        for r in json.loads(review.read_text()):
            try:
                e=indexed[r['source_sha256']]
                # Original bytes must be present locally or restored by their
                # checkpoint hash before a review can become an input.
                raw=[p for p in (out/'raw').glob('*.bin') if sha(p.read_bytes())==r['source_sha256']]
                if not raw:raise ValueError('Original source bytes unavailable for review audit')
                records.append(reviewed(r,e,(out/(r['source_sha256']+'.txt')).read_text()))
            except Exception as exc:rejected.append({'source_sha256':r.get('source_sha256'),'error':str(exc)})
    d=frame.copy();links=[]
    for i,r in d.iterrows():
        cut=(pd.Timestamp(r.date).tz_localize('Asia/Kolkata')+pd.Timedelta(hours=15,minutes=30)).tz_convert('UTC')
        q=[z for z in records if z['isin']==r['isin'] and cut-pd.Timedelta(days=450)<=pd.Timestamp(z['available_at_utc'])<=cut]
        commissioned=[z for z in q if z.get('commissioned') is True]
        orders={z['order_id']:z for z in q if z.get('firm_order') is True and z.get('order_id') and z.get('order_due') and
          pd.Timestamp(r.date)<=pd.Timestamp(z['order_due'])<=pd.Timestamp(r.date)+pd.DateOffset(months=12)}
        recent=[z for z in commissioned if z.get('document_date') and pd.Timestamp(r.date)-pd.Timestamp(z['document_date'])<=pd.Timedelta(days=90)]
        if recent:d.at[i,'commissioned_document_count90']=len({z['source_sha256'] for z in recent})
        capacity=[z for z in commissioned if z.get('capacity_units_per_year') is not None]
        if capacity:d.at[i,'reported_commissioned_capacity_units_per_year']=max(capacity,key=lambda z:z['available_at_utc'])['capacity_units_per_year']
        utilisation=[z for z in q if z.get('utilisation_pct') is not None]
        if utilisation:d.at[i,'utilisation_disclosed_pct']=max(utilisation,key=lambda z:z['available_at_utc'])['utilisation_pct']
        # Revenue is unknown here unless original annual revenue exists. Do not
        # substitute PAT, order book or a guessed denominator for sales.
        valued=[z for z in orders.values() if z.get('order_value_INR') is not None]
        if valued:
            value=sum(z['order_value_INR'] for z in valued)
            d.at[i,'executable_order_value12_INR']=value
            revenue=r.get('annual_reported_revenue_INR',np.nan)
            if np.isfinite(revenue) and revenue>0:d.at[i,'disclosed_order_value_due12_to_sales']=value/revenue
        chain=[z for z in [*commissioned,*valued] if z.get('funding_sufficient') is True and
          z.get('approval_completed') is True and z.get('expected_margin_pct') is not None and z['expected_margin_pct']>0 and
          z.get('execution_stage') not in {'ANNOUNCED_INTENT','CERTIFICATE_PENDING'}]
        if chain and bool(r.get('financial_complete',False)):
            d.at[i,'evidence_status']='FINANCIAL_AND_EXECUTION_SUPPORTED'
        for z in q:links.append({k:z.get(k) for k in ('isin','symbol','attachment_url','source_sha256','published_at_utc','available_at_utc','publication_time_status','unknown_fields',*FIELDS)})
    out.mkdir(exist_ok=True);(out/'source_links.json').write_text(json.dumps(links,indent=2));(out/'rejected_reviews.json').write_text(json.dumps(rejected,indent=2))
    (out/'quantified_execution_reviews.json').write_text(json.dumps(records,indent=2))
    return d
