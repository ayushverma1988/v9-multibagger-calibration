"""Rehydrate already registered catalyst-review originals after raw-byte retirement."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, re
from urllib.parse import urlsplit

import pandas as pd
import requests
from v12_dated_source_collection import source_response

KEY=('source_sha256','symbol','isin','attachment_url')
def key(row):return tuple(row.get(k) for k in KEY)
def digest(b):return hashlib.sha256(b).hexdigest()
def key_id(row):return digest(json.dumps(key(row),separators=(',',':')).encode())
def current_time():return datetime.now(timezone.utc).isoformat()
def fetch_original(url):
    u=urlsplit(url)
    if u.scheme!='https' or u.username or u.password or u.port not in (None,443) or u.query or u.fragment:raise ValueError('Unapproved original URL')
    with requests.Session() as s:
        s.headers.update({'User-Agent':'Mozilla/5.0','Referer':'https://www.nseindia.in/'})
        if u.hostname in ('nsearchives.nseindia.com','nsearchives.nseindia.in') and u.path.startswith('/corporate/'):
            response,_=source_response(s,url)
        elif u.hostname=='www.aeroflexindia.com' and u.path.startswith('/uploads/topics/'):
            response=s.get(url,allow_redirects=False,timeout=(8,25))
            if response.url!=url:raise ValueError('Issuer URL mismatch')
        else:raise ValueError('Unapproved original host/path')
        with response as r:
            if r.status_code!=200:raise ValueError('Original unavailable (HTTP %s)'%r.status_code)
            body=r.content
            if len(body)>32*1024*1024 or not body.startswith(b'%PDF'):raise ValueError('Unexpected original PDF')
            return body,r.url

def source_index(docs):
    index={}
    for doc in docs:index.setdefault(key(doc),[]).append(doc)
    return index
def bound_doc(index,r):
    matches=index.get(key(r),[])
    if not matches:raise ValueError('No original with matching hash, URL, symbol and ISIN')
    fields=('text_sha256','issuer_identity_read','available_at_utc','first_retrieved_utc','first_reviewed_utc')
    if len({tuple(m.get(k) for k in fields) for m in matches})!=1:raise ValueError('Conflicting saved original provenance')
    return matches[0]

def receipt_path(out):return out/'source_rehydration_receipts.json'
def receipts(out):
    p=receipt_path(out)
    return json.loads(p.read_text()) if p.exists() else {}
def floor_clock(record,r,journal):
    rec=journal.get(key_id(r))
    if rec is None:return record
    if key(rec)!=key(r):raise ValueError('Recovery receipt identity mismatch')
    orig=pd.Timestamp(record['available_at_utc']); first=pd.Timestamp(rec['first_revalidated_utc'])
    if first.tzinfo is None or orig.tzinfo is None:raise ValueError('Unknown original recovery clock')
    return {**record,'original_available_at_utc':orig.isoformat(),'first_source_revalidated_utc':first.isoformat(),'available_at_utc':max(orig,first).isoformat()}

def atomic(path,data):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_bytes(data);tmp.replace(path)

def restore(out,seeds,stored,docs,verify,extract,fetch=fetch_original,clock=current_time):
    """Only exact checked-in sources of existing reviews; never replace unverified bytes."""
    if not stored:return {'restored_originals':0,'errors':[]}
    rawdir=out/'raw';rawdir.mkdir(parents=True,exist_ok=True)
    trusted={key(s)+(s.get('project_id'),s.get('order_id')) for s in seeds}
    groups={}
    for r in stored:
        if key(r)+(r.get('project_id'),r.get('order_id')) in trusted:groups.setdefault(key(r),[]).append(r)
    idx=source_index(docs);j=receipts(out);errors=[];count=0;checked=0
    for (h,symbol,isin,url),rows in groups.items():
        try:
            if not re.fullmatch('[0-9a-f]{64}',h or ''):raise ValueError('Invalid reviewed original digest')
            doc=bound_doc(idx,rows[0]);rawpath=rawdir/(h+'.bin');cached=rawdir/(digest(url.encode())+'.bin');tp=out/(h+'.txt')
            if rawpath.is_file() and digest(rawpath.read_bytes())!=h:raise ValueError('Corrupted recovered original; not overwritten')
            if tp.is_file() and digest(tp.read_bytes())!=doc['text_sha256']:raise ValueError('Retained extraction differs; not overwritten')
            have=rawpath if rawpath.is_file() else cached if cached.is_file() and digest(cached.read_bytes())==h else None
            if have is not None and tp.exists():continue
            raw,final=(have.read_bytes(),url) if have else fetch(url)
            if digest(raw)!=h:raise ValueError('Fetched original differs from stored SHA-256')
            text=extract(raw)
            if digest(text.encode())!=doc['text_sha256']:raise ValueError('Original extraction differs from stored SHA-256')
            for r in rows:verify(r,doc,text)
            today=clock();rid=key_id(rows[0]);previous=j.get(rid);first=today
            if previous:
                if key(previous)!=key(rows[0]) or pd.Timestamp(previous['first_revalidated_utc']).tzinfo is None:raise ValueError('Invalid recovery receipt')
                first=previous['first_revalidated_utc']
            j[rid]={**dict(zip(KEY,key(rows[0]))),'first_revalidated_utc':first,'last_revalidated_utc':today,'final_url':final,'bytes':len(raw),'text_sha256':doc['text_sha256'],'review_rows_checked':len(rows)}
            # Persist the date floor before exposing recovered proof, including after interruption.
            atomic(receipt_path(out),json.dumps(j,indent=2).encode())
            if not tp.exists():atomic(tp,text.encode())
            if not have:atomic(rawpath,raw);count+=1
            checked+=len(rows)
        except Exception as exc:errors.append({'symbol':symbol,'source_sha256':h,'error':str(exc)[:200]})
    return {'restored_originals':count,'revalidated_review_rows':checked,'errors':errors,'private_forecasts_modified':False}

def present_hashes(rawdir,wanted):
    if not wanted:return set()
    found=set()
    for p in rawdir.glob('*.bin'):
        h=digest(p.read_bytes())
        if h in wanted:found.add(h)
        if found==wanted:break
    return found
