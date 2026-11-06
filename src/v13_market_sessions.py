"""NSE special sessions require a reviewed original, never a weekday guess."""
from functools import lru_cache
from pathlib import Path
import base64,hashlib,json,re
import pandas as pd

SEED=Path(__file__).resolve().parents[1]/'v13_evidence_seed/trading_session_reviews.json'

@lru_cache(maxsize=4)
def reviewed_sessions(seed=SEED):
    from v11_4_primary_document_catalysts import extract_pdf
    seed=Path(seed)
    if not seed.exists():raise ValueError('Missing reviewed special-session configuration')
    records=json.loads(seed.read_text());result={}
    for item in records:
        date=str(pd.Timestamp(item['date']).date())
        if item.get('timezone')!='Asia/Kolkata':raise ValueError('Special-session timezone must match the exchange')
        if date in result:raise ValueError('Duplicate special-session date')
        if not re.fullmatch(r'https://nsearchives\.nseindia\.com/content/circulars/[A-Z]+\d+\.pdf',item['source_url']):raise ValueError('Unapproved special-session original')
        raw=base64.b64decode(item['original_pdf_base64'],validate=True)
        if not raw.startswith(b'%PDF') or len(raw)!=item['bytes'] or hashlib.sha256(raw).hexdigest()!=item['source_sha256']:raise ValueError('Special-session original hash/length conflict')
        text=re.sub(r'\s+',' ',extract_pdf(raw))
        quotes=item['field_excerpts']
        for key in ('date','open','close'):
            if not quotes.get(key) or re.sub(r'\s+',' ',quotes[key]).strip() not in text:raise ValueError('Unbound special-session '+key)
        literal=pd.Timestamp(date).strftime('%B %d, %Y')
        if literal not in quotes['date']:raise ValueError('Special-session date conflicts with original')
        for field in ('open','close'):
            if not re.fullmatch(r'\d{2}:\d{2}',item[field]) or item[field] not in quotes[field]:raise ValueError('Special-session time conflicts with original')
        if not re.search('Normal Market.*Open time',quotes['open']):raise ValueError('Special-session open must be the normal market open')
        if 'Trade Modification cut-off time' not in quotes['close']:raise ValueError('Special-session completion bound must include the final cut-off')
        if item['open']>=item['close']:raise ValueError('Invalid special-session clock order')
        retrieved=pd.Timestamp(item['retrieved_at_utc']);review=pd.Timestamp(item['reviewed_at_utc'])
        if retrieved.tzinfo is None or review.tzinfo is None or review<retrieved or review>pd.Timestamp.now(tz='UTC'):raise ValueError('Invalid special-session observation clocks')
        publication=pd.Timestamp(item['publication_date']).date()
        if publication>retrieved.tz_convert('Asia/Kolkata').date():raise ValueError('Special-session publication date is in the future')
        result[date]={k:v for k,v in item.items() if k!='original_pdf_base64'}
    return result

def session_open(date):
    day=pd.Timestamp(date).normalize();item=reviewed_sessions().get(str(day.date()))
    hour,minute=map(int,(item['open'] if item else '09:15').split(':'))
    return day.tz_localize('Asia/Kolkata')+pd.Timedelta(hours=hour,minutes=minute)

def session_close(date):
    day=pd.Timestamp(date).normalize();item=reviewed_sessions().get(str(day.date()))
    hour,minute=map(int,(item['close'] if item else '15:30').split(':'))
    return day.tz_localize('Asia/Kolkata')+pd.Timedelta(hours=hour,minutes=minute)

def open_clocks(dates):
    values=pd.DatetimeIndex([session_open(d) for d in pd.to_datetime(dates)],tz='Asia/Kolkata')
    return pd.Series(values,index=dates.index)

def retain_originals(folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    records=json.loads(SEED.read_text());reviewed_sessions()
    for item in records:
        path=folder/(item['source_sha256']+'.bin')
        raw=base64.b64decode(item['original_pdf_base64'],validate=True)
        if path.exists() and path.read_bytes()!=raw:raise ValueError('Special-session retained original changed')
        if not path.exists():path.write_bytes(raw)
    return list(reviewed_sessions().values())
