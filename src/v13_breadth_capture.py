"""Capture a public adjusted NSE file before the prospective cutoff.

No token or private account is used. Original bytes and the immutable public
publisher revision are retained together with the actual retrieval clock.
"""
from datetime import datetime,timezone
from pathlib import Path
from urllib.request import Request,urlopen
from zoneinfo import ZoneInfo
import hashlib,json,sys,time

import pandas as pd

API='https://huggingface.co/api/datasets/tejhq/indian-markets/tree/main/prices_adjusted?recursive=false&expand=true'
BASE='https://huggingface.co/datasets/tejhq/indian-markets/resolve/'
MAX_BYTES=60_000_000


def get(url,limit):
    with urlopen(Request(url,headers={'User-Agent':'V13-public-research/1.0'}),timeout=45) as r:
        raw=r.read(limit+1)
    if not raw or len(raw)>limit:raise ValueError('Empty or oversized public dataset response')
    return raw


def capture(asof,folder):
    name='prices_adjusted/nse_'+asof[:4]+'.parquet'
    tree=json.loads(get(API,2_000_000))
    matches=[x for x in tree if x.get('path')==name]
    if len(matches)!=1:raise ValueError('Public dataset year is unavailable or ambiguous')
    commit=matches[0].get('lastCommit',{})
    revision=str(commit.get('id',''))
    if len(revision)!=40 or any(c not in '0123456789abcdef' for c in revision):raise ValueError('Invalid immutable source revision')
    published=pd.Timestamp(commit['date'])
    close=pd.Timestamp(asof).tz_localize('Asia/Kolkata')+pd.Timedelta(hours=15,minutes=30)
    ceiling=close.normalize()+pd.Timedelta(hours=22)
    if published.tzinfo is None or not close<=published<=ceiling:
        raise ValueError('Current source-day public revision is not available before the cutoff')
    url=BASE+revision+'/'+name+'?download=true'
    raw=get(url,MAX_BYTES)
    captured=datetime.now(timezone.utc).isoformat()
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    temp=folder/'nse_adjusted.parquet.tmp';temp.write_bytes(raw)
    # Date-only scan, before the bytes are made eligible for the worker.
    dates=pd.read_parquet(temp,columns=['date'])['date']
    dates=pd.to_datetime(dates)
    if not dates.eq(pd.Timestamp(asof)).any() or dates.max()>pd.Timestamp(asof):
        temp.unlink(missing_ok=True);raise ValueError('Public file lacks the day or contains future sessions')
    digest=hashlib.sha256(raw).hexdigest()
    receipt={'source_asof':asof,'source_url':url,'publisher_revision':revision,
      'publisher_commit_utc':published.isoformat(),'first_retrieved_utc':captured,
      'sha256':digest,'bytes':len(raw),'scope':'PUBLIC_ADJUSTED_PRICES_SCORING_ONLY',
      'entry_or_outcome_certified':False}
    from v13_breadth_fallback import MIN_CURRENT_YEAR_SESSIONS
    if dates.nunique()<MIN_CURRENT_YEAR_SESSIONS:
        previous_year=int(asof[:4])-1
        old_name='prices_adjusted/nse_'+str(previous_year)+'.parquet'
        old_matches=[x for x in tree if x.get('path')==old_name]
        if len(old_matches)!=1:raise ValueError('Preceding-year public dataset unavailable')
        old_commit=old_matches[0].get('lastCommit',{});old_revision=str(old_commit.get('id',''))
        if len(old_revision)!=40 or any(c not in '0123456789abcdef' for c in old_revision):raise ValueError('Invalid preceding-year revision')
        old_url=BASE+old_revision+'/'+old_name+'?download=true'
        old_raw=get(old_url,MAX_BYTES)
        old_path=folder/'nse_adjusted_previous.parquet';old_path.write_bytes(old_raw)
        receipt['previous_year']={'year':previous_year,'source_url':old_url,'publisher_revision':old_revision,
          'publisher_commit_utc':old_commit['date'],'first_retrieved_utc':datetime.now(timezone.utc).isoformat(),
          'sha256':hashlib.sha256(old_raw).hexdigest(),'bytes':len(old_raw)}
    from v13_breadth_fallback import validate_receipt
    validate_receipt(receipt,temp,asof)
    temp.replace(folder/'nse_adjusted.parquet')
    (folder/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


def main():
    folder=Path('v13_public_breadth_snapshot')
    local=datetime.now(ZoneInfo('Asia/Kolkata'))
    asof=local.date().isoformat()
    start=pd.Timestamp(asof).tz_localize('Asia/Kolkata')+pd.Timedelta(hours=15,minutes=30)
    end=pd.Timestamp(asof).tz_localize('Asia/Kolkata')+pd.Timedelta(hours=21,minutes=58)
    last_error='Not attempted'
    if pd.Timestamp.now(tz='Asia/Kolkata')<start:
        last_error='Exchange session has not closed'
    while start<=pd.Timestamp.now(tz='Asia/Kolkata')<end:
        try:
            receipt=capture(asof,folder)
            print(json.dumps({'status':'CAPTURED','source_asof':asof,'sha256':receipt['sha256'],'first_retrieved_utc':receipt['first_retrieved_utc']}),flush=True)
            return 0
        except Exception as exc:
            last_error=type(exc).__name__+': '+str(exc)[:200]
            print(json.dumps({'status':'NOT_YET_AVAILABLE','source_asof':asof,'reason':last_error}),flush=True)
            remaining=(end-pd.Timestamp.now(tz='Asia/Kolkata')).total_seconds()
            if remaining<=0:break
            time.sleep(min(120,remaining))
    folder.mkdir(exist_ok=True)
    (folder/'unavailable.json').write_text(json.dumps({'source_asof':asof,'status':'NO_TIMELY_PUBLIC_SNAPSHOT','reason':last_error,'checked_at_utc':datetime.now(timezone.utc).isoformat()},indent=2)+'\n')
    print('No timely public adjusted snapshot; existing collection remains unchanged.',flush=True)
    return 0


if __name__=='__main__':sys.exit(main())
