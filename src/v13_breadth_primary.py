"""Retain one dated NSE original per capture even if the independent supplier is late."""
import csv
from datetime import date, datetime, time, timedelta, timezone
from hashlib import sha256
from io import BytesIO, TextIOWrapper
import json
from pathlib import Path
from urllib.error import HTTPError
from zoneinfo import ZoneInfo
from zipfile import ZipFile

IST = ZoneInfo('Asia/Kolkata')
LIMIT = 10_000_000
CLOCK = lambda: datetime.now(timezone.utc)

def url(day):
    return 'https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_' + date.fromisoformat(day).strftime('%Y%m%d') + '_F_0000.csv.zip'

def window(day):
    d = date.fromisoformat(day)
    return datetime.combine(d, time(15,30), IST), datetime.combine(d, time(22), IST)

def rows_in_original(raw, day):
    with ZipFile(BytesIO(raw)) as z:
        parts = z.infolist()
        if len(parts) != 1 or not parts[0].filename.lower().endswith('.csv') or parts[0].file_size > 40_000_000:
            raise ValueError('Unexpected NSE archive structure')
        with z.open(parts[0]) as f:
            rows = csv.DictReader(TextIOWrapper(f, encoding='utf-8-sig', newline=''))
            if not {'TradDt','BizDt','Sgmt','Src','ISIN','TckrSymb','SctySrs','OpnPric','HghPric','LwPric','ClsPric','TtlTradgVol'}.issubset(rows.fieldnames or []):
                raise ValueError('NSE file lacks required identity or price columns')
            count = 0
            for row in rows:
                if row['TradDt'] != day or row['BizDt'] != day or row['Src'] != 'NSE' or row['Sgmt'] != 'CM':
                    raise ValueError('NSE market or trade date mismatch')
                count += 1
    if not count: raise ValueError('Empty NSE session')
    return count

def _aware(value):
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None: raise ValueError('Missing observation timezone')
    return dt

def read_primary(day, folder, clock=CLOCK):
    folder = Path(folder)
    if not (folder/'primary.json').exists(): return None
    obj = json.loads((folder/'primary.json').read_bytes())
    raw = (folder/'nse.zip').read_bytes()
    start,end = window(day)
    stamp = _aware(obj['first_retrieved_utc'])
    if (obj.get('scope') != 'NSE_PRIMARY_ONLY_TIMED' or obj.get('source_asof') != day
        or obj.get('source_url') != url(day) or not start <= stamp <= end or stamp > clock()
        or obj.get('bytes') != len(raw) or obj.get('sha256') != sha256(raw).hexdigest()
        or obj.get('nse_rows') != rows_in_original(raw,day)):
        raise ValueError('Invalid or out-of-window primary receipt')
    return raw,obj

def retain(day, folder, fetch, *, timely=True, clock=CLOCK):
    folder=Path(folder); start,end=window(day)
    if timely:
        cached=read_primary(day,folder,clock)
        if cached is not None: return cached
    before=clock()
    if before.tzinfo is None or (timely and not start <= before <= end) or (not timely and before <= end):
        raise ValueError('Wrong capture window')
    raw=fetch(url(day),LIMIT); stamp=clock()
    if stamp.tzinfo is None or stamp<before or (timely and stamp>end): raise ValueError('Receipt outside capture window')
    n=rows_in_original(raw,day)
    # Always preserve the time the actual bytes finished downloading; parsing is later.
    obj={'source_asof':day,'source_url':url(day),'first_retrieved_utc':stamp.astimezone(timezone.utc).isoformat(),
         'bytes':len(raw),'sha256':sha256(raw).hexdigest(),'nse_rows':n,
         'scope':'NSE_PRIMARY_ONLY_TIMED' if timely else 'NSE_LATE_REFERENCE_ONLY',
         'adjusted_prices_or_forecast_certified':False,'entry_or_outcome_certified':False}
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'primary.json').exists(): raise ValueError('Refusing to replace existing evidence')
    temp=folder/'nse.zip.part';temp.write_bytes(raw);temp.replace(folder/'nse.zip')
    temp=folder/'primary.json.part';temp.write_text(json.dumps(obj,indent=2)+'\n');temp.replace(folder/'primary.json')
    return raw,obj

def late_reference(folder, fetch, *, clock=CLOCK):
    """Try earlier days only for an explicit 404/410; surface a blocked or invalid original."""
    local=clock().astimezone(IST);day=local.date()
    if local<=window(day.isoformat())[1]: day-=timedelta(days=1)
    attempts=[]
    for back in range(8):
        target=(day-timedelta(days=back)).isoformat()
        # Do not skip weekends: NSE can hold a special session.
        try: return retain(target,folder,fetch,timely=False,clock=clock)[1]
        except HTTPError as exc:
            if exc.code not in (404,410):raise
            attempts.append(target)
    raise ValueError('No published exchange session among '+','.join(attempts))
