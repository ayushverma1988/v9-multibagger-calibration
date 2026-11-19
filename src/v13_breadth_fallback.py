"""A strictly timed, independently reconciled, scoring-only price fallback.

The public alternative carries NSE daily bars and its own adjustment factors.
It never replaces stored provider histories or supplies entry/outcome evidence.
Each resumed series needs 126 matching adjusted closes, an exact current
identity, the complete intervening exchange sessions and exact raw OHLCV.
"""
from pathlib import Path
import hashlib
import json
import re

import numpy as np
import pandas as pd

MIN_FACTOR_OVERLAP = 126
FACTOR_RTOL = 0.00001
MAX_MISSING_SESSIONS = 5
MIN_CURRENT_YEAR_SESSIONS = 150
SOURCE_BASE='https://huggingface.co/datasets/tejhq/indian-markets/resolve/'


def _aware(value):
    result = pd.Timestamp(value)
    if result.tzinfo is None:
        raise ValueError('Source clock is not timezone-aware')
    return result.tz_convert('UTC')


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def validate_receipt(receipt, path, asof):
    """Require original bytes actually captured between close and the cutoff."""
    if receipt.get('source_asof') != asof:
        raise ValueError('Fallback snapshot has a different source session')
    close = pd.Timestamp(asof).tz_localize('Asia/Kolkata') + pd.Timedelta(hours=15, minutes=30)
    ceiling = close.normalize() + pd.Timedelta(hours=22)
    captured = _aware(receipt.get('first_retrieved_utc'))
    published = _aware(receipt.get('publisher_commit_utc'))
    if not (close.tz_convert('UTC') <= published <= captured <= ceiling.tz_convert('UTC')):
        raise ValueError('Fallback source was published or captured outside the permitted information window')
    if not re.fullmatch(r'[0-9a-f]{40}', str(receipt.get('publisher_revision', ''))):
        raise ValueError('Missing immutable public source revision')
    expected=SOURCE_BASE+receipt['publisher_revision']+'/prices_adjusted/nse_'+asof[:4]+'.parquet?download=true'
    if receipt.get('source_url')!=expected:raise ValueError('Fallback source URL does not bind immutable revision')
    if receipt.get('bytes')!=Path(path).stat().st_size or not re.fullmatch(r'[0-9a-f]{64}', str(receipt.get('sha256', ''))) or _sha(path) != receipt['sha256']:
        raise ValueError('Fallback source hash mismatch')
    previous=receipt.get('previous_year')
    if previous is not None:
        if not isinstance(previous,dict) or previous.get('year') != int(asof[:4])-1:
            raise ValueError('Invalid preceding-year source identity')
        old_published=_aware(previous.get('publisher_commit_utc'))
        old_captured=_aware(previous.get('first_retrieved_utc'))
        if not (old_published<=old_captured and close.tz_convert('UTC')<=old_captured<=ceiling.tz_convert('UTC')):
            raise ValueError('Preceding-year source captured outside the permitted window')
        if not re.fullmatch(r'[0-9a-f]{40}',str(previous.get('publisher_revision',''))):
            raise ValueError('Missing immutable preceding-year revision')
        expected_old=SOURCE_BASE+previous['publisher_revision']+'/prices_adjusted/nse_'+str(previous['year'])+'.parquet?download=true'
        if previous.get('source_url')!=expected_old:raise ValueError('Preceding-year URL does not bind immutable revision')
        old_path=Path(path).parent/'nse_adjusted_previous.parquet'
        if not old_path.exists() or previous.get('bytes')!=old_path.stat().st_size or not re.fullmatch(r'[0-9a-f]{64}',str(previous.get('sha256',''))) or _sha(old_path)!=previous['sha256']:
            raise ValueError('Preceding-year source hash mismatch')
    return captured


def load(root, asof):
    """Read a captured public snapshot; a missing/late source simply blocks it."""
    folder = Path(root).parent / 'v13_public_breadth_snapshot'
    receipt_path = folder / 'receipt.json'
    data_path = folder / 'nse_adjusted.parquet'
    if not receipt_path.exists() or not data_path.exists():
        return None, {'status':'NO_TIMELY_PUBLIC_FALLBACK_SNAPSHOT'}
    try:
        receipt = json.loads(receipt_path.read_text())
        validate_receipt(receipt, data_path, asof)
        columns = ['date','symbol','isin','series','open','high','low','close','volume','adj_close','adj_factor_cumulative']
        data = pd.read_parquet(data_path, columns=columns)
        data['date'] = pd.to_datetime(data.date)
        if data.date.nunique()<MIN_CURRENT_YEAR_SESSIONS:
            if not receipt.get('previous_year'):raise ValueError('Preceding-year source required for adjustment overlap')
            previous=pd.read_parquet(folder/'nse_adjusted_previous.parquet',columns=columns)
            previous['date']=pd.to_datetime(previous.date)
            data=pd.concat([previous,data],ignore_index=True)
        low = pd.Timestamp(asof) - pd.DateOffset(years=4)
        if (data.date > pd.Timestamp(asof)).any():
            raise ValueError('Snapshot contains future sessions')
        data = data[data.date.between(low, pd.Timestamp(asof))].copy()
        if not data.date.eq(pd.Timestamp(asof)).any():
            raise ValueError('Snapshot lacks the source session')
        if data.duplicated(['date','symbol','isin','series']).any():
            raise ValueError('Duplicate alternative source identity/date')
        return (data, receipt), {'status':'TIMELY_SNAPSHOT_VERIFIED', 'source_sha256':receipt['sha256'],
          'first_retrieved_utc':receipt['first_retrieved_utc'], 'publisher_commit_utc':receipt['publisher_commit_utc']}
    except (ValueError, KeyError, TypeError, OSError) as exc:
        return None, {'status':'BLOCKED_SNAPSHOT', 'reason':str(exc)[:200]}


def _calendar_and_official(root, start, asof):
    """All business-day sessions and original official files must be present."""
    from v13_action_audit import expected_sessions
    holidays=[]
    for year in range(pd.Timestamp(start).year, pd.Timestamp(asof).year + 1):
        p=Path(root)/'action_audit'/f'calendar_{year}.json'
        if not p.exists(): raise ValueError('Missing original NSE session calendar')
        c=json.loads(p.read_text())
        if c.get('year') != year or not re.fullmatch(r'[0-9a-f]{64}',str(c.get('source_sha256',''))):
            raise ValueError('Invalid NSE session calendar receipt')
        holidays.extend(c['holidays'])
    years=list(range(pd.Timestamp(start).year,pd.Timestamp(asof).year+1))
    sessions=expected_sessions(start,asof,{'year':years[-1],'years':years,'holidays':holidays})
    if len(sessions)>MAX_MISSING_SESSIONS:
        raise ValueError('Stale provider gap exceeds bounded fallback limit')
    files=[]
    for date in sessions:
        p=Path(root)/'official'/(str(date.date())+'_official_market.parquet')
        if not p.exists(): raise ValueError('Missing original exchange session '+str(date.date()))
        files.append(pd.read_parquet(p))
    return sessions, pd.concat(files,ignore_index=True) if files else pd.DataFrame()


def extend(previous, alternative, official, sessions, asof, symbol, isin, source_sha256):
    """Build an in-memory scoring view after factor, identity and raw checks."""
    if previous.empty or set(previous.symbol)!={symbol} or set(previous['isin'])!={isin}:
        raise ValueError('Original provider history has ambiguous identity')
    old=previous.copy(); old['date']=pd.to_datetime(old.date)
    if old.date.duplicated().any() or old.date.max() >= pd.Timestamp(asof):
        raise ValueError('Original provider dates are duplicate or already current')
    alt=alternative[alternative.symbol.eq(symbol)].copy()
    if alt.empty or set(alt['isin'])!={isin} or alt.series.nunique()!=1:
        raise ValueError('Alternative identity or series changed')
    if alt.date.duplicated().any(): raise ValueError('Alternative has duplicate dates')
    joined=old[['date','adj_close']].merge(alt[['date','adj_close']],on='date',suffixes=('_old','_new'),validate='1:1')
    joined=joined.sort_values('date').tail(252)
    if len(joined)<MIN_FACTOR_OVERLAP: raise ValueError('Insufficient independent adjusted-close overlap')
    if set(old.sort_values('date').tail(MIN_FACTOR_OVERLAP).date)!=set(joined.tail(MIN_FACTOR_OVERLAP).date):
        raise ValueError('Alternative omitted a recent accepted provider session')
    a=pd.to_numeric(joined.adj_close_old,errors='coerce').to_numpy(float)
    b=pd.to_numeric(joined.adj_close_new,errors='coerce').to_numpy(float)
    if not (np.isfinite(a).all() and np.isfinite(b).all() and (a>0).all() and (b>0).all()):
        raise ValueError('Missing adjustment factors in overlap')
    if not np.allclose(a,b,rtol=FACTOR_RTOL,atol=.001):
        raise ValueError('Provider adjustment factors disagree')
    incoming=alt[alt.date.gt(old.date.max()) & alt.date.le(pd.Timestamp(asof))].sort_values('date').copy()
    if incoming.empty or incoming.date.max()!=pd.Timestamp(asof):
        raise ValueError('Alternative lacks the completed source session')
    for col in ('open','high','low','close','adj_close','adj_factor_cumulative','volume'):
        incoming[col]=pd.to_numeric(incoming[col],errors='coerce')
    numeric=incoming[['open','high','low','close','adj_close','adj_factor_cumulative','volume']].to_numpy(float)
    if not np.isfinite(numeric).all() or (numeric[:,:6]<=0).any() or (numeric[:,6]<0).any():
        raise ValueError('Alternative gap contains missing or invalid prices or factors')
    if not np.allclose(incoming.adj_close,incoming.close*incoming.adj_factor_cumulative,rtol=FACTOR_RTOL,atol=.001):
        raise ValueError('Alternative adjusted close and declared factor disagree')
    valid=official[(official.symbol==symbol)&(official['isin']==isin)].copy()
    valid['date']=pd.to_datetime(valid.date)
    valid=valid[valid.date.isin(sessions)&valid.source_status.eq('VALID_POSITIVE_VOLUME_SESSION')]
    if valid.date.duplicated().any() or set(incoming.date)!=set(valid.date):
        raise ValueError('Alternative and official intervening sessions differ')
    q=incoming.merge(valid,on='date',suffixes=('_alt','_official'),validate='1:1')
    if not q.symbol_alt.eq(q.symbol_official).all() or not q.isin_alt.eq(q.isin_official).all() or not q.series_alt.eq(q.series_official).all():
        raise ValueError('Alternative/exchange identity mismatch')
    for col in ('open','high','low','close'):
        if not (pd.to_numeric(q[col+'_alt'],errors='coerce')-pd.to_numeric(q[col+'_official'],errors='coerce')).abs().le(.011).all():
            raise ValueError('Alternative/exchange OHLC mismatch')
    if not pd.to_numeric(q.volume_alt,errors='coerce').eq(pd.to_numeric(q.volume_official,errors='coerce')).all():
        raise ValueError('Alternative/exchange volume mismatch')
    current=incoming[incoming.date.eq(pd.Timestamp(asof))].iloc[0]
    if not np.isfinite(float(current.adj_close)) or not np.isfinite(float(current.adj_factor_cumulative)) or abs(float(current.adj_factor_cumulative)-1)>FACTOR_RTOL or abs(float(current.adj_close)-float(current.close))>.011:
        raise ValueError('Source-day adjustment factor is unavailable or inconsistent')
    incoming['source_sha256']=source_sha256
    # Keep only the existing provider schema; no provenance is written back.
    out=pd.concat([old,incoming.reindex(columns=old.columns)],ignore_index=True).sort_values('date')
    return out, {'overlap_sessions':len(joined),'new_sessions':len(incoming),
      'source_sha256':source_sha256,'source_status':'ALTERNATIVE_ADJUSTED_SCORING_ONLY',
      'action_interval_certified':False,'entry_or_outcome_certified':False}
