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
import shutil

import numpy as np
import pandas as pd

MIN_FACTOR_OVERLAP = 126
FACTOR_RTOL = 0.00001
MAX_MISSING_SESSIONS = 5
MIN_CURRENT_YEAR_SESSIONS = 150
SOURCE_BASE='https://huggingface.co/datasets/tejhq/indian-markets/resolve/'
NSE_CALENDAR_URL='https://www.nseindia.com/api/holiday-master?type=trading'
NSE_ACTIONS='https://www.nseindia.com/api/corporates-corporateActions?index=equities&from_date={day}&to_date={day}'


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
    if not (published <= captured and close.tz_convert('UTC') <= captured <= ceiling.tz_convert('UTC')):
        raise ValueError('Fallback source was published or captured outside the permitted information window')
    if not re.fullmatch(r'[0-9a-f]{40}', str(receipt.get('publisher_revision', ''))):
        raise ValueError('Missing immutable public source revision')
    expected=SOURCE_BASE+receipt['publisher_revision']+'/prices_adjusted/nse_'+asof[:4]+'.parquet?download=true'
    if receipt.get('source_url')!=expected:raise ValueError('Fallback source URL does not bind immutable revision')
    if receipt.get('bytes')!=Path(path).stat().st_size or not re.fullmatch(r'[0-9a-f]{64}', str(receipt.get('sha256', ''))) or _sha(path) != receipt['sha256']:
        raise ValueError('Fallback source hash mismatch')
    expected_session='https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_'+asof.replace('-','')+'_F_0000.csv.zip'
    for key,name,url in [('nse_original','nse_source_session.zip',expected_session),
                         ('nse_calendar','nse_holiday_calendar.json',NSE_CALENDAR_URL)]:
        obj=receipt.get(key)
        if not isinstance(obj,dict) or obj.get('source_url')!=url:
            raise ValueError('Missing timed primary NSE source: '+key)
        clock=_aware(obj.get('first_retrieved_utc'))
        if not close.tz_convert('UTC')<=clock<=ceiling.tz_convert('UTC'):
            raise ValueError('Primary NSE source was captured outside the cutoff: '+key)
        original=Path(path).parent/name
        if not original.exists() or obj.get('bytes')!=original.stat().st_size or not re.fullmatch(r'[0-9a-f]{64}',str(obj.get('sha256',''))) or _sha(original)!=obj['sha256']:
            raise ValueError('Primary NSE source hash mismatch: '+key)
    action_day=pd.Timestamp(asof).strftime('%d-%m-%Y')
    key='nse_corporate_actions';name='nse_corporate_actions.json';url=NSE_ACTIONS.format(day=action_day)
    obj=receipt.get(key)
    if not isinstance(obj,dict) or obj.get('source_url')!=url or obj.get('no_action_interval_certified') is not False:
        raise ValueError('Missing timed primary NSE source: '+key)
    clock=_aware(obj.get('first_retrieved_utc'))
    original=Path(path).parent/name
    if not close.tz_convert('UTC')<=clock<=ceiling.tz_convert('UTC') or not original.exists() or obj.get('bytes')!=original.stat().st_size or _sha(original)!=obj.get('sha256'):
        raise ValueError('Primary NSE corporate-action directory mismatch')
    actions=json.loads(original.read_text())
    if not isinstance(actions,list) or obj.get('rows')!=len(actions):raise ValueError('Primary NSE corporate-action directory schema changed')
    adjusted_through=pd.Timestamp(receipt.get('adjusted_through_asof'))
    if adjusted_through>pd.Timestamp(asof) or adjusted_through<pd.Timestamp(asof)-pd.Timedelta(days=10):
        raise ValueError('Adjusted source is future-dated or too stale')
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
        adjusted_through=pd.Timestamp(receipt['adjusted_through_asof'])
        if data.date.max().normalize()!=adjusted_through:
            raise ValueError('Adjusted source terminal date changed')
        if data.duplicated(['date','symbol','isin','series']).any():
            raise ValueError('Duplicate alternative source identity/date')
        calendar=json.loads((folder/'nse_holiday_calendar.json').read_text())
        dates=calendar.get('CM') if isinstance(calendar,dict) else None
        if not isinstance(dates,list):raise ValueError('Captured exchange calendar schema changed')
        holidays=[]
        for row in dates:
            value=pd.to_datetime(row['tradingDate'],format='%d-%b-%Y')
            if value.year==int(asof[:4]):holidays.append(str(value.date()))
        if not holidays:raise ValueError('Captured exchange calendar omits the current year')
        actions=json.loads((folder/'nse_corporate_actions.json').read_text())
        action_symbols=sorted({str(r.get('symbol') or r.get('symbolName') or '').strip() for r in actions if isinstance(r,dict) and (r.get('symbol') or r.get('symbolName'))})
        receipt={**receipt,'source_day_action_symbols':action_symbols}
        status='TIMELY_SNAPSHOT_VERIFIED' if adjusted_through==pd.Timestamp(asof) else 'TIMELY_LAGGED_SNAPSHOT_VERIFIED_FOR_SCORING_ONLY'
        return (data, receipt, {'year':int(asof[:4]),'years':[int(asof[:4])],'holidays':holidays}), {'status':status, 'source_sha256':receipt['sha256'],
          'first_retrieved_utc':receipt['first_retrieved_utc'], 'publisher_commit_utc':receipt['publisher_commit_utc']}
    except (ValueError, KeyError, TypeError, OSError) as exc:
        return None, {'status':'BLOCKED_SNAPSHOT', 'reason':str(exc)[:200]}


def _calendar_and_official(root, start, asof, receipt, calendar):
    """All business-day sessions and original official files must be present."""
    from v13_action_audit import expected_sessions
    if pd.Timestamp(start).year!=pd.Timestamp(asof).year:
        raise ValueError('Cross-year gap needs an independently timed exchange calendar')
    sessions=expected_sessions(start,asof,calendar)
    if len(sessions)>MAX_MISSING_SESSIONS:
        raise ValueError('Stale provider gap exceeds bounded fallback limit')
    files=[]
    for date in sessions:
        p=Path(root)/'official'/(str(date.date())+'_official_market.parquet')
        if not p.exists(): raise ValueError('Missing original exchange session '+str(date.date()))
        raw=pd.read_parquet(p)
        if 'source_sha256' not in raw or raw.empty:raise ValueError('Missing source-bound exchange session')
        if str(date.date())==asof:
            if set(raw.source_sha256)!={receipt['nse_original']['sha256']}:
                raise ValueError('Post-capture exchange bytes changed')
        else:
            if 'first_retrieved_utc' not in raw:raise ValueError('Older exchange session has no retrieval clock')
            cut=pd.Timestamp(asof).tz_localize('Asia/Kolkata')+pd.Timedelta(hours=22)
            clocks=pd.to_datetime(raw.first_retrieved_utc,utc=True,errors='coerce')
            if clocks.isna().any() or (clocks>cut.tz_convert('UTC')).any():
                raise ValueError('Older exchange session was observed after the cutoff')
        files.append(raw)
    return sessions, pd.concat(files,ignore_index=True) if files else pd.DataFrame()


def timely_history_receipts(root,asof):
    """Use immutable old collection receipts, not a newly observed factor."""
    ceiling=(pd.Timestamp(asof).tz_localize('Asia/Kolkata')+pd.Timedelta(hours=22)).tz_convert('UTC')
    result={}
    for path in (Path(root)/'sources').glob('*/history_refresh_*.json'):
        try:
            items=json.loads(path.read_text())
            if not isinstance(items,list):continue
            for item in items:
                if not isinstance(item,dict):continue
                info=item.get('info') or {};digest=info.get('source_sha256');symbol=info.get('symbol');isin=info.get('isin')
                if item.get('symbol')!=symbol or not re.fullmatch(r'[0-9a-f]{64}',str(digest or '')):continue
                observed=_aware(info.get('first_retrieved_utc'))
                if observed>ceiling:continue
                key=(symbol,isin,digest)
                if key not in result or observed<result[key]:result[key]=observed
        except (ValueError,TypeError,KeyError,OSError):continue
    return result


def snapshot_prior_histories(root,asof,destination):
    """Preserve pre-refresh, time-bound histories in a disposable scoring area."""
    root=Path(root);destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    receipts=timely_history_receipts(root,asof)
    wanted=None
    if (root/'universe.parquet').exists():
        universe=pd.read_parquet(root/'universe.parquet')
        symbol_col='nse_current_symbol' if 'nse_current_symbol' in universe else 'symbol'
        wanted=set(zip(universe['isin'],universe[symbol_col]))
    from v13_history_views import histories
    selected,_=histories(root/'history'/'securities',wanted)
    index={};skipped=0
    for (isin,symbol),(path,bars) in selected.items():
        if bars.empty or pd.to_datetime(bars.date).max()>=pd.Timestamp(asof) or 'source_sha256' not in bars:
            skipped+=1;continue
        hashes=set(bars.source_sha256)
        if len(hashes)!=1:skipped+=1;continue
        source_hash=next(iter(hashes));seen=receipts.get((symbol,isin,source_hash))
        last_close=pd.to_datetime(bars.date).max().tz_localize('Asia/Kolkata')+pd.Timedelta(hours=15,minutes=30)
        if seen is None or seen<last_close.tz_convert('UTC'):skipped+=1;continue
        key=isin+'|'+symbol;name=hashlib.sha256(key.encode()).hexdigest()+'.parquet'
        target=destination/name;shutil.copyfile(path,target)
        index[key]={'file':name,'file_sha256':_sha(target),'source_sha256':source_hash,'first_retrieved_utc':seen.isoformat()}
    (destination/'index.json').write_text(json.dumps(index,indent=2))
    return {'status':'TIMELY_ORIGINAL_HISTORIES_SNAPSHOTTED','histories':len(index),'skipped':skipped,
      'private_checkpoints_or_original_histories_modified':False}


def prior_history(folder,isin,symbol,receipts,index=None):
    if not folder:return None
    root=Path(folder);p=root/'index.json'
    if not p.exists():return None
    item=(json.loads(p.read_text()) if index is None else index).get(isin+'|'+symbol)
    if not item:return None
    name=item.get('file','')
    if not re.fullmatch(r'[0-9a-f]{64}\.parquet',name):raise ValueError('Unsafe prior scoring history path')
    original=root/name
    if not original.exists() or _sha(original)!=item.get('file_sha256'):raise ValueError('Prior scoring history changed')
    source_hash=item.get('source_sha256');seen=receipts.get((symbol,isin,source_hash))
    if seen is None or seen!=_aware(item.get('first_retrieved_utc')):raise ValueError('Prior provider receipt is unavailable')
    bars=pd.read_parquet(original)
    if bars.empty or set(bars.symbol)!={symbol} or set(bars['isin'])!={isin} or set(bars.source_sha256)!={source_hash}:
        raise ValueError('Prior scoring history identity changed')
    return bars


def extend(previous, alternative, official, sessions, asof, symbol, isin, source_sha256, source_day_action_symbols=()):
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
    lagged=incoming.empty or incoming.date.max()!=pd.Timestamp(asof)
    if lagged:
        if symbol in set(source_day_action_symbols):
            raise ValueError('Primary exchange corporate action requires a current adjusted factor')
        valid=official[(official.symbol==symbol)&(official['isin']==isin)].copy()
        valid['date']=pd.to_datetime(valid.date)
        valid=valid[valid.date.isin(sessions)&valid.source_status.eq('VALID_POSITIVE_VOLUME_SESSION')]
        if valid.empty or valid.date.duplicated().any() or set(valid.date)!=set(sessions):
            raise ValueError('Lagged adjusted source and official intervening sessions differ')
        series=set(alt.series.dropna())
        if len(series)!=1 or set(valid.series)!=series:raise ValueError('Lagged alternative/exchange series changed')
        incoming=valid[['date','symbol','isin','series','open','high','low','close','volume']].copy()
        # Current adjusted close equals the official raw close only for a name
        # absent from the timed NSE action directory. This remains a scoring
        # view and never certifies a no-action interval, entry or outcome.
        incoming['adj_factor_cumulative']=1.0
        incoming['adj_close']=incoming['close']
    if incoming.empty or incoming.date.max()!=pd.Timestamp(asof):raise ValueError('Alternative lacks the completed source session')
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
      'lagged_adjusted_source_extended_with_primary_raw':bool(lagged),
      'action_interval_certified':False,'entry_or_outcome_certified':False}


def single_source_view(alternative, official, sessions, asof, symbol, isin,
                       source_sha256, source_day_action_symbols=(), reference=None):
    """Use one timely adjusted source for a provisional *scoring* view.

    The caller must first validate the capture receipt and the current NSE
    source hash. A lack of another provider is not called independent
    adjustment verification. Any observed overlap disagreement is still a
    conflict. Neither this view nor its receipt can certify an outcome.
    """
    if not re.fullmatch(r'[0-9a-f]{64}', str(source_sha256)):
        raise ValueError('Missing alternative source hash')
    alt = alternative.copy()
    alt['date'] = pd.to_datetime(alt.date)
    if alt.empty or set(alt.symbol) != {symbol} or set(alt['isin']) != {isin}:
        raise ValueError('Single-source identity is ambiguous')
    if not alt.series.isin(['EQ','BE','BZ','SM','ST']).all():
        raise ValueError('Single-source series is unsupported or missing')
    if alt.date.duplicated().any() or (alt.date > pd.Timestamp(asof)).any():
        raise ValueError('Single-source dates are duplicate or future')
    alt = alt.sort_values('date').reset_index(drop=True)
    fields = ['open','high','low','close','adj_close','adj_factor_cumulative','volume']
    for col in fields:
        alt[col] = pd.to_numeric(alt[col], errors='coerce')
    numbers = alt[fields].to_numpy(float)
    if not np.isfinite(numbers).all() or (numbers[:,:6] <= 0).any() or (numbers[:,6] < 0).any():
        raise ValueError('Single-source prices or factors are invalid')
    if not np.allclose(alt.adj_close, alt.close * alt.adj_factor_cumulative,
                       rtol=FACTOR_RTOL, atol=.001):
        raise ValueError('Single-source declared adjustment factors disagree')
    if symbol in set(source_day_action_symbols):
        raise ValueError('Primary exchange corporate action requires adjustment review')

    overlap = 0
    if reference is not None:
        old = reference.copy()
        old['date'] = pd.to_datetime(old.date)
        if old.empty or set(old.symbol) != {symbol} or set(old['isin']) != {isin} or old.date.duplicated().any():
            raise ValueError('Reference history has ambiguous identity')
        joined = old[['date','adj_close']].merge(alt[['date','adj_close']], on='date',
                                                   suffixes=('_reference','_alternative'), validate='1:1')
        joined = joined.sort_values('date').tail(126)
        overlap = len(joined)
        if overlap:
            a = pd.to_numeric(joined.adj_close_reference, errors='coerce').to_numpy(float)
            b = pd.to_numeric(joined.adj_close_alternative, errors='coerce').to_numpy(float)
            if (not np.isfinite(a).all() or (a <= 0).any()
                    or not np.allclose(a, b, rtol=FACTOR_RTOL, atol=.001)):
                raise ValueError('Observed provider adjustment conflict')

    valid = official[(official.symbol == symbol) & (official['isin'] == isin)].copy()
    valid['date'] = pd.to_datetime(valid.date)
    valid = valid[valid.date.isin(sessions) & valid.source_status.eq('VALID_POSITIVE_VOLUME_SESSION')]
    if valid.date.duplicated().any() or set(valid.date) != set(sessions):
        raise ValueError('Single-source exchange session or identity is missing')
    # Past trading-series changes may belong to the same ISIN. Retain them
    # for the user's wider paper scan, with an explicit unresolved-transition
    # flag. The current series must still match the primary exchange.
    if set(valid.series) != {alt.iloc[-1].series}:
        raise ValueError('Single-source current exchange series changed')
    series_transition = alt.series.nunique() > 1
    lagged = alt.date.max() < pd.Timestamp(asof)
    if lagged:
        # Only the immediately following exchange session may be appended:
        # the capture contains a source-day action directory, not an audit of
        # all earlier action intervals.
        if list(sessions) != [pd.Timestamp(asof)]:
            raise ValueError('Single-source lag exceeds one exchange session')
        incoming = valid[['date','symbol','isin','series','open','high','low','close','volume']].copy()
        incoming['adj_close'] = incoming.close
        incoming['adj_factor_cumulative'] = 1.0
        alt = pd.concat([alt, incoming], ignore_index=True)
    current = alt[alt.date.eq(pd.Timestamp(asof))]
    if len(current) != 1:
        raise ValueError('Single-source current session missing')
    current = current.iloc[0]
    original = valid[valid.date.eq(pd.Timestamp(asof))]
    if len(original) != 1:
        raise ValueError('Single-source official current session missing')
    original = original.iloc[0]
    for col in ('open','high','low','close'):
        if abs(float(current[col]) - float(original[col])) > .011:
            raise ValueError('Single-source exchange OHLC conflict')
    if float(current.volume) != float(original.volume):
        raise ValueError('Single-source exchange volume conflict')
    if abs(float(current.adj_factor_cumulative) - 1) > FACTOR_RTOL:
        raise ValueError('Single-source current adjustment factor is not one')
    alt['source_sha256'] = source_sha256
    return alt, {'source_sha256':source_sha256, 'overlap_sessions':overlap,
      'source_status':'SINGLE_SOURCE_ADJUSTED_SCORING_ONLY',
      'reference_status':('OVERLAP_MATCHED' if overlap else 'NO_TIMELY_COMPARABLE_PROVIDER'),
      'historical_series_transition_unverified':bool(series_transition),
      'lagged_adjusted_source_extended_with_primary_raw':bool(lagged),
      'action_interval_certified':False, 'entry_or_outcome_certified':False}
