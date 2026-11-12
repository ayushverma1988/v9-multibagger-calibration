"""Append an actual NSE session only over a verified action-free interval.

This is a separate single-source research view. It does not manufacture a
vendor observation or modify the preserved historical source files.
"""
import numpy as np
import pandas as pd

ANCHOR=pd.Timestamp('2026-11-09')
TARGET=pd.Timestamp('2026-11-11')

def extend(bars,anchor,current,action_symbols,feed_verified,calendar_verified):
    if not feed_verified or not calendar_verified:raise ValueError('Missing interval evidence')
    d=bars.copy().sort_values('date').reset_index(drop=True)
    d['date']=pd.to_datetime(d.date)
    if d.empty or d.date.duplicated().any():raise ValueError('Invalid history')
    if d.date.max()!=ANCHOR:raise ValueError('History must end on exact anchor session')
    if pd.Timestamp(anchor['date'])!=ANCHOR or pd.Timestamp(current['date'])!=TARGET:raise ValueError('Wrong primary session')
    last=d.iloc[-1]
    for c in ['isin','symbol']:
        if d[c].nunique()!=1 or last[c]!=anchor[c] or last[c]!=current[c]:raise ValueError('Identity changed or ambiguous')
    if anchor['series'] not in {'EQ','BE','BZ'} or current['series'] not in {'EQ','BE','BZ'}:
        raise ValueError('Action feed scope is main-board equity only')
    if last['symbol'] in action_symbols:raise ValueError('Corporate action requires explicit adjustment')
    for q in [anchor,current]:
        if q['source_status']!='VALID_POSITIVE_VOLUME_SESSION':raise ValueError('Invalid primary session')
    if any(abs(float(last[c])-float(anchor[c]))>.011 for c in ['open','high','low','close']):
        raise ValueError('Anchor raw prices differ')
    if last['volume']!=anchor['volume']:raise ValueError('Anchor volume differs')
    factor=float(last['adj_close'])/float(last['close'])
    if not np.isfinite(factor) or factor<=0:raise ValueError('Invalid original adjustment scale')
    row={c:last[c] for c in d.columns}
    row.update({c:current[c] for c in ['date','isin','symbol','open','high','low','close','volume']})
    row['date']=pd.Timestamp(current['date'])
    row['adj_close']=float(current['close'])*factor
    row['source_sha256']=current['source_sha256']
    result=pd.concat([d,pd.DataFrame([row])],ignore_index=True)
    return result,{'method':'NSE_PRIMARY_SESSION_APPENDED_WITH_VERIFIED_NO_ACTION_INTERVAL',
        'anchor_date':str(ANCHOR.date()),'new_date':str(TARGET.date()),'isin':last['isin'],'symbol':last['symbol'],
        'anchor_primary_sha256':anchor['source_sha256'],'new_primary_sha256':current['source_sha256'],
        'original_adjustment_scale':factor,'latest_session_independently_vendor_verified':False,
        'single_primary_source_for_new_bar':True,'old_history_values_changed':False}
