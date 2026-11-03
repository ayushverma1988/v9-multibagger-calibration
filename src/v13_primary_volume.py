"""Optional research-only final exchange volume overlay with two-source lineage.

Never fills a missing date/price, changes adjusted prices, or claims independent
provider/exchange agreement on volume. Original histories remain untouched.
"""
import re
import numpy as np
import pandas as pd

def reconcile(history,official,asof):
    d=history.copy()
    if d.empty or official.empty:return d,False
    date=pd.Timestamp(asof);ix=d.index[pd.to_datetime(d.date).eq(date)]
    if len(ix)!=1:return d,False
    i=ix[0];r=d.loc[i]
    q=official[(pd.to_datetime(official.date).eq(date))&official['isin'].eq(r['isin'])&official.symbol.eq(r.symbol)]
    if len(q)!=1:return d,False
    o=q.iloc[0]
    if o.source_status!='VALID_POSITIVE_VOLUME_SESSION':return d,False
    if not all(re.fullmatch('[0-9a-f]{64}',str(z)) for z in (r.source_sha256,o.source_sha256)):return d,False
    try:
        clock=pd.Timestamp(o.first_retrieved_utc)
        if clock.tzinfo is None or clock<date.tz_localize('Asia/Kolkata')+pd.Timedelta(hours=15,minutes=30):return d,False
    except (ValueError,TypeError,AttributeError):return d,False
    if not np.isfinite(o.volume) or o.volume<=0 or not np.isfinite(r.adj_close) or r.adj_close<=0:return d,False
    for c in ('open','high','low','close'):
        if not np.isfinite(r[c]) or not np.isfinite(o[c]) or r[c]<=0 or abs(r[c]-o[c])>.011:return d,False
    if r.volume==o.volume:return d,False
    d.loc[i,'provider_original_volume']=float(r.volume)
    d.loc[i,'volume']=o.volume
    d.loc[i,'volume_primary_source_sha256']=o.source_sha256
    d.loc[i,'volume_source']='PRIMARY_FINAL_VOLUME_PROVIDER_OHLC_MATCHED'
    d.loc[i,'volume_available_at_utc']=o.first_retrieved_utc
    return d,True
