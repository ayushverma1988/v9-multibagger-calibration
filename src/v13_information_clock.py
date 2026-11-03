"""Explicit information clocks for closing-price/next-session forecasts."""
import pandas as pd

def cutoff(row,default_date=None):
    close=pd.Timestamp(row.get('date',default_date)).tz_localize('Asia/Kolkata')+pd.Timedelta(hours=15,minutes=30)
    value=row.get('information_cutoff_utc')
    if value is None or pd.isna(value):return close.tz_convert('UTC')
    clock=pd.Timestamp(value)
    ceiling=close.normalize()+pd.Timedelta(hours=22)
    if clock.tzinfo is None or not close<=clock<=ceiling:raise ValueError('Information cutoff must be aware and between source close and 22:00 IST')
    return clock.tz_convert('UTC')

def research_cutoffs(d,observed_at):
    d=d.copy();seen=pd.Timestamp(observed_at)
    if seen.tzinfo is None:raise ValueError('Observation clock needs timezone')
    dates=pd.to_datetime(d.date).dt.tz_localize('Asia/Kolkata')+pd.Timedelta(hours=22)
    d['information_cutoff_utc']=[min(x,seen).tz_convert('UTC').isoformat() for x in dates]
    for row in d.to_dict('records'):cutoff(row)
    return d
