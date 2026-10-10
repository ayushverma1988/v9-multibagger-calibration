"""A date-only market outcome becomes observable at that session's close.

The frozen target archive stores many maturity dates without a clock. Treating
them as UTC midnight falsely makes the final session's outcome available at
05:30 IST. Explicit aware timestamps keep their actual reported instant.
"""
from __future__ import annotations
import pandas as pd


def maturity_utc(values):
    series=pd.Series(values,copy=False)
    if pd.api.types.is_datetime64_any_dtype(series.dtype):
        if isinstance(series.dtype,pd.DatetimeTZDtype):
            return series.dt.tz_convert("UTC")
        if (series.notna() & series.ne(series.dt.normalize())).any():
            raise ValueError("Naive intraday maturity needs an explicit source timezone")
        return (series.dt.tz_localize("Asia/Kolkata")+pd.Timedelta(hours=15,minutes=30)).dt.tz_convert("UTC")
    def convert(value):
        if pd.isna(value):return pd.NaT
        try:t=pd.Timestamp(value)
        except (ValueError,TypeError):return pd.NaT
        if pd.isna(t):return pd.NaT
        if t.tzinfo is None:
            if t!=t.normalize():
                raise ValueError("Naive intraday maturity needs an explicit source timezone")
            t=t.tz_localize("Asia/Kolkata")+pd.Timedelta(hours=15,minutes=30)
        return t.tz_convert("UTC")
    return pd.to_datetime(series.map(convert),utc=True,errors="coerce")
