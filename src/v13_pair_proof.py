"""Fail-closed, read-only comparison of independently observed entry proofs."""
import numpy as np
import pandas as pd

def entry_match(a,b):
    required={'entry_verified','first_entry_date','entry_open_INR','entry_official_sha256','entry_provider_sha256'}
    if not required.issubset(a) or not required.issubset(b):return pd.Series(False,index=a.index)
    good=a.entry_verified.eq(True)&b.entry_verified.eq(True)
    good&=a.first_entry_date.notna()&a.first_entry_date.eq(b.first_entry_date)
    x=pd.to_numeric(a.entry_open_INR,errors='coerce');y=pd.to_numeric(b.entry_open_INR,errors='coerce')
    good&=np.isfinite(x)&np.isfinite(y)&x.gt(0)&y.gt(0)&x.sub(y).abs().le(.011)
    for key in ('entry_official_sha256','entry_provider_sha256'):
        good&=a[key].astype(str).str.fullmatch('[0-9a-f]{64}')&a[key].eq(b[key])
    return good.fillna(False)

def same_metadata(a,b):
    good=pd.Series(True,index=a.index)
    for key in ('comparison_contract','information_cutoff_utc'):
        if key in a or key in b:
            if key not in a or key not in b:return pd.Series(False,index=a.index)
            good&=a[key].notna()&a[key].eq(b[key])
    return good.fillna(False)

def baseline_match(a,b,h):
    key='baseline_p'+str(h)
    if key not in a or key not in b:return pd.Series(False,index=a.index)
    x=pd.to_numeric(a[key],errors='coerce');y=pd.to_numeric(b[key],errors='coerce')
    return (x.between(0,1)&y.between(0,1)&x.sub(y).abs().le(1e-12)).fillna(False)

def sides(joined):
    return tuple(joined[[c for c in joined if c.endswith('_'+s)]].rename(columns=lambda c:c[:-2]) for s in ('a','b'))
