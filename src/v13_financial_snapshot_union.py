"""Union independent source snapshots without losing known observations.

Exact original/period/basis/availability keys only. Conflicting numeric
extractions of the same original become unknown in the derived union and
remain preserved in their original snapshots for review.
"""
import json
import numpy as np
import pandas as pd

KEYS=['source_sha256','period_kind','period_end','security_isin','reporting_mode','available_at_utc']

def union(primary,supplement):
    if primary.duplicated(KEYS).any() or supplement.duplicated(KEYS).any():raise ValueError('Duplicate original financial snapshot identities')
    a=primary.set_index(KEYS);b=supplement.set_index(KEYS)
    result=a.combine_first(b);common=a.index.intersection(b.index);conflicts=[];blocked={}
    for previous in (a,b):
        if '_union_conflict_metrics' not in previous:continue
        for key,value in previous['_union_conflict_metrics'].dropna().items():
            if value!='[]':blocked.setdefault(key,set()).update(json.loads(value))
    numeric=set(primary.select_dtypes(include='number'))&set(supplement.select_dtypes(include='number'))
    for c in sorted(numeric):
        av=a.loc[common,c];bv=b.loc[common,c]
        different=av.notna()&bv.notna()&~np.isclose(av.to_numpy(float),bv.to_numpy(float),rtol=1e-10,atol=1e-6,equal_nan=True)
        for key in common[different]:
            conflicts.append({**dict(zip(KEYS,key)),'metric':c,'primary_extraction':float(a.loc[key,c]),'supplement_extraction':float(b.loc[key,c])})
            blocked.setdefault(key,set()).add(c)
        result.loc[common[different],c]=np.nan
    if blocked:
        result['_union_conflict_metrics']='[]'
        for key,metrics in blocked.items():
            for metric in metrics:result.loc[key,metric]=np.nan
            result.loc[key,'_union_conflict_metrics']=json.dumps(sorted(metrics))
    return result.reset_index(),conflicts
