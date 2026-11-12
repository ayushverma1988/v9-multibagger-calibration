"""Predeclared research ranking candidate; no forecast probability changes.

Weights are preferences for testing, not fitted or calibrated probabilities.
No early-breakout bonus: today's study did not establish one.
"""
import numpy as np
import pandas as pd

VERSION='V13_RISK_EXTENSION_RANKING_CANDIDATE_20261108'
WEIGHTS={'p6_percentile':.60,'joint12_percentile':.20,
         'lower_loss_risk_percentile':.10,'price_discipline':.05,'optional_dma200':.05}

def rank_candidate(frame,eligible,weights=None):
    weights=dict(WEIGHTS if weights is None else weights)
    if set(weights)!=set(WEIGHTS) or any(not np.isfinite(v) or v<0 for v in weights.values()) or not np.isclose(sum(weights.values()),1):
        raise ValueError('Candidate weights must be nonnegative, complete and sum to one')
    d=frame.copy()
    valid=eligible.fillna(False)&d.loss12.between(0,.95)&d.p6.between(0,1)&d.joint12.between(0,1)
    ret20=np.expm1(pd.to_numeric(d.log_ret20,errors='coerce'))
    ret63=np.expm1(pd.to_numeric(d.log_ret63,errors='coerce'))
    dma=pd.to_numeric(d.distance_ma200,errors='coerce')
    extension=pd.concat([ret20.clip(lower=0)/.25,ret63.clip(lower=0)/.50,dma.clip(lower=0)/.40],axis=1).clip(0,1)
    discipline=(1-extension.mean(axis=1)).where(extension.notna().all(axis=1),0.)
    components={'p6_percentile':d.p6.where(valid).rank(pct=True),
        'joint12_percentile':d.joint12.where(valid).rank(pct=True),
        'lower_loss_risk_percentile':(-d.loss12).where(valid).rank(pct=True),
        'price_discipline':discipline,'optional_dma200':dma.abs().le(.10+1e-12).astype(float)}
    d['candidate_price_discipline']=discipline
    d['candidate_extension_known']=extension.notna().all(axis=1)
    d['candidate_score']=sum(weights[k]*v for k,v in components.items()).where(valid)
    d['candidate_rank']=np.nan
    pool=d[valid].sort_values(['candidate_score','p6','joint12','isin'],ascending=[False,False,False,True]).head(25)
    d.loc[pool.index,'candidate_rank']=np.arange(1,len(pool)+1)
    return d

