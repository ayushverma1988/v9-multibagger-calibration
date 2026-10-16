"""Causal breakout diagnostics and separately evaluated research selection.

Thresholds/weights are declared research preferences, not accuracy claims.
Historical adjusted prices remain subject to provider revisions/survivor bias.
No frozen head, registry or historical information cutoff is rewritten.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
import v13_model as model
from v13_research_selection import screen

VERSION='V13_EXTENSION_BREAKOUT_EVIDENCE_RISK_20261016'
MARKET_FEATURES=['rd_return20','rd_return63','rd_extension200','rd_prior_range_ratio',
 'rd_prior_vol_ratio','rd_volume_ratio','rd_breakout_gap','rd_breakout_age',
 'rd_extension_since_breakout','rd_first_breakout']
EVIDENCE_FEATURES=['owner_PAT_yoy','owner_PAT_margin','reported_diluted_EPS_yoy_same_document',
 'order_disclosure_count90','disclosed_order_value_due12_to_sales','verified_recurring_diluted_EPS_growth']
WEIGHTS={'joint':.35,'risk':.15,'price_discipline':.25,'breakout':.10,'financial':.10,'dma_band':.05}

def daily(bars):
    """Use only prior observations for resistance/base/volume denominators."""
    d=bars[bars.volume.gt(0)].copy().sort_values('date').reset_index(drop=True)
    if d.date.duplicated().any():raise ValueError('Duplicate daily dates')
    p=pd.to_numeric(d.adj_close);factor=p/d.close
    hi=d.high*factor;lo=d.low*factor;ret=np.log(p).diff()
    resistance=hi.shift().rolling(63).max()
    range20=(hi.rolling(20).max()-lo.rolling(20).min()).shift()
    range63=(hi.rolling(63).max()-lo.rolling(63).min()).shift()
    vr=ret.rolling(20).std().shift()/ret.rolling(63).std().shift()
    volume=d.volume/d.volume.shift().rolling(20).median().replace(0,np.nan)
    gap=p/resistance-1
    cross=gap.gt(.005)&gap.shift().le(.005)
    first=(cross&cross.shift().rolling(20).sum().eq(0)&gap.shift().rolling(20).count().eq(20)
      &(range20/range63).le(.65)&vr.le(.8)&volume.ge(1.5))
    known=resistance.notna()&vr.notna()&range63.gt(0)&volume.notna()
    event_idx=pd.Series(np.arange(len(d)),index=d.index).where(first).ffill()
    age=pd.Series(np.arange(len(d)),index=d.index)-event_idx
    reference=resistance.where(first).ffill()
    d['rd_return20']=p/p.shift(20)-1;d['rd_return63']=p/p.shift(63)-1
    d['rd_extension200']=p/p.rolling(200).mean()-1
    d['rd_prior_range_ratio']=range20/range63;d['rd_prior_vol_ratio']=vr
    d['rd_volume_ratio']=volume;d['rd_breakout_gap']=gap
    d['rd_breakout_age']=age.where(age.le(20))
    d['rd_extension_since_breakout']=(p/reference-1).where(age.le(20))
    d['rd_first_breakout']=first.astype(float).where(known)
    return d.replace([np.inf,-np.inf],np.nan)

def attach(frame,history):
    """Join exact ISIN/date only; never substitute another date/security."""
    wanted=set(frame['isin']);pieces=[]
    for path in sorted(Path(history).glob('*.parquet')):
        bars=pd.read_parquet(path)
        if bars.empty or not set(bars['isin'])&wanted:continue
        x=daily(bars);pieces.append(x[['date','isin',*MARKET_FEATURES]])
    out=frame.copy();out['date']=pd.to_datetime(out.date)
    if pieces:
        x=pd.concat(pieces,ignore_index=True);x['date']=pd.to_datetime(x.date)
        if x.duplicated(['date','isin']).any():raise ValueError('Ambiguous history identity')
        out=out.merge(x,on=['date','isin'],how='left',validate='one_to_one',sort=False)
    for name in MARKET_FEATURES+EVIDENCE_FEATURES:
        if name not in out:out[name]=np.nan
    return out

def shrink_calibration(raw,y,train_y):
    """Monotone Platt fit regularized toward the training event-rate prior.

    Fit only the final chronological calibration date; no test labels used.
    Conditional heads preserve nested horizons and competing event coherence.
    """
    x=logit(np.clip(np.asarray(raw,float),1e-5,1-1e-5));y=np.asarray(y,float)
    prior=(np.sum(train_y)+1)/(len(train_y)+2)
    center=float(logit(prior)-np.mean(x))
    def loss(z):
        pred=z[0]*x+z[1]
        return np.logaddexp(0,pred).sum()-np.dot(y,pred)+25*(z[0]-1)**2+10*(z[1]-center)**2
    result=minimize(loss,[1.,center],bounds=[(.05,3.),(-15.,15.)],method='L-BFGS-B')
    if not result.success:raise RuntimeError('Research calibration failed: '+result.message)
    return {'method':'regularized_positive_slope_Platt','slope':float(result.x[0]),
      'intercept':float(result.x[1]),'cal_rows':len(y),'cal_events':int(y.sum()),
      'slope_penalty':25,'intercept_penalty':10,'approved':False}

def fit_heads(data,date,variant):
    heads={};clocks=[]
    features=list(dict.fromkeys([*model.FEATURES,*MARKET_FEATURES,*EVIDENCE_FEATURES]))
    for target in model.TARGETS:
        parts=model.parts(data,date,target)
        if parts is None:continue
        train,cal=parts
        p=model.fit(train,cal,target,feature_names=features if variant!='baseline' else None)
        if variant=='redeveloped':
            final=cal[cal.date.eq(cal.date.max())]
            p['calibration']=shrink_calibration(model.raw_score(p,final),final[target],train[target])
        heads[target]=p
        clocks.append({k:p[k] for k in ('target','method','train_rows','cal_rows','train_maturity_max','cal_start','cal_maturity_max','calibration')})
    return heads,clocks

def select(scored):
    d,old=screen(scored)
    def col(name):return pd.to_numeric(d.get(name,pd.Series(np.nan,index=d.index)),errors='coerce')
    eligible=d.research_eligible
    # Missing extension observations receive no positive stability credit.
    extension=pd.concat([(col('rd_return63').clip(lower=0)/.50),
      col('rd_extension200').clip(lower=0)/.40,col('rd_return20').clip(lower=0)/.25],axis=1).clip(0,1)
    discipline=1-extension.mean(axis=1);discipline=discipline.where(extension.notna().all(axis=1),0)
    early=col('rd_breakout_age').between(0,5)&col('rd_extension_since_breakout').between(0,.10)
    observed=col('rd_first_breakout').notna()
    d['redevelopment_breakout_status']=np.select([~observed,early,col('rd_breakout_age').notna()],
      ['UNKNOWN_HISTORY','EARLY_CONFIRMED_BREAKOUT','RECENT_BREAKOUT_NOT_EARLY'],default='NO_CONFIRMED_RECENT_FIRST_BREAKOUT')
    d['redevelopment_price_discipline']=discipline
    d['redevelopment_extension_status']=np.select([~extension.notna().all(axis=1),col('rd_return63').gt(.5)|col('rd_extension200').gt(.4)|col('rd_return20').gt(.25)],
      ['UNKNOWN_EXTENSION','EXTENDED'],default='WITHIN_DECLARED_EXTENSION_LIMITS')
    d['redevelopment_score']=(WEIGHTS['joint']*col('joint12').where(eligible).rank(pct=True)+
      WEIGHTS['risk']*(-col('loss12')).where(eligible).rank(pct=True)+WEIGHTS['price_discipline']*discipline+
      WEIGHTS['breakout']*early+WEIGHTS['financial']*d.research_financial_input_fraction+
      WEIGHTS['dma_band']*d.research_within_200dma_band).where(eligible)
    pool=d[eligible].sort_values(['redevelopment_score','isin'],ascending=[False,True]).head(model.TOP_K)
    d['redevelopment_rank']=d['isin'].map({v:i+1 for i,v in enumerate(pool['isin'])})
    d['redevelopment_selected']=d.redevelopment_rank.notna()
    # Observed announcement counts are discovery cues, never execution proof.
    d['redevelopment_earnings_status']=np.where(col('owner_PAT_yoy').notna()&col('owner_PAT_margin').notna(),
      'OWNER_EARNINGS_OBSERVED_RECURRING_EPS_UNPROVEN','PARTIAL_OR_UNKNOWN_OWNER_EARNINGS')
    d['redevelopment_catalyst_status']=np.where(d.research_execution_supported,'COMPLETE_EXECUTION_SUPPORTED',
      np.where(col('order_disclosure_count90').gt(0),'DATED_ANNOUNCEMENT_ONLY','NO_COMPLETE_EXECUTION_SUPPORT'))
    return d,{'version':VERSION,'weights':WEIGHTS,'weights_are_probabilities':False,
      'baseline_gate_counts':old['gate_counts'],'selected':len(pool),
      'early_breakouts_selected':int(early[d.redevelopment_selected].sum()),
      'source_date':str(pd.Timestamp(d.date.max()).date()),'production_accuracy_approved':False}

def export(scored,root,replay_receipt):
    """Daily diagnostics reuse exact champion probabilities, with new ranks."""
    root=Path(root);d=attach(scored,root/'history'/'securities');d,audit=select(d)
    audit['prediction_replay']=replay_receipt
    audit['probabilities_recalibrated']=False
    audit['records']=json.loads(d.to_json(orient='records',date_format='iso',double_precision=15))
    out=root/'redevelopment';out.mkdir(exist_ok=True)
    raw=json.dumps(audit,sort_keys=True,indent=2).encode();key=hashlib.sha256(raw).hexdigest()
    path=out/(key+'.json')
    if not path.exists():path.write_bytes(raw)
    (out/'latest.json').write_bytes(raw)
    return {k:v for k,v in audit.items() if k not in ('records','prediction_replay')}
