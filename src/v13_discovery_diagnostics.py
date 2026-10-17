"""Causal breakout diagnosis, separate from frozen probabilities and gates."""
import numpy as np
import pandas as pd
import v13_redevelopment as rd
FEATURES=['dd_rsi_cross_age','dd_return_since_rsi_cross','dd_cross_age','dd_cross_extension',
          'dd_base_range','dd_base_volatility','dd_volume_confirmed','dd_prior_crosses','dd_first_breakout']
def daily(bars):
    d=rd.daily(bars);p=d.adj_close;change=p.diff()
    gains=change.clip(lower=0).rolling(14).mean();losses=-change.clip(upper=0).rolling(14).mean()
    rsi=(100-100/(1+gains/losses)).where(~((gains==0)&(losses==0)))
    index=pd.Series(np.arange(len(d)),index=d.index,dtype=float)
    rc=rsi.gt(70)&rsi.shift().le(70)
    d['dd_rsi_cross_age']=index-index.where(rc).ffill()
    d['dd_return_since_rsi_cross']=p/p.where(rc).ffill()-1
    gap=d.rd_breakout_gap;cross=gap.gt(.005)&gap.shift().le(.005)
    prior=cross.shift().rolling(20).sum().where(gap.shift().rolling(20).count().eq(20))
    age=index-index.where(cross).ffill();d['dd_cross_age']=age.where(age.le(20))
    reference=(p/(1+gap)).where(cross).ffill()
    d['dd_cross_extension']=(p/reference-1).where(age.le(20))
    for source,target,threshold in [('rd_prior_range_ratio','dd_base_range',.65),('rd_prior_vol_ratio','dd_base_volatility',.8)]:
        d[target]=d[source].le(threshold).astype(float).where(d[source].notna())
    d['dd_volume_confirmed']=d.rd_volume_ratio.ge(1.5).astype(float).where(d.rd_volume_ratio.notna())
    d['dd_prior_crosses']=prior;d['dd_first_breakout']=d.rd_first_breakout.where(prior.notna())
    reasons=pd.Series(None,index=d.index,dtype=object)
    for i in d.index[cross]:
        failed=[]
        if pd.isna(prior.loc[i]):failed.append('INSUFFICIENT_RESISTANCE_WARMUP')
        elif prior.loc[i]>0:failed.append('REPEAT_RESISTANCE_CROSS')
        for c,label in [('dd_base_range','BASE_RANGE_NOT_COMPRESSED'),('dd_base_volatility','BASE_VOLATILITY_NOT_COMPRESSED'),('dd_volume_confirmed','VOLUME_NOT_CONFIRMED')]:
            if pd.isna(d.loc[i,c]):failed.append('UNKNOWN_'+label)
            elif d.loc[i,c]==0:failed.append(label)
        reasons.loc[i]='|'.join(failed) if failed else 'FIRST_BREAKOUT_CONFIRMED'
    reason=reasons.ffill().where(age.le(20))
    d['dd_breakout_reason']=reason.fillna('NO_RESISTANCE_CROSS_IN_LAST_20_SESSIONS')
    d.loc[gap.isna(),'dd_breakout_reason']='UNKNOWN_HISTORY'
    d['dd_breakout_state']=np.select([d.dd_first_breakout.eq(1),d.rd_breakout_age.between(0,5)&d.rd_extension_since_breakout.between(0,.10),d.dd_cross_age.notna()&reason.eq('FIRST_BREAKOUT_CONFIRMED'),d.dd_cross_age.notna(),gap.isna()],['FIRST_BREAKOUT_TODAY','EARLY_FIRST_BREAKOUT','OLDER_OR_EXTENDED_FIRST_BREAKOUT','UNCONFIRMED_OR_CONTINUATION_CROSS','UNKNOWN_HISTORY'],default='NO_RECENT_CROSS')
    return d.replace([np.inf,-np.inf],np.nan)
def attach(frame,history):
    from pathlib import Path
    cols=list(dict.fromkeys([*rd.MARKET_FEATURES,*FEATURES,'dd_breakout_reason','dd_breakout_state']))
    wanted=set(frame['isin']);pieces=[]
    for path in sorted(Path(history).glob('*.parquet')):
        b=pd.read_parquet(path)
        if b.empty or not set(b['isin'])&wanted:continue
        pieces.append(daily(b)[['date','isin',*cols]])
    out=frame.drop(columns=cols,errors='ignore').copy();out['date']=pd.to_datetime(out.date)
    if not pieces:
        for c in cols:out[c]=np.nan
        return out
    x=pd.concat(pieces,ignore_index=True);x['date']=pd.to_datetime(x.date)
    if x.duplicated(['date','isin']).any():raise ValueError('Ambiguous history identity')
    return out.merge(x,on=['date','isin'],how='left',validate='one_to_one')
