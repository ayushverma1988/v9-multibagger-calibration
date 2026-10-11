"""NSE coverage expansion and a fixed, causal feature catalogue.

No pre-boundary warmup, future sector memberships, or fabricated fundamentals.
The original dated feature matrix is preserved; new market rows have UNKNOWN
fundamentals until an exact-ISIN, dated filing supplies them.
"""
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
import v12_four_year_model as core
from v12_public_history import bounds

MIN_TURNOVER = 2_500_000  # INR 25 lakh; an execution constraint, not a fitted cutoff.
EXTRA = ('momentum126', 'momentum252_skip21', 'relative_strength126',
 'relative_strength252_skip21', 'volatility_contraction', 'range_compression',
 'atr14_fraction', 'close_location20', 'accumulation_distribution63',
 'turnover_growth20_63', 'amihud_illiquidity63', 'breakout_volume_interaction',
 'trend_regime_interaction', 'earnings_market_confirmation')

def daily_features(bars, asof):
    d=core.make_security_features(bars,asof)
    p=d.adj_close;ret=np.log(p).diff();factor=p/d.close
    hi=d.high*factor;lo=d.low*factor;previous=p.shift(1)
    tr=pd.concat([hi-lo,(hi-previous).abs(),(lo-previous).abs()],axis=1).max(axis=1)
    d['momentum126']=np.log(p/p.shift(126))
    d['momentum252_skip21']=np.log(p.shift(21)/p.shift(252))
    d['volatility_contraction']=ret.rolling(20).std()/ret.rolling(63).std()
    span20=hi.rolling(20).max()-lo.rolling(20).min()
    span63=hi.rolling(63).max()-lo.rolling(63).min()
    d['range_compression']=span20/span63.replace(0,np.nan)
    d['atr14_fraction']=tr.rolling(14).mean()/p
    location=(2*p-hi-lo)/(hi-lo).replace(0,np.nan)
    d['close_location20']=location.rolling(20,min_periods=15).mean()
    d['accumulation_distribution63']=(location*d.volume).rolling(63,min_periods=50).sum()/d.volume.rolling(63).sum()
    turnover=d.close*d.volume
    d['turnover_growth20_63']=np.log(turnover.rolling(20).mean()/turnover.rolling(63).mean())
    d['amihud_illiquidity63']=(ret.abs()/turnover).rolling(63).mean()*1e7
    d['breakout_close_vs_prior63_high']=p/hi.shift(1).rolling(63).max()-1
    d['breakout_volume_interaction']=d.breakout_close_vs_prior63_high*d.volume_confirmation
    d['feature_eligible']=(d.median_turnover63.ge(MIN_TURNOVER)&d.median_turnover20.ge(MIN_TURNOVER)
       &d.gap_max63.le(10)&d.jump_max63.le(np.log(1.4))&d.observed_sessions35d.ge(18)&d.history_start_used.notna())
    return d.replace([np.inf,-np.inf],np.nan)

def build(history_path, previous_matrix, output, asof):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    old=pd.read_parquet(previous_matrix).set_index(['date','isin'])
    sources=[pd.read_parquet(p) for p in sorted((Path(history_path)/'securities').glob('*.parquet'))]
    counts=pd.concat([x[['date']] for x in sources]).groupby('date').size()
    calendar=pd.DatetimeIndex(counts[counts.ge(20)].index).sort_values()
    decisions=set(pd.Series(calendar,index=calendar).resample('ME').max().dropna())|{pd.Timestamp(asof)}
    records=[]
    for i,source in enumerate(sources,1):
        d=daily_features(source,asof)
        for t in d.index[d.date.isin(decisions)&d.feature_eligible]:
            row=d.loc[t].to_dict();key=(row['date'],row['isin'])
            if key in old.index:
                prev=old.loc[key].to_dict()
                row={**prev,**row}
            else:
                for months in (6,12):
                    result=core.forward_outcome(d,t,months,asof)
                    for field in ('hit2','endpoint_return','peak_return','loss30','hit2_before_loss30','mature_date'):
                        row[field+'_'+str(months)]=result[field] if result else None
            records.append(row)
        if i%300==0:print('V13 market issuers',i,'rows',len(records),flush=True)
    x=pd.DataFrame(records)
    if x.duplicated(['date','isin']).any():raise ValueError('Ambiguous security identity')
    for c in old.columns:
        if c not in x:x[c]=np.nan
    groups=x.groupby('date')
    for source,target in [('log_ret63','relative_strength63'),('log_ret20','relative_strength20'),
       ('momentum126','relative_strength126'),('momentum252_skip21','relative_strength252_skip21')]:
        x[target]=groups[source].rank(pct=True)
    x['market_breadth63']=groups.log_ret63.transform(lambda s:s.gt(0).mean())
    x['market_median_ret63']=groups.log_ret63.transform('median')
    x['market_dispersion63']=groups.log_ret63.transform('std')
    x['trend_regime_interaction']=x.relative_strength63*(x.market_breadth63-.5)
    if 'quarter_pat_yoy' not in x:x['quarter_pat_yoy']=np.nan
    x['earnings_market_confirmation']=x.quarter_pat_yoy*x.relative_strength63
    x['liquidity_volume_interaction']=x.volume_confirmation*x.log_liquidity63
    for c in ('mature_date_6','mature_date_12'):x[c]=pd.to_datetime(x[c])
    x.to_parquet(out/'matrix.parquet',index=False)
    summary={'asof':asof,'four_year_start':str(bounds(asof)[0].date()),'rows':len(x),
       'current_scored':int(x.date.eq(pd.Timestamp(asof)).sum()),'minimum_turnover_INR':MIN_TURNOVER,
       'new_features':list(EXTRA),'new_row_fundamentals':'UNKNOWN_until_dated_original_filing',
       'survivor_bias_current_listings':True,'corporate_actions_independently_verified':False}
    (out/'coverage.json').write_text(json.dumps(summary,indent=2));print(summary,flush=True)
    return x

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--history',required=True);p.add_argument('--previous-matrix',required=True);p.add_argument('--output',required=True);p.add_argument('--asof',required=True);a=p.parse_args()
    build(a.history,a.previous_matrix,a.output,a.asof)
