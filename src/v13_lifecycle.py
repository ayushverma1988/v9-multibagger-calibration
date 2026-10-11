"""Append-only prospective forecasts, complete-cohort scoring, guarded learning."""
from pathlib import Path
import hashlib,json
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss
import v12_four_year_model as core

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def register(frame, model_sha, folder, registered_at=None):
    out=Path(folder);out.mkdir(parents=True,exist_ok=True)
    now=pd.Timestamp(registered_at or datetime.now(timezone.utc))
    if now.tzinfo is None:raise ValueError('Registration needs timezone')
    if frame.duplicated('isin').any():raise ValueError('Duplicate forecast identity')
    records=[]
    for r in frame.to_dict('records'):
        decision=pd.Timestamp(r['date']).tz_localize('Asia/Kolkata')+pd.Timedelta(hours=15,minutes=30)
        if decision>now:raise ValueError('Future decision')
        records.append({'isin':r['isin'],'symbol':r['symbol'],'source_asof':str(pd.Timestamp(r['date']).date()),
          'p6':float(r['p6']),'p12':float(r['p12']),'loss12':float(r['loss12']),'joint12':float(r['joint12']),
          'selected_verified':bool(r.get('selected_verified',False)),'selected_discovery':bool(r.get('selected_discovery',False))})
    obj={'version':'V13-prospective-research','registered_at_utc':now.isoformat(),'model_sha256':model_sha,
      'first_entry':'FIRST_VERIFIED_NSE_SESSION_OPEN_STRICTLY_AFTER_REGISTRATION',
      'all_scoreable_NSE_names_registered':True,'accuracy_approved':False,'records':sorted(records,key=lambda r:r['isin'])}
    key=digest(obj);path=out/(key+'.json')
    if not path.exists():path.write_text(json.dumps(obj,indent=2,allow_nan=False))
    elif json.loads(path.read_text())!=obj:raise ValueError('Registry changed')
    return path

def validate_registry(path):
    obj=json.loads(Path(path).read_text())
    if Path(path).stem!=digest(obj):raise ValueError('Registry hash mismatch')
    return obj

def adjudicate(registry,official,history,asof):
    """Mature provider labels stay provisional until independent action proof.

    Missing sessions/identities are UNKNOWN; a same-day entry is impossible.
    Monitoring a raw-price doubling is separate from mature target success.
    """
    r=validate_registry(registry);clock=pd.Timestamp(r['registered_at_utc']);rows=[]
    official=official.copy();official['date']=pd.to_datetime(official.date)
    open_clocks=official.date.dt.tz_localize('Asia/Kolkata')+pd.Timedelta(hours=9,minutes=15)
    official=official[open_clocks.gt(clock)&official.source_status.eq('VALID_POSITIVE_VOLUME_SESSION')]
    for item in r['records']:
        q=official[official['isin'].eq(item['isin'])].sort_values('date')
        row={**item,'registry_sha256':Path(registry).stem,'model_sha256':r['model_sha256'],
          'registered_at_utc':r['registered_at_utc'],'model_role':Path(registry).parent.name,
          'status':'UNKNOWN_ENTRY','first_entry_date':None,
          'raw_price_2x_observed':False,'six_month_mature':False,'twelve_month_mature':False,
          'independent_outcome_verified':False}
        if not q.empty:
            entry=q.iloc[0];row['first_entry_date']=str(entry.date.date());row['status']='TRACKING'
            row['raw_price_2x_observed']=bool((q.close/entry.open*(1-core.COST_PER_SIDE)/(1+core.COST_PER_SIDE)>=2).any())
            source=history.get(item['isin'])
            if source is not None:
                source=source.sort_values('date').reset_index(drop=True);source['date']=pd.to_datetime(source.date)
                ix=source.index[source.date.eq(entry.date)]
                expected=source[(source.date.dt.tz_localize('Asia/Kolkata')+pd.Timedelta(hours=9,minutes=15)).gt(clock)]
                # The adjusted target starts at the observed entry, not at the
                # earlier source-asof date of a weekend registration.
                if len(ix)==1 and ix[0]>0:
                    check=source.loc[ix[0]]
                    if expected.empty or expected.date.iloc[0]!=entry.date:
                        row['status']='UNKNOWN_MISSING_FIRST_SESSION'
                    elif check.symbol!=entry.symbol or abs(check.open-entry.open)>.011:
                        row['status']='UNKNOWN_ENTRY_CONFLICT'
                    else:
                        for h in (6,12):
                            result=core.forward_outcome(source,int(ix[0])-1,h,asof)
                            if result:
                                row['six_month_mature' if h==6 else 'twelve_month_mature']=True
                                row['provisional_hit2_'+str(h)]=result['hit2']
                        if row['six_month_mature'] or row['twelve_month_mature']:
                            row['status']='PROVISIONAL_MATURE_REQUIRES_INDEPENDENT_ACTION_RECONCILIATION'
        rows.append(row)
    return rows

def paired_observations(observations,challenger_sha):
    d=observations.copy()
    if d.empty:return pd.DataFrame()
    required={'model_role','model_sha256','registry_sha256','registered_at_utc','source_asof','isin','p12','selected_verified','provisional_hit2_12','twelve_month_mature','first_entry_date','independent_outcome_verified'}
    if not required.issubset(d):return pd.DataFrame()
    d=d[d.twelve_month_mature.eq(True)&d.status.ne('UNKNOWN_ENTRY_CONFLICT')]
    b=d[d.model_role.eq('challenger')&d.model_sha256.eq(challenger_sha)]
    a=d[d.model_role.eq('champion')]
    if a.duplicated(['source_asof','isin']).any() or b.duplicated(['source_asof','isin']).any():return pd.DataFrame()
    x=a.merge(b,on=['source_asof','isin'],suffixes=('_a','_b'),validate='1:1')
    if x.empty:return pd.DataFrame()
    x=x[x.first_entry_date_a.eq(x.first_entry_date_b)&x.provisional_hit2_12_a.eq(x.provisional_hit2_12_b)]
    return pd.DataFrame({'date':x.source_asof,'registered_at_utc':x.registered_at_utc_b,
      'mature_at_utc':pd.to_datetime(x.first_entry_date_b,utc=True)+pd.DateOffset(months=12),
      'y':x.provisional_hit2_12_b,'champion':x.p12_a,'challenger':x.p12_b,
      'selected_champion':x.selected_verified_a,'selected_challenger':x.selected_verified_b,
      'independent_outcome_verified':x.independent_outcome_verified_a&x.independent_outcome_verified_b})

def promotion_gate(paired,registered_after,source_complete=False):
    """One untouched, date-clustered prospective comparison, never winner-only."""
    reasons=[];required={'date','registered_at_utc','mature_at_utc','y','champion','challenger','selected_champion','selected_challenger','independent_outcome_verified'}
    if not required.issubset(paired):return {'promote':False,'reasons':['missing_paired_prospective_schema']}
    d=paired.copy();d['date']=pd.to_datetime(d.date)
    d['registered_at_utc']=pd.to_datetime(d.registered_at_utc,utc=True);d['mature_at_utc']=pd.to_datetime(d.mature_at_utc,utc=True)
    cutoff=pd.Timestamp(registered_after)
    if cutoff.tzinfo is None:raise ValueError('Challenger freeze clock needs timezone')
    d=d[d.registered_at_utc.gt(cutoff)&d.mature_at_utc.le(pd.Timestamp.now(tz='UTC'))&d.y.isin([0,1])]
    if not source_complete:reasons.append('source_or_action_coverage_incomplete')
    if not d.independent_outcome_verified.all():reasons.append('unverified_outcomes')
    if d.date.nunique()<6:reasons.append('fewer_than_six_independent_decision_dates')
    if len(d)<500:reasons.append('fewer_than_500_mature_paired_rows')
    if d.y.eq(1).sum()<20:reasons.append('fewer_than_20_positive_outcomes')
    for label in ('champion','challenger'):
        if d['selected_'+label].sum()<60:reasons.append('fewer_than_60_selected_'+label)
    if len(d):
        delta=(d.champion-d.y)**2-(d.challenger-d.y)**2
        means=delta.groupby(d.date).mean();lower=float(means.mean()-1.96*means.std(ddof=1)/np.sqrt(len(means))) if len(means)>1 else -1
        if not lower>0:reasons.append('date_clustered_brier_gain_not_positive')
        a=d[d.selected_champion];b=d[d.selected_challenger]
        if a.empty or b.empty or b.y.mean()<a.y.mean()+.02:reasons.append('precision_gain_below_two_percentage_points')
    return {'promote':not reasons,'reasons':reasons,'mature_dates':int(d.date.nunique()),'mature_rows':len(d)}
