"""Append-only prospective forecasts, complete-cohort scoring, guarded learning."""
from pathlib import Path
import hashlib,json
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss
from scipy.stats import t as student_t
import v12_four_year_model as core

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def register(frame, model_sha, folder, registered_at=None, context=None):
    out=Path(folder);out.mkdir(parents=True,exist_ok=True)
    now=pd.Timestamp(registered_at or datetime.now(timezone.utc))
    if now.tzinfo is None:raise ValueError('Registration needs timezone')
    if frame.duplicated('isin').any():raise ValueError('Duplicate forecast identity')
    records=[]
    for r in frame.to_dict('records'):
        decision=pd.Timestamp(r['date']).tz_localize('Asia/Kolkata')+pd.Timedelta(hours=15,minutes=30)
        if decision>now:raise ValueError('Future decision')
        values=[float(r[c]) for c in ('p6','p12','loss12','joint12')]
        if not all(np.isfinite(v) and 0<=v<=1 for v in values):raise ValueError('Incomplete forecast heads')
        if values[0]>values[1]+1e-12 or values[3]>values[1]+1e-12 or values[3]+values[2]>1+1e-12:raise ValueError('Incoherent probabilities')
        records.append({'isin':r['isin'],'symbol':r['symbol'],'source_asof':str(pd.Timestamp(r['date']).date()),
          'p6':float(r['p6']),'p12':float(r['p12']),'loss12':float(r['loss12']),'joint12':float(r['joint12']),
          'selected_verified':bool(r.get('selected_verified',False)),'selected_discovery':bool(r.get('selected_discovery',False)),
          **{k:r[k] for k in ('financial_tier','market_regime','liquidity_tier','evidence_status','baseline_p6','baseline_p12','series','source_close_INR','official_source_sha256') if k in r}})
    obj={'version':'V13-prospective-research','registered_at_utc':now.isoformat(),'model_sha256':model_sha,
      'first_entry':'FIRST_VERIFIED_NSE_SESSION_OPEN_STRICTLY_AFTER_REGISTRATION',
      'all_scoreable_NSE_names_registered':True,'accuracy_approved':False,'records':sorted(records,key=lambda r:r['isin'])}
    if context:obj['paired_context']=context
    key=digest(obj);path=out/(key+'.json')
    if not path.exists():path.write_text(json.dumps(obj,indent=2,allow_nan=False))
    elif json.loads(path.read_text())!=obj:raise ValueError('Registry changed')
    return path

def validate_registry(path):
    obj=json.loads(Path(path).read_text())
    if Path(path).stem!=digest(obj):raise ValueError('Registry hash mismatch')
    return obj

def adjudicate(registry,official,history,asof,verification=None):
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
          'cohort_id':r.get('paired_context',{}).get('cohort_id'),
          'comparison_contract':r.get('paired_context',{}).get('comparison_contract','V13_LEGACY_PAIR_20261012'),
          'status':'UNKNOWN_ENTRY','first_entry_date':None,
          'raw_price_2x_observed':False,'six_month_mature':False,'twelve_month_mature':False,
          'independent_outcome_verified':False,'entry_verified':False,
          'provisional_hit2_6':None,'provisional_hit2_12':None,'provisional_loss30_12':None,
          'verified_hit2_6':None,'verified_hit2_12':None,'verified_loss30_12':None,
          'confirmed_early_2x':False,'confirmed_risk_event':False}
        if not q.empty:
            entry=q.iloc[0];row['first_entry_date']=str(entry.date.date());row['status']='TRACKING'
            horizon=q[q.date.le(entry.date+pd.DateOffset(months=12))]
            row['raw_price_2x_observed']=bool((horizon.close/entry.open*(1-core.COST_PER_SIDE)/(1+core.COST_PER_SIDE)>=2).any())
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
                                if h==12:row['provisional_loss30_12']=result['loss30']
                        if row['six_month_mature'] or row['twelve_month_mature']:
                            row['status']='PROVISIONAL_MATURE_REQUIRES_INDEPENDENT_ACTION_RECONCILIATION'
            if verification is not None:
                proof=verification.get((Path(registry).stem,item['isin']),{})
                row.update({k:v for k,v in proof.items() if k in {
                  'entry_verified','independent_outcome_verified','verification_status',
                  'verified_hit2_6','verified_hit2_12','verified_loss30_12',
                  'confirmed_early_2x','confirmed_risk_event','action_audit_sha256'}})
                if proof.get('entry_verified') is True:
                    if row['status']=='UNKNOWN_ENTRY_CONFLICT' and proof.get('verification_status')=='INDEPENDENT_PRIMARY_PRICES_ACTIONS_AND_PROVIDER_FACTORS_MATCHED':
                        row['status']='INDEPENDENT_PRIMARY_ENTRY_PROVIDER_ACTION_RECONCILED'
                    if proof.get('independent_outcome_verified') is True:
                        for h in (6,12):
                            if proof.get('verified_hit2_'+str(h)) in (0,1):row['six_month_mature' if h==6 else 'twelve_month_mature']=True
        rows.append(row)
    return rows

PAIRED_COLUMNS=['date','isin','cohort_id','registered_at_utc','mature_at_utc','y','champion','challenger',
 'selected_champion','selected_challenger','independent_outcome_verified','financial_tier','market_regime','liquidity_tier','baseline']

def paired_observations(observations,challenger_sha,comparison_contract=None):
    d=observations.copy()
    empty=pd.DataFrame(columns=PAIRED_COLUMNS)
    if d.empty:return empty
    required={'cohort_id','model_role','model_sha256','registry_sha256','registered_at_utc','source_asof','isin','p12','selected_verified','verified_hit2_12','twelve_month_mature','first_entry_date','independent_outcome_verified'}
    if not required.issubset(d):return empty
    d=d[d.cohort_id.notna()]
    if comparison_contract is not None:
        if 'comparison_contract' not in d:d['comparison_contract']='V13_LEGACY_PAIR_20261012'
        d=d[d.comparison_contract.eq(comparison_contract)]
    d=d[d.twelve_month_mature.eq(True)&d.status.ne('UNKNOWN_ENTRY_CONFLICT')]
    b=d[d.model_role.eq('challenger')&d.model_sha256.eq(challenger_sha)]
    a=d[d.model_role.eq('champion')]
    keys=['cohort_id','source_asof','isin']
    if a.duplicated(keys).any() or b.duplicated(keys).any():raise ValueError('Duplicate paired observation')
    x=a.merge(b,on=keys,suffixes=('_a','_b'),validate='1:1')
    if x.empty:return empty
    x=x[x.registered_at_utc_a.eq(x.registered_at_utc_b)&x.first_entry_date_a.eq(x.first_entry_date_b)&x.verified_hit2_12_a.eq(x.verified_hit2_12_b)&x.independent_outcome_verified_a&x.independent_outcome_verified_b]
    if x.empty:return empty
    return pd.DataFrame({'date':x.source_asof,'isin':x['isin'],'cohort_id':x.cohort_id,'registered_at_utc':x.registered_at_utc_b,
      'mature_at_utc':pd.to_datetime(x.first_entry_date_b,utc=True)+pd.DateOffset(months=12),
      'y':x.verified_hit2_12_b,'champion':x.p12_a,'challenger':x.p12_b,
      'baseline':x.baseline_p12_a if 'baseline_p12_a' in x else np.nan,
      'selected_champion':x.selected_verified_a,'selected_challenger':x.selected_verified_b,
      'independent_outcome_verified':x.independent_outcome_verified_a&x.independent_outcome_verified_b,
      **{k:x[k+'_a'] if k+'_a' in x else 'UNKNOWN' for k in ('financial_tier','market_regime','liquidity_tier')}})

def promotion_gate(paired,registered_after,source_complete=False,require_baseline=False):
    """One untouched, date-clustered prospective comparison, never winner-only."""
    reasons=[];required={'date','registered_at_utc','mature_at_utc','y','champion','challenger','selected_champion','selected_challenger','independent_outcome_verified'}
    if not required.issubset(paired):return {'promote':False,'reasons':['missing_paired_prospective_schema']}
    d=paired.copy();d['date']=pd.to_datetime(d.date)
    d['registered_at_utc']=pd.to_datetime(d.registered_at_utc,utc=True);d['mature_at_utc']=pd.to_datetime(d.mature_at_utc,utc=True)
    cutoff=pd.Timestamp(registered_after)
    if cutoff.tzinfo is None:raise ValueError('Challenger freeze clock needs timezone')
    d=d[d.registered_at_utc.gt(cutoff)&d.mature_at_utc.le(pd.Timestamp.now(tz='UTC'))&d.y.isin([0,1])]
    # A month contributes its first registered decision cohort only. Repeated
    # daily forecasts of substantially the same names are not six independent
    # months of evidence. Overlapping horizons still limit inference strength.
    first_dates=d.groupby(d.date.dt.to_period('M')).date.transform('min')
    d=d[d.date.eq(first_dates)]
    if not source_complete:reasons.append('source_or_action_coverage_incomplete')
    if not d.independent_outcome_verified.all():reasons.append('unverified_outcomes')
    if d.date.dt.to_period('M').nunique()<6:reasons.append('fewer_than_six_mature_decision_months')
    if len(d)<500:reasons.append('fewer_than_500_mature_paired_rows')
    if d.y.eq(1).sum()<20:reasons.append('fewer_than_20_positive_outcomes')
    for label in ('champion','challenger'):
        if d['selected_'+label].sum()<60:reasons.append('fewer_than_60_selected_'+label)
    if require_baseline:
        if d.empty or 'baseline' not in d or not d.baseline.between(0,1).all():reasons.append('missing_verified_future_frozen_baseline_comparison')
        else:
            gain=((d.baseline-d.y)**2-(d.challenger-d.y)**2).groupby(d.date).mean()
            critical=student_t.ppf(.975,len(gain)-1) if len(gain)>1 else np.inf
            lower=float(gain.mean()-critical*gain.std(ddof=1)/np.sqrt(len(gain))) if len(gain)>1 else -1
            if not lower>0:reasons.append('date_clustered_gain_over_simple_baseline_not_positive')
    if len(d):
        delta=(d.champion-d.y)**2-(d.challenger-d.y)**2
        means=delta.groupby(d.date).mean()
        critical=student_t.ppf(.975,len(means)-1) if len(means)>1 else np.inf
        lower=float(means.mean()-critical*means.std(ddof=1)/np.sqrt(len(means))) if len(means)>1 else -1
        if not lower>0:reasons.append('date_clustered_brier_gain_not_positive')
        a=d[d.selected_champion];b=d[d.selected_challenger]
        if a.empty or b.empty or b.y.mean()<a.y.mean()+.02:reasons.append('precision_gain_below_two_percentage_points')
    return {'promote':not reasons,'reasons':reasons,'mature_dates':int(d.date.nunique()),'mature_rows':len(d)}
