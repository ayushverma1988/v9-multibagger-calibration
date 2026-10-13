"""Immutable same-clock paired cohorts, with truthful missing-head reporting."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
import pandas as pd
import v13_model as model
from v13_lifecycle import digest,register,validate_registry

COMPARISON_CONTRACT='V13_FINANCIAL_BUNDLES_FUNDED_APPROVED_MARGIN_POLICY_20261013'
LEGACY_COMPARISON_CONTRACT='V13_LEGACY_PAIR_20261012'

def scored(heads,frame,policy='legacy'):
    d=frame.copy();pred=model.probabilities(heads,d)
    if not {'p6','p12','loss12','joint12'}.issubset(pred):raise ValueError('All four conditional heads required for paired forecasting')
    for k,v in pred.items():d[k]=v
    v,q,b=model.gates(d)
    d['financial_tier']=['verified' if a else 'known_failure' if c else 'discovery' for a,c in zip(v,b)]
    d['market_regime']=d.market_breadth63.map(lambda x:'UNKNOWN' if pd.isna(x) else 'BROAD_UPTREND' if x>=.6 else 'BROAD_DOWNTREND' if x<=.4 else 'MIXED')
    d['liquidity_tier']=d.median_turnover63.map(lambda x:'GE_1_CRORE' if x>=1e7 else '25_LAKH_TO_1_CRORE')
    for tier in ('verified','discovery'):
        picks=model.ranked(d,tier) if policy=='legacy' else selective(d,tier)
        d['selected_'+tier]=d['isin'].isin(picks['isin'])
    return d

def promotion_coverage(root,observations,challenger_sha,now=None,comparison_contract=COMPARISON_CONTRACT):
    """All elapsed paired cohorts count, including missing/delisted outcomes."""
    now=pd.Timestamp(now or datetime.now(timezone.utc));elapsed=[]
    for path in (Path(root)/'paired_cohorts').glob('*.json'):
        obj=json.loads(path.read_text())
        if digest(obj)!=path.stem:raise ValueError('Paired manifest changed')
        if obj['challenger_sha256']!=challenger_sha:continue
        if obj.get('comparison_contract',LEGACY_COMPARISON_CONTRACT)!=comparison_contract:continue
        if pd.Timestamp(obj['registered_at_utc'])+pd.DateOffset(months=12)>now:continue
        elapsed.append(obj)
        d=observations[observations.cohort_id.eq(obj['cohort_id'])]
        if len(d)!=2*obj['rows_per_model'] or not d.twelve_month_mature.eq(True).all() or not d.independent_outcome_verified.eq(True).all():return False
    return bool(elapsed)

def selective(d,tier):
    """Evidence policy may abstain. Thresholds are preferences, not fitted gains."""
    v,q,_=model.gates(d);pool=d[v if tier=='verified' else q].copy()
    pool=pool[pool.p12.notna()&pool.loss12.le(model.RISK_CEILING)]
    # Do not pad a portfolio with weak/unknown evidence. Discovery is kept in
    # the complete ledger but requires reviewed execution to become selected.
    if 'evidence_status' not in pool:return pool.iloc[:0]
    pool=pool[pool.evidence_status.eq('FINANCIAL_AND_EXECUTION_SUPPORTED')]
    if 'scenario_net_multiple' in pool:pool=pool[pool.scenario_net_multiple.ge(2)]
    elif 'scenario_base_multiple' in pool:pool=pool[pool.scenario_base_multiple.ge(2*1.005/.995)]
    else:return pool.iloc[:0]
    pool=pool.sort_values(['joint12','isin'],ascending=[False,True])
    return pool.head(model.TOP_K)

def register_pair(frame,champion,challenger,champion_sha,challenger_sha,root,registered_at=None):
    root=Path(root);folder=root/'paired_cohorts';folder.mkdir(exist_ok=True)
    date=str(pd.Timestamp(frame.date.max()).date())
    if frame.date.nunique()!=1 or frame['isin'].duplicated().any():raise ValueError('Pair needs one date and unique ISINs')
    # Idempotent by decision and fixed head pair, even if later data collection
    # grows the universe on the same source date. New evidence cannot overwrite.
    for p in sorted(folder.glob('*.json')):
        rec=json.loads(p.read_text())
        if digest(rec)!=p.stem:raise ValueError('Paired cohort manifest changed')
        if (rec['source_asof'],rec['champion_sha256'],rec['challenger_sha256'])==(date,champion_sha,challenger_sha):
            for label in ('champion','challenger'):validate_registry(root/'registries'/label/rec[label+'_registry'])
            return {**rec,'already_registered':True}
    now=pd.Timestamp(registered_at or datetime.now(timezone.utc))
    if now.tzinfo is None:raise ValueError('Pair clock needs timezone')
    raw=frame.sort_values('isin').to_json(orient='records',date_format='iso',double_precision=15).encode()
    identity={'source_asof':date,'registered_at_utc':now.isoformat(),'input_sha256':hashlib.sha256(raw).hexdigest(),
      'champion_sha256':champion_sha,'challenger_sha256':challenger_sha,'isins':sorted(frame['isin'].tolist()),
      'comparison_contract':COMPARISON_CONTRACT}
    cohort_id=digest(identity);context={'cohort_id':cohort_id,'input_sha256':identity['input_sha256'],'comparison':'SAME_CLOCK_SAME_INPUTS_ALL_NAMES','comparison_contract':COMPARISON_CONTRACT}
    registries={}
    for label,d in [('champion',champion),('challenger',challenger)]:
        if sorted(d['isin'])!=identity['isins']:raise ValueError('Unequal paired universe')
        sha=champion_sha if label=='champion' else challenger_sha
        path=register(d,sha,root/'registries'/label,now,context)
        registries[label+'_registry']=path.name
    obj={**identity,**registries,'cohort_id':cohort_id,'rows_per_model':len(frame),'unselected_included':True}
    # Manifest is the commit point: orphan registries after interruption do not
    # count as committed paired evidence. Frozen original registries are kept.
    p=folder/(digest(obj)+'.json');p.write_text(json.dumps(obj,indent=2));return obj

def export_watchlists(root,cohort):
    for role in ('champion','challenger'):
        obj=validate_registry(Path(root)/'registries'/role/cohort[role+'_registry'])
        picks=[r for r in obj['records'] if r['selected_verified'] or r['selected_discovery']]
        (Path(root)/(role+'_latest_watchlist.json')).write_text(json.dumps({'registry_sha256':cohort[role+'_registry'][:-5],
          'registered_at_utc':obj['registered_at_utc'],'source_asof':cohort['source_asof'],'records':picks},indent=2))
