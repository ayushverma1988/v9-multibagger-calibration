"""Separate, versioned research screen; never overwrites the formal challenger.

Unknown execution/valuation evidence remains visible and is not 2x support.
RSI uses the existing simple 14-session model input (not Wilder's smoothing).
"""
import json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
import v13_model as model

POLICY='V13_FINANCIAL_MOMENTUM_RSI_GT70_RESEARCH_20261015'

def screen(scored):
    d=scored.copy();verified,_,bad=model.gates(d)
    def col(name):return pd.to_numeric(d.get(name,pd.Series(np.nan,index=d.index)),errors='coerce')
    masks={'all_fresh_scored':pd.Series(True,index=d.index),
      'financial_market_gate':verified&~bad,
      'financial_quality':verified&~bad&d.get('financial_quality_pass',pd.Series(False,index=d.index)).eq(True)}
    owner_bad=col('owner_PAT_yoy').lt(0)|col('owner_PAT_margin').le(0)|d.get('current_owner_profit_status',pd.Series('',index=d.index)).eq('UNKNOWN_OWNER_PLUS_MINORITY_DOES_NOT_RECONCILE_TOTAL_PAT')
    masks['no_observed_owner_earnings_failure']=masks['financial_quality']&~owner_bad
    masks['RSI_above_70']=masks['no_observed_owner_earnings_failure']&col('rsi14').gt(70)
    masks['finite_risk_and_probabilities']=masks['RSI_above_70']&np.isfinite(col('p6'))&np.isfinite(col('p12'))&np.isfinite(col('joint12'))&col('loss12').between(0,model.RISK_CEILING)
    pool=d[masks['finite_risk_and_probabilities']].sort_values(['joint12','isin'],ascending=[False,True]).head(model.TOP_K)
    d['research_selected']=d['isin'].isin(pool['isin'])
    d['research_policy']=POLICY
    d['research_status']='RESEARCH_ONLY_ACCURACY_UNVALIDATED'
    d['research_owner_evidence']=np.where(col('owner_PAT_yoy').ge(0)&col('owner_PAT_margin').gt(0),
      'OWNER_PAT_GROWTH_AND_MARGIN_OBSERVED_NOT_RECURRING_EPS','UNKNOWN_OWNER_PAT_GROWTH_OR_MARGIN')
    counts={k:int(v.sum()) for k,v in masks.items()}
    counts['selected']=len(pool)
    strict=masks['finite_risk_and_probabilities']&d.get('evidence_status',pd.Series('',index=d.index)).eq('FINANCIAL_AND_EXECUTION_SUPPORTED')
    counts['strict_complete_execution']=int(strict.sum())
    counts['strict_net_2x']=int((strict&col('scenario_net_multiple').ge(2)).sum())
    return d,{'policy':POLICY,'RSI_definition':'simple rolling 14 session gains/losses; strict >70',
      'gate_counts':counts,'accuracy_approved':False,'formal_selection_changed':False,
      'missing_execution_or_valuation_is_not_negative_upside_probability':True}

def export(scored,root,replay_receipt):
    d,audit=screen(scored);out=Path(root)/'research_selection';out.mkdir(exist_ok=True)
    if not replay_receipt.get('replay_verified') or replay_receipt.get('rows')!=len(d):raise ValueError('Research selection needs matching exact replay receipt')
    from v13_lifecycle import digest
    records=json.loads(d[d.research_selected].to_json(orient='records',date_format='iso',double_precision=15))
    obj={**audit,'source_asof':str(pd.Timestamp(d.date.max()).date()),'records':records,
      'registered_at_utc':datetime.now(timezone.utc).isoformat(),'prediction_replay':replay_receipt,
      'information_cutoffs_utc':sorted(d.information_cutoff_utc.dropna().unique().tolist()),
      'all_names_selection':json.loads(d[['isin','research_selected','research_owner_evidence']].sort_values('isin').to_json(orient='records')),
      'unselected_included':True}
    key=digest(obj);path=out/(key+'.json')
    if not path.exists():path.write_text(json.dumps(obj,indent=2))
    (out/'latest.json').write_text(json.dumps({**obj,'snapshot_sha256':key},indent=2));return audit
