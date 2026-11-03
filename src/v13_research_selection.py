"""Available-data research ranking, separate from frozen forecasts.

Weights are declared preferences, not fitted or calibrated probabilities.
RSI retains the model's simple rolling 14-session definition.
"""
import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
import v13_model as model
from v13_financial_evidence import REQUIRED,financial_quality

POLICY='V13_AVAILABLE_DATA_RSI_GE50_NO_PRICE_BAND_20261021'
WEIGHTS={'joint12_percentile':.75,'financial_input_fraction':.10,
         'complete_execution':.05,'source_supported_net_2x':.05,'within_200dma_band':.05}
DMA_BAND=.10
RSI_MIN=50

def screen(scored):
    d=scored.copy();verified,discovery,_=model.gates(d)
    def col(name):return pd.to_numeric(d.get(name,pd.Series(np.nan,index=d.index)),errors='coerce')
    financial=pd.DataFrame({k:col(k) for k in REQUIRED},index=d.index)
    quality=[financial_quality(r) for r in financial.to_dict('records')]
    known_failure=pd.Series([q['status']=='KNOWN_FINANCIAL_FAILURE' for q in quality],index=d.index,dtype=bool)
    complete=pd.Series([q['quality_pass'] for q in quality],index=d.index,dtype=bool)
    owner_bad=col('owner_PAT_yoy').lt(0)|col('owner_PAT_margin').le(0)|d.get(
      'current_owner_profit_status',pd.Series('',index=d.index)).eq('UNKNOWN_OWNER_PLUS_MINORITY_DOES_NOT_RECONCILE_TOTAL_PAT')
    market=verified|discovery
    probability_ok=(col('p6').between(0,1)&col('p12').between(0,1)&col('joint12').between(0,1)
      &col('p6').le(col('p12')+1e-12)&col('joint12').le(col('p12')+1e-12)
      &(col('joint12')+col('loss12')).le(1+1e-12)&col('loss12').between(0,model.RISK_CEILING))
    masks={'all_fresh_scored':pd.Series(True,index=d.index),'market_and_no_observed_financial_failure':market&~known_failure}
    masks['no_observed_owner_earnings_failure']=masks['market_and_no_observed_financial_failure']&~owner_bad
    masks['RSI_at_least_50']=masks['no_observed_owner_earnings_failure']&col('rsi14').ge(RSI_MIN)
    masks['finite_risk_and_probabilities']=masks['RSI_at_least_50']&probability_ok
    eligible=masks['finite_risk_and_probabilities']
    dma=col('distance_ma200');dma_known=np.isfinite(dma)&dma.gt(-1)
    within=dma_known&dma.abs().le(DMA_BAND+1e-12)
    execution=d.get('evidence_status',pd.Series('',index=d.index)).eq('FINANCIAL_AND_EXECUTION_SUPPORTED')
    valuation=d.get('earnings_scenario_status',pd.Series('',index=d.index)).eq('SCENARIO_NOT_PROBABILITY')&col('scenario_net_multiple').ge(2)
    d['research_eligible']=eligible
    d['research_prior_return63']=np.expm1(col('log_ret63'))
    d['research_financial_input_fraction']=np.isfinite(financial).mean(axis=1)
    d['research_financial_tier']=np.where(known_failure,'OBSERVED_FINANCIAL_FAILURE',
      np.where(complete,'PASS_AVAILABLE_SEVEN_CHECKS','INCOMPLETE_FINANCIAL_INPUTS'))
    d['research_dma200_status']=np.where(~dma_known,'UNKNOWN_200DMA',np.where(within,'WITHIN_PLUS_MINUS_10_PERCENT','OUTSIDE_PLUS_MINUS_10_PERCENT'))
    d['research_within_200dma_band']=within
    d['research_joint12_percentile']=col('joint12').where(eligible).rank(pct=True)
    d['research_execution_supported']=execution
    d['research_source_supported_net_2x']=valuation
    components={'joint12_percentile':d.research_joint12_percentile,
      'financial_input_fraction':d.research_financial_input_fraction,'complete_execution':execution,
      'source_supported_net_2x':valuation,'within_200dma_band':within}
    d['research_ranking_score']=sum(WEIGHTS[k]*v for k,v in components.items()).where(eligible)
    pool=d[eligible].sort_values(['research_ranking_score','joint12','isin'],ascending=[False,False,True]).head(model.TOP_K)
    d['research_selected']=d['isin'].isin(pool['isin'])
    d['research_rank']=d['isin'].map({isin:i+1 for i,isin in enumerate(pool['isin'])})
    d['research_policy']=POLICY
    d['research_status']='RESEARCH_ONLY_ACCURACY_UNVALIDATED'
    d['research_owner_evidence']=np.where(col('owner_PAT_yoy').ge(0)&col('owner_PAT_margin').gt(0),
      'OWNER_PAT_GROWTH_AND_MARGIN_OBSERVED_NOT_RECURRING_EPS','UNKNOWN_OWNER_PAT_GROWTH_OR_MARGIN')
    reasons={'MARKET_OR_EXISTING_FINANCIAL_GATE':~market,'OBSERVED_FINANCIAL_FAILURE':known_failure,
      'OBSERVED_OWNER_EARNINGS_FAILURE_OR_CONFLICT':owner_bad,'RSI_BELOW_50_OR_MISSING':~col('rsi14').ge(RSI_MIN),
      'INVALID_PROBABILITIES_OR_RISK_LIMIT':~probability_ok}
    d['research_exclusion_reasons']=[json.dumps([k for k,v in reasons.items() if v.loc[i]]) for i in d.index]
    counts={k:int(v.sum()) for k,v in masks.items()}
    counts.update(selected=len(pool),eligible_complete_financial=int((eligible&complete).sum()),
      eligible_incomplete_financial=int((eligible&~complete).sum()),eligible_within_200dma_band=int((eligible&within).sum()),
      selected_within_200dma_band=int((d.research_selected&within).sum()),eligible_unknown_200dma=int((eligible&~dma_known).sum()),
      strict_complete_execution=int((eligible&execution).sum()),strict_net_2x=int((eligible&execution&valuation).sum()))
    return d,{'policy':POLICY,'RSI_definition':'simple rolling 14 session gains/losses; inclusive >=50',
      'DMA_definition':'adjusted close / trailing 200-observation adjusted-close mean - 1; inclusive +/-10%; optional bonus only',
      'ranking_weights':WEIGHTS.copy(),'ranking_score_is_probability':False,'weights_fitted_to_outcomes':False,
      'gate_counts':counts,'accuracy_approved':False,'formal_selection_changed':False,
      'missing_execution_or_valuation_is_not_negative_upside_probability':True,
      'missing_financial_inputs_permitted_when_no_observed_failure':True,
      '200dma_proximity_proves_no_prior_bull_run_or_future_upside':False}

def export(scored,root,replay_receipt):
    d,audit=screen(scored);out=Path(root)/'research_selection';out.mkdir(exist_ok=True)
    if not replay_receipt.get('replay_verified') or replay_receipt.get('rows')!=len(d):raise ValueError('Research selection needs matching exact replay receipt')
    from v13_lifecycle import digest
    records=json.loads(d[d.research_selected].sort_values('research_rank').to_json(orient='records',date_format='iso',double_precision=15))
    columns=['isin','information_cutoff_utc',*[c for c in d if c.startswith('research_')]]
    obj={**audit,'source_asof':str(pd.Timestamp(d.date.max()).date()),'records':records,
      'registered_at_utc':datetime.now(timezone.utc).isoformat(),'prediction_replay':replay_receipt,
      'selector_code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      'information_cutoffs_utc':sorted(d.information_cutoff_utc.dropna().unique().tolist()),
      'all_names_selection':json.loads(d[columns].sort_values('isin').to_json(orient='records')),
      'unselected_included':True}
    key=digest(obj);path=out/(key+'.json')
    if not path.exists():path.write_text(json.dumps(obj,indent=2))
    (out/'latest.json').write_text(json.dumps({**obj,'snapshot_sha256':key},indent=2));return audit
