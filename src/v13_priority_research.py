"""Six-month-priority research ranking with unchanged exact-head probabilities."""
import hashlib,json
from pathlib import Path
import numpy as np,pandas as pd
from v13_available_financials import screen as available_screen,screen_many,validate_exact_replay

POLICY='V13_SIX_MONTH_PRIORITY_RSI_GT70_NO_PRICE_BAND_20261103'
WEIGHTS={'p6_percentile':.70,'joint12_percentile':.25,'optional_dma200':.05}

def rank(d,eligible,loss_ceiling=.95):
    d=d.copy();valid=eligible&d.loss12.le(loss_ceiling)
    within=pd.to_numeric(d.get('distance_ma200',pd.Series(np.nan,index=d.index)),errors='coerce').abs().le(.10+1e-12)
    d['priority_score']=(.70*d.p6.where(valid).rank(pct=True)+.25*d.joint12.where(valid).rank(pct=True)+.05*within).where(valid)
    d['priority_rank']=np.nan
    pool=d[valid].sort_values(['priority_score','p6','joint12','isin'],ascending=[False,False,False,True]).head(25)
    d.loc[pool.index,'priority_rank']=np.arange(1,len(pool)+1)
    return d

def _export_one(scored,wide,root,replay_receipt,rsi_min=70,assessed=None):
    root=Path(root)
    validate_exact_replay(scored,root,replay_receipt)
    p=root/'available_financial_research/latest.json'
    prior=json.loads(p.read_text())
    if not replay_receipt.get('replay_verified') or prior['prediction_replay']!=replay_receipt:
        raise ValueError('Six-month ranking requires matching exact financial replay')
    d,assessments,audit=assessed if assessed is not None else available_screen(scored,wide,rsi_min=rsi_min)
    d=rank(d,d.available_research_eligible)
    cols=['isin','symbol','series','source_close_INR','rsi14','p6','p12','joint12','loss12',
      'available_financial_tier','available_research_group','priority_score','priority_rank']
    for c in cols:
        if c not in d:d[c]=None
    records=json.loads(d[d.priority_rank.notna()].sort_values('priority_rank')[cols].to_json(orient='records',double_precision=15))
    sensitivities={str(x):int((d.available_research_eligible&d.loss12.le(x)).sum()) for x in [.95,.8,.7,.6,.5]}
    policy=POLICY if rsi_min==70 else 'V13_SIX_MONTH_PRIORITY_RSI_GT50_NO_PRICE_BAND_20261112'
    obj={'policy':policy,'source_asof':str(pd.Timestamp(d.date.max()).date()),'prediction_replay':replay_receipt,
      'RSI_rule':f'strict >{rsi_min}; simple rolling 14 sessions','rsi_strictly_greater_than':rsi_min,'price_band':None,'DMA_band_optional':True,
      'ranking_weights':WEIGHTS,'ranking_score_is_probability':False,'accuracy_approved':False,
      'model_retrained':False,'frozen_probabilities_changed':False,'unsupported_24month_probability_invented':False,
      'eligible':int(d.available_research_eligible.sum()),'selected':len(records),'records':records,
      'loss_ceiling_sensitivity':sensitivities,'default_loss_ceiling_unchanged':.95,
      'historical_validation_required_before_model_promotion':True}
    out=root/'priority_research';out.mkdir(exist_ok=True)
    raw=json.dumps(obj,sort_keys=True,indent=2).encode();key=hashlib.sha256(raw).hexdigest()
    immutable=out/(key+'.json')
    if immutable.exists() and immutable.read_bytes()!=raw:raise ValueError('Immutable priority ranking changed')
    if not immutable.exists():immutable.write_bytes(raw)
    from v13_capture_policy import atomic_json
    stem='latest' if rsi_min==70 else 'latest_RSI50'
    atomic_json(out/(stem+'.json'),{**obj,'snapshot_sha256':key})
    if rsi_min==70:atomic_json(out/'latest_RSI70.json',{**obj,'snapshot_sha256':key})
    lines=[f"# Six-month-priority research — {obj['source_asof']}",'',
      f'Frozen-model estimates, not validated probabilities or trade instructions. RSI >{rsi_min}; no nominal price band; 200-DMA proximity is optional.',
      'Ranking: 70% six-month probability percentile, 25% twelve-month joint-outcome percentile, 5% optional 200-DMA proximity. No 24-month probability is invented.','',
      '| Rank | Share | Series | Source close INR | RSI | 2x / 6m | 2x / 12m | 30% loss event | Financial evidence |',
      '|---:|---|---|---:|---:|---:|---:|---:|---|']
    for r in records:
        price=r['source_close_INR'];price='—' if price is None else f'{price:,.2f}'
        lines.append(f"| {int(r['priority_rank'])} | {r['symbol']} | {r['series']} | {price} | {r['rsi14']:.2f} | {100*r['p6']:.2f}% | {100*r['p12']:.2f}% | {100*r['loss12']:.2f}% | {r['available_financial_tier']} |")
    lines+=['','Missing fundamentals, complete execution and forward valuation proof remain explicitly separate. A changed ranking is not evidence of improved returns.']
    (out/(stem+'.md')).write_text('\n'.join(lines)+'\n')
    if rsi_min==70:(out/'latest_RSI70.md').write_text('\n'.join(lines)+'\n')
    return {k:v for k,v in obj.items() if k not in ['records','prediction_replay']}


def export(scored,wide,root,replay_receipt):
    """Always export RSI >70 and RSI >50 from the same verified model replay."""
    validate_exact_replay(scored,root,replay_receipt)
    assessed=screen_many(scored,wide,thresholds=(70,50))
    eligible={k:set(v[0].loc[v[0].available_research_eligible,'isin']) for k,v in assessed.items()}
    if not eligible[70].issubset(eligible[50]):raise ValueError('RSI eligibility nesting changed')
    summaries={str(k):_export_one(scored,wide,root,replay_receipt,k,assessed[k]) for k in (70,50)}
    out=Path(root)/'priority_research'
    lists={str(k):json.loads((out/f'latest_RSI{k}.json').read_text()) for k in (70,50)}
    from v13_capture_policy import atomic_json
    atomic_json(out/'latest_dual.json',{'source_asof':lists['70']['source_asof'],
      'required_RSI_thresholds':[70,50],'lists':lists,'eligible_70_is_subset_of_50':True,
      'screens_overlap':True,'same_model_and_probabilities':True,'frozen_forecasts_changed':False})
    (out/'latest_dual.md').write_text('\n\n'.join((out/f'latest_RSI{k}.md').read_text() for k in (70,50)))
    return {**summaries['70'],'dual_RSI_required':True,'dual_RSI':summaries}


def unavailable(root,asof,status):
    """Keep both list labels explicit when no current model replay is usable."""
    from v13_capture_policy import atomic_json
    out=Path(root)/'priority_research';out.mkdir(parents=True,exist_ok=True)
    lists={str(k):{'source_asof':asof,'rsi_strictly_greater_than':k,'status':status,
      'selected':0,'records':[],'not_a_no_opportunity_conclusion':True} for k in (70,50)}
    for k,obj in lists.items():
        atomic_json(out/f'latest_RSI{k}.json',obj)
        (out/f'latest_RSI{k}.md').write_text(f"# V13 RSI >{k} — source {asof}\n\n{status}. No current eligible-list conclusion is available.\n")
    result={'source_asof':asof,'status':status,'dual_RSI_required':True,'dual_RSI':lists}
    atomic_json(out/'latest_dual.json',result)
    (out/'latest_dual.md').write_text('\n\n'.join((out/f'latest_RSI{k}.md').read_text() for k in (70,50)))
    return result
