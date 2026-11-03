"""Fixed profit-first DEVELOPMENT experiment; never edits frozen forecasts.

Reported profits are not proof of recurring earnings or future profitability.
Missing/conflicting source facts block the accounting gate. All rankings are
research only: the inherited probabilities are not calibrated for this subset.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from v12_3_point_in_time import coherent_period

VERSION = 'V12.4-profit-first-development'
VARIANTS = ('profit_first_top5', 'profit_momentum_top5', 'profit_first_legacy_risk_veto')
RULES = {'four_consecutive_quarter_PAT_positive': True,
         'annual_PAT_positive': True, 'annual_CFO_PAT_min': .6,
         'positive_equity': True, 'debt_equity_max': 1.,
         'latest_quarter_sales_growth_positive': True,
         'latest_quarter_PAT_growth_positive': True,
         'quarter_freshness_days': 190, 'annual_balance_freshness_days': 450,
         'price_range': [20, 2000], 'above_MA50': True,
         'prior63_return_max': .75, 'RSI_hard_threshold': None,
         'rank_weight_on_loss': 0., 'legacy_risk_ablation_max': .45}

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def profit_evidence(facts, date, isin):
    """All chosen periods/versions must exist before this market-close cutoff."""
    date = pd.Timestamp(date)
    cut = (date.tz_localize('Asia/Kolkata') + pd.Timedelta(hours=15, minutes=30)).tz_convert('UTC')
    result = {'profit_evidence_status': 'UNKNOWN', 'profit_evidence_reasons': '',
              'positive_quarters': np.nan, 'four_quarter_PAT_INR': np.nan,
              'annual_PAT_INR': np.nan, 'annual_CFO_INR': np.nan,
              'financial_source_count': 0, 'financial_latest_available': None}
    known = facts[facts.security_isin.eq(isin) & facts.available.le(cut)
                  & facts.period.le(date)]
    recent = known[known.period_kind.eq('quarter') & (date-known.period).dt.days.le(190)]
    if recent.empty:
        result['profit_evidence_reasons'] = 'UNKNOWN:recent_quarter'
        return result
    basis = 'consolidated' if recent.reporting_mode.eq('consolidated').any() else 'standalone'
    known = known[known.reporting_mode.eq(basis)]
    result['profit_reporting_basis'] = basis
    latest = recent[recent.reporting_mode.eq(basis)].period.max()
    values, used, unknown, fail = [], [], [], []
    for offset in [0, 3, 6, 9]:
        period = latest-pd.offsets.MonthEnd(offset)
        q = coherent_period(known, 'quarter', period)
        v = q.set_index('metric').value_INR if len(q) else pd.Series(dtype=float)
        if 'pat' not in v or not np.isfinite(v['pat']):
            unknown.append('quarter_PAT_'+period.strftime('%Y-%m-%d'))
        else:
            values.append(float(v['pat'])); used.append(q)
            if v['pat'] <= 0: fail.append('nonpositive_PAT_'+period.strftime('%Y-%m-%d'))
    if len(values)==4:
        result['positive_quarters'] = sum(v>0 for v in values)
        result['four_quarter_PAT_INR'] = sum(values)
    a = coherent_period(known, 'annual')
    if len(a) and (date-a.period.iloc[0]).days<=450:
        used.append(a); av=a.set_index('metric').value_INR
        for metric,key in [('pat','annual_PAT_INR'),('cfo','annual_CFO_INR')]:
            if metric not in av or not np.isfinite(av[metric]):unknown.append(key)
            else:
                result[key]=float(av[metric])
                if av[metric]<=0:fail.append('nonpositive_'+key)
    else: unknown.append('fresh_annual_PAT_and_CFO')
    if used:
        s=pd.concat(used)
        result['financial_source_count']=int(s.source_sha256.nunique())
        result['financial_latest_available']=s.available.max().isoformat()
    result['profit_evidence_status']='REJECT' if fail else 'UNKNOWN' if unknown else 'PASS_REPORTED_PROFIT'
    result['profit_evidence_reasons']=';'.join(['FAIL:'+x for x in fail]+['UNKNOWN:'+x for x in unknown])
    return result

def gate(d):
    cols=['quarter_sales_yoy','quarter_pat_yoy','quarter_pat_margin','annual_CFO_PAT','dated_debt_equity',
          'close','distance_ma50','log_ret63']
    finite=pd.Series(np.isfinite(d[cols].to_numpy(float)).all(axis=1),index=d.index)
    return (finite & d.profit_evidence_status.eq('PASS_REPORTED_PROFIT')
            & d.quarter_sales_yoy.gt(0) & d.quarter_pat_yoy.gt(0)
            & d.quarter_pat_margin.gt(0) & d.annual_CFO_PAT.ge(.6)
            & d.dated_debt_equity.between(0,1)
            & d.close.between(20,2000) & d.distance_ma50.gt(0)
            & d.log_ret63.le(np.log(1.75)))

def choose(d, variant, months):
    if variant not in VARIANTS:raise ValueError('Unregistered variant')
    probability='p_hit2_'+str(months)
    pool=d[gate(d) & d[probability].between(0,1)].copy()
    if variant=='profit_first_legacy_risk_veto':
        pool=pool[pool.p_loss30_12.between(0,.45)]
    column='relative_strength63' if variant=='profit_momentum_top5' else probability
    return pool[np.isfinite(pool[column])].sort_values([column,'isin'],ascending=[False,True]).head(5)

def run(scores_path, facts_path, output, asof='2026-10-09', bse_path=None):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    protocol={'version':VERSION,'registered_at_utc':datetime.now(timezone.utc).isoformat(),
              'variants':list(VARIANTS),'rules':RULES,'scores_sha256':sha(scores_path),
              'facts_sha256':sha(facts_path),'code_sha256':sha(__file__),
              'asof':asof,'window_start':'2022-10-09',
              'no_weight_refit':True,'historical_results':'DEVELOPMENT_ONLY_ALREADY_EXAMINED_PERIOD',
              'future_outcomes':0,'probabilities_validated_for_profit_subset':False,
              'production_approved':False,'existing_frozen_artifacts_modified':False}
    with (out/'registered_protocol.json').open('x') as f:json.dump(protocol,f,indent=2)
    scores=pd.read_parquet(scores_path)
    facts=pd.read_parquet(facts_path)
    facts['available']=pd.to_datetime(facts.available_at_utc,utc=True,format='mixed')
    facts['period']=pd.to_datetime(facts.period_end)
    starts=pd.to_datetime(facts.period_start,errors='coerce')
    boundary=pd.Timestamp('2022-10-09')
    if ((facts.period<boundary) | (starts.notna() & starts.lt(boundary))).any():
        raise ValueError('Financial fact beyond registered four-year boundary')
    groups={str(k):g for k,g in facts.groupby('security_isin')};empty=facts.iloc[:0]
    frames=[]
    for date,d in scores.groupby('date',sort=True):
        records=[profit_evidence(groups.get(str(r['isin']),empty),date,r['isin']) for r in d.to_dict('records')]
        frames.append(pd.concat([d.reset_index(drop=True),pd.DataFrame(records)],axis=1))
        print('profit evidence',str(pd.Timestamp(date).date()),len(d),flush=True)
    enriched=pd.concat(frames,ignore_index=True)
    enriched['profit_first_gate']=gate(enriched)
    enriched.to_parquet(out/'profit_evidence_scores.parquet',index=False)
    summaries=[];selected=[];current=[]
    for h in [6,12]:
        y='hit2_'+str(h);m='mature_date_'+str(h);p='p_hit2_'+str(h)
        for variant in VARIANTS:
            counts={'variant':variant,'months':h,'decision_dates':0,'selection_dates':0,
                    'selected':0,'known':0,'hits':0,'unknown':0,'price_loss_events_known':0,
                    'price_loss_known_rows':0}
            for date,d in enriched.groupby('date',sort=True):
                picks=choose(d,variant,h)
                if date==pd.Timestamp(asof):
                    current.extend(picks.assign(variant=variant,months=h,approved_for_trading=False).to_dict('records'))
                    continue
                # Use the pre-existing original panel's calendar maturity eligibility,
                # not the candidate's results, to retain abstention/unknown dates.
                if not (d[m].le(pd.Timestamp(asof)) & d[y].isin([0,1]) & d[p].notna()).any():continue
                counts['decision_dates']+=1;counts['selection_dates']+=int(len(picks)>0)
                known=picks[picks[m].le(pd.Timestamp(asof)) & picks[y].isin([0,1])]
                counts['selected']+=len(picks);counts['known']+=len(known)
                counts['hits']+=int(known[y].sum());counts['unknown']+=len(picks)-len(known)
                risk=known[known.loss30_before_hit2_12.isin([0,1])] if h==12 else known.iloc[:0]
                counts['price_loss_known_rows']+=len(risk)
                counts['price_loss_events_known']+=int(risk.loss30_before_hit2_12.eq(1).sum())
                for r in picks.to_dict('records'):
                    selected.append({'variant':variant,'months':h,'date':date,'symbol':r['symbol'],'isin':r['isin'],
                                     'score':r[p],'hit2':r[y] if pd.notna(r[m]) and r[m]<=pd.Timestamp(asof) else None,
                                     'profit_evidence_status':r['profit_evidence_status']})
            counts['hit_fraction_all_selected']=counts['hits']/counts['selected'] if counts['selected'] else None
            summaries.append(counts)
    cur=enriched[enriched.date.eq(pd.Timestamp(asof))]
    pd.DataFrame(selected).to_parquet(out/'selected_development_outcomes.parquet',index=False)
    pd.DataFrame(current).to_json(out/'current_research_selections.json',orient='records',date_format='iso',indent=2)
    report={'version':VERSION,'historical_development':summaries,'current_rows':len(cur),
            'current_profit_evidence':cur.profit_evidence_status.value_counts().to_dict(),
            'current_profit_and_growth_gate_pass':int(cur.profit_first_gate.sum()),
            'current_profit_gate_and_legacy_risk_pass':int((cur.profit_first_gate & cur.p_loss30_12.between(0,.45)).sum()),
            'all_rows_profit_evidence':enriched.profit_evidence_status.value_counts().to_dict(),
            'future_outcomes':0,'investment_accuracy_improvement_proven':False,
            'new_variables_fitted':False,'production_approved':False}
    if bse_path:
        b=pd.read_parquet(bse_path)
        report['BSE_current_rows']=len(b);report['BSE_profit_evidence_pass']=0
        report['BSE_status']='UNKNOWN_NO_DATED_FINANCIAL_PROOF'
    (out/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(report,indent=2),flush=True)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scores',required=True);p.add_argument('--facts',required=True)
    p.add_argument('--output',required=True);p.add_argument('--asof',default='2026-10-09');p.add_argument('--bse')
    a=p.parse_args();run(a.scores,a.facts,a.output,a.asof,a.bse)
