"""Audit exported study results and add clearly labelled partial case paths."""
import argparse, gzip, importlib.util, json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

def review(root):
    root=Path(root)
    spec=importlib.util.spec_from_file_location('executed_study',root/'v13_signal_timing_study.py')
    study=importlib.util.module_from_spec(spec);spec.loader.exec_module(study)
    receipt=json.loads((root/'run_receipt.json').read_text());asof=pd.Timestamp(receipt['source_asof'])
    cases=json.loads((root/'case_first_signals.json').read_text())
    with gzip.open(root/'all_signal_events.jsonl.gz','rt') as f:events=pd.DataFrame([json.loads(l) for l in f])
    if events.duplicated(['date','isin','rsi_threshold']).any():raise ValueError('Duplicate events')
    if not events.rsi14.gt(events.rsi_threshold).all():raise ValueError('RSI threshold failure')
    if not pd.to_datetime(events.date).le(asof).all():raise ValueError('Future signal')
    financial_sources=0;catalyst_sources=0
    with gzip.open(root/'dated_evidence_snapshots.jsonl.gz','rt') as f:
        for line in f:
            row=json.loads(line);cut=pd.Timestamp(row['information_cutoff_utc'])
            for source in row['financial_sources']:
                if pd.Timestamp(source['available'])>cut:raise ValueError('Future financial source')
                financial_sources+=1
            for source in row['catalyst_sources']:
                if pd.Timestamp(source['available_at_utc'])>cut:raise ValueError('Future catalyst source')
                catalyst_sources+=1
    for m in study.HORIZONS:
        valid=events[events['status_'+str(m)].eq('MATURE_VALID_PROVIDER_PROXY')]
        if not pd.to_datetime(valid['target_date_'+str(m)]).le(asof).all():raise ValueError('Immature endpoint counted')
        if not (pd.to_datetime(valid['entry_date_'+str(m)])>pd.to_datetime(valid.date)).all():raise ValueError('Same-day entry')
    price_cache={};checks=0;rows=[]
    def partial(d,t):
        if t+1>=len(d):return {'status':'PENDING_ENTRY'}
        entry=d.iloc[t+1];f=d.iloc[t+1:]
        if (entry.date-d.iloc[t].date).days>5:return {'status':'UNKNOWN_ENTRY_GAP'}
        if (asof-f.date.iloc[-1]).days>5:return {'status':'UNKNOWN_STALE_FOLLOWUP'}
        if f.date.diff().dt.days.max()>10 or np.log(f.adj_close).diff().abs().max()>np.log(1.4):return {'status':'UNKNOWN_PATH_QUALITY'}
        eo=entry.open*entry.adj_close/entry.close;mult=(1-study.COST)/(1+study.COST)
        w=f.adj_close.to_numpy(float)/eo*mult;low=f.low*f.adj_close/f.close/eo*mult
        path=np.r_[1.,w];dd=path/np.maximum.accumulate(path)-1
        return {'status':'OBSERVED_TO_ASOF_NOT_FIXED_HORIZON','entry_date':entry.date,'observed_through':f.date.iloc[-1],
            'observed_sessions':len(f),'endpoint_return':w[-1]-1,'max_close_drawdown':dd.min(),
            'worst_adjusted_low_return':low.min()-1}
    def pct(v):return 'pending/unknown' if v is None else f'{100*v:+.2f}%'
    for case in cases:
        isin=case['isin'];path=root/'case_histories'/(isin+'.parquet')
        if isin not in price_cache:
            price_cache[isin]=pd.read_parquet(path);price_cache[isin]['date']=pd.to_datetime(price_cache[isin].date)
        d=price_cache[isin]
        for label in ['first_observable_signal','first_signal_trailing12m','last_signal_before_trailing6m',
                      'first_technical_and_permitted_financials','first_technical_and_permitted_financials_trailing12m']:
            signal=case.get(label)
            if not signal:continue
            ix=np.flatnonzero(d.date.eq(pd.Timestamp(signal['date'])))
            if len(ix)!=1:raise ValueError('Missing or ambiguous case signal date')
            t=int(ix[0]);signal['partial_to_source_asof']=partial(d,t)
            for m in study.HORIZONS:
                recomputed=study.outcome(d,t,m,asof)
                if recomputed['status']!=signal['status_'+str(m)]:raise ValueError('Case outcome status differs')
                for k in ['endpoint_return','best_close_return','worst_adjusted_low_return','max_close_drawdown']:
                    a=recomputed.get(k);b=signal.get(k+'_'+str(m))
                    if (a is None)!=(b is None) or (a is not None and not np.isclose(a,b,atol=1e-12,rtol=0)):
                        raise ValueError('Case path recomputation differs')
                checks+=1
        s=case.get('first_signal_trailing12m')
        if s:
            rows.append({'symbol':case['symbol'],'isin':isin,'rsi_threshold':case['rsi_threshold'],
                'first_signal_date':s['date'],'stage':s['entry_stage'],'earnings':s['earnings_evidence'],
                'catalyst':s['catalyst_evidence'],'return_6m':s['endpoint_return_6'],'status_6m':s['status_6'],
                'max_drawdown_6m':s['max_close_drawdown_6'],
                'partial_to_asof':s['partial_to_source_asof'],
                'qualified_before_6m_anchor':pd.Timestamp(s['date'])<=pd.Timestamp(receipt['prior_6m_start']),
                'prior_63_session_return_at_signal':s['rd_return63'],'MA200_extension_at_signal':s['rd_extension200']})
    study.save_json(root/'case_first_signals_reviewed.json',cases)
    study.save_json(root/'case_trace_summary.json',rows)
    audit={'event_rows_audited':len(events),'case_horizon_paths_recomputed':checks,
        'exported_financial_source_references_clock_checked':financial_sources,
        'exported_catalyst_source_references_clock_checked':catalyst_sources,
        'future_source_or_maturity_violations':0,'same_day_entry_violations':0,
        'case_price_files':len(price_cache),'partial_paths_added':True,'partial_paths_are_not_mature_horizon_outcomes':True,
        'raw_retired_original_source_bytes_independently_checked':False,
        'financial_source_clock_audit_scope':'Exported feature references; historical archive availability metadata, not independent vintage reconstruction',
        'source_receipt_sha256':hashlib.sha256((root/'run_receipt.json').read_bytes()).hexdigest()}
    study.save_json(root/'independent_review_audit.json',audit)
    lines=['# V13 signal timing: reviewed case traces','',f"Source as of {asof.date()}. See study_report.md for the full-universe comparison.",'',
        'Dates below are first reconstructed technical qualification in the trailing 12 months. Exact historical full V13 qualification remains unproven; no current trained head is applied backwards.',
        'Net returns start at the next observed open and deduct 0.5% per side. Adjusted-close data are a total-return proxy. Partial paths end on the source date and are not substitutes for mature 6/12-month outcomes.','',
        '| Share | RSI > | First signal | Entry stage | Mature net 6m | 6m max drawdown | Partial return to source date | Partial max drawdown |',
        '|---|---:|---|---|---:|---:|---:|---:|']
    for r in rows:
        p=r['partial_to_asof'];lines.append(f"| {r['symbol']} | {r['rsi_threshold']} | {pd.Timestamp(r['first_signal_date']).date()} | {r['stage']} | {pct(r['return_6m'])} | {pct(r['max_drawdown_6m'])} | {pct(study.plain(p.get('endpoint_return')))} | {pct(study.plain(p.get('max_close_drawdown')))} |")
    lines+=['','`case_first_signals_reviewed.json` also records first retained-history signals, the last qualification before the six-month measurement window, first technical+permitted-financial signals, and the first date observed in each entry stage.',
        '', 'Full-universe losing and failed-breakout events remain in all_signal_events.jsonl.gz and failed_and_adverse_signals.jsonl.gz. No survivorship-free or independent original filing-vintage certification is claimed.','']
    (root/'reviewed_case_traces.md').write_text('\n'.join(lines))
    print(json.dumps(study.plain(audit),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);review(p.parse_args().root)
