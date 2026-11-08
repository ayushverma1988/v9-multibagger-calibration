"""Read-only retrospective timing research. Never a historical frozen-head replay.

Protocol is committed before execution. Every event, including missing-data and
losing events, remains in the output; only matured valid windows enter metrics.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import re
import zipfile
import numpy as np
import pandas as pd

VERSION = 'V13_SIGNAL_TIMING_RECONSTRUCTION_20261108_A'
HORIZONS = (1, 3, 6, 12)
COOLDOWN = 63
COST = .005
STAGES = ('EARLY_BREAKOUT', 'CONSOLIDATION_AFTER_RALLY', 'EXTENDED_MOMENTUM',
          'OTHER_TREND', 'UNKNOWN_STAGE')
PROTOCOL = {
    'version': VERSION, 'research_only': True,
    'signal_clock': '15:30 Asia/Kolkata; after-close filings wait until the following signal',
    'entry': 'Next observed positive-volume session open; reject entry gaps >5 calendar days',
    'technical_gate': 'Existing V13 feature_eligible; positive close; above MA50; 63-session gain <=150%; RSI14 strictly >50 or >70',
    'rsi': 'Existing V13 simple rolling 14-session gains/losses, not Wilder smoothing',
    'universe': 'All unambiguous archived whole histories, including historical-only ISINs; current alias preferred when unique',
    'event': 'First qualifying day after a nonqualifying day; accept next such event only >=63 observed sessions later',
    'first_signal': 'First observable technical qualification in the retained history; not first-ever or proof of original V13 qualification',
    'case_window': 'First observable qualification in trailing 12 calendar months, also report first retained-history and last before trailing-6-month start',
    'stage_priority': ['UNKNOWN_STAGE if insufficient 200-session history', 'EXTENDED_MOMENTUM', 'CONSOLIDATION_AFTER_RALLY', 'EARLY_BREAKOUT', 'OTHER_TREND'],
    'extended': '20-session gain >25% OR 63-session gain >50% OR extension above MA200 >40%',
    'early': 'Existing causal first-breakout detector age 0..5 sessions and 0..10% above breakout resistance; not extended',
    'first_breakout_detector': 'Close crosses >0.5% above prior 63-session high; no such cross in prior 20 sessions; prior 20/63 range <=0.65 and volatility <=0.80; volume >=1.5x prior-20 median',
    'consolidation': 'Earlier 63-session rally ending 20 sessions ago >=25%; prior 20-session range <=15%; current 20-session return -10..+10%; close >=85% of prior-63 high; not extended',
    'breakout_failure': 'For signals at/above 99% of prior 63-session resistance, a close >3% below that fixed resistance within next 20 observed sessions; may subsequently recover',
    'adverse_excursion': 'All signals also retain -10% adjusted-low breach over next 20 sessions',
    'outcomes': '1/3/6/12 calendar months from next-session open; endpoint total-return proxy, best close, worst adjusted low, close-to-close maximum drawdown',
    'cost_per_side': COST, 'net_cost_multiplier': (1-COST)/(1+COST),
    'outcome_quality': 'Require >=16 observed sessions per month, last session <=5 days from target, no >10-day gaps or >40% absolute log-price jumps',
    'censoring': 'No endpoint/complete-window drawdown if horizon has not matured; partial latest observation exported separately',
    'financial_clock': 'Exact ISIN; valid source URL/hash; source available timestamp <= signal close; period within prior four years; latest known version at that cutoff, retain ties',
    'financial_vintage_limit': 'Archived availability timestamps are used; full historical filing revisions and complete original bytes have not been independently reconstructed',
    'catalysts': 'Exact ISIN+symbol, deduplicate archived events; publication timestamp <= signal close and within prior 90 days; order/capacity disclosures are not execution proof; no observation means unknown',
    'current_30': 'Diagnostic cases chosen in November 2026; never an unbiased validation sample',
    'comparison': 'Report wider universe, both RSI thresholds, entry stages, earnings/catalyst evidence and signal-year groups; thresholds are predeclared research preferences',
    'dependent_observations': '63-session cooldown still permits overlapping long-horizon returns; first-per-ISIN sensitivity also reported; no IID confidence or significance claim',
    'not_replayed': 'No current model probabilities are projected into the past; no trained head, forecast registry, archive or production eligibility is modified',
    'limitations': ['Current-listing and retained-history survivor bias; delisted/suspended histories can be missing',
        'Provider historical price/action vintages can be revised; full action-interval certification absent',
        'First retained signal is left-censored by history start/warmup',
        'Missing earnings/catalysts remain unknown; reconstruction is not a contemporaneous recorded forecast',
        'No liquidity slippage/limit-fill/market-impact model beyond fixed costs; a next-session open is a proxy',
        'This retrospective study cannot establish a calibrated probability of doubling or guarantee future performance']}

def plain(x):
    if isinstance(x, dict): return {str(k): plain(v) for k,v in x.items()}
    if isinstance(x, (list,tuple)): return [plain(v) for v in x]
    if isinstance(x, (np.integer,np.bool_)): return x.item()
    if isinstance(x, (np.floating,float)): return float(x) if np.isfinite(x) else None
    if isinstance(x, (pd.Timestamp,datetime)): return x.isoformat()
    if x is pd.NA or x is pd.NaT: return None
    return x

def save_json(path, obj):
    Path(path).write_text(json.dumps(plain(obj), indent=2, allow_nan=False))

def write_lines(path, rows):
    with gzip.open(path, 'wt', encoding='utf-8') as f:
        for r in rows: f.write(json.dumps(plain(r), allow_nan=False)+'\n')

def signal_cutoff(date):
    return (pd.Timestamp(date).normalize().tz_localize('Asia/Kolkata')+
            pd.Timedelta(hours=15, minutes=30)).tz_convert('UTC')

def stage_frame(d):
    """All rolling inputs end at the current or prior session, never after it."""
    x=d.copy();p=x.adj_close.astype(float);factor=p/x.close
    hi=x.high*factor;lo=x.low*factor
    x['prior_resistance63']=hi.shift().rolling(63).max()
    x['prior_base_range20']=hi.shift().rolling(20).max()/lo.shift().rolling(20).min()-1
    x['prior_rally63_ending20']=p.shift(20)/p.shift(83)-1
    unknown=x.rd_extension200.isna()|x.rd_return63.isna()|x.rd_return20.isna()
    extended=(x.rd_return20.gt(.25)|x.rd_return63.gt(.50)|x.rd_extension200.gt(.40))
    early=x.rd_breakout_age.between(0,5)&x.rd_extension_since_breakout.between(0,.10)
    base=(x.prior_rally63_ending20.ge(.25)&x.prior_base_range20.le(.15)&
          x.rd_return20.between(-.10,.10)&p.ge(.85*x.prior_resistance63))
    x['entry_stage']=np.select([unknown,extended,base,early],
        ['UNKNOWN_STAGE','EXTENDED_MOMENTUM','CONSOLIDATION_AFTER_RALLY','EARLY_BREAKOUT'], default='OTHER_TREND')
    return x

def eligible(d, threshold):
    return (d.feature_eligible.fillna(False)&d.close.gt(0)&d.distance_ma50.gt(0)&
            d.log_ret63.le(np.log(2.5))&d.rsi14.gt(threshold))

def event_indices(mask, cooldown=COOLDOWN):
    starts=mask&~mask.shift(fill_value=False)
    accepted=[];last=-cooldown
    for i in np.flatnonzero(starts.to_numpy()):
        if i-last>=cooldown: accepted.append(int(i));last=int(i)
    return accepted

def known_financials(w, date):
    if w is None or w.empty:return pd.DataFrame()
    cut=signal_cutoff(date);date=pd.Timestamp(date)
    q=w[w.available.le(cut)&w.period.between(date-pd.DateOffset(years=4),date)].copy()
    if q.empty:return q
    latest=q.groupby(['reporting_mode','period_kind','period']).available.transform('max')
    return q[q.available.eq(latest)].copy()

def financial_snapshot(w, date):
    from v13_available_financials import assess
    from v13_financial_evidence import balancing_quarters
    from v12_5_features import features
    known=known_financials(w,date)
    if not known.empty: known=balancing_quarters(known)
    assessment=assess(known,date,'UNKNOWN')
    values,sources=features(known,pd.Timestamp(date)) if not known.empty else ({},[])
    owner=values.get('owner_PAT_yoy',np.nan);sales=values.get('quarter_sales_yoy',np.nan)
    margin=values.get('owner_PAT_margin',np.nan)
    observed=all(np.isfinite(v) for v in (owner,sales,margin))
    earning=('DATED_GROWTH_AND_MARGIN_OBSERVED' if observed and owner>=.15 and sales>=.10 and margin>=.08
             else 'DATED_OBSERVED_BELOW_GROWTH_OR_MARGIN_THRESHOLD' if observed else 'UNKNOWN_OR_INCOMPLETE_EARNINGS')
    if assessment['source_conflict']:earning='UNKNOWN_SOURCE_CONFLICT'
    cols=['quarter_sales_yoy','quarter_pat_yoy','owner_PAT_yoy','owner_PAT_margin',
          'reported_diluted_EPS_yoy_same_document','annual_CFO_PAT','dated_debt_equity']
    clocks=[pd.Timestamp(s['available']) for s in sources]
    if clocks and max(clocks)>signal_cutoff(date):raise ValueError('Future financial source leaked')
    used=[{k:s.get(k) for k in ['period','period_kind','available','source_sha256','source_url','derived_balancing_quarter','input_source_sha256','input_source_urls']} for s in sources]
    return {'financial_tier':assessment['tier'],'financial_check_complete':assessment['strict_financial_complete'],
        'financial_policy_permitted':assessment['tier'] in {'AVAILABLE_SEVEN_CHECK_PASS_RESEARCH','AVAILABLE_CURRENT_FINANCIALS_RESEARCH','TURNAROUND_RESEARCH','INCOMPLETE_FINANCIAL_DISCOVERY'},
        'financial_check_pass':assessment['strict_financial_quality_pass'],'earnings_evidence':earning,
        'financial_rows_known':len(known),'latest_used_financial_available':max(clocks).isoformat() if clocks else None,
        'financial_core_missing':assessment['core_missing_fields'],'financial_safety_failures':assessment['current_safety_failures'],
        **{k:values.get(k,np.nan) for k in cols},'financial_sources':used}

def catalyst_snapshot(e,date):
    cut=signal_cutoff(date)
    q=e[e.available.between(cut-pd.Timedelta(days=90),cut)] if e is not None and not e.empty else pd.DataFrame()
    if q.empty:return {'catalyst_evidence':'UNKNOWN_CATALOGUE_COVERAGE','dated_order_count90':None,
        'dated_capacity_count90':None,'catalyst_sources':[]}
    if q.available.max()>cut:raise ValueError('Future catalyst source leaked')
    orders=int(q.study_order.sum());capacity=int(q.study_capacity.sum())
    sources=[{k:r.get(k) for k in ['available_at_utc','headline','event_id','attachment_url','source_sha256','study_order','study_capacity']}
             for r in q[q.study_order|q.study_capacity].to_dict('records')]
    return {'catalyst_evidence':'DATED_ORDER_OR_CAPACITY_DISCLOSURE' if orders+capacity else 'OTHER_DATED_EVENTS_ONLY_COVERAGE_INCOMPLETE',
        'dated_order_count90':orders,'dated_capacity_count90':capacity,'catalyst_sources':sources}

def outcome(d,t,months,asof):
    """Censor before calculating; evaluate paths from next-session adjusted open."""
    if t+1>=len(d):return {'status':'PENDING_ENTRY'}
    entry=d.iloc[t+1];target=entry.date+pd.DateOffset(months=months)
    common={'entry_date':entry.date,'target_date':target}
    if (entry.date-d.iloc[t].date).days>5:return {**common,'status':'UNKNOWN_ENTRY_GAP'}
    if target>pd.Timestamp(asof):return {**common,'status':'CENSORED_NOT_MATURE'}
    if d.date.max()<target:return {**common,'status':'UNKNOWN_FOLLOWUP_ENDED'}
    f=d[d.date.between(entry.date,target)]
    if len(f)<16*months or (target-f.date.iloc[-1]).days>5:return {**common,'status':'UNKNOWN_SPARSE_FOLLOWUP'}
    if f.date.diff().dt.days.max()>10:return {**common,'status':'UNKNOWN_FOLLOWUP_GAP'}
    if np.log(f.adj_close).diff().abs().max()>np.log(1.4):return {**common,'status':'UNKNOWN_LARGE_PRICE_JUMP'}
    if not np.isfinite(f[['adj_close','open','high','low','close']].to_numpy(float)).all() or (f[['adj_close','open','low','close']]<=0).any().any():
        return {**common,'status':'UNKNOWN_PRICE'}
    entry_open=float(entry.open*entry.adj_close/entry.close)
    multiplier=(1-COST)/(1+COST)
    wealth=f.adj_close.to_numpy(float)/entry_open*multiplier
    low=f.low.to_numpy(float)*f.adj_close.to_numpy(float)/f.close.to_numpy(float)/entry_open*multiplier
    path=np.r_[1.,wealth];dd=path/np.maximum.accumulate(path)-1
    return {**common,'status':'MATURE_VALID_PROVIDER_PROXY','exit_date':f.date.iloc[-1],
        'endpoint_return':wealth[-1]-1,'best_close_return':wealth.max()-1,
        'worst_adjusted_low_return':low.min()-1,'max_close_drawdown':dd.min(),
        'hit_2x_close':bool((wealth>=2).any()),'loss_30_low':bool((low<=.7).any()),'sessions':len(f)}

def failure20(d,t):
    if t+20>=len(d):return {'failure20_status':'CENSORED_20_SESSIONS','failed_breakout20':None,'low_loss10_20':None}
    f=d.iloc[t+1:t+21];r=d.iloc[t];res=r.prior_resistance63
    if (f.date-r.date).dt.days.max()>45 or f.date.diff().dt.days.max()>10 or np.log(f.adj_close).diff().abs().max()>np.log(1.4):
        return {'failure20_status':'UNKNOWN_PATH_QUALITY','failed_breakout20':None,'low_loss10_20':None}
    if (f.date.iloc[0]-r.date).days>5:return {'failure20_status':'UNKNOWN_ENTRY_GAP','failed_breakout20':None,'low_loss10_20':None}
    is_breakout=bool(np.isfinite(res) and r.adj_close>=.99*res)
    breach=f.adj_close.lt(.97*res) if is_breakout else pd.Series(False,index=f.index)
    entry=f.iloc[0];eo=entry.open*entry.adj_close/entry.close
    low=f.low*f.adj_close/f.close/eo*(1-COST)/(1+COST)-1
    return {'failure20_status':'MATURE_20_SESSIONS','at_breakout_resistance':is_breakout,
        'failed_breakout20':bool(breach.any()) if is_breakout else None,
        'breakout_failure_date':f.loc[breach,'date'].iloc[0] if breach.any() else None,
        'low_loss10_20':bool(low.min()<=-.1)}

def summarize(events, keys):
    d=events.copy();rows=[]
    grouped=[('ALL',d)] if not keys else d.groupby(keys,dropna=False,sort=True)
    for key,g in grouped:
        if not isinstance(key,tuple):key=(key,)
        base={k:v for k,v in zip(keys,key)};base.update(events=len(g),issuers=g['isin'].nunique())
        fail=g[g.failure20_status.eq('MATURE_20_SESSIONS')]
        fden=fail.failed_breakout20.notna()
        base.update(failure20_mature=len(fail),breakout_failure_denominator=int(fden.sum()),
            failed_breakouts=int(fail.loc[fden,'failed_breakout20'].astype(bool).sum()),
            low_loss10_20_count=int(fail.low_loss10_20.fillna(False).astype(bool).sum()))
        for m in HORIZONS:
            h=g[g[f'status_{m}'].eq('MATURE_VALID_PROVIDER_PROXY')]
            vals=pd.to_numeric(h[f'endpoint_return_{m}'],errors='coerce')
            base.update({f'n_{m}m':len(h),f'censored_{m}m':int(g[f'status_{m}'].eq('CENSORED_NOT_MATURE').sum()),
                f'other_unknown_{m}m':len(g)-len(h)-int(g[f'status_{m}'].eq('CENSORED_NOT_MATURE').sum()),
                f'median_return_{m}m':vals.median(),f'mean_return_{m}m':vals.mean(),
                f'win_rate_{m}m':float(vals.gt(0).mean()) if len(h) else np.nan,
                f'endpoint_ge50_{m}m':float(vals.ge(.5).mean()) if len(h) else np.nan,
                f'hit2_rate_{m}m':pd.to_numeric(h[f'hit_2x_close_{m}'],errors='coerce').mean(),
                f'loss30_rate_{m}m':pd.to_numeric(h[f'loss_30_low_{m}'],errors='coerce').mean(),
                f'median_max_drawdown_{m}m':pd.to_numeric(h[f'max_close_drawdown_{m}'],errors='coerce').median(),
                f'median_worst_low_{m}m':pd.to_numeric(h[f'worst_adjusted_low_return_{m}'],errors='coerce').median()})
        rows.append(base)
    return plain(rows)

def protected_hashes(root):
    names=['active_heads.joblib','active_model.json','challenger/research_heads.joblib','challenger/freeze.json','prospective_observations.parquet']
    paths=[root/n for n in names]+list(root.glob('registries/*/*.json'))+list(root.glob('paired_cohorts/*.json'))
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}

def run(root,out,case_zip,asof_override=None):
    from v13_history_views import canonical_histories
    from v13_features import daily_features
    from v13_redevelopment import daily as breakout_features
    from v13_available_financials import bound_source
    from v13_source_quality import business_order_status,capacity_disclosure
    root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    before=protected_hashes(root)
    manifest=hashlib.sha256((root/'remote_manifest.json').read_bytes()).hexdigest()
    report=json.loads((root/'latest_run.json').read_text());asof=asof_override or report['asof']
    case_start=pd.Timestamp(asof)-pd.DateOffset(years=1);six_start=pd.Timestamp(asof)-pd.DateOffset(months=6)
    case_records={}
    with zipfile.ZipFile(case_zip) as z:
        for threshold in (50,70):
            obj=json.loads(z.read(f'RSI_GT{threshold}.json'))
            for r in obj['records']:
                case_records.setdefault(r['isin'],{**r,'current_lists':[]})['current_lists'].append(threshold)
    save_json(out/'protocol.json',PROTOCOL)
    wide=pd.read_parquet(root/'financial_wide.parquet')
    wide['available']=pd.to_datetime(wide.available,utc=True,format='mixed')
    wide['period']=pd.to_datetime(wide.period)
    bound=wide.apply(bound_source,axis=1);rejected_financial_rows=int((~bound).sum());wide=wide[bound]
    financials=dict(tuple(wide.groupby('security_isin')))
    epaths=sorted((root/'sources').glob('*/events.parquet'))
    eventsrc=pd.concat([pd.read_parquet(p) for p in epaths],ignore_index=True) if epaths else pd.DataFrame()
    source_event_rows=len(eventsrc)
    if not eventsrc.empty:
        ids=[c for c in ['venue_index','event_id','available_at_utc','isin','symbol','attachment_url'] if c in eventsrc]
        eventsrc=eventsrc.drop_duplicates(ids).copy()
        eventsrc['available']=pd.to_datetime(eventsrc.available_at_utc,utc=True,format='mixed',errors='coerce')
        eventsrc=eventsrc[eventsrc.available.notna()]
        eventsrc['study_order']=[business_order_status(r.get('text',''),r.get('headline',''))=='BUSINESS_ORDER_DISCLOSURE_NOT_REVENUE' for r in eventsrc.to_dict('records')]
        eventsrc['study_capacity']=[capacity_disclosure(str(v)) for v in eventsrc.get('text',pd.Series('',index=eventsrc.index)).fillna('')]
    catalysts=dict(tuple(eventsrc.groupby(['isin','symbol']))) if not eventsrc.empty else {}
    universe=pd.read_parquet(root/'universe.parquet');sym='nse_current_symbol' if 'nse_current_symbol' in universe else 'symbol'
    wanted=set(zip(universe['isin'],universe[sym]))
    histories,audit=canonical_histories(root/'history/securities',wanted)
    records=[];case_rows=[];source_receipts=[];history_receipts=[];coverage=Counter();snapshots={}
    case_daily=out/'case_histories';case_daily.mkdir(exist_ok=True)
    print('Study source',asof,'histories',len(histories),'financial rows',len(wide),'events',len(eventsrc),flush=True)
    for number,((isin,symbol),(path,bars)) in enumerate(histories.items(),1):
        d=daily_features(bars,asof)
        if d.empty:coverage['empty_history']+=1;continue
        # Attach causal features to the exact same trimmed rows, not a differently warmed series.
        rd=breakout_features(d)
        if not rd.date.equals(d.date):raise ValueError('Feature rows differ')
        for c in rd.columns:
            if c.startswith('rd_'):d[c]=rd[c]
        d=stage_frame(d)
        history_receipts.append({'isin':isin,'symbol':symbol,'path':str(path.relative_to(root)),
            'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'first_date':d.date.min(),'last_date':d.date.max(),'sessions':len(d),
            'historical_only_identity':(isin,symbol) not in wanted})
        coverage['histories_processed']+=1
        coverage['sessions_processed']+=len(d)
        case=case_records.get(isin)
        if case is not None:d.to_parquet(case_daily/(isin+'.parquet'),index=False,compression='zstd')
        def detail(t,threshold):
            r=d.iloc[t];key=(isin,int(pd.Timestamp(r.date).value))
            if key not in snapshots:
                f=financial_snapshot(financials.get(isin),r.date)
                c=catalyst_snapshot(catalysts.get((isin,symbol)),r.date)
                snapshots[key]={**f,**c}
                source_receipts.append({'isin':isin,'symbol':symbol,'date':r.date,'information_cutoff_utc':signal_cutoff(r.date),**snapshots[key]})
            s=snapshots[key]
            row={'isin':isin,'symbol':symbol,'date':r.date,'rsi_threshold':threshold,
                'rsi14':r.rsi14,'entry_stage':r.entry_stage,'provider_close_INR':r.close,
                'information_cutoff_utc':signal_cutoff(r.date),'history_first_date':d.date.iloc[0],
                'left_censored_history':True,'current_case':case is not None,
                'exact_original_V13_qualification':'UNPROVEN_NO_HISTORICAL_FROZEN_PREDICTION_REPLAY',
                **{k:r[k] for k in ['rd_return20','rd_return63','rd_extension200','rd_first_breakout','rd_breakout_age','rd_extension_since_breakout','rd_volume_ratio','prior_resistance63','prior_base_range20','prior_rally63_ending20']},
                **{k:v for k,v in s.items() if k not in ['financial_sources','catalyst_sources']},**failure20(d,t)}
            for months in HORIZONS:row.update({k+'_'+str(months):v for k,v in outcome(d,t,months,asof).items()})
            # Pending horizons have the same columns in every row.
            for months in HORIZONS:
                for k in ['endpoint_return','best_close_return','worst_adjusted_low_return','max_close_drawdown','hit_2x_close','loss_30_low']:
                    row.setdefault(k+'_'+str(months),None)
            return row
        for threshold in (50,70):
            mask=eligible(d,threshold);coverage['qualifying_days_'+str(threshold)]+=int(mask.sum())
            ix=event_indices(mask)
            for j,t in enumerate(ix):
                row=detail(t,threshold);row['first_signal_per_isin_threshold']=j==0;row['signal_year']=int(d.iloc[t].date.year)
                records.append(row)
            if case is not None:
                allix=np.flatnonzero(mask.to_numpy())
                in_window=[int(t) for t in allix if d.iloc[t].date>=case_start]
                before_six=[int(t) for t in allix if d.iloc[t].date<=six_start]
                first=int(allix[0]) if len(allix) else None
                window=in_window[0] if in_window else None
                pre=before_six[-1] if before_six else None
                trace={'isin':isin,'symbol':symbol,'current_lists':case['current_lists'],'rsi_threshold':threshold,
                    'history_first_date':d.date.iloc[0],'first_observable_signal':detail(first,threshold) if first is not None else None,
                    'first_signal_trailing12m':detail(window,threshold) if window is not None else None,
                    'last_signal_before_trailing6m':detail(pre,threshold) if pre is not None else None,
                    'qualifying_days':int(mask.sum()),'episode_count':len(ix),'first_stage_dates':{}}
                for label,candidates in [('first_technical_and_permitted_financials',[int(t) for t in allix]),
                                         ('first_technical_and_permitted_financials_trailing12m',in_window)]:
                    match=None
                    for t in candidates:
                        v=detail(t,threshold)
                        if v['financial_policy_permitted']:
                            match=v;break
                    trace[label]=match
                for stage in STAGES:
                    v=d[mask&d.entry_stage.eq(stage)]
                    trace['first_stage_dates'][stage]=str(v.date.min().date()) if not v.empty else None
                case_rows.append(trace)
        if number%100==0:print('Study progress',number,'events',len(records),'dated snapshots',len(snapshots),flush=True)
    if not records:raise ValueError('Study produced no signals')
    ev=pd.DataFrame(records).sort_values(['date','isin','rsi_threshold']).reset_index(drop=True)
    if ev.duplicated(['date','isin','rsi_threshold']).any():raise ValueError('Duplicate event identity')
    ev.to_parquet(out/'all_signal_events.parquet',index=False,compression='zstd')
    write_lines(out/'all_signal_events.jsonl.gz',ev.to_dict('records'))
    write_lines(out/'dated_evidence_snapshots.jsonl.gz',source_receipts)
    write_lines(out/'history_source_receipts.jsonl.gz',history_receipts)
    failed=ev[ev.failed_breakout20.eq(True)|ev.low_loss10_20.eq(True)]
    write_lines(out/'failed_and_adverse_signals.jsonl.gz',failed.to_dict('records'))
    save_json(out/'case_first_signals.json',case_rows)
    save_json(out/'history_identity_audit.json',audit)
    first=ev[ev.first_signal_per_isin_threshold]
    summary={'all_events':summarize(ev,['rsi_threshold']),
        'financial_policy_permitted_by_stage':summarize(ev[ev.financial_policy_permitted],['rsi_threshold','entry_stage']),
        'by_stage':summarize(ev,['rsi_threshold','entry_stage']),
        'first_per_isin_by_stage':summarize(first,['rsi_threshold','entry_stage']),
        'by_stage_earnings':summarize(ev,['rsi_threshold','entry_stage','earnings_evidence']),
        'by_stage_catalyst':summarize(ev,['rsi_threshold','entry_stage','catalyst_evidence']),
        'by_signal_year':summarize(ev,['rsi_threshold','signal_year']),
        'by_financial_tier':summarize(ev,['rsi_threshold','financial_tier'])}
    save_json(out/'continuation_summary.json',summary)
    # Restrict official source sampling to the diagnostic 12-month first signals,
    # never present this as validation of every historical interval.
    primary=[]
    from v12_3_official_monitor import collect
    samples={}
    for r in case_rows:
        v=r['first_signal_trailing12m']
        if v:samples.setdefault(str(pd.Timestamp(v['date']).date()),[]).append(v)
    for day,rs in sorted(samples.items()):
        try:
            q=collect(day,out/'primary_sessions');p=pd.read_parquet(out/'primary_sessions'/(day+'_official_market.parquet'))
            for r in rs:
                found=p[p['isin'].eq(r['isin'])&p.symbol.eq(r['symbol'])]
                status='UNKNOWN_EXACT_IDENTITY_MISSING_OR_MULTIPLE';price=None;series=None
                if len(found)==1:
                    f=found.iloc[0];price=f.close;series=f.series
                    status='SAME_ISIN_SYMBOL_CLOSE_MATCH' if abs(price-r['provider_close_INR'])<=.011 else 'CLOSE_MISMATCH_POSSIBLE_ACTION_BASIS'
                primary.append({'isin':r['isin'],'symbol':r['symbol'],'rsi_threshold':r['rsi_threshold'],'date':day,'status':status,
                    'official_close_INR':price,'provider_close_INR':r['provider_close_INR'],'historical_series':series,'source_sha256':q['source_sha256']})
        except Exception as exc:
            for r in rs:primary.append({'isin':r['isin'],'symbol':r['symbol'],'rsi_threshold':r['rsi_threshold'],'date':day,'status':'PRIMARY_SOURCE_UNAVAILABLE','error_type':type(exc).__name__})
        print('Primary sample',day,len(rs),flush=True)
    save_json(out/'primary_session_sample_audit.json',primary)
    missing_cases=sorted(set(case_records)-{r['isin'] for r in case_rows})
    after=protected_hashes(root)
    if before!=after:raise ValueError('Protected model/forecast files changed')
    if hashlib.sha256((root/'remote_manifest.json').read_bytes()).hexdigest()!=manifest:raise ValueError('Checkpoint descriptor changed')
    receipt={'version':VERSION,'source_asof':asof,'workflow_run':__import__('os').environ.get('GITHUB_RUN_ID'),
        'completed_at_utc':datetime.now(timezone.utc).isoformat(),'coverage':dict(coverage),
        'history_count':len(histories),'event_count':len(ev),'event_isins':ev['isin'].nunique(),
        'diagnostic_cases_requested':len(case_records),'diagnostic_cases_reconstructed':len({r['isin'] for r in case_rows}),
        'missing_case_isins':missing_cases,'case_window_start':case_start,'prior_6m_start':six_start,
        'financial_rows_source_bound':len(wide),'financial_rows_rejected_unbound':rejected_financial_rows,
        'financial_available_min':wide.available.min(),'financial_available_max':wide.available.max(),
        'catalyst_rows_before_dedup':source_event_rows,'catalyst_rows_after_dedup':len(eventsrc),
        'catalyst_available_min':eventsrc.available.min() if not eventsrc.empty else None,
        'catalyst_available_max':eventsrc.available.max() if not eventsrc.empty else None,
        'dated_evidence_snapshot_count':len(snapshots),'future_source_violations':0,
        'evidence_counts':{c:ev[c].value_counts().to_dict() for c in ['earnings_evidence','catalyst_evidence','financial_tier']},
        'primary_sample_status_counts':dict(Counter(r['status'] for r in primary)),
        'protected_files_count':len(before),'protected_files_unchanged':True,'protected_hashes':before,
        'source_manifest_sha256':manifest,'archive_writes':False,'production_changed':False,'frozen_probabilities_replayed_backwards':False,
        'accuracy_validated':False,'point_in_time_reconstruction_complete':False,
        'retained_source_originals_independently_verified':False,'protocol':PROTOCOL}
    save_json(out/'run_receipt.json',receipt)
    lines=['# V13 first-signal and continuation study','',f'Source: {asof}. Research reconstruction; not historical frozen-model recommendations.','',
        f'{len(histories):,} histories; {ev["isin"].nunique():,} issuers with signals; {len(ev):,} threshold-specific events; {len(failed):,} failed/adverse event rows.',
        '', '## Continuation by entry stage', '', '| RSI > | Stage | Events | Mature 6m | Median net 6m | Median maximum drawdown 6m | 2x close within 6m |',
        '|---:|---|---:|---:|---:|---:|---:|']
    def pct(v):return 'unknown' if v is None else f'{v*100:.2f}%'
    for r in summary['by_stage']:
        lines.append(f"| {r['rsi_threshold']} | {r['entry_stage']} | {r['events']} | {r['n_6m']} | {pct(r['median_return_6m'])} | {pct(r['median_max_drawdown_6m'])} | {pct(r['hit2_rate_6m'])} |")
    lines+=['','## Diagnostic first signals in trailing 12 months','','| Share | RSI > | Signal date | Stage | Net 6m | Status | Financial evidence |',
        '|---|---:|---|---|---:|---|---|']
    for r in case_rows:
        v=r['first_signal_trailing12m']
        if v:lines.append(f"| {r['symbol']} | {r['rsi_threshold']} | {pd.Timestamp(v['date']).date()} | {v['entry_stage']} | {pct(plain(v['endpoint_return_6']))} | {v['status_6']} | {v['earnings_evidence']} |")
        else:lines.append(f"| {r['symbol']} | {r['rsi_threshold']} | None observed | | | | |")
    lines+=['','## Interpretation and limits','',
        'Read protocol.json before comparisons. Earnings and catalysts use only source dates on/before signal close; unknown is not a failure or a passing fact. All failed and censored events remain in the data.',
        'Stage membership is reconstructed from a predeclared current rule, not an original V13 forecast. The 30 current names are selected after the historical period and cannot validate accuracy.',
        'Prices are a provider adjusted-close total-return proxy after 0.5% costs per side. Adjusted-low excursion differs from close-to-close maximum drawdown. Full action intervals are not certified.',
        'Current-listing/retained-history survivor bias, original filing-vintage gaps, overlapping outcomes and missing suspended/delisted histories limit the results. No production model or archive was changed.',
        'See continuation_summary.json for 1/3/6/12-month maturity denominators, failures, evidence strata, year groups and first-per-ISIN sensitivity; case_first_signals.json includes earliest retained and pre-six-month signals.', '']
    (out/'study_report.md').write_text('\n'.join(lines))
    print(json.dumps(plain({k:v for k,v in receipt.items() if k not in ['protected_hashes','protocol']}),indent=2),flush=True)
    return receipt

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);p.add_argument('--case-zip',required=True);p.add_argument('--asof')
    a=p.parse_args();run(a.root,a.out,a.case_zip,a.asof)
