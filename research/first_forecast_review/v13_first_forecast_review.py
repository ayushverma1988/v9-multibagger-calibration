"""Review the first actual V13 frozen cohort; run a separate ranking candidate.

No past forecast is rescored with today's head and no model/registry is changed.
Short elapsed returns are not mature six-month performance.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import argparse,hashlib,json,math,re
from pathlib import Path
from urllib.parse import quote,urlencode
import joblib,numpy as np,pandas as pd,requests

def plain(x):
    if isinstance(x,dict):return {str(k):plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [plain(v) for v in x]
    if isinstance(x,(np.integer,np.bool_)):return x.item()
    if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
    if isinstance(x,(pd.Timestamp,datetime)):return x.isoformat()
    if x is pd.NA or x is pd.NaT:return None
    return x

def save(path,obj):Path(path).write_text(json.dumps(plain(obj),indent=2,allow_nan=False))
def protected(root):
    paths=list(root.glob('registries/*/*.json'))+list(root.glob('paired_cohorts/*.json'))
    paths += [root/p for p in ['active_heads.joblib','active_model.json','challenger/research_heads.joblib','challenger/freeze.json','prospective_observations.parquet']]
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}

def run(root,out):
    from v13_lifecycle import validate_registry
    from v12_3_official_monitor import collect
    from v13_model import probabilities
    import v13_available_financials as available
    from v13_priority_research import rank,WEIGHTS as BASELINE_WEIGHTS
    from v13_ranking_candidate import rank_candidate,WEIGHTS as CANDIDATE_WEIGHTS,VERSION
    root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    started=datetime.now(timezone.utc);before=protected(root)
    latest=json.loads((root/'latest_run.json').read_text());end=pd.Timestamp(latest['asof'])
    if end!=pd.Timestamp('2026-11-06'):raise ValueError('Review endpoint needs explicit fresh-source review')
    registers=[]
    for path in (root/'registries/champion').glob('*.json'):
        obj=validate_registry(path);registers.append((pd.Timestamp(obj['registered_at_utc']),path,obj))
    if not registers:raise ValueError('No original champion registries')
    clock,path,reg=min(registers,key=lambda x:(x[0],str(x[1])))
    selected=[r for r in reg['records'] if r.get('selected_verified') or r.get('selected_discovery')]
    if len(selected)!=24 or len(reg['records'])!=670:raise ValueError('First original cohort differs from independent publication')
    source=pd.Timestamp(selected[0]['source_asof']);entry_day=pd.Timestamp('2026-10-12')
    if source!=pd.Timestamp('2026-10-09') or clock.tz_convert('Asia/Kolkata').date()!=pd.Timestamp('2026-10-11').date():
        raise ValueError('Unexpected first registration/source dates')
    if entry_day.tz_localize('Asia/Kolkata')+pd.Timedelta(hours=9,minutes=15)<=clock:
        raise ValueError('Entry must follow actual registration')
    save(out/'original_first_registry.json',reg)
    ledger=pd.read_parquet(root/'prospective_observations.parquet')
    initial=ledger[ledger.registry_sha256.eq(path.stem)].copy()
    if len(initial)!=670 or initial['isin'].duplicated().any():raise ValueError('First frozen ledger cohort incomplete')
    for item in reg['records']:
        q=initial[initial['isin'].eq(item['isin'])].iloc[0]
        for k in ['p6','p12','joint12','loss12','selected_verified','selected_discovery']:
            if k.startswith('selected'):
                if bool(q[k])!=bool(item[k]):raise ValueError('Frozen selected flag differs')
            elif not np.isclose(q[k],item[k],atol=1e-12,rtol=0):raise ValueError('Frozen probability differs')
    initial.to_parquet(out/'first_cohort_ledger.parquet',index=False,compression='zstd')
    save(out/'first_cohort_ledger.json',json.loads(initial.to_json(orient='records',date_format='iso',double_precision=15)))
    print('First cohort',clock.isoformat(),source.date(),len(reg['records']),'selected',len(selected),flush=True)
    source_dir=out/'sources';source_dir.mkdir(exist_ok=True)
    def primary(day):
        collect(str(day.date()),source_dir/'official')
        return str(day.date()),pd.read_parquet(source_dir/'official'/(str(day.date())+'_official_market.parquet'))
    with ThreadPoolExecutor(max_workers=3) as pool:primary_frames=dict(pool.map(primary,[source,entry_day,end]))
    def capture(url,name,session=None):
        caller=session or requests
        response=caller.get(url,headers={'User-Agent':'Mozilla/5.0','Accept':'application/json','Referer':'https://www.nseindia.com/'},timeout=(10,35))
        raw=response.content;(source_dir/(name+'.raw')).write_bytes(raw)
        proof={'url':url,'retrieved_at_utc':datetime.now(timezone.utc).isoformat(),'status':response.status_code,
            'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
        save(source_dir/(name+'_receipt.json'),proof);response.raise_for_status()
        return response.json(),proof
    action_rows=None;action_receipt=None;action_error=None
    url='https://www.nseindia.com/api/corporate-actions?'+urlencode({'index':'equities','from_date':source.strftime('%d-%m-%Y'),'to_date':end.strftime('%d-%m-%Y')})
    try:
        session=requests.Session()
        try:session.get('https://www.nseindia.com/',timeout=(10,15))
        except requests.RequestException:pass
        actions,action_receipt=capture(url,'nse_corporate_actions',session)
        action_rows=actions if isinstance(actions,list) else actions.get('data')
        if not isinstance(action_rows,list):raise ValueError('Unknown action catalogue schema')
    except Exception as exc:action_error=type(exc).__name__;action_rows=None
    low=source-pd.Timedelta(days=14)
    p1=int(low.tz_localize('Asia/Kolkata').timestamp());p2=int((end+pd.Timedelta(days=1)).tz_localize('Asia/Kolkata').timestamp())
    def provider(item):
        symbol=item['symbol'];url='https://query1.finance.yahoo.com/v8/finance/chart/'+quote(symbol+'.NS',safe='')+f'?interval=1d&period1={p1}&period2={p2}&events=div%2Csplits'
        try:
            body,proof=capture(url,'provider_'+symbol);v=body['chart']['result']
            if not isinstance(v,list) or len(v)!=1:raise ValueError('Ambiguous provider result')
            v=v[0];meta=v['meta']
            if meta.get('symbol')!=symbol+'.NS' or meta.get('exchangeName') not in ('NSI','NSE') or meta.get('currency')!='INR' or meta.get('instrumentType')!='EQUITY':
                raise ValueError('Provider identity mismatch')
            q=v['indicators']['quote'][0]
            d=pd.DataFrame({'date':pd.to_datetime(v['timestamp'],unit='s',utc=True).tz_convert('Asia/Kolkata').tz_localize(None).normalize(),
                **{c:q[c] for c in ['open','high','low','close','volume']},'adj_close':v['indicators']['adjclose'][0]['adjclose']})
            if d.date.duplicated().any():raise ValueError('Duplicate provider dates')
            splits=[]
            for e in v.get('events',{}).get('splits',{}).values():
                dt=pd.Timestamp(e['date'],unit='s',tz='UTC').tz_convert('Asia/Kolkata').tz_localize(None).normalize()
                multiplier=float(e['numerator'])/float(e['denominator'])
                if multiplier<=0:raise ValueError('Invalid share factor')
                splits.append({'ex_date':str(dt.date()),'share_multiplier':multiplier})
            return symbol,{'frame':d,'splits':splits,'receipt':proof},None
        except Exception as exc:return symbol,None,type(exc).__name__
    with ThreadPoolExecutor(max_workers=5) as pool:downloaded=list(pool.map(provider,selected))
    providers={k:v for k,v,e in downloaded if v is not None}
    provider_errors=[{'symbol':k,'error_type':e} for k,v,e in downloaded if e]
    def official(item,day):
        f=primary_frames[str(day.date())]
        q=f[f['isin'].eq(item['isin'])&f.symbol.eq(item['symbol'])]
        if len(q)!=1 or q.iloc[0].source_status!='VALID_POSITIVE_VOLUME_SESSION':return None
        return q.iloc[0]
    def known_actions(item,start):
        found=[];factors=[];unsupported=[]
        if action_rows is None:return found,None,unsupported
        for a in action_rows:
            if str(a.get('symbol','')).strip()!=item['symbol']:continue
            dt=pd.to_datetime(a.get('exDate') or a.get('ex_date'),dayfirst=True,errors='coerce')
            if pd.isna(dt) or not start<dt<=end:continue
            subject=str(a.get('subject',''));found.append(a)
            if re.search('bonus',subject,re.I):
                m=re.search(r'(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)',subject)
                if m:factors.append(1+float(m[1])/float(m[2]))
                else:unsupported.append(subject)
            elif re.search(r'split|sub.?division|consolidation',subject,re.I):
                m=re.search(r'from[^0-9]*(\d+(?:\.\d+)?).*?to[^0-9]*(\d+(?:\.\d+)?)',subject,re.I)
                if m:factors.append(float(m[1])/float(m[2]))
                else:unsupported.append(subject)
            elif re.search(r'rights|demerger|merger|reduction|arrangement',subject,re.I):unsupported.append(subject)
        return found,float(np.prod(factors)),unsupported
    records=[]
    for item in selected:
        start=official(item,source);entry=official(item,entry_day);last=official(item,end);pdata=providers.get(item['symbol'])
        r={**item,'frozen_tier':'financial_checks' if item['selected_verified'] else 'discovery',
            'registered_at_utc':clock,'source_date':source,'intended_entry_date':entry_day,'latest_price_date':end,
            'calendar_days_from_source':int((end-source).days),'calendar_days_from_entry':int((end-entry_day).days),
            'calendar_days_from_registration':int((end-clock.tz_convert('Asia/Kolkata').tz_localize(None).normalize()).days),
            'source_close_INR':float(start.close) if start is not None else None,
            'entry_open_INR':float(entry.open) if entry is not None else None,
            'latest_close_INR':float(last.close) if last is not None else None,
            'entry_series':str(entry.series) if entry is not None else None,
            'six_month_deadline':entry_day+pd.DateOffset(months=6),'twelve_month_deadline':entry_day+pd.DateOffset(months=12),
            'six_month_mature':False,'twelve_month_mature':False,'provider_receipt':pdata['receipt'] if pdata else None,
            'frozen_entry_verified':bool(initial.loc[initial['isin'].eq(item['isin']),'entry_verified'].iloc[0])}
        for label,day,base,field in [('source',source,start,'close'),('entry',entry_day,entry,'open')]:
            acts,primary_factor,unsupported=known_actions(item,day)
            r[label+'_primary_actions']=acts;r[label+'_unsupported_actions']=unsupported
            r[label+'_price_change_percent']=None;r[label+'_price_return_status']='UNKNOWN_ENDPOINT_OR_ACTION_DATA'
            if base is None or last is None:continue
            provider_factor=math.prod(e['share_multiplier'] for e in pdata['splits'] if str(day.date())<e['ex_date']<=str(end.date())) if pdata else None
            factor=primary_factor if primary_factor is not None and not unsupported else provider_factor
            if unsupported:r[label+'_price_return_status']='UNSUPPORTED_REORGANISATION_REVIEW_REQUIRED';continue
            if primary_factor is not None and provider_factor is not None and not np.isclose(primary_factor,provider_factor):
                r[label+'_price_return_status']='SHARE_FACTOR_SOURCE_CONFLICT';continue
            if factor is None:continue
            value=100*(last.close*factor/float(base[field])-1);r[label+'_share_multiplier']=factor
            r[label+'_price_change_percent']=value
            r[label+'_price_return_status']='PRIMARY_NSE_ENDPOINTS_WITH_EXCHANGE_ACTION_CATALOGUE' if primary_factor is not None else 'PRIMARY_NSE_ENDPOINTS_WITH_PROVIDER_SHARE_ADJUSTMENTS_PROVISIONAL'
            if pdata:
                d=pdata['frame'];a=d[d.date.eq(day)];b=d[d.date.eq(end)]
                if len(a)==1 and len(b)==1 and a.iloc[0][field]>0:
                    proxy=100*(b.iloc[0].close/a.iloc[0][field]-1)
                    r[label+'_provider_price_change_percent']=proxy
                    r[label+'_provider_reconciles']=bool(abs(value-proxy)<=.20)
                    if not r[label+'_provider_reconciles']:
                        r[label+'_unreconciled_candidate_percent']=value;r[label+'_price_change_percent']=None
                        r[label+'_price_return_status']='PRIMARY_PROVIDER_RETURN_CONFLICT_REVIEW_REQUIRED'
        if pdata and entry is not None and last is not None:
            d=pdata['frame'];a=d[d.date.eq(entry_day)];f=d[d.date.between(entry_day,end)].dropna(subset=['open','high','low','close','adj_close'])
            if len(a)==1 and not f.empty and f.date.iloc[-1]==end:
                a=a.iloc[0];factor=a.adj_close/a.close;eo=a.open*factor
                wealth=f.adj_close.to_numpy(float)/eo*.995/1.005
                low=f.low.to_numpy(float)*f.adj_close.to_numpy(float)/f.close.to_numpy(float)/eo*.995/1.005
                path_wealth=np.r_[1.,wealth]
                r.update(net_total_return_proxy_percent=100*(wealth[-1]-1),
                    max_close_drawdown_percent=100*(path_wealth/np.maximum.accumulate(path_wealth)-1).min(),
                    worst_adjusted_low_excursion_percent=100*(low.min()-1),observed_sessions=len(f),
                    net_2x_close_observed_provider_proxy=bool((wealth>=2).any()),
                    provider_entry_open_matches_primary=bool(abs(a.open-entry.open)<=.011),
                    partial_path_status='PROVISIONAL_PROVIDER_TOTAL_RETURN_PATH_NOT_MATURE_6M')
        records.append(r)
    save(out/'first_forecast_performance.json',records)
    frozen_inputs=latest['exact_prediction_replay']['champion'];folder=root/'prediction_replay'/frozen_inputs['input_parquet_sha256']
    raw=(folder/'inputs.parquet').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=frozen_inputs['input_parquet_sha256']:raise ValueError('Current replay input hash differs')
    model=root/'active_heads.joblib'
    if hashlib.sha256(model.read_bytes()).hexdigest()!=frozen_inputs['model_sha256']:raise ValueError('Current frozen model differs')
    frame=pd.read_parquet(folder/'inputs.parquet');p=probabilities(joblib.load(model),frame)
    saved=json.loads((folder/('champion_'+frozen_inputs['receipt_sha256']+'.json')).read_text())
    if frame['isin'].tolist()!=saved['isins']:raise ValueError('Current replay identities differ')
    for k in ['p6','p12','joint12','loss12']:
        if not np.allclose(p[k],saved['probabilities'][k],atol=1e-12,rtol=0):raise ValueError('Current head replay differs')
        frame[k]=p[k]
    available.validate_exact_replay(frame,root,frozen_inputs)
    wide=pd.read_parquet(root/'financial_wide.parquet');ranking=[];original_threshold=available.RSI_MIN
    try:
        for threshold in [70,50]:
            available.RSI_MIN=threshold
            assessed,checks,audit=available.screen(frame,wide)
            old=rank(assessed,assessed.available_research_eligible)
            new=rank_candidate(assessed,assessed.available_research_eligible)
            pd.testing.assert_frame_equal(old[['isin','p6','p12','joint12','loss12']],new[['isin','p6','p12','joint12','loss12']])
            oldpool=old[old.priority_rank.notna()].sort_values('priority_rank');newpool=new[new.candidate_rank.notna()].sort_values('candidate_rank')
            if threshold==70:
                previous=json.loads((root/'priority_research/latest.json').read_text())
                if oldpool['isin'].tolist()!=[v['isin'] for v in previous['records']]:raise ValueError('Baseline RSI70 ranking does not reproduce saved result')
            cols=['isin','symbol','series','source_close_INR','rsi14','p6','p12','joint12','loss12','available_financial_tier',
                'candidate_score','candidate_rank','candidate_price_discipline','candidate_extension_known','distance_ma200','log_ret20','log_ret63']
            chosen=json.loads(newpool[cols].to_json(orient='records',double_precision=15))
            result={'RSI_strictly_greater_than':threshold,'source_asof':str(end.date()),'eligible':int(assessed.available_research_eligible.sum()),
                'baseline_weights':BASELINE_WEIGHTS,'candidate_weights':CANDIDATE_WEIGHTS,'candidate_version':VERSION,
                'baseline_selected':oldpool['symbol'].tolist(),'candidate_selected':[r['symbol'] for r in chosen],
                'added':[r['symbol'] for r in chosen if r['isin'] not in set(oldpool['isin'])],
                'removed':[r['symbol'] for _,r in oldpool.iterrows() if r['isin'] not in set(newpool['isin'])],
                'baseline_median_loss12':oldpool.loss12.median(),'candidate_median_loss12':newpool.loss12.median(),
                'records':chosen,'frozen_probabilities_unchanged':True,'historical_candidate_performance_validated':False,
                'weights_are_fitted_or_optimal':False,'production_changed':False,'new_model_registered':False}
            save(out/f'ranking_candidate_RSI{threshold}.json',result);ranking.append({k:v for k,v in result.items() if k!='records'})
    finally:available.RSI_MIN=original_threshold
    if protected(root)!=before:raise ValueError('Protected forecasts or heads changed')
    def group_stats(rs):
        vals=[r['entry_price_change_percent'] for r in rs if r['entry_price_change_percent'] is not None]
        return {'rows':len(rs),'known_entry_to_latest_price_returns':len(vals),'unknown':len(rs)-len(vals),
            'positive':sum(x>0 for x in vals),'negative':sum(x<0 for x in vals),'ge75':sum(x>=75 for x in vals),'ge100':sum(x>=100 for x in vals),
            'mean_price_change_percent':float(np.mean(vals)) if vals else None,'median_price_change_percent':float(np.median(vals)) if vals else None}
    result={'started_at_utc':started,'completed_at_utc':datetime.now(timezone.utc),'workflow_run':__import__('os').environ.get('GITHUB_RUN_ID'),
        'source_asof':str(end.date()),'first_forecast_source_date':str(source.date()),'first_registered_at_utc':clock,
        'first_registered_at_IST':clock.tz_convert('Asia/Kolkata'),'first_registry_sha256':path.stem,
        'first_model_sha256':reg['model_sha256'],'first_cohort_records':len(reg['records']),'first_selected':len(selected),
        'first_verified_selected':sum(bool(r['selected_verified']) for r in selected),
        'first_discovery_selected':sum(bool(r['selected_discovery']) for r in selected),'entry_date':str(entry_day.date()),
        'calendar_days_entry_to_price':int((end-entry_day).days),'calendar_days_registration_to_review':int((pd.Timestamp(started).tz_convert('Asia/Kolkata').normalize()-clock.tz_convert('Asia/Kolkata').normalize()).days),
        'earliest_six_month_deadline':str((entry_day+pd.DateOffset(months=6)).date()),
        'earliest_twelve_month_deadline':str((entry_day+pd.DateOffset(months=12)).date()),
        'actual_six_month_mature_forecasts':0,'actual_twelve_month_mature_forecasts':0,
        'performance_groups':{'all':group_stats(records),**{tier:group_stats([r for r in records if r['frozen_tier']==tier]) for tier in ['financial_checks','discovery']}},
        'provider_errors':provider_errors,'action_catalogue_receipt':action_receipt,'action_catalogue_error':action_error,
        'full_corporate_action_intervals_independently_certified':False,
        'protected_files_count':len(before),'protected_files_unchanged':True,'protected_hashes':before,
        'archive_writes':False,'production_changed':False,'heads_retrained':False,'past_forecasts_rescored':False,
        'first_registry_matches_complete_670_row_frozen_ledger':True,'ranking_experiments':ranking,
        'market_date_scope':'6 November official close; 8 November Muhurat session not final at review start; no incomplete-session prices mixed into returns'}
    save(out/'review_receipt.json',result)
    lines=['# First V13 forecast: performance since publication','',f"First frozen registration: {clock.tz_convert('Asia/Kolkata').isoformat()}. Source close: {source.date()}. First eligible entry: {entry_day.date()}.",
        f"Price endpoint: {end.date()}, {(end-entry_day).days} calendar days after the entry date. Six-month maturity begins {(entry_day+pd.DateOffset(months=6)).date()}.",'',
        'The 8 November Muhurat session was unfinished at review start. These are dated closing-price comparisons, not live quotes or mature six-month forecast accuracy.','',
        '| Share | Frozen tier | 2x/6m estimate | 2x/12m estimate | Source close INR | Entry open INR | Latest close INR | Price change from source | Price change from entry | Net total-return proxy from entry |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    fmt=lambda x:'unknown' if x is None else f'{x:.2f}'
    for r in records:
        lines.append(f"| {r['symbol']} | {r['frozen_tier']} | {100*r['p6']:.2f}% | {100*r['p12']:.2f}% | {fmt(r['source_close_INR'])} | {fmt(r['entry_open_INR'])} | {fmt(r['latest_close_INR'])} | {fmt(r['source_price_change_percent'])}% | {fmt(r['entry_price_change_percent'])}% | {fmt(r.get('net_total_return_proxy_percent'))}% |")
    lines+=['','Share-adjusted price returns exclude cash dividends. Net total-return proxies include provider dividend/split adjustment and 0.5% assumed costs per side. Action/identity conflicts remain unknown. Full independent action-interval proof is incomplete.',
        '', '## Separate ranking candidate', '',str(CANDIDATE_WEIGHTS),'',
        'The same current model probabilities and eligibility are retained. More weight is given to lower predicted loss risk and less price extension; large prior rallies remain eligible. No early-breakout label bonus is introduced. This candidate is implemented only for research, has no validated historical improvement, and has not changed production.','']
    (out/'first_forecast_performance.md').write_text('\n'.join(lines))
    print(json.dumps(plain({k:v for k,v in result.items() if k not in ['protected_hashes','ranking_experiments']}),indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);a=p.parse_args();run(a.root,a.out)
