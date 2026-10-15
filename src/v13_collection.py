"""Bounded public-universe collection; immutable raw versions and restart queue."""
from pathlib import Path
import json
import pandas as pd
from v12_dated_source_collection import Store,sha
from v12_dated_training_inputs import normalize_financial_catalog,dated_events
from v12_5_expanded_facts import parse_extra
from v12_5_features import source_wide,features,FINANCIAL,UNSUPPORTED,BASE_FIELDS
from v12_public_history import bounds,parse_chart
from v11_4_exchange_reference_universe import NSE_MASTER,NSE_SME_MASTER,parse_nse_master
from urllib.parse import quote
import numpy as np
import time

def financial_wide(extra):
    keys=['symbol','security_isin','reporting_mode','period_kind','period_end','available_at_utc','source_sha256']
    x=extra.drop_duplicates(keys+['metric','value'])
    if (x.groupby(keys+['metric'],dropna=False).value.nunique()>1).any():raise ValueError('Conflicting original facts')
    w=x.pivot(index=keys,columns='metric',values='value').reset_index()
    # Preserve source URL, explicit economic start and retrieval time through
    # the pivot; comparative periods retain their original document identity.
    meta=[c for c in ('period_start','source_url','first_retrieved_utc') if c in x]
    if meta:
        m=x[keys+meta].drop_duplicates()
        if m.duplicated(keys).any():raise ValueError('Conflicting financial source metadata')
        w=w.merge(m,on=keys,validate='1:1')
    w['period']=pd.to_datetime(w.period_end);w['available']=pd.to_datetime(w.available_at_utc,utc=True,format='mixed')
    return w

def enrich(matrix,wide,only_unknown=False):
    from v13_financial_evidence import balancing_quarters
    wide=balancing_quarters(wide)
    d=matrix.copy();groups={k:g for k,g in wide.groupby('security_isin')}
    columns=[*FINANCIAL,*UNSUPPORTED,*BASE_FIELDS,'four_profitable_quarters','expanded_financial_available_at']
    for c in columns:
        if c not in d:d[c]=None if c=='expanded_financial_available_at' else np.nan
    mask=d.expanded_financial_available_at.isna() if only_unknown else pd.Series(True,index=d.index)
    for i in d.index[mask]:
        date=pd.Timestamp(d.at[i,'date'])
        from v13_information_clock import cutoff
        cut=cutoff(d.loc[i])
        w=groups.get(d.at[i,'isin'])
        if w is None:continue
        known=w[w.available.le(cut)&w.period.le(date)&w.period.ge(bounds(d.date.max())[0])]
        if known.empty:continue
        f,s=features(known,date)
        for c,v in f.items():d.at[i,c]=v
        if s:d.at[i,'expanded_financial_available_at']=max(x['available'] for x in s).isoformat()
    if 'earnings_market_confirmation' in d:d['earnings_market_confirmation']=d.quarter_pat_yoy*d.relative_strength63
    return d

def current_catalogs(root,asof):
    from datetime import datetime,timezone
    folder=Path(root)/'sources'/asof
    # Catalogues change within a trade date. Preserve each run's raw response
    # rather than reusing a morning cache after evening results are published.
    vintage=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    store=Store(folder/'raw'/vintage);low,high=bounds(asof);financial=[];events=[];errors=[]
    for venue in ('equities','sme'):
        for period in ('Annual','Quarterly'):
            try:
                raw,proof=store.get('https://www.nseindia.com/api/corporates-financial-results',{'index':venue,'period':period,'from_date':low.strftime('%d-%m-%Y'),'to_date':high.strftime('%d-%m-%Y')})
                data=json.loads(raw)
                if not isinstance(data,list):raise ValueError('Financial catalog schema')
                financial.extend({'schema':'legacy','venue_index':venue,'catalog_sha256':proof['sha256'],'record':r} for r in data)
            except Exception as e:errors.append({'feed':venue+'_'+period,'error':str(e)[:160]})
        seen=set()
        for page in range(1,101):
            try:
                raw,proof=store.get('https://www.nseindia.com/api/integrated-filing-results',{'index':venue,'type':'Integrated Filing- Financials','page':page,'size':1000,'from_date':low.strftime('%d-%m-%Y'),'to_date':high.strftime('%d-%m-%Y')})
                data=json.loads(raw);rows=data['data'];keys={str(r.get('seq_Id')) for r in rows}
                if int(data.get('page',-1))!=page-1 or (keys and keys.issubset(seen)):raise ValueError('Catalog pagination conflict')
                financial.extend({'schema':'integrated','venue_index':venue,'catalog_sha256':proof['sha256'],'record':r} for r in rows);seen.update(keys)
                if not rows or page*1000>=int(data['totalCount']):break
            except Exception as e:errors.append({'feed':venue+'_integrated','error':str(e)[:160]});break
        try:
            raw,proof=store.get('https://www.nseindia.com/api/corporate-announcements',{'index':venue,'from_date':(high-pd.Timedelta(days=7)).strftime('%d-%m-%Y'),'to_date':high.strftime('%d-%m-%Y')})
            data=json.loads(raw)
            if not isinstance(data,list):raise ValueError('Announcement catalog schema')
            events.extend({'venue_index':venue,'catalog_sha256':proof['sha256'],'record':r} for r in data)
        except Exception as e:errors.append({'feed':venue+'_announcements','error':str(e)[:160]})
        try:
            raw,proof=store.get('https://www.nseindia.com/api/corporates-corporateActions',{'index':venue,'from_date':(high-pd.Timedelta(days=28)).strftime('%d-%m-%Y'),'to_date':high.strftime('%d-%m-%Y')})
            actions=json.loads(raw)
            if not isinstance(actions,list):raise ValueError('Corporate action catalog schema')
            (folder/(venue+'_actions.json')).write_text(json.dumps({'start':str((high-pd.Timedelta(days=28)).date()),'end':str(high.date()),
              'catalog_sha256':proof['sha256'],'requested_url':proof['requested_url'],'retrieved_at_utc':proof['first_retrieved_utc'],
              'venue_index':venue,'records':actions,'absence_is_not_independent_no_action_proof':True},indent=2))
        except Exception as e:errors.append({'feed':venue+'_actions','error':str(e)[:160]})
    folder.mkdir(parents=True,exist_ok=True);(folder/'errors.json').write_text(json.dumps(errors,indent=2))
    if financial:
        catalog=normalize_financial_catalog(financial,asof);path=Path(root)/'catalog.parquet'
        previous=pd.read_parquet(path) if path.exists() else pd.DataFrame()
        catalog,inventory=merge_catalog(previous,catalog,asof)
        catalog.to_parquet(path,index=False)
        (folder/'financial_inventory.json').write_text(json.dumps(inventory,indent=2))
    if events:
        e=dated_events(events,asof);path=folder/'events.parquet'
        if path.exists():e=pd.concat([pd.read_parquet(path),e],ignore_index=True).drop_duplicates(['venue_index','event_id','available_at_utc','symbol','attachment_url'],keep='first')
        e.to_parquet(path,index=False)
    return {'financial_catalog_records':len(financial),'announcement_records':len(events),'errors':errors,'catalog_absence_does_not_verify_no_catalyst':True}

def merge_catalog(previous,fresh,asof):
    """A transient thinner source response cannot erase pending filing versions."""
    keys=['source_url','available_at_utc','reporting_mode']
    low,high=bounds(asof)
    combined=pd.concat([previous,fresh],ignore_index=True).drop_duplicates(keys,keep='first')
    period=pd.to_datetime(combined.period_end)
    available=pd.to_datetime(combined.available_at_utc,utc=True,format='mixed')
    combined=combined[period.between(low,high)&available.ge(low.tz_localize('Asia/Kolkata'))&available.le(high.tz_localize('Asia/Kolkata')+pd.Timedelta(days=1))]
    prior_keys=set(map(tuple,previous[keys].to_numpy())) if not previous.empty else set()
    fresh_keys=set(map(tuple,fresh[keys].to_numpy()))
    return combined,{'source_asof':asof,'previous_versions':len(previous),'fresh_response_versions':len(fresh),
      'retained_versions':len(combined),'previous_versions_absent_from_response':len(prior_keys-fresh_keys),
      'absence_does_not_delete_or_verify_no_filing':True,'outside_four_year_window_retained':False}

def recover_public_catalogue(root,seed_path,asof):
    """Recover earlier public fiscal metadata; originals still need validation."""
    import base64,gzip,hashlib,io,re
    root=Path(root);seed=Path(seed_path);mp=seed/'manifest.json'
    if not mp.exists():return {'status':'NO_RECOVERY_SNAPSHOT','new_versions':0}
    m=json.loads(mp.read_text())
    if m.get('format')!='V13-public-primary-catalogue-recovery-1':raise ValueError('Unknown public catalogue recovery format')
    parts=m.get('parts',[])
    if not parts or any(Path(p).name!=p or not re.fullmatch(r'primary_catalog_[0-9]{8}_[0-9]{2}\.b64',p) for p in parts):raise ValueError('Unsafe public recovery part')
    encoded=''.join((seed/p).read_text() for p in parts)
    if hashlib.sha256(encoded.encode()).hexdigest()!=m['encoded_sha256']:raise ValueError('Public catalogue recovery encoded hash mismatch')
    raw=gzip.decompress(base64.b64decode(encoded,validate=True))
    if len(raw)>12_000_000 or hashlib.sha256(raw).hexdigest()!=m['sha256']:raise ValueError('Public catalogue recovery hash mismatch')
    d=pd.read_parquet(io.BytesIO(raw))
    allowed={'symbol','period_end','available_at_utc','broadcast_at_utc','reporting_mode','source_url','catalog_sha256','source_schema','venue_index','catalog_isin','filing_id','filed_period','filing_type'}
    if set(d.columns)!=allowed or len(d)!=m['rows']:raise ValueError('Public catalogue recovery schema/row count mismatch')
    if not d.source_url.str.fullmatch(r'https://nsearchives\.nseindia\.com/corporate/xbrl/[^?]+\.xml',case=False).all() or not d.catalog_sha256.str.fullmatch(r'[0-9a-f]{64}').all():raise ValueError('Unapproved recovered primary source')
    if not d.venue_index.isin(['equities','sme']).all() or not d.source_schema.isin(['legacy','integrated']).all():raise ValueError('Unapproved recovered venue/schema')
    if not d.available_at_utc.str.contains(r'(?:Z|[+-][0-9]{2}:[0-9]{2})$').all():raise ValueError('Recovered publication clock lacks timezone')
    available=pd.to_datetime(d.available_at_utc,utc=True,format='mixed');period=pd.to_datetime(d.period_end)
    if (available<period.dt.tz_localize('Asia/Kolkata')).any():raise ValueError('Recovered publication precedes financial period')
    path=root/'catalog.parquet';previous=pd.read_parquet(path) if path.exists() else d.iloc[:0].copy()
    merged,_=merge_catalog(previous,d,asof);merged.to_parquet(path,index=False)
    report={'status':'PUBLIC_METADATA_RECOVERED','source_asof':asof,'origin':m['origin'],'source_snapshot_sha256':m['sha256'],
      'public_snapshot_rows':len(d),'new_versions':len(merged)-len(previous),'retained_versions':len(merged),
      'facts_still_require_original_document_validation':True,'forecasts_or_outcomes_changed':False}
    (root/'catalogue_recovery.json').write_text(json.dumps(report,indent=2));return report

def announcement_counts(frame,root,asof):
    paths=list((Path(root)/'sources').glob('*/events.parquet'))
    if not paths:return frame
    e=pd.concat([pd.read_parquet(p) for p in paths],ignore_index=True)
    # Repeated daily catalogue snapshots are not independent order wins.
    identity=[k for k in ('venue_index','event_id','available_at_utc','symbol','attachment_url') if k in e]
    e=e.drop_duplicates(identity)
    e['available']=pd.to_datetime(e.available_at_utc,utc=True,format='mixed')
    from v13_information_clock import cutoff
    cut=max((cutoff(row,asof) for row in frame.to_dict('records')),default=(pd.Timestamp(asof).tz_localize('Asia/Kolkata')+pd.Timedelta(hours=15,minutes=30)).tz_convert('UTC'))
    e=e[e.available.le(cut)&e.available.ge(bounds(asof)[0].tz_localize('Asia/Kolkata'))]
    from v13_source_quality import business_order_status,capacity_disclosure
    # Old stored catalogues retain their original bytes. Recompute the derived
    # flag in memory so cached broad regex flags cannot revive legal orders.
    e['business_order_status']=[business_order_status(z.get('text',''),z.get('headline','')) for z in e.to_dict('records')]
    e['order_disclosure']=e.business_order_status.eq('BUSINESS_ORDER_DISCLOSURE_NOT_REVENUE')
    e['capacity_disclosure']=[capacity_disclosure(z) for z in e.text]
    indexed={}
    for key,g in e.groupby(['isin','symbol']):
        g=g.sort_values('available');values=g.available.astype('int64').to_numpy()
        indexed[key]=(values,np.r_[0,np.cumsum(g.order_disclosure.to_numpy(int))],
          np.r_[0,np.cumsum(g.capacity_disclosure.to_numpy(int))],
          np.r_[0,np.cumsum(g.business_order_status.eq('LEGAL_REGULATORY_NOT_BUSINESS_ORDER').to_numpy(int))])
    d=frame.copy()
    for i,r in d.iterrows():
        rowdate=pd.Timestamp(r.get('date',asof))
        rowcut=cutoff(r,asof)
        stored=indexed.get((r['isin'],r['symbol']))
        if stored is None:continue
        clocks,order,capacity,legal=stored
        left=np.searchsorted(clocks,(rowcut-pd.Timedelta(days=90)).value,side='left');right=np.searchsorted(clocks,rowcut.value,side='right')
        # Observed counts may be known with incomplete catalogue coverage.
        # Absence cannot be made a synthetic zero.
        if right>left:
            d.at[i,'order_disclosure_count90']=int(order[right]-order[left])
            d.at[i,'observed_legal_regulatory_disclosure_count90']=int(legal[right]-legal[left])
            d.at[i,'capacity_disclosure_count90']=int(capacity[right]-capacity[left])
    return d

def historical_event_backfill(root,asof,budget=2):
    """Bounded whole-public-venue windows, confined to the trailing four years."""
    root=Path(root);low,high=bounds(asof);windows=stable_event_windows(asof)
    errors=[];completed=0
    for start,end in windows:
        folder=root/'sources'/('backfill_'+str(start.date())+'_'+str(end.date()))
        if (folder/'complete.json').exists():continue
        if completed>=budget:break
        store=Store(folder/'raw');events=[];ok=True
        for venue in ('equities','sme'):
            try:
                raw,proof=store.get('https://www.nseindia.com/api/corporate-announcements',{'index':venue,'from_date':start.strftime('%d-%m-%Y'),'to_date':end.strftime('%d-%m-%Y')})
                rows=json.loads(raw)
                if not isinstance(rows,list):raise ValueError('Historical event schema')
                events.extend({'record':r,'venue_index':venue,'catalog_sha256':proof['sha256']} for r in rows)
            except Exception as exc:ok=False;errors.append({'start':str(start.date()),'venue':venue,'error':str(exc)[:160]})
        if events:dated_events(events,asof).to_parquet(folder/'events.parquet',index=False)
        if ok:(folder/'complete.json').write_text(json.dumps({'start':str(start.date()),'end':str(end.date()),'records':len(events),'absence_does_not_verify_no_catalyst':True}))
        completed+=1
    return {'windows_attempted':completed,'total_windows':len(windows),'errors':errors,'stable_partitions_prevent_daily_backfill_starvation':True}

def stable_event_windows(asof):
    """Fixed 29-day complete windows first; clipping obeys four-year limits.

    Moving every boundary with asof meant each day started from two new recent
    windows and never progressed into older history. Current announcements are
    already collected separately; stable completed blocks are not refetched.
    """
    low,high=bounds(asof);anchor=pd.Timestamp('2000-01-03')
    start=anchor+pd.Timedelta(days=((low-anchor).days//29)*29);full=[];partial=[]
    while start<=high:
        end=start+pd.Timedelta(days=28);a=max(low,start);b=min(high,end)
        (full if a==start and b==end else partial).append((a,b));start=end+pd.Timedelta(days=1)
    return sorted(full,reverse=True)+sorted(partial,reverse=True)

def universe(root,asof):
    store=Store(Path(root)/'sources'/asof/'raw');frames=[]
    for url in (NSE_MASTER,NSE_SME_MASTER):
        raw,_=store.get(url);frames.append(parse_nse_master(raw))
    d=pd.concat(frames).drop_duplicates('isin')
    if d.nse_current_symbol.duplicated().any():raise ValueError('Public universe symbol conflict')
    d.to_parquet(Path(root)/'universe.parquet',index=False);return d

def collect_histories(root,asof,budget=800):
    root=Path(root);u=pd.read_parquet(root/'universe.parquet');low,high=bounds(asof)
    store=Store(root/'sources'/asof/'history_raw');folder=root/'history'/'securities';folder.mkdir(parents=True,exist_ok=True)
    attempt_path=root/'history_attempts.json'
    attempts=json.loads(attempt_path.read_text()) if attempt_path.exists() else {}
    queue=[]
    for r in u.to_dict('records'):
        path=folder/(sha(r['nse_current_symbol'].encode())+'.parquet')
        h=pd.read_parquet(path) if path.exists() else pd.DataFrame()
        age=str(h.date.max().date()) if not h.empty else '0000'
        age=max(age,attempts.get(r['nse_current_symbol'],'0000'))
        # Public, unranked execution/history evidence prioritises refreshes;
        # private forecasts, selections and realised outcomes are not read.
        liquid=not h.empty and len(h)>=127 and {'close','volume'}.issubset(h) and h.close.iloc[-1]>=20 and h.close.iloc[-1]<=2000 and (h.close*h.volume).tail(63).median()>=2_500_000
        queue.append((age,sha(r['nse_current_symbol'].encode()),r,path.exists(),bool(liquid)))
    errors=[];count=0;started=time.monotonic();attempted=0
    ordered=sorted(queue,key=lambda q:(q[0],q[1]))
    # Keep known usable public-universe histories fresh while expanding cold
    # coverage. No private ranked/selected stock list enters this queue.
    refresh=sorted([q for q in ordered if q[3] and q[0]<asof],key=lambda q:(not q[4],q[0],q[1]))[:budget*3//4]
    chosen={q[1] for q in refresh}
    batch=refresh+[q for q in ordered if q[1] not in chosen][:max(0,budget-len(refresh))]
    def fetch(job):
        r=job[2]
        symbol=r['nse_current_symbol'];isin=r['isin']
        url='https://query1.finance.yahoo.com/v8/finance/chart/'+quote(symbol+'.NS',safe='')+f'?interval=1d&period1={int(low.tz_localize("Asia/Kolkata").timestamp())}&period2={int((high+pd.Timedelta(days=1)).tz_localize("Asia/Kolkata").timestamp())}&events=div%2Csplits'
        try:
            raw,receipt=store.get(url);d,info=parse_chart(raw,symbol,isin,asof,receipt);d.to_parquet(folder/(sha(symbol.encode())+'.parquet'),index=False)
            return {'info':info}
        except Exception as e:return {'error':str(e)[:160]}
    from v13_parallel_collection import bounded_map
    for job,result in bounded_map(batch,fetch,workers=4,seconds=600):
        attempted+=1;symbol=job[2]['nse_current_symbol']
        if result.get('info'):count+=1
        else:errors.append({'symbol':symbol,'error':result['error']})
        attempts[symbol]=asof
    attempt_path.write_text(json.dumps(attempts,indent=2))
    return {'requested':attempted,'usable':count,'errors':errors,'public_universe':len(u),'total_histories':len(list(folder.glob('*.parquet'))),
      'reserved_refresh_slots':len(refresh),'cold_expansion_slots':len(batch)-len(refresh),
      'source_workers':4,'source_rate_limited':True,'partial_coverage_is_not_full_NSE':True}

def collect_financials(root,asof,budget=800):
    root=Path(root);catalog_path=root/'catalog.parquet'
    if not catalog_path.exists():return {'requested':0,'errors':['financial_catalog_missing']}
    cat=pd.read_parquet(catalog_path);symbols=set(pd.read_parquet(root/'universe.parquet').nse_current_symbol)
    cat=cat[cat.symbol.isin(symbols)&cat.reporting_mode.ne('unknown')]
    accepted_path=root/'accepted_filings.json';accepted=set(json.loads(accepted_path.read_text())) if accepted_path.exists() else set()
    cat['version_key']=cat.source_url+'|'+cat.available_at_utc
    # Parser upgrades must revisit accepted originals; a successful old
    # quarter-only extraction does not mean nine-month/annual inputs exist.
    parsed_path=root/'financial_parser_owner_v2.json'
    parsed=set(json.loads(parsed_path.read_text())) if parsed_path.exists() else set()
    attempts_path=root/'filing_attempts.json'
    attempts=json.loads(attempts_path.read_text()) if attempts_path.exists() else {}
    queue=cat[~cat.version_key.isin(parsed)].copy()
    queue['last_attempt']=queue.version_key.map(lambda k:attempts.get(k,'0000'))
    # Complete public-cohort queue, newest disclosures first. No query uses
    # private model picks. Accepted URLs keep immutable original-byte hashes.
    from v13_financial_evidence import priority_queue,completion_queue,owner_completion_queue,balancing_quarters
    wide=pd.read_parquet(root/'financial_wide.parquet') if (root/'financial_wide.parquet').exists() else pd.DataFrame()
    expected_hashes={}
    if not wide.empty:
        for k,g in wide.groupby(['symbol','reporting_mode','available_at_utc']):expected_hashes[k]=set(g.source_sha256)
    remote_path=root/'remote_manifest.json'
    original_hash_by_path={r['path']:r['sha256'] for r in json.loads(remote_path.read_text())['files']} if remote_path.exists() else {}
    public_history_symbols=set()
    for p in (root/'history'/'securities').glob('*.parquet'):
        h=pd.read_parquet(p,columns=['symbol'])
        if not h.empty:public_history_symbols.add(h.symbol.iloc[0])
    fresh_symbols=fresh_public_price_symbols(root,asof)
    scoreable_symbols=fresh_public_price_symbols(root,asof,require_feature_eligible=True)
    # Complete same-basis fiscal bundles, rather than granting one recent
    # filing to each of hundreds of issuers without completing any issuer.
    # Selection/ranking flags are neither read nor sent to a data source.
    expanded=balancing_quarters(wide)
    owner_public=queue[queue.symbol.isin(scoreable_symbols)]
    owner_completion=owner_completion_queue(owner_public,expanded,asof,budget//4)
    owner_keys=set(owner_completion.version_key)
    queue=queue[~queue.version_key.isin(owner_keys)]
    completion_budget=budget*3//4-len(owner_completion)
    scoreable_completion=completion_queue(queue[queue.symbol.isin(scoreable_symbols)],expanded,asof,completion_budget)
    raw_completion=completion_queue(queue[queue.symbol.isin(fresh_symbols)&~queue.version_key.isin(scoreable_completion.version_key)],expanded,asof,max(0,completion_budget-len(scoreable_completion)))
    fresh_completion=pd.concat([scoreable_completion,raw_completion],ignore_index=True)
    other=queue[queue.symbol.isin(public_history_symbols)&~queue.version_key.isin(fresh_completion.version_key)]
    warm_completion=completion_queue(other,expanded,asof,max(0,completion_budget-len(fresh_completion)))
    completion=pd.concat([owner_completion,fresh_completion,warm_completion],ignore_index=True)
    remainder=queue[~queue.version_key.isin(completion.version_key)]
    queue=pd.concat([completion,priority_queue(remainder,wide,asof,max(0,budget-len(completion)))],ignore_index=True)
    rows=[];errors=[];started=time.monotonic();attempted=0
    def fetch(r):
        try:
            store=Store(root/'filing_raw'/sha(r['available_at_utc'].encode()))
            raw,proof=store.get(r['source_url'])
            expected=expected_hashes.get((r['symbol'],r['reporting_mode'],r['available_at_utc']))
            original_path='filing_raw/'+sha(r['available_at_utc'].encode())+'/'+sha(r['source_url'].encode())+'.bin'
            if original_path in original_hash_by_path:expected={original_hash_by_path[original_path]}
            if r['version_key'] in accepted and expected and proof['sha256'] not in expected:
                raise ValueError('Previously accepted original filing bytes changed; revision review required')
            facts=parse_extra(raw,r,asof,include_ytd=True)
            if not facts:raise ValueError('No usable exact-identity financial facts; remains queued')
            return {'facts':[{**r,**f,'source_sha256':proof['sha256'],'first_retrieved_utc':proof['first_retrieved_utc']} for f in facts]}
        except Exception as e:return {'error':str(e)[:160]}
    from v13_parallel_collection import bounded_map
    # A URL/publication is parsed once, avoiding concurrent mode duplicates.
    for r,result in bounded_map(queue.drop_duplicates('version_key').to_dict('records'),fetch,workers=4,seconds=600):
        attempted+=1
        if result.get('facts'):
            rows.extend(result['facts']);accepted.add(r['version_key']);parsed.add(r['version_key'])
        else:errors.append({'source_url':r['source_url'],'error':result['error']})
        attempts[r['version_key']]=asof
    if rows:
        fresh=financial_wide(pd.DataFrame(rows));path=root/'financial_wide.parquet'
        previous=pd.read_parquet(path) if path.exists() else pd.DataFrame()
        combined=pd.concat([previous,fresh],ignore_index=True).drop_duplicates(['source_sha256','period_kind','period_end','security_isin','reporting_mode'],keep='last')
        combined.to_parquet(path,index=False)
    accepted_path.write_text(json.dumps(sorted(accepted)))
    parsed_path.write_text(json.dumps(sorted(parsed)))
    attempts_path.write_text(json.dumps(attempts))
    return {'requested':attempted,'remaining':int(len(cat[~cat.version_key.isin(accepted)])),'accepted_total':len(accepted),'new_facts':len(rows),'errors':errors,
      'completion_bundle_filings':len(completion),'completion_bundle_issuers':int(completion.symbol.nunique()),
      'reserved_owner_earnings_filings':len(owner_completion),'owner_queue_does_not_infer_recurring_EPS':True,
      'fresh_public_price_completion_filings':len(fresh_completion),'fresh_public_price_symbols':len(fresh_symbols),
      'scoreable_public_price_completion_filings':len(scoreable_completion),'scoreable_public_price_symbols':len(scoreable_symbols),
      'parser_owner_v2_completed_versions':len(parsed),'new_facts_include_explicit_ytd':True,
      'source_workers':4,'source_rate_limited':True,'owner_PAT_is_not_recurring_EPS':True}

def fresh_public_price_symbols(root,asof,require_feature_eligible=False):
    """Unranked public raw-price coverage, never a private prediction list."""
    root=Path(root);path=root/'official'/(asof+'_official_market.parquet')
    if not path.exists():return set()
    official=pd.read_parquet(path)
    official=official[official.source_status.eq('VALID_POSITIVE_VOLUME_SESSION')&official.close.between(20,2000)]
    official=official[~official['isin'].duplicated(keep=False)].set_index('isin');symbols=set()
    for p in (root/'history'/'securities').glob('*.parquet'):
        h=pd.read_parquet(p) if require_feature_eligible else pd.read_parquet(p,columns=['symbol','isin','date','close','volume'])
        if h.empty:continue
        r=h.sort_values('date').iloc[-1]
        if pd.Timestamp(r.date)!=pd.Timestamp(asof) or r['isin'] not in official.index:continue
        o=official.loc[r['isin']]
        if not (r.symbol==o.symbol and abs(r.close-o.close)<=.011 and r.volume==o.volume):continue
        if require_feature_eligible:
            # Deterministic public price/liquidity/history checks only. No
            # probability, selection flag, forecast rank or future label.
            from v13_features import daily_features
            if not {'open','high','low','adj_close'}.issubset(h.columns):continue
            if any(abs(r[c]-o[c])>.011 for c in ('open','high','low')):continue
            bars=h[pd.to_datetime(h.date).ge(bounds(asof)[0])]
            d=daily_features(bars,asof);current=d[d.date.eq(pd.Timestamp(asof))]
            if len(current)!=1 or not bool(current.feature_eligible.iloc[0]):continue
        symbols.add(r.symbol)
    return symbols
