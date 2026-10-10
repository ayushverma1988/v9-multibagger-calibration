"""Dated primary financial/catalyst research inputs, never today's backfill.

Financial observations retain source versions, economic periods, exact units,
reporting basis and max dissemination/revision clocks. Catalyst counts mean
disclosures, not earnings transformations. Missing coverage stays UNKNOWN.
"""
from __future__ import annotations
import argparse,json,re,hashlib
from datetime import datetime
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from xml.etree import ElementTree as ET
import numpy as np
import pandas as pd
from v12_dated_source_collection import Store,sha
from v12_public_history import bounds
from v11_4_verified_current_financials import TAGS,INSTANT,scalar,mode,numeric
from v11_4_longterm_fundamentals import context_map,local
from v11_4_strict_annual_numeric_features import parse_units,dimensional_contexts

FACT_TAGS={**TAGS,'exceptional_items':['ExceptionalItemsBeforeTax','ExceptionalItems'],'other_income':['OtherIncome']}

def exchange_clock(v):
    if v is None or not str(v).strip() or str(v).lower() in {'none','nan','nat'}:return pd.NaT
    for f in ('%d-%b-%Y %H:%M:%S','%d-%b-%Y %H:%M','%Y-%m-%d %H:%M:%S'):
        try:x=datetime.strptime(str(v),f)
        except ValueError:continue
        return pd.Timestamp(x.replace(tzinfo=ZoneInfo('Asia/Kolkata'))).tz_convert('UTC')
    return pd.NaT

def normalize_financial_catalog(items,asof):
    lo,hi=bounds(asof);end=(hi.tz_localize('Asia/Kolkata')+pd.Timedelta(days=1)).tz_convert('UTC');rows=[];errors=[]
    for item in items:
        r=item['record'];integrated=item['schema']=='integrated'
        clocks=[exchange_clock(r.get(k)) for k in (['broadcast_Date','creation_Date','revised_Date'] if integrated else ['broadCastDate','filingDate','exchdisstime','revisedDate'])]
        valid=[t for t in clocks if pd.notna(t)]
        period=pd.to_datetime(r.get('qe_Date') if integrated else r.get('toDate'),format='%d-%b-%Y',errors='coerce')
        # Required broadcast/dissemination times cannot be replaced with a fiscal date.
        if pd.isna(clocks[0]) or pd.isna(period) or not valid:continue
        av=max(valid)
        if period<lo or period>hi or av>end or av<period.tz_localize('Asia/Kolkata'):continue
        url=str(r.get('xbrl') or '')
        if not re.fullmatch(r'https://nsearchives\.nseindia\.com/corporate/xbrl/[^?]+\.xml',url,re.I):continue
        rows.append({'symbol':str(r.get('symbol','')).upper(),'period_end':str(period.date()),'available_at_utc':av.isoformat(),'broadcast_at_utc':clocks[0].isoformat(),'reporting_mode':mode(r.get('consolidated')),'source_url':url,'catalog_sha256':item['catalog_sha256'],'source_schema':item['schema'],'venue_index':item['venue_index'],'catalog_isin':str(r.get('isin') or ''),'filing_id':str(r.get('seq_Id') if integrated else r.get('seqNumber')),'filed_period':str(r.get('period') or ''),'filing_type':str(r.get('type_Sub') or 'legacy')})
    d=pd.DataFrame(rows)
    if d.empty:raise ValueError('No dated primary financial catalog')
    # Same URL with a later revision must become known at its LATEST clock.
    d=d.sort_values('available_at_utc').drop_duplicates('source_url',keep='last')
    return d

def exact_facts(raw,record,asof):
    root=ET.fromstring(raw)
    if local(root.tag)!='xbrl':raise ValueError('Not primary XBRL instance')
    if scalar(root,{'Symbol','NSESymbol'}).upper()!=record['symbol']:raise ValueError('XBRL issuer symbol mismatch')
    document_isin=scalar(root,{'ISIN'}).upper()
    catalog_isin=record.get('catalog_isin','').upper()
    isin=document_isin or catalog_isin
    if not re.fullmatch(r'INE[A-Z0-9]{8}[0-9]',isin):raise ValueError('Original document/catalog equity ISIN unavailable')
    if document_isin and catalog_isin and document_isin!=catalog_isin:raise ValueError('Document/catalog ISIN conflict')
    basis=mode(scalar(root,{'NatureOfReportStandaloneConsolidated'}))
    if basis=='unknown' or basis!=record['reporting_mode']:raise ValueError('XBRL reporting basis mismatch')
    target=pd.Timestamp(record['period_end']).date();lo,hi=bounds(asof)
    ctx=context_map(root);units=parse_units(root);dims=dimensional_contexts(root)
    # Legacy instances may omit xbrli contexts but label exact fiscal dates.
    # Named dates override nothing: contradictions reject the whole document.
    rejected_contexts=[];explicit={}
    for e in root.iter():
        tag=local(e.tag);ref=e.get('contextRef')
        if tag in {'DateOfStartOfReportingPeriod','DateOfEndOfReportingPeriod'} and ref and e.text and not dims.get(ref,0):
            explicit.setdefault(ref,{}).setdefault(tag,set()).add(str(e.text).strip())
    for ref,metadata in explicit.items():
        dates={}
        for tag,key in [('DateOfStartOfReportingPeriod','start'),('DateOfEndOfReportingPeriod','end')]:
            values=metadata.get(tag,set())
            if len(values)>1:raise ValueError('Conflicting explicit economic dates')
            if values:
                v=pd.to_datetime(next(iter(values)),errors='coerce')
                if pd.notna(v):dates[key]=v.date()
        if set(dates)=={'start','end'}:
            c=ctx.get(ref)
            if c and (c.get('start')!=dates['start'] or c.get('end')!=dates['end']):
                rejected_contexts.append(ref)
                continue
            if not c:ctx[ref]={**dates,'duration':(dates['end']-dates['start']).days}
    rows=[]
    for metric,tags in FACT_TAGS.items():
        choices={}
        for e in root.iter():
            tag=local(e.tag);ref=e.get('contextRef');c=ctx.get(ref,{})
            if tag not in tags or ref in rejected_contexts or dims.get(ref,0) or c.get('end')!=target or units.get(e.get('unitRef'))!=['INR']:continue
            value=numeric(e.text)
            if value is None:continue
            start=c.get('start');days=c.get('duration')
            kind='instant' if days is None else 'annual' if 330<=days<=400 else 'quarter' if 60<=days<=120 else 'ytd'
            if pd.Timestamp(target)<lo or pd.Timestamp(target)>hi:continue
            if start is not None and pd.Timestamp(start)<lo:continue
            if metric in INSTANT and kind!='instant':continue
            if metric not in INSTANT and kind not in {'annual','quarter'}:continue
            if metric=='cfo' and kind!='annual':continue
            choices.setdefault(kind,[]).append({'metric':metric,'value_INR':value,'tag':tag,'context':ref,'period_start':str(start or target),'period_end':str(target),'period_kind':kind,'document_isin':document_isin,'security_isin':isin,'identity_source':'DOCUMENT_ISIN' if document_isin else 'ORIGINAL_EXCHANGE_CATALOG_ISIN','reporting_mode':basis,'unit':'INR','decimals':str(e.get('decimals','')),'primary_XBRL_exact_period':True,'rendered_statement_independently_reconciled':False})
        for kind,options in choices.items():
            priority=min(tags.index(x['tag']) for x in options);best=[x for x in options if tags.index(x['tag'])==priority]
            if len({x['value_INR'] for x in best})!=1:continue
            rows.append(best[0])
    if not rows:raise ValueError('No finite exact-period facts inside four-year window')
    return rows,{'document_symbol':record['symbol'],'document_isin':document_isin,'security_isin':isin,'identity_source':'DOCUMENT_ISIN' if document_isin else 'ORIGINAL_EXCHANGE_CATALOG_ISIN','reporting_mode':basis,'bse_scrip_code':scalar(root,{'ScripCode'}),'contexts_with_conflicting_economic_dates_rejected':rejected_contexts,'historical_revision_integrity_independently_proven':False}

def collect_financials(catalogs,universe,output,asof,companies=128):
    out=Path(output);out.mkdir(parents=True,exist_ok=True);store=Store(out/'raw')
    cat=normalize_financial_catalog(json.loads((Path(catalogs)/'financial_catalog.json').read_text()),asof)
    cat.to_parquet(out/'dated_full_financial_catalog.parquet',index=False)
    ref=pd.read_parquet(universe).rename(columns={'nse_current_symbol':'symbol'});symbols=sorted(set(ref.symbol),key=lambda s:sha(str(s).encode()))[:companies]
    public=ref[ref.symbol.isin(symbols)].copy();public.to_parquet(out/'public_hash_control_financial_universe.parquet',index=False)
    selected=cat[cat.symbol.isin(symbols)&cat.reporting_mode.ne('unknown')].copy()
    selected.to_parquet(out/'requested_dated_original_filings.parquet',index=False)
    facts=[];audit=[];errors=[]
    def fetch(r):
        try:
            raw,proof=store.get(r['source_url']);fs,identity=exact_facts(raw,r,asof)
            meta={**r,**identity,'source_sha256':proof['sha256'],'first_retrieved_utc':proof['first_retrieved_utc']}
            return [{**meta,**f} for f in fs],meta,None
        except Exception as e:return [],None,{**r,'error':str(e)[:200]}
    with ThreadPoolExecutor(max_workers=6) as pool:
        for i,(fs,info,error) in enumerate(pool.map(fetch,selected.to_dict('records')),1):
            facts.extend(fs)
            if info:audit.append(info)
            if error:errors.append(error)
            if i%100==0:print('financial originals',i,'usable',len(audit),'facts',len(facts),'errors',len(errors),flush=True)
    pd.DataFrame(facts).to_parquet(out/'dated_primary_financial_facts.parquet',index=False)
    pd.DataFrame(audit).to_parquet(out/'original_filing_identity_audit.parquet',index=False)
    (out/'financial_errors.json').write_text(json.dumps(errors,indent=2))
    (out/'financial_summary.json').write_text(json.dumps({'public_control_issuers':len(public),'full_catalog_filings':len(cat),'original_filings_requested':len(selected),'original_filings_parsed':len(audit),'fact_observations':len(facts),'distinct_issuers_with_facts':len({f['symbol'] for f in facts}),'errors':len(errors),'financial_model_training_completed':False,'production_approved':False},indent=2))

def dated_events(items,asof):
    lo,hi=bounds(asof);rows=[]
    for item in items:
        r=item['record'];a=exchange_clock(r.get('an_dt'));b=exchange_clock(r.get('exchdisstime'))
        if pd.isna(a) or pd.isna(b):continue
        av=max(a,b)
        if av<lo.tz_localize('Asia/Kolkata') or av>hi.tz_localize('Asia/Kolkata')+pd.Timedelta(days=1):continue
        text=str(r.get('desc') or '')+' '+str(r.get('attchmntText') or '')
        rows.append({'symbol':str(r.get('symbol') or '').upper(),'isin':str(r.get('sm_isin') or ''),'available_at_utc':av.isoformat(),'event_id':str(r.get('seq_id') or ''),'headline':str(r.get('desc') or ''),'issuer_name':str(r.get('sm_name') or ''),'text':text,'attachment_url':str(r.get('attchmntFile') or ''),'catalog_sha256':item['catalog_sha256'],'venue_index':item['venue_index'],'order_disclosure':bool(re.search(r'order|contract|award',text,re.I)),'capacity_disclosure':bool(re.search(r'capacity|commission|commercial production',text,re.I)),'earnings_impact_verified':False})
    x=pd.DataFrame(rows)
    if x.empty:raise ValueError('No dated primary announcements')
    return x.sort_values('available_at_utc').drop_duplicates(['venue_index','event_id'],keep='last')

def join_training_inputs(panel_path,financial_path,catalogs,output,asof):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    panel=pd.read_parquet(panel_path);financial=pd.read_parquet(financial_path)
    events=dated_events(json.loads((Path(catalogs)/'announcement_catalog.json').read_text()),asof)
    events.to_parquet(out/'dated_primary_event_metadata.parquet',index=False)
    financial['available']=pd.to_datetime(financial.available_at_utc,utc=True,format='mixed');financial['period']=pd.to_datetime(financial.period_end)
    events['available']=pd.to_datetime(events.available_at_utc,utc=True,format='mixed')
    coverage=json.loads((Path(catalogs)/'coverage.json').read_text());event_groups={s:g for s,g in events.groupby('symbol')};fin_groups={s:g for s,g in financial.groupby('symbol')};rows=[];links=[]
    complete_by_date={}
    for date0 in pd.to_datetime(panel.date).unique():
        date0=pd.Timestamp(date0);start0=date0-pd.Timedelta(days=90);complete0=True
        for venue in ['equities','sme']:
            expected=set(pd.date_range(start0,date0));covered=set()
            for c in coverage:
                if c['venue_index']==venue and c['status']=='CATALOG_RESPONSE' and pd.Timestamp(c['end'])>=start0 and pd.Timestamp(c['start'])<=date0:
                    covered.update(pd.date_range(max(start0,pd.Timestamp(c['start'])),min(date0,pd.Timestamp(c['end']))))
            if not expected.issubset(covered):complete0=False
        complete_by_date[date0]=complete0
    for _,p in panel.iterrows():
        date=pd.Timestamp(p['date']);cut=(date.tz_localize('Asia/Kolkata')+pd.Timedelta(hours=15,minutes=30)).tz_convert('UTC');symbol=p['symbol'];isin=p['isin']
        r={'date':str(date.date()),'symbol':symbol,'isin':isin,'decision_at_utc':cut.isoformat(),'financial_status':'UNKNOWN','catalyst_status':'UNKNOWN'}
        f=fin_groups.get(symbol)
        if f is not None:
            known=f[(f.available<=cut)&f.security_isin.eq(isin)&((date-f.period).dt.days<=450)].copy()
            basis='consolidated' if known.reporting_mode.eq('consolidated').any() else 'standalone';known=known[known.reporting_mode.eq(basis)]
            q=known[known.period_kind.eq('quarter')&known.metric.isin(['revenue','pat'])]
            if len(q):
                end=q.period.max();q=q[q.period.eq(end)].sort_values('available').drop_duplicates('metric',keep='last')
                vals=q.set_index('metric').value_INR
                if {'revenue','pat'}.issubset(vals.index) and vals['revenue']>0 and (date-end).days<=190:
                    r.update({'quarter_pat_margin':vals['pat']/vals['revenue'],'quarter_age_days':(date-end).days,'financial_status':'PARTIAL_DATED_PRIMARY_FACTS'})
                    old=known[known.period_kind.eq('quarter')&known.period.eq(end-pd.DateOffset(years=1))&known.metric.isin(['revenue','pat'])].sort_values('available').drop_duplicates('metric',keep='last').set_index('metric').value_INR
                    for metric,label in [('revenue','quarter_sales_yoy'),('pat','quarter_pat_yoy')]:
                        if metric in old.index and old[metric]>0:r[label]=vals[metric]/old[metric]-1
            annual=known[known.period_kind.eq('annual')&known.metric.isin(['revenue','pat','cfo'])]
            if len(annual):
                end=annual.period.max();annual=annual[annual.period.eq(end)].sort_values('available').drop_duplicates('metric',keep='last');vals=annual.set_index('metric').value_INR
                if {'cfo','pat'}.issubset(vals.index) and vals['pat']>0:r['annual_CFO_PAT']=vals['cfo']/vals['pat'];r['financial_status']='PARTIAL_DATED_PRIMARY_FACTS'
            instant=known[known.period_kind.eq('instant')&known.metric.isin(['equity','borrowings_current','borrowings_noncurrent'])]
            if len(instant):
                end=instant.period.max();iv=instant[instant.period.eq(end)].sort_values('available').drop_duplicates('metric',keep='last').set_index('metric').value_INR
                if {'equity','borrowings_current','borrowings_noncurrent'}.issubset(iv.index) and iv['equity']>0:r['dated_debt_equity']=(iv['borrowings_current']+iv['borrowings_noncurrent'])/iv['equity']
            if len(known):
                r['financial_available_at_utc']=known.available.max().isoformat();r['reporting_mode']=basis
                for _,v in known.iterrows():links.append({'date':r['date'],'symbol':symbol,'kind':'financial','source_sha256':v.source_sha256,'catalog_sha256':v.catalog_sha256,'available_at_utc':v.available_at_utc,'metric':v.metric,'period_end':v.period_end,'source_url':v.source_url})
        start=date-pd.Timedelta(days=90)
        # Both feeds must have successful responses covering the full lookback.
        complete=complete_by_date[date]
        ev=event_groups.get(symbol)
        known_events=ev[(ev.available<=cut)&(ev.available>=start.tz_localize('Asia/Kolkata').tz_convert('UTC'))&ev['isin'].eq(isin)] if ev is not None else pd.DataFrame()
        if complete:
            r['catalyst_status']='DATED_METADATA_ONLY';r['order_disclosure_count90']=int(known_events.order_disclosure.sum()) if len(known_events) else 0;r['capacity_disclosure_count90']=int(known_events.capacity_disclosure.sum()) if len(known_events) else 0
        if len(known_events):links.append({'date':r['date'],'symbol':symbol,'kind':'announcement_metadata','source_sha256':None,'catalog_sha256':None,'catalog_sha256_list':json.dumps(sorted(set(known_events.catalog_sha256))),'event_ids':json.dumps(known_events.event_id.tolist()),'available_at_utc':known_events.available.max().isoformat(),'metric':'OBSERVED_DISCLOSURE_COUNTS','period_end':None,'source_url':None})
        rows.append(r)
    x=pd.DataFrame(rows);x.to_parquet(out/'dated_financial_catalyst_training_input_matrix.parquet',index=False);pd.DataFrame(links).drop_duplicates().to_parquet(out/'dated_training_row_source_links.parquet',index=False)
    summary={'market_panel_rows':len(panel),'dated_financial_partial_rows':int(x.financial_status.ne('UNKNOWN').sum()),'dated_financial_distinct_issuers':int(x.loc[x.financial_status.ne('UNKNOWN'),'symbol'].nunique()),'dated_event_metadata_rows':int(x.catalyst_status.ne('UNKNOWN').sum()),'source_event_records':len(events),'quantified_complete_catalyst_training_rows':0,'full_financial_gate_complete_rows':0,'old_model_weights_modified':False,'unseen_outcomes':0,'production_approved':False,'financial_status_counts':x.financial_status.value_counts().to_dict(),'catalyst_status_counts':x.catalyst_status.value_counts().to_dict()}
    (out/'dated_training_input_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['collect','join'],required=True);p.add_argument('--catalogs',required=True);p.add_argument('--output',required=True);p.add_argument('--asof',default='2026-10-09');p.add_argument('--universe');p.add_argument('--companies',type=int,default=128);p.add_argument('--panel');p.add_argument('--financial');a=p.parse_args()
    if a.kind=='collect':collect_financials(a.catalogs,a.universe,a.output,a.asof,a.companies)
    else:join_training_inputs(a.panel,a.financial,a.catalogs,a.output,a.asof)
if __name__=='__main__':main()
