"""Exploratory retrospective cases; no fitted model, forecast or accuracy claim."""
import argparse
import calendar
import csv
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
import hashlib
import io
import json
import math
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

NSE_MASTER = 'https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv'


class AccessLimited(RuntimeError):
    pass


def capture(url, folder, name):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    for attempt in range(3):
        try:
            request=Request(url,headers={'User-Agent':'Mozilla/5.0 V13 retrospective research','Accept':'*/*'})
            with urlopen(request,timeout=30) as response:raw=response.read()
            break
        except HTTPError as exc:
            if exc.code in (401,403,429):raise AccessLimited('Public source access limited: HTTP '+str(exc.code)) from None
            if exc.code not in (408,425,500,502,503,504) or attempt==2:raise
            time.sleep(2**attempt)
        except (URLError,TimeoutError):
            if attempt==2:raise
            time.sleep(2**attempt)
    (folder/(name+'.raw')).write_bytes(raw)
    receipt={'source_url':url,'observed_at_utc':datetime.now(timezone.utc).isoformat(),
        'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
    (folder/(name+'.receipt.json')).write_text(json.dumps(receipt,indent=2))
    return raw,receipt


def master_records(raw):
    rows={}
    for row in csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))):
        row={k.strip():v.strip() for k,v in row.items()}
        if row.get('SYMBOL'):rows[row['SYMBOL']]=row
    if not rows:raise ValueError('Empty official identity master')
    return rows


def bind_identity(case, rows):
    item=rows.get(case['symbol'])
    if not item:raise ValueError('Symbol absent from official identity master')
    isin=item.get('ISIN NUMBER')
    if not isin or len(isin)!=12:raise ValueError('Invalid official current ISIN')
    if case.get('expected_current_isin') and isin!=case['expected_current_isin']:
        raise ValueError('Official current ISIN conflicts with intake evidence')
    return {'symbol':case['symbol'],'current_isin':isin,'current_company':item['NAME OF COMPANY'],
        'current_series':item['SERIES'],'current_listing_date':item.get('DATE OF LISTING'),
        'historical_identity_chain_verified':False}


def history_url(symbol, start, asof):
    def stamp(day):return int(datetime.combine(day,datetime.min.time(),timezone.utc).timestamp())
    params={'period1':stamp(start),'period2':stamp(asof+timedelta(days=1)),
        'interval':'1d','events':'div,splits','includeAdjustedClose':'true'}
    return 'https://query1.finance.yahoo.com/v8/finance/chart/'+quote(symbol,safe='')+'?'+urlencode(params)


def decode_history(raw, expected_symbol, asof, index=False):
    payload=json.loads(raw)['chart']
    if payload.get('error') or len(payload.get('result') or [])!=1:raise ValueError('History unavailable')
    data=payload['result'][0];meta=data['meta']
    if (meta.get('symbol')!=expected_symbol or meta.get('currency')!='INR' or
        meta.get('instrumentType')!=('INDEX' if index else 'EQUITY') or meta.get('exchangeName')!='NSI'):
        raise ValueError('Provider security, exchange, instrument or currency mismatch')
    quotes=data['indicators']['quote'][0]
    adjusted=data['indicators'].get('adjclose',[{}])[0].get('adjclose',[])
    rows=[];rejected=0;seen=set()
    for i,stamp in enumerate(data['timestamp']):
        day=datetime.fromtimestamp(stamp,timezone.utc).astimezone(ZoneInfo('Asia/Kolkata')).date()
        if day>asof:continue
        if day in seen:raise ValueError('Duplicate provider trading date')
        seen.add(day)
        values={c:quotes.get(c,[])[i] if i<len(quotes.get(c,[])) else None for c in ('open','high','low','close','volume')}
        adj=adjusted[i] if i<len(adjusted) else (values['close'] if index else None)
        price=[values[c] for c in ('open','high','low','close')]+[adj]
        if any(v is None or not math.isfinite(v) or v<=0 for v in price):rejected+=1;continue
        if values['volume'] is None or values['volume']<0 or not math.isfinite(values['volume']):rejected+=1;continue
        if not index and values['volume']==0:rejected+=1;continue
        if values['high']<max(values['open'],values['close'],values['low']) or values['low']>min(values['open'],values['close']):
            rejected+=1;continue
        rows.append({'date':day,**values,'adj_close':adj})
    rows.sort(key=lambda r:r['date'])
    if not rows:raise ValueError('No valid daily provider history')
    return rows,{'provider_meta':meta,'events':data.get('events',{}),'rejected_rows':rejected,
        'adjustment_basis':'PROVIDER_RECONSTRUCTED; FULL_ACTION_INTERVAL_NOT_INDEPENDENTLY_VERIFIED'}


def add_months(day, months):
    total=day.year*12+day.month-1+months;year,month=divmod(total,12);month+=1
    return date(year,month,min(day.day,calendar.monthrange(year,month)[1]))


def features(rows, i):
    closes=[r['adj_close'] for r in rows[:i+1]]
    rsi=None
    if len(closes)>=15:
        changes=[closes[j]-closes[j-1] for j in range(len(closes)-14,len(closes))]
        up=sum(max(d,0) for d in changes);down=sum(max(-d,0) for d in changes)
        if up+down>0:rsi=100*up/(up+down)
    ma50=sum(closes[-50:])/50 if len(closes)>=50 else None
    ma200=sum(closes[-200:])/200 if len(closes)>=200 else None
    prior=[r['adj_close'] for r in rows[max(0,i-63):i]]
    vols=[r['volume'] for r in rows[max(0,i-20):i]]
    volume_ratio=rows[i]['volume']/(sum(vols)/len(vols)) if len(vols)==20 and sum(vols)>0 else None
    return {'rsi14_simple':rsi,'rsi_at_least_50':rsi is not None and rsi>=50,
        'above_ma50':ma50 is not None and closes[-1]>ma50,
        'ma50_above_ma200':ma50 is not None and ma200 is not None and ma50>ma200,
        'distance_ma200':closes[-1]/ma200-1 if ma200 else None,
        'close_above_prior63_closing_high':len(prior)==63 and closes[-1]>max(prior),
        'volume_ratio_prior20':volume_ratio,
        'financial_and_catalyst_inputs_asof_decision':'NOT_RECONSTRUCTED_IN_THIS_PRICE_STUDY'}


def outcome(rows, decision_i, months, asof, calendar_dates, cost=.005):
    base={'horizon_months':months,'entry_date':None,'status':'UNKNOWN_ENTRY',
        'hit2':None,'loss30_before_double_or_deadline':None,'double_before_loss30':None}
    if decision_i+1>=len(rows):return base
    entry=rows[decision_i+1];deadline=add_months(entry['date'],months)
    base.update(entry_date=entry['date'].isoformat(),deadline=deadline.isoformat())
    if asof<deadline:base['status']='IMMATURE';return base
    if calendar_dates is None:base['status']='UNKNOWN_BENCHMARK_CALENDAR';return base
    expected=[d for d in calendar_dates if rows[decision_i]['date']<d<=deadline]
    if not expected or expected[0]!=entry['date']:
        base['status']='UNKNOWN_FIRST_ENTRY_SESSION';return base
    window=[r for r in rows if entry['date']<=r['date']<=deadline]
    if set(expected)!={r['date'] for r in window}:
        base['status']='UNKNOWN_MISSING_OR_NONMATCHING_SESSIONS';return base
    adjusted_entry=entry['open']*entry['adj_close']/entry['close']
    net=lambda value:value/adjusted_entry*(1-cost)/(1+cost)
    hit=next((r['date'] for r in window if net(r['adj_close'])>=2),None)
    stop=next((r['date'] for r in window if net(r['low']*r['adj_close']/r['close'])<=.7),None)
    base.update(status='PROVISIONAL_MATURE_PROVIDER_ACTIONS_UNVERIFIED',
        hit2=int(hit is not None),
        loss30_before_double_or_deadline=int(stop is not None and (hit is None or stop<=hit)),
        double_before_loss30=int(hit is not None and (stop is None or hit<stop)),
        first_double_date=hit.isoformat() if hit else None,
        first_loss30_date=stop.isoformat() if stop else None,
        best_observed_net_closing_multiple=max(net(r['adj_close']) for r in window),
        worst_observed_net_daily_low_multiple=min(net(r['low']*r['adj_close']/r['close']) for r in window),
        costs_and_gap_execution_verified=False)
    return base


def audit_cases(manifest_path, output):
    source=Path(manifest_path).read_bytes();manifest=json.loads(source);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    asof=date.fromisoformat(manifest['analysis_asof']);start=date.fromisoformat(manifest['price_fetch_start'])
    analysis_start=date.fromisoformat(manifest['analysis_start'])
    report={'format':'V13_HISTORICAL_CASES_PRICE_AUDIT_1','asof':asof.isoformat(),
        'observed_at_utc':datetime.now(timezone.utc).isoformat(),'intake_sha256':hashlib.sha256(source).hexdigest(),
        'case_count':len(manifest['cases']),'selected_after_outcomes':True,'accuracy_estimated':False,
        'model_training_or_promotion_performed':False,'production_state_written':False,
        'full_corporate_action_intervals_verified':False,'matched_control_study_completed':False,
        'formal_covid_exclusion_unchanged':True,'cases':[],'errors':[]}
    (out/'intake.json').write_bytes(source)
    official={};calendar_dates=None
    try:
        raw,receipt=capture(NSE_MASTER,out/'sources','nse_identity_master');official=master_records(raw)
        report['identity_master_receipt']=receipt
    except Exception as exc:report['errors'].append({'scope':'identity_master','error_type':type(exc).__name__,'message':str(exc)[:180]})
    try:
        raw,receipt=capture(history_url('^NSEI',start,asof),out/'sources','nifty50_calendar')
        index,meta=decode_history(raw,'^NSEI',asof,index=True);calendar_dates=[r['date'] for r in index]
        if calendar_dates[-1]<(asof-timedelta(days=4)):raise ValueError('Benchmark calendar is stale')
        report['benchmark_calendar_receipt']=receipt
    except Exception as exc:
        calendar_dates=None;report['errors'].append({'scope':'benchmark_calendar','error_type':type(exc).__name__,'message':str(exc)[:180]})
    observations=[];access_limited=False
    for case in manifest['cases']:
        symbol=case['symbol'];item={**case,'implied_quoted_multiple':str(Decimal(case['claimed_end_INR'])/Decimal(case['claimed_start_INR'])),
            'claimed_return_verified':False,'history_collected':False}
        report['cases'].append(item)
        try:item['current_identity']=bind_identity(case,official)
        except ValueError as exc:item['identity_issue']=str(exc)
        if access_limited:
            item['collection_status']='PENDING_SOURCE_ACCESS_LIMIT';continue
        try:
            raw,receipt=capture(history_url(symbol+'.NS',start,asof),out/'sources',symbol)
            rows,meta=decode_history(raw,symbol+'.NS',asof)
            folder=out/'histories';folder.mkdir(exist_ok=True)
            with (folder/(symbol+'.csv')).open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=['date','open','high','low','close','volume','adj_close']);writer.writeheader();writer.writerows(rows)
            (folder/(symbol+'.provider_metadata.json')).write_text(json.dumps(meta,indent=2,allow_nan=False))
            active=[r for r in rows if r['date']>=analysis_start]
            if not active:raise ValueError('No requested analysis-period bars')
            item.update(history_collected=True,collection_status='PROVISIONAL_PROVIDER_HISTORY',receipt=receipt,
                history_rows=len(rows),analysis_rows=len(active),first_analysis_date=active[0]['date'].isoformat(),
                last_analysis_date=active[-1]['date'].isoformat(),first_provider_close=active[0]['close'],
                last_provider_close=active[-1]['close'],start_to_end_provider_adjusted_multiple=active[-1]['adj_close']/active[0]['adj_close'],
                full_requested_start_coverage=active[0]['date']<=analysis_start+timedelta(days=7),
                historical_identity_chain_verified=False)
            month_ends={}
            for i,row in enumerate(rows):
                day=row['date']
                if day>=analysis_start and (day.year,day.month)<(asof.year,asof.month):month_ends[(day.year,day.month)]=i
            for i in month_ends.values():
                f=features(rows,i)
                for months in manifest['horizons_months']:
                    observations.append({'symbol':symbol,'decision_date':rows[i]['date'].isoformat(),
                        'covid_2020_case_study_only':rows[i]['date'].year==2020,**f,
                        **outcome(rows,i,months,asof,calendar_dates,manifest['cost_per_side'])})
            print(json.dumps({'case':symbol,'bars':len(rows),'first':item['first_analysis_date'],'last':item['last_analysis_date']}),flush=True)
        except Exception as exc:
            access_limited=isinstance(exc,AccessLimited)
            item['collection_status']='SOURCE_COLLECTION_FAILED';item['error_type']=type(exc).__name__;item['error']=str(exc)[:180]
            report['errors'].append({'scope':symbol,'error_type':type(exc).__name__,'message':str(exc)[:180]})
        time.sleep(.3)
    report['histories_collected']=sum(r['history_collected'] for r in report['cases'])
    report['current_official_identities_matched']=sum('current_identity' in r for r in report['cases'])
    report['month_end_horizon_observations']=len(observations)
    report['outcome_status_counts']={s:sum(r['status']==s for r in observations) for s in sorted({r['status'] for r in observations})}
    report['qualified_next_step']='Review complete action and identity intervals, reconstruct dated financial/catalyst evidence and compare against a historical matched control universe before any model fitting.'
    (out/'month_end_observations.json').write_text(json.dumps(observations,indent=2,allow_nan=False))
    (out/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps({k:v for k,v in report.items() if k not in ('cases','errors')},indent=2),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    report=audit_cases(a.manifest,a.output)
    if report['histories_collected']!=report['case_count']:raise SystemExit(1)
