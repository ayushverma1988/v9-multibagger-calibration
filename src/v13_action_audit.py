"""Primary entry/return reconstruction; incomplete actions remain UNKNOWN.

Supported reviewed events: splits, bonuses and cash dividends. Rights,
mergers/demergers and symbol/ISIN transitions require a dedicated economic
review. A provider factor or a catalogue hash alone cannot certify an outcome.
"""
from pathlib import Path
import json,hashlib
import re
import numpy as np
import pandas as pd
from v12_dated_source_collection import Store
from v13_lifecycle import validate_registry,digest
import v12_four_year_model as core

def validate_action_notice(action,raw):
    """A reviewed factor is bound to an issuer and exact primary excerpts."""
    from v11_4_primary_document_catalysts import extract_pdf
    if hashlib.sha256(raw).hexdigest()!=action['source_sha256']:raise ValueError('Action original hash conflict')
    text=extract_pdf(raw) if raw.startswith(b'%PDF') else raw.decode('utf-8')
    text=re.sub(r'\s+',' ',text)
    symbol=action.get('symbol')
    if not symbol or not (action['isin'] in text or re.search(r'\b'+re.escape(symbol)+r'\b',text,re.I)):raise ValueError('Action original issuer identity not matched')
    needed=['ex_date','cash_per_old_share' if action['kind']=='dividend' else 'share_multiplier']
    for field in needed:
        excerpt=action.get('field_excerpts',{}).get(field)
        if not excerpt or re.sub(r'\s+',' ',excerpt).strip() not in text:raise ValueError('Unbound action field: '+field)
    date_quote=action['field_excerpts']['ex_date']
    if re.search(r'\bproposed|\bplanned|subject\s+to\s+approval',date_quote+' '+action['field_excerpts'].get('share_multiplier',''),re.I):raise ValueError('Conditional intended action is not certified completed')
    if not re.search(r'\bex[ -]?date\b',date_quote,re.I):raise ValueError('Record/payment date is not an exchange ex-date')
    date_pattern=r'\d{4}-\d{2}-\d{2}|\d{1,2}(?:st|nd|rd|th)?[ -](?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[ ,-]+\d{4}|\d{1,2}[/.-]\d{1,2}[/.-]\d{4}'
    match=re.search(r'\bex[ -]?date\b([^0-9]{0,50})('+date_pattern+r')',date_quote,re.I)
    if not match or re.search(r'record|payment|book closure',match.group(1),re.I):raise ValueError('Ex-date field is ambiguous or bound to another date label')
    literal=re.sub(r'(?<=\d)(?:st|nd|rd|th)\b','',match.group(2),flags=re.I)
    parsed=pd.to_datetime(literal,dayfirst=True) if not re.match(r'^\d{4}-',literal) else pd.Timestamp(literal)
    if not action.get('ex_date') or parsed.date()!=pd.Timestamp(action['ex_date']).date():raise ValueError('Action ex-date does not match the original ex-date value')
    from v13_source_quality import bound_number
    if action['kind']=='dividend':
        quote=action['field_excerpts']['cash_per_old_share']
        match=re.search(r'(?:dividend\s*(?:of\s*)?)(?:INR|Rs\.?|₹)\s*([\d,]+(?:\.\d+)?)',quote,re.I)
        if not match or not np.isclose(action.get('cash_per_old_share',np.nan),float(match.group(1).replace(',','')),rtol=1e-9,atol=1e-9):raise ValueError('Dividend amount/unit does not match original dividend clause')
    else:
        ratio=action.get('source_share_ratio',{});quote=action['field_excerpts']['share_multiplier']
        if action['kind']=='split':
            old=ratio.get('old');new=ratio.get('new');kind=ratio.get('kind')
            if kind not in {'face_value','share_counts'} or not old or not new:raise ValueError('Split needs an explicit original share/face-value ratio')
            if not re.search(r'split|sub.?division',quote,re.I):raise ValueError('Original does not identify a split')
            if kind=='face_value' and not re.search(r'face\s+value',quote,re.I):raise ValueError('Face-value ratio not identified')
            match=re.search(r'from\s*(?:INR|Rs\.?|₹)?\s*([\d,]+(?:\.\d+)?)\s*(?:shares?|each)?\s*(?:to|into)\s*(?:INR|Rs\.?|₹)?\s*([\d,]+(?:\.\d+)?)',quote,re.I)
            if not match or not np.isclose(old,float(match.group(1).replace(',',''))) or not np.isclose(new,float(match.group(2).replace(',',''))):raise ValueError('Split ratio direction is not unambiguously bound to original')
            factor=old/new if kind=='face_value' else new/old
        elif action['kind']=='bonus':
            new=ratio.get('new');old=ratio.get('old')
            if new is None or old is None or old<=0 or new<0 or not re.search(r'bonus',quote,re.I):raise ValueError('Bonus needs an explicit new-to-old share ratio')
            match=re.search(r'bonus\s*(?:shares?\s*)?(?:in\s+(?:the\s+)?ratio\s*(?:of\s*)?)?([\d,]+)\s*:\s*([\d,]+)',quote,re.I)
            if not match or new!=int(match.group(1).replace(',','')) or old!=int(match.group(2).replace(',','')):raise ValueError('Bonus ratio direction needs an explicit new:old source ratio')
            factor=1+new/old
        else:raise ValueError('Unsupported primary action kind')
        if not bound_number(old,quote) or not bound_number(new,quote) or not np.isclose(action.get('share_multiplier',np.nan),factor,rtol=1e-9,atol=1e-9):raise ValueError('Action factor does not reconcile the original ratio')
    return True

def collect_calendar(root,asof):
    out=Path(root)/'action_audit';out.mkdir(exist_ok=True)
    store=Store(out/'raw'/asof)
    raw,proof=store.get('https://www.nseindia.com/api/holiday-master',{'type':'trading'})
    data=json.loads(raw)
    if not isinstance(data,dict) or not isinstance(data.get('CM'),list):raise ValueError('Unknown official holiday calendar schema')
    days=[]
    for r in data['CM']:
        d=pd.to_datetime(r['tradingDate'],format='%d-%b-%Y');days.append(str(d.date()))
    year=pd.Timestamp(asof).year
    if year not in {pd.Timestamp(d).year for d in days}:raise ValueError('Official calendar does not cover the requested year')
    obj={'year':year,'holidays':days,'source_url':proof['requested_url'],
      'source_sha256':proof['sha256'],'retrieved_at_utc':proof['first_retrieved_utc'],'special_sessions_require_official_snapshot':True}
    (out/('calendar_'+str(year)+'.json')).write_text(json.dumps(obj,indent=2))
    prior=[json.loads(p.read_text()) for p in out.glob('calendar_*.json')]
    combined={**obj,'years':sorted({z['year'] for z in prior}),'holidays':sorted({d for z in prior for d in z['holidays']}),
      'source_receipts':[{k:z[k] for k in ('year','source_url','source_sha256','retrieved_at_utc')} for z in prior]}
    (out/'calendar.json').write_text(json.dumps(combined,indent=2));return combined

def expected_sessions(start,end,calendar):
    start=pd.Timestamp(start);end=pd.Timestamp(end)
    if calendar is None or not set(range(start.year,end.year+1)).issubset(set(calendar.get('years',[calendar['year']]))):return None
    holiday=set(calendar['holidays'])
    return [d for d in pd.bdate_range(start,end) if str(d.date()) not in holiday]

def backfill(root,asof,budget=8):
    from v12_3_official_monitor import collect
    root=Path(root);errors=[]
    try:calendar=collect_calendar(root,asof)
    except Exception as exc:calendar=None;errors.append({'phase':'calendar','error':str(exc)[:180]})
    regs=list((root/'registries').glob('*/*.json'))
    if not regs:return {'downloaded_sessions':0,'errors':errors}
    start=min(pd.Timestamp(validate_registry(p)['registered_at_utc']).tz_convert('Asia/Kolkata').tz_localize(None).normalize() for p in regs)
    dates=expected_sessions(start,asof,calendar)
    # Business-day attempts are safe as a fallback; failed holiday requests
    # are unknown and never evidence of non-trading.
    if dates is None:dates=list(pd.bdate_range(start,asof))
    count=0
    for date in dates:
        if (root/'official'/(str(date.date())+'_official_market.parquet')).exists():continue
        if count>=budget:break
        count+=1
        try:collect(str(date.date()),root/'official')
        except Exception as exc:errors.append({'phase':'official_backfill','date':str(date.date()),'error':str(exc)[:180]})
    return {'attempted_missing_sessions':count,'calendar_verified':calendar is not None,'errors':errors}

def reconstruct(official,provider,actions):
    """Compare independent total-return ratios with provider adjusted ratios."""
    o=official.sort_values('date').copy();p=provider.sort_values('date').copy()
    if o.empty or o.duplicated('date').any() or p.duplicated('date').any():raise ValueError('Ambiguous price identity')
    columns=['date','open','close','adj_close']+[c for c in ('high','low') if c in p]
    x=o.merge(p[columns],on='date',suffixes=('_official','_provider'),validate='1:1')
    if len(x)!=len(o):raise ValueError('Missing provider sessions')
    if not np.isfinite(x[['open_official','close_official','open_provider','close_provider','adj_close']].to_numpy(float)).all():raise ValueError('Missing prices')
    if not x.source_status.eq('VALID_POSITIVE_VOLUME_SESSION').all():raise ValueError('Unknown official observations')
    if not (x[['open_official','close_official','open_provider','close_provider','adj_close']]>0).all().all():raise ValueError('Nonpositive prices')
    bydate={}
    for a in actions:
        if a['kind'] not in {'split','bonus','dividend'}:raise ValueError('Unsupported rights/reorganisation/identity action')
        bydate.setdefault(pd.Timestamp(a['ex_date']),[]).append(a)
    shares=1.;previous=float(x.close_official.iloc[0]);wealth=[previous]
    for r in x.iloc[1:].itertuples():
        events=bydate.get(r.date,[]);mult=1.;cash=0.
        for a in events:
            if a['kind'] in {'split','bonus'}:
                factor=float(a['share_multiplier'])
                if not factor>0:raise ValueError('Invalid share multiplier')
                mult*=factor
            else:cash+=float(a['cash_per_old_share'])
        wealth.append(wealth[-1]*(float(r.close_official)*mult+cash)/previous)
        previous=float(r.close_official)
    independent=np.asarray(wealth)/wealth[0]
    proxy=x.adj_close.to_numpy(float)/float(x.adj_close.iloc[0])
    if not np.allclose(independent,proxy,rtol=.002,atol=.002):raise ValueError('Primary actions and provider adjustment factors disagree')
    # Provider OHLC may be split adjusted; require SAME within-day raw factor
    # and independently reconstructed close-ratio agreement, not a guessed split.
    rawfactor=x.close_provider/x.close_official
    if not np.allclose(x.open_provider/x.open_official,rawfactor,rtol=.002,atol=.002):raise ValueError('Provider unadjusted OHLC conflict')
    expected=[]
    for date in x.date:
        factor=1.
        for action in actions:
            if action['kind'] in {'split','bonus'} and pd.Timestamp(action['ex_date'])>date:factor/=float(action['share_multiplier'])
        expected.append(factor)
    if not (np.allclose(rawfactor,1.,rtol=.002,atol=.002) or np.allclose(rawfactor,expected,rtol=.002,atol=.002)):
        raise ValueError('Provider raw-price factors are not explained by reviewed share actions')
    return x,independent

def verified_return_paths(x,total_returns,actions):
    """Match original model targets and separately track price-only returns.

    The frozen heads predict dividend-adjusted closing-price doubling and a
    daily-low loss before doubling/deadline. Also expose any-horizon losses
    separately; never relabel the frozen first-event probability as that risk.
    """
    required=['high_official','low_official','high_provider','low_provider']
    if not set(required).issubset(x):raise ValueError('Original/provider OHLC lows required for loss-risk verification')
    values=x[required].to_numpy(float)
    if not np.isfinite(values).all() or not (values>0).all():raise ValueError('Invalid official/provider low or high')
    factor=x.close_provider/x.close_official
    for c in ('high','low'):
        if not np.allclose(x[c+'_provider']/x[c+'_official'],factor,rtol=.002,atol=.002):raise ValueError('Provider '+c+' adjustment not explained by original actions')
    if not (x.low_official<=x[['open_official','close_official']].min(axis=1)).all() or not (x.high_official>=x[['open_official','close_official']].max(axis=1)).all():raise ValueError('Original OHLC order conflict')
    cost=(1-core.COST_PER_SIDE)/(1+core.COST_PER_SIDE)
    total=np.asarray(total_returns)*float(x.close_official.iloc[0])/float(x.open_official.iloc[0])*cost
    low=total*x.low_official.to_numpy(float)/x.close_official.to_numpy(float)
    shares=1.;price=[]
    for index,r in enumerate(x.itertuples()):
        # Shares are acquired at the ex-date open on the entry date. An entry-
        # date split/bonus must not grant pre-entry entitlements a second time.
        if index:
            for a in actions:
                if pd.Timestamp(a['ex_date'])==r.date and a['kind'] in {'split','bonus'}:shares*=float(a['share_multiplier'])
        price.append(shares*float(r.close_official)/float(x.open_official.iloc[0])*cost)
    price=np.asarray(price);price_low=price*x.low_official.to_numpy(float)/x.close_official.to_numpy(float)
    return {'total_return_close':total,'total_return_low':low,'price_only_close':price,'price_only_low':price_low}

def audit(root,official,history,asof):
    root=Path(root);out=root/'action_audit';out.mkdir(exist_ok=True)
    cp=out/'calendar.json';calendar=json.loads(cp.read_text()) if cp.exists() else None
    # Coverage reviews are explicit, original-source-bound intervals. Default
    # is absent: incomplete catalogue monitoring is not certification.
    rp=out/'reviewed_actions.json';reviews=json.loads(rp.read_text()) if rp.exists() else []
    coverage_path=out/'reviewed_coverage.json';coverage=json.loads(coverage_path.read_text()) if coverage_path.exists() else []
    rows=[];lookup={}
    committed={z for p in (root/'paired_cohorts').glob('*.json') for z in (json.loads(p.read_text())['champion_registry'],json.loads(p.read_text())['challenger_registry'])}
    for reg in sorted((root/'registries').glob('*/*.json')):
        obj=validate_registry(reg);clock=pd.Timestamp(obj['registered_at_utc'])
        if 'paired_context' in obj and reg.name not in committed:continue
        for r in obj['records']:
            proof={'registry_sha256':reg.stem,'isin':r['isin'],'entry_verified':False,'independent_outcome_verified':False,
              'verification_status':'UNKNOWN_ENTRY_OR_SOURCE_COVERAGE','confirmed_early_2x':False,'confirmed_risk_event':False}
            p=history.get(r['isin']);o=official[official['isin'].eq(r['isin'])].copy();o['date']=pd.to_datetime(o.date)
            o=o[(o.date.dt.tz_localize('Asia/Kolkata')+pd.Timedelta(hours=9,minutes=15)).gt(clock)&o.date.le(pd.Timestamp(asof))].sort_values('date')
            start=clock.tz_convert('Asia/Kolkata').tz_localize(None).normalize()
            dates=expected_sessions(start,asof,calendar)
            if dates is not None:
                dates=[d for d in dates if d.tz_localize('Asia/Kolkata')+pd.Timedelta(hours=9,minutes=15)>clock]
            if p is not None and dates and not o.empty:
                q=p.copy();q['date']=pd.to_datetime(q.date)
                first=q[(q.date.dt.tz_localize('Asia/Kolkata')+pd.Timedelta(hours=9,minutes=15)).gt(clock)].sort_values('date')
                if o.date.iloc[0]==dates[0] and not first.empty and first.date.iloc[0]==dates[0] and first.symbol.iloc[0]==r['symbol'] and o.symbol.iloc[0]==r['symbol'] and o.source_status.iloc[0]=='VALID_POSITIVE_VOLUME_SESSION':
                    proof['entry_verified']=bool(abs(first.open.iloc[0]-o.open.iloc[0])<=.011)
                    proof['entry_date']=str(dates[0].date());proof['entry_open_INR']=float(o.open.iloc[0]);proof['entry_official_sha256']=o.source_sha256.iloc[0]
                    proof['entry_provider_sha256']=first.source_sha256.iloc[0] if 'source_sha256' in first else None
                    proof['entry_evidence_recorded_at_utc']=pd.Timestamp.now(tz='UTC').isoformat()
                    proof['verification_status']='ENTRY_VERIFIED_ACTION_COVERAGE_UNKNOWN' if proof['entry_verified'] else 'ENTRY_PROVIDER_OPEN_MISMATCH_ACTION_PROOF_REQUIRED'
                    matched=[z for z in coverage if z.get('isin')==r['isin'] and z.get('symbol')==r['symbol'] and pd.Timestamp(z['start'])<=dates[0] and pd.Timestamp(z['end'])>=pd.Timestamp(asof)]
                    try:
                        if not matched:raise ValueError('No independently reviewed full action interval')
                        cv=matched[-1]
                        if cv.get('identity_transitions_reviewed') is not True or cv.get('all_primary_notices_reviewed') is not True:raise ValueError('Unreviewed identity/action interval')
                        # Require originals saved by hash, not just unchecked
                        # user/collector booleans saying evidence is complete.
                        refs=cv.get('primary_source_hashes',[])
                        originals={hashlib.sha256(p.read_bytes()).hexdigest() for p in (out/'raw').rglob('*.bin')}
                        if not refs or not set(refs).issubset(originals):raise ValueError('Coverage originals not available')
                        acts=[a for a in reviews if a['isin']==r['isin'] and dates[0]<=pd.Timestamp(a['ex_date'])<=pd.Timestamp(asof)]
                        if any(a.get('source_sha256') not in originals or not str(a.get('source_url','')).startswith('https://nsearchives.nseindia.com/') for a in acts):raise ValueError('Missing original primary action notice')
                        for a in acts:
                            original=next(p for p in (out/'raw').rglob('*.bin') if hashlib.sha256(p.read_bytes()).hexdigest()==a['source_sha256'])
                            validate_action_notice(a,original.read_bytes())
                        if not set(dates).issubset(set(o.date)):raise ValueError('Missing official sessions')
                        if not o.symbol.eq(r['symbol']).all():raise ValueError('Identity transition requires dedicated economic mapping')
                        horizon_end=dates[0]+pd.DateOffset(months=12)
                        o=o[o.date.le(horizon_end)]
                        x,returns=reconstruct(o,q,acts)
                        proof['entry_verified']=True
                        paths=verified_return_paths(x,returns,acts)
                        ratio=paths['total_return_close'];lows=paths['total_return_low']
                        hit=np.flatnonzero(ratio>=2);loss=np.flatnonzero(lows<=.7)
                        proof['confirmed_early_2x']=bool(len(hit))
                        proof['confirmed_risk_event']=bool(len(loss) and (not len(hit) or loss[0]<=hit[0]))
                        proof['confirmed_any_horizon_loss30']=bool(len(loss))
                        proof['confirmed_price_only_2x']=bool((paths['price_only_close']>=2).any())
                        proof['confirmed_price_only_loss30']=bool((paths['price_only_low']<=.7).any())
                        proof['outcome_target_contract']='FROZEN_DIVIDEND_ADJUSTED_CLOSE_2X_AND_LOW_LOSS30_BEFORE_2X'
                        proof['verification_status']='INDEPENDENT_PRIMARY_PRICES_ACTIONS_AND_PROVIDER_FACTORS_MATCHED'
                        for h in (6,12):
                            end=dates[0]+pd.DateOffset(months=h)
                            if pd.Timestamp(asof)>=end:
                                mask=x.date.le(end)
                                proof['verified_hit2_'+str(h)]=int((ratio[mask]>=2).any());proof['independent_outcome_verified']=True
                                proof['verified_price_only_hit2_'+str(h)]=int((paths['price_only_close'][mask]>=2).any())
                                if h==12:
                                    period_hits=np.flatnonzero(ratio[mask]>=2);period_losses=np.flatnonzero(lows[mask]<=.7)
                                    loss_first=bool(len(period_losses) and (not len(period_hits) or period_losses[0]<=period_hits[0]))
                                    proof['verified_loss30_12']=int(loss_first)
                                    proof['verified_any_horizon_loss30_12']=int(bool(len(period_losses)))
                                    proof['verified_price_only_loss30_12']=int((paths['price_only_low'][mask]<=.7).any())
                                    proof['verified_joint12']=int(bool(len(period_hits)) and not loss_first)
                                    proof['independent_outcome_verified']=True
                    except Exception as exc:proof['verification_status']='UNKNOWN_ACTION_OR_RETURN_RECONCILIATION: '+str(exc)
            lookup[(reg.stem,r['isin'])]=proof;rows.append(proof)
    report={'asof':asof,'observations':len(rows),'verified_entries':sum(z['entry_verified'] for z in rows),
      'confirmed_early_2x':sum(z['confirmed_early_2x'] for z in rows),'independent_mature_outcomes':sum(z['independent_outcome_verified'] for z in rows),'records':rows}
    raw=json.dumps(report,sort_keys=True,indent=2).encode();path=out/(asof+'_audit.json');path.write_bytes(raw)
    audit_sha=hashlib.sha256(raw).hexdigest()
    for proof in lookup.values():proof['action_audit_sha256']=audit_sha
    backlog=[{**z,'required_checks':['complete_primary_action_interval','original_notices_and_identity_transitions','official_unadjusted_sessions','provider_adjustment_factor_reconciliation']} for z in rows if not z['independent_outcome_verified']]
    (out/'action_review_backlog.json').write_text(json.dumps({'source_asof':asof,'complete_registered_cohorts':True,
      'records':backlog,'unknown_is_not_no_action':True},indent=2))
    return lookup,{k:v for k,v in report.items() if k!='records'}
