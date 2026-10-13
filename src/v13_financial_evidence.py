"""Dated financial completeness, explicit earnings feasibility and source links."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from v12_5_features import features,pick,number
from v12_public_history import bounds

REQUIRED=('four_profitable_quarters','quarter_sales_yoy','quarter_pat_yoy','quarter_pat_margin',
          'annual_CFO_PAT','dated_debt_equity','annual_EBIT_proxy_interest_coverage')

def balancing_quarters(wide):
    """Explicit annual minus nine-month amounts; never subtract EPS/shares.

    Same issuer, basis and economic start are mandatory. This is a balancing
    quarter reconstructed at the later original availability clock, not an
    independently reported/recurring quarter or a claim of revision integrity.
    Original rows are unchanged and both source hashes remain attached.
    """
    if wide.empty or 'period_start' not in wide:return wide.copy()
    derived=[]
    for _,g in wide.groupby(['security_isin','reporting_mode']):
        for a in g[g.period_kind.eq('annual')].to_dict('records'):
            start=pd.to_datetime(a.get('period_start'),errors='coerce')
            if pd.isna(start) or not 330<=(a['period']-start).days<=400:continue
            y=g[g.period_kind.eq('ytd')&g.period_start.eq(a['period_start'])&
                (a['period']-g.period).dt.days.between(60,120)]
            if y.empty:continue
            for z in y.to_dict('records'):
                if not 240<=(z['period']-start).days<=310:continue
                available=max(a['available'],z['available'])
                explicit=g[g.period_kind.eq('quarter')&g.period.eq(a['period'])&g.available.le(available)]
                if not explicit.empty:continue
                row={k:a[k] for k in ('symbol','security_isin','reporting_mode','period_end','period','source_sha256')}
                row.update(period_kind='quarter',period_start=str((z['period']+pd.Timedelta(days=1)).date()),
                  available=available,available_at_utc=available.isoformat(),derived_balancing_quarter=True,
                  source_url=a.get('source_url'),input_source_sha256=json.dumps([a['source_sha256'],z['source_sha256']]),
                  input_source_urls=json.dumps([a.get('source_url'),z.get('source_url')]),
                  historical_revision_integrity_independently_proven=False)
                for key in ('revenue','pat','pbt','finance_cost','depreciation','exceptional_items','other_income'):
                    av=number(a,key);yv=number(z,key)
                    if np.isfinite(av) and np.isfinite(yv):row[key]=av-yv
                if 'pat' in row or 'revenue' in row:derived.append(row)
    if not derived:return wide.copy()
    add=pd.DataFrame(derived).drop_duplicates(['security_isin','reporting_mode','period_end','available_at_utc','input_source_sha256'])
    return pd.concat([wide,add],ignore_index=True)

def completion_queue(queue,wide,asof,budget):
    """Bounded issuer bundles, independent of forecast selections or returns.

    Breadth-only round robin can leave every issuer with just one quarter.
    Complete recent quarters, their year-ago comparisons and two annual
    statements on ONE reporting basis before moving to the next issuer.
    """
    if queue.empty or budget<=0:return queue.iloc[:0].copy()
    date=pd.Timestamp(asof);low,_=bounds(asof)
    cut=(date.tz_localize('Asia/Kolkata')+pd.Timedelta(hours=15,minutes=30)).tz_convert('UTC')
    q=queue.copy();q['_period']=pd.to_datetime(q.period_end)
    q=q[q['_period'].between(low,date)&pd.to_datetime(q.available_at_utc,utc=True,format='mixed').le(cut)]
    groups={} if wide.empty else dict(tuple(wide.groupby('symbol')))
    bundles=[]
    for symbol,g in q.groupby('symbol'):
        w=groups.get(symbol,pd.DataFrame())
        if not w.empty:w=w[w.available.le(cut)&w.period.between(low,date)]
        recent=g[g['_period'].ge(date-pd.Timedelta(days=190))]
        observed_recent=w[w.period_kind.eq('quarter') & (date-w.period).dt.days.between(0,190)] if not w.empty else w
        # The pending queue no longer contains an already parsed current
        # filing. Its observed quarter must still anchor older comparison
        # periods; otherwise completing the newest filing starves its issuer.
        if recent.empty and observed_recent.empty:continue
        anchor=observed_recent if not observed_recent.empty else recent
        basis='consolidated' if anchor.reporting_mode.eq('consolidated').any() else 'standalone'
        g=g[g.reporting_mode.eq(basis)]
        if not w.empty:w=w[w.reporting_mode.eq(basis)]
        endpoints=[g['_period'].max()]
        if not w.empty:endpoints.append(w.loc[w.period_kind.eq('quarter'),'period'].max())
        latest=max(z for z in endpoints if pd.notna(z))
        if pd.isna(latest):continue
        periods={latest-pd.offsets.MonthEnd(k) for k in (0,3,6,9,12,15,18)}
        annual=g[g.filed_period.str.contains('annual',case=False,na=False)&g['_period'].ge(date-pd.DateOffset(months=27))] if 'filed_period' in g else g.iloc[:0]
        periods.update(sorted(annual['_period'].unique(),reverse=True)[:2])
        target=g[g['_period'].isin(periods)].copy()
        if target.empty:continue
        # One latest pending version per fiscal endpoint; versions already
        # accepted remain in the immutable wide/provenance store.
        target=target.sort_values('available_at_utc',ascending=False).drop_duplicates('_period')
        # A parser-version upgrade is not itself a missing economic period.
        # Complete absent facts first; otherwise accepted current filings can
        # consume every slot while year-ago originals remain untouched.
        def economic_gap(period):
            if w.empty:return True
            quarter=pick(w,'quarter',period)
            if not quarter or any(not np.isfinite(number(quarter,k)) for k in ('revenue','pat')):return True
            if period in set(pd.to_datetime(annual['_period'])):
                a=pick(w,'annual',period);b=pick(w,'instant',period)
                if not a or any(not np.isfinite(number(a,k)) for k in ('revenue','pat','cfo','pbt','finance_cost')):return True
                if not b or any(not np.isfinite(number(b,k)) for k in ('equity','borrowings_current','borrowings_noncurrent')):return True
            # A parsed annual without its March quarter needs a nine-month
            # original too, even when December's quarter is already present.
            for a in w[w.period_kind.eq('annual')].to_dict('records'):
                if a['period']-pd.offsets.MonthEnd(3)==period and not pick(w,'quarter',a['period']) and not pick(w,'ytd',period):return True
            return False
        target=target[[economic_gap(p) for p in target['_period']]]
        if target.empty:continue
        f,_=features(w,date) if not w.empty else ({},[])
        missing=sum(not np.isfinite(f.get(k,np.nan)) for k in REQUIRED)
        if not missing:continue
        target=target.sort_values(['last_attempt','_period'],ascending=[True,False]).head(8)
        bundles.append((missing,str(target.last_attempt.min()),__import__('hashlib').sha256(symbol.encode()).hexdigest(),target))
    chosen=[];count=0
    for _,_,_,bundle in sorted(bundles,key=lambda z:z[:3]):
        room=budget-count
        if room<=0:break
        selected=bundle.head(room);chosen.append(selected);count+=len(selected)
    return pd.concat(chosen,ignore_index=True).drop(columns='_period') if chosen else queue.iloc[:0].copy()

def priority_queue(queue,wide,asof,budget):
    """Fill public-issuer quarter/annual gaps, then rotate remaining attempts.

    Uses the complete public catalogue, not outbound requests for private picks.
    Diversity keeps one large issuer from consuming an entire collection batch.
    """
    q=queue.copy();low,_=bounds(asof)
    if q.empty:return q
    existing={}
    if not wide.empty:
        for symbol,g in wide.groupby('symbol'):existing[symbol]=set(g.period_end.astype(str))
    period=pd.to_datetime(q.period_end)
    q['missing_period']=[str(p.date()) not in existing.get(s,set()) for s,p in zip(q.symbol,period)]
    q['recent_period']=period.ge(pd.Timestamp(asof)-pd.DateOffset(months=27))&period.ge(low)
    q=q.sort_values(['last_attempt','recent_period','missing_period','available_at_utc','symbol'],ascending=[True,False,False,False,True])
    q['issuer_round']=q.groupby('symbol').cumcount()
    return q.sort_values(['last_attempt','issuer_round','recent_period','missing_period','available_at_utc','symbol'],ascending=[True,True,False,False,False,True]).head(budget)

def enrich_evidence(frame,wide,root,asof):
    wide=balancing_quarters(wide)
    d=frame.copy();links=[];coverage=[];low,_=bounds(asof)
    for c in ('ttm_reported_PAT_INR','annual_reported_revenue_INR','latest_issued_shares','latest_diluted_shares','ttm_reported_eps','implied_reported_PE','required_eps_at_current_PE_for_2x','scenario_base_multiple','scenario_net_multiple','historical_sector_relative_PE'):
        if c not in d:d[c]=np.nan
    d['financial_complete']=False;d['financial_unknown_fields']='';d['evidence_status']='MISSING_FINANCIAL_OR_EXECUTION'
    d['earnings_scenario_status']='UNKNOWN_FORWARD_EARNINGS_AND_SECTOR_VALUATION'
    for i,r in d.iterrows():
        date=pd.Timestamp(r.date);cut=(date.tz_localize('Asia/Kolkata')+pd.Timedelta(hours=15,minutes=30)).tz_convert('UTC')
        w=wide[wide.security_isin.eq(r['isin'])&wide.available.le(cut)&wide.period.le(date)&wide.period.ge(low)] if not wide.empty else pd.DataFrame()
        f,used=features(w,date) if not w.empty else ({},[])
        unknown=[k for k in REQUIRED if not np.isfinite(f.get(k,np.nan))]
        d.at[i,'financial_complete']=not unknown;d.at[i,'financial_unknown_fields']='|'.join(unknown)
        coverage.append({'isin':r['isin'],'symbol':r['symbol'],'source_asof':str(date.date()),'unknown_fields':unknown,'financial_complete':not unknown})
        for z in used:
            item={'isin':r['isin'],'source_asof':str(date.date()),'period_end':z['period_end'],
              'reporting_mode':z['reporting_mode'],'available_at_utc':z['available_at_utc'],
              'source_sha256':z['source_sha256'],'source_url':z.get('source_url'),
              'derived_balancing_quarter':bool(z.get('derived_balancing_quarter') is True),
              'input_source_sha256':z.get('input_source_sha256'),'input_source_urls':z.get('input_source_urls')}
            links.append({k:None if isinstance(v,float) and not np.isfinite(v) else v for k,v in item.items()})
        if w.empty:continue
        recent=w[w.period_kind.eq('quarter')&(date-w.period).dt.days.between(0,190)]
        basis='consolidated' if recent.reporting_mode.eq('consolidated').any() else 'standalone'
        w=w[w.reporting_mode.eq(basis)];q=pick(w,'quarter')
        if not q or (date-q['period']).days>190:continue
        annual=pick(w,'annual')
        if annual and (date-annual['period']).days<=450:d.at[i,'annual_reported_revenue_INR']=number(annual,'revenue')
        capital=number(q,'paidup_equity_capital');face=number(q,'face_value')
        if capital>0 and face>0:d.at[i,'latest_issued_shares']=capital/face
        quarters=[pick(w,'quarter',q['period']-pd.offsets.MonthEnd(k)) for k in (0,3,6,9)]
        pats=[number(z,'pat') for z in quarters];shares=number(q,'weighted_diluted_shares')
        if all(np.isfinite(x) for x in pats):d.at[i,'ttm_reported_PAT_INR']=sum(pats)
        if all(np.isfinite(x) for x in pats) and shares>0:
            pat=sum(pats);eps=pat/shares
            d.at[i,'latest_diluted_shares']=shares;d.at[i,'ttm_reported_eps']=eps
            if eps>0:
                d.at[i,'implied_reported_PE']=float(r.close)/eps
                d.at[i,'required_eps_at_current_PE_for_2x']=2*eps*1.005/.995
                d.at[i,'earnings_scenario_status']='REPORTED_EARNINGS_ONLY_FORWARD_GROWTH_UNKNOWN'
        # Latest sector memberships or guessed growth are never applied to
        # history. A reviewed scenario must bind every input to dated sources.
    out=Path(root)/'evidence';out.mkdir(exist_ok=True)
    (out/'financial_completeness.json').write_text(json.dumps({'asof':asof,'rows':len(d),
      'complete_rows':int(d.financial_complete.sum()),'issuer_backlog':coverage,
      'reported_EPS_is_not_recurring_EPS':True},indent=2))
    (out/'financial_source_links.json').write_text(json.dumps(links,indent=2,default=str))
    # Reverse arithmetic, explicitly hypothetical; it does not assign a PE,
    # forward margin or earnings growth to any company.
    target=2*1.005/.995
    stress=[{'assumed_PE_multiple':pe,'assumed_forward_share_multiple':shares,
             'required_PAT_growth':target*shares/pe-1} for pe in (.8,1.,1.2,1.4) for shares in (1.,1.1,1.25)]
    (out/'earnings_feasibility.json').write_text(json.dumps({'source_asof':asof,'status':'REVERSE_SCENARIO_ARITHMETIC_NOT_PREDICTION',
      'assumptions_not_source_verified':True,'net_return_target':2,'required_growth_grid':stress,
      'reported_TTM_PAT_available':int(d.ttm_reported_PAT_INR.notna().sum()),
      'issued_share_count_is_not_diluted_weighted_share_count':True,
      'independently_supported_forward_PAT_and_sector_PE_required':True},indent=2))
    return d

def earnings_scenario(price,current_pat,current_shares,forward_pat,forward_shares,sector_pe):
    """Arithmetic only; callers must supply source-bound forward assumptions."""
    vals=[price,current_pat,current_shares,forward_pat,forward_shares,sector_pe]
    if not all(x is not None and np.isfinite(x) and x>0 for x in vals):return {'status':'UNKNOWN_MISSING_OR_NONPOSITIVE_INPUT'}
    eps=forward_pat/forward_shares;target=eps*sector_pe
    return {'status':'SCENARIO_NOT_PROBABILITY','forward_diluted_eps':eps,'scenario_price':target,
      'scenario_multiple':target/price,'net_multiple_after_registered_costs':target/price*.995/1.005,
      'earnings_growth':forward_pat/current_pat-1,'share_growth':forward_shares/current_shares-1}

def apply_scenarios(frame,root):
    """Reviewed scenario assumptions, never guessed growth or PE targets."""
    root=Path(root);path=root/'evidence'/'reviewed_earnings_scenarios.json';d=frame.copy();audit=[]
    if not path.exists():return d
    originals={__import__('hashlib').sha256(p.read_bytes()).hexdigest():p for p in root.rglob('*.bin')}
    for r in json.loads(path.read_text()):
        cut=pd.Timestamp(r['available_at_utc'])
        hashes=r.get('input_source_sha256',{})
        excerpts=r.get('input_source_excerpts',{})
        reviewed_at=pd.Timestamp(r.get('first_reviewed_utc'))
        if cut.tzinfo is None or pd.isna(reviewed_at) or reviewed_at.tzinfo is None or set(hashes)!={'forward_PAT','forward_diluted_shares','sector_PE'} or not set(hashes.values()).issubset(originals) or set(excerpts)!=set(hashes):
            audit.append({'isin':r.get('isin'),'status':'UNKNOWN_UNBOUND_SCENARIO_INPUTS'});continue
        import re
        from v11_4_primary_document_catalysts import extract_pdf
        bound=True
        for name,digest in hashes.items():
            try:
                raw=originals[digest].read_bytes();text=extract_pdf(raw) if raw.startswith(b'%PDF') else raw.decode('utf-8')
            except Exception:
                bound=False;break
            excerpt=excerpts[name]
            if not isinstance(excerpt,str) or not excerpt.strip() or re.sub(r'\s+',' ',excerpt).strip() not in re.sub(r'\s+',' ',text):bound=False
        if not bound:
            audit.append({'isin':r.get('isin'),'status':'UNKNOWN_UNBOUND_SCENARIO_EXCERPTS'});continue
        cut=max(cut,reviewed_at)
        for i,row in d[d['isin'].eq(r['isin'])].iterrows():
            decision=pd.Timestamp(row.date).tz_localize('Asia/Kolkata')+pd.Timedelta(hours=15,minutes=30)
            if cut>decision:continue
            z=earnings_scenario(row.close,row.ttm_reported_PAT_INR,row.latest_diluted_shares,r.get('forward_PAT_INR'),r.get('forward_diluted_shares'),r.get('sector_PE'))
            audit.append({'isin':r['isin'],'source_asof':str(pd.Timestamp(row.date).date()),'available_at_utc':cut.isoformat(),'input_source_sha256':hashes,**z})
            d.at[i,'earnings_scenario_status']=z['status']
            if 'scenario_multiple' in z:
                d.at[i,'scenario_base_multiple']=z['scenario_multiple'];d.at[i,'scenario_net_multiple']=z['net_multiple_after_registered_costs']
    (root/'evidence'/'earnings_scenario_audit.json').write_text(json.dumps(audit,indent=2));return d
