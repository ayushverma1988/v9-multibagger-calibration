"""Dated financial completeness, explicit earnings feasibility and source links."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from v12_5_features import features,pick,number,owner_pat,owner_profit_status,FINANCIAL,UNSUPPORTED,preferred_reporting_basis,reporting_basis_status
from v12_public_history import bounds

REQUIRED=('four_profitable_quarters','quarter_sales_yoy','quarter_pat_yoy','quarter_pat_margin',
          'annual_CFO_PAT','dated_debt_equity','annual_EBIT_proxy_interest_coverage')

def financial_quality(f):
    """Audit existing seven-input checks, without changing forecasts or policy.

    Complete data can show a weak issuer. Missing checks are kept distinct from
    observed failures; a zero denominator remains unknown in source features.
    This is not a forward-profitability or investment-accuracy certification.
    """
    checks={
      'four_profitable_quarters':lambda x:x==1,
      'quarter_sales_yoy':lambda x:x>=0,
      'quarter_pat_yoy':lambda x:x>0,
      'quarter_pat_margin':lambda x:x>0,
      'annual_CFO_PAT':lambda x:x>=.4,
      'dated_debt_equity':lambda x:x<=1.5,
      'annual_EBIT_proxy_interest_coverage':lambda x:x>=1.5}
    unknown=[k for k in REQUIRED if not np.isfinite(f.get(k,np.nan))]
    failed=[k for k,test in checks.items() if k not in unknown and not test(f[k])]
    status='KNOWN_FINANCIAL_FAILURE' if failed else 'UNKNOWN_FINANCIAL_INPUTS' if unknown else 'PASS_AVAILABLE_SEVEN_CHECKS'
    return {'status':status,'unknown_fields':unknown,'failed_checks':failed,
            'complete':not unknown,'quality_pass':not unknown and not failed}

def financial_readiness(known,date,f):
    """Distinguish missing source data from mathematically undefined ratios.

    Zero interest or a nonpositive comparative PAT must not be fabricated into
    an interest cover/growth number. Such cases need interpretation, not more
    downloads of an already observed original. No financial gate is relaxed.
    """
    missing=[k for k in REQUIRED if not np.isfinite(f.get(k,np.nan))]
    reasons={k:'MISSING_OR_AMBIGUOUS_ORIGINAL_PERIOD_OR_FACT' for k in missing}
    status=reporting_basis_status(known,date)
    empty={**status,'unknown_reasons':reasons,'underlying_required_facts_complete':False,'latest_reported_diluted_EPS_available':False}
    if known.empty:return empty
    basis=preferred_reporting_basis(known,date)
    d=known[known.reporting_mode.eq(basis)];q=pick(d,'quarter')
    if not q or (date-q['period']).days>190:
        if basis=='consolidated':reasons.update({k:status['reporting_basis_status'] for k in missing})
        return empty
    previous=pick(d,'quarter',q['period']-pd.DateOffset(years=1));annual=pick(d,'annual');bal=pick(d,'instant')
    if annual and (date-annual['period']).days>450:annual={}
    if bal and (date-bal['period']).days>450:bal={}
    denominator={'quarter_sales_yoy':number(previous,'revenue'),'quarter_pat_yoy':number(previous,'pat'),
      'quarter_pat_margin':number(q,'revenue'),'annual_CFO_PAT':number(annual,'pat'),
      'dated_debt_equity':number(bal,'equity'),'annual_EBIT_proxy_interest_coverage':number(annual,'finance_cost')}
    for k,value in denominator.items():
        if k in reasons and np.isfinite(value) and value<=0:reasons[k]='OBSERVED_NONPOSITIVE_DENOMINATOR_NOT_MISSING_SOURCE'
    quarter_pats=[number(pick(d,'quarter',q['period']-pd.offsets.MonthEnd(k)),'pat') for k in (0,3,6,9)]
    values=[number(q,'revenue'),number(q,'pat'),number(previous,'revenue'),number(previous,'pat'),*quarter_pats,
      number(annual,'cfo'),number(annual,'pat'),number(annual,'pbt'),number(annual,'finance_cost'),
      number(bal,'equity'),number(bal,'borrowings_current'),number(bal,'borrowings_noncurrent')]
    return {**status,'unknown_reasons':reasons,'underlying_required_facts_complete':bool(all(np.isfinite(x) for x in values)),
      'latest_reported_diluted_EPS_available':bool(np.isfinite(number(q,'eps_diluted_continuing'))),
      'reported_EPS_is_not_recurring_or_action_reconciled':True,'owner_earnings_source_status':owner_profit_status(q)}

def balancing_quarters(wide):
    """Explicit fiscal cumulative differences; never subtract EPS/shares.

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
                for key in ('revenue','pat','pat_owners','pat_noncontrolling','pbt','finance_cost','depreciation','exceptional_items','other_income'):
                    av=number(a,key);yv=number(z,key)
                    if np.isfinite(av) and np.isfinite(yv):row[key]=av-yv
                if 'pat' in row or 'revenue' in row:derived.append(row)
    if derived:
        add=pd.DataFrame(derived).drop_duplicates(['security_isin','reporting_mode','period_end','available_at_utc','input_source_sha256'])
        result=pd.concat([wide,add],ignore_index=True)
    else:result=wide.copy()
    return cumulative_quarters(result)

def cumulative_quarters(wide):
    """H1 minus Q1 and nine months minus H1, on one explicit fiscal basis.

    Some originals provide the half-year cumulative statement but no separate
    September-quarter context. The amount difference supplies that quarter.
    Both originals, their later availability and an unverified revision flag
    remain attached. This does not reconstruct per-share earnings.
    """
    derived=[]
    flows=('revenue','pat','pat_owners','pat_noncontrolling','pbt','finance_cost',
           'depreciation','exceptional_items','other_income')
    for _,g in wide.groupby(['security_isin','reporting_mode']):
        for end in g[g.period_kind.eq('ytd')].to_dict('records'):
            start=pd.to_datetime(end.get('period_start'),errors='coerce')
            if pd.isna(start):continue
            days=(end['period']-start).days
            if not (150<=days<=210 or 240<=days<=310):continue
            earlier=g[g.period_start.eq(end['period_start'])&
                (end['period']-g.period).dt.days.between(60,120)&
                g.period_kind.isin(['quarter','ytd'])]
            for prev in earlier.to_dict('records'):
                prior_days=(prev['period']-start).days
                if not (150<=days<=210 and 60<=prior_days<=120 or
                        240<=days<=310 and 150<=prior_days<=210):continue
                available=max(end['available'],prev['available'])
                explicit=g[g.period_kind.eq('quarter')&g.period.eq(end['period'])&g.available.le(available)]
                if not explicit.empty:continue
                row={k:end[k] for k in ('symbol','security_isin','reporting_mode','period_end','period','source_sha256')}
                row.update(period_kind='quarter',period_start=str((prev['period']+pd.Timedelta(days=1)).date()),
                    available=available,available_at_utc=available.isoformat(),derived_balancing_quarter=True,
                    derived_quarter_method='HALF_YEAR_MINUS_Q1' if days<=210 else 'NINE_MONTHS_MINUS_HALF_YEAR',
                    source_url=end.get('source_url'),input_source_sha256=json.dumps([end['source_sha256'],prev['source_sha256']]),
                    input_source_urls=json.dumps([end.get('source_url'),prev.get('source_url')]),
                    historical_revision_integrity_independently_proven=False)
                for key in flows:
                    a=number(end,key);b=number(prev,key)
                    if np.isfinite(a) and np.isfinite(b):row[key]=a-b
                if 'pat' in row or 'revenue' in row:derived.append(row)
    if not derived:return wide
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
    from v13_information_clock import research_cutoffs,cutoff
    cut=cutoff(research_cutoffs(pd.DataFrame({'date':[date]}),pd.Timestamp.now(tz='UTC')).iloc[0])
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
        pending=g.rename(columns={'_period':'period'})
        anchor=pd.concat([w,pending],ignore_index=True)
        basis=preferred_reporting_basis(anchor,date)
        g=g[g.reporting_mode.eq(basis)]
        if not w.empty:w=w[w.reporting_mode.eq(basis)]
        endpoints=[g['_period'].max()]
        if not w.empty:endpoints.append(w.loc[w.period_kind.eq('quarter'),'period'].max())
        endpoints=[z for z in endpoints if pd.notna(z)]
        # A current standalone filing can coexist with a recent group annual
        # but no group quarter or pending group filing. The separate basis
        # queue handles that gap; it supplies no endpoint for this bundle.
        if not endpoints:continue
        latest=max(endpoints)
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
        readiness=financial_readiness(w,date,f)
        missing=sum(reason!='OBSERVED_NONPOSITIVE_DENOMINATOR_NOT_MISSING_SOURCE' for reason in readiness['unknown_reasons'].values())
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

def owner_completion_queue(queue,wide,asof,budget):
    """Reserve original-filing work for owner earnings after basic readiness.

    Only pending originals are requested. Missing owner facts stay missing;
    total PAT, a standalone result and a zero minority entry are not substituted.
    Current and year-ago owners are prioritised, followed by the TTM quarters.
    Nothing depends on model ranks, selections or realised outcomes.
    """
    if queue.empty or budget<=0:return queue.iloc[:0].copy()
    from v13_information_clock import research_cutoffs,cutoff
    date=pd.Timestamp(asof);low,_=bounds(asof)
    cut=cutoff(research_cutoffs(pd.DataFrame({'date':[date]}),pd.Timestamp.now(tz='UTC')).iloc[0])
    q=queue.copy();q['_period']=pd.to_datetime(q.period_end)
    q=q[q['_period'].between(low,date)&pd.to_datetime(q.available_at_utc,utc=True,format='mixed').le(cut)]
    groups={} if wide.empty else dict(tuple(wide.groupby('symbol')))
    bundles=[]
    for symbol,g in q.groupby('symbol'):
        w=groups.get(symbol,pd.DataFrame())
        if not w.empty:w=w[w.available.le(cut)&w.period.between(low,date)]
        recent=g[g['_period'].ge(date-pd.Timedelta(days=190))]
        observed=w[w.period_kind.eq('quarter')&(date-w.period).dt.days.between(0,190)] if not w.empty else w
        if recent.empty and observed.empty:continue
        anchor=observed if not observed.empty else recent
        # Parent-attributable consolidated earnings cannot be inferred from
        # standalone PAT. No consolidated publication means an explicit gap.
        if not anchor.reporting_mode.eq('consolidated').any():continue
        g=g[g.reporting_mode.eq('consolidated')]
        if not w.empty:w=w[w.reporting_mode.eq('consolidated')]
        endpoints=list(recent.loc[recent.reporting_mode.eq('consolidated'),'_period'])
        if not observed.empty:endpoints.extend(observed.loc[observed.reporting_mode.eq('consolidated'),'period'])
        if not endpoints:continue
        latest=max(endpoints)
        targets=[latest,latest-pd.DateOffset(years=1)]+[latest-pd.offsets.MonthEnd(k) for k in (3,6,9)]
        missing={period:rank for rank,period in enumerate(targets) if not np.isfinite(owner_pat(pick(w,'quarter',period)))} if not w.empty else dict((p,i) for i,p in enumerate(targets))
        target=g[g['_period'].isin(missing)].copy()
        if target.empty:continue
        target=target.sort_values('available_at_utc',ascending=False).drop_duplicates('_period')
        target['_owner_priority']=target['_period'].map(missing)
        target=target.sort_values(['last_attempt','_owner_priority']).head(5)
        bundles.append((str(target.last_attempt.min()),__import__('hashlib').sha256(symbol.encode()).hexdigest(),target))
    chosen=[];count=0
    for _,_,bundle in sorted(bundles,key=lambda z:z[:2]):
        if count>=budget:break
        selected=bundle.head(budget-count);chosen.append(selected);count+=len(selected)
    return pd.concat(chosen,ignore_index=True).drop(columns=['_period','_owner_priority']) if chosen else queue.iloc[:0].copy()

def basis_completion_queue(queue,wide,asof,budget):
    """Recover missing current group results across the public universe.

    This queue is independent of fresh price coverage. One newest original
    per issuer first; failures rotate with the recorded source-attempt clock.
    """
    if queue.empty or wide.empty or budget<=0:return queue.iloc[:0].copy()
    date=pd.Timestamp(asof);groups=dict(tuple(wide.groupby('symbol')));chosen=[]
    for symbol,g in queue.groupby('symbol'):
        w=groups.get(symbol)
        if w is None or preferred_reporting_basis(w,date)!='consolidated':continue
        current=pick(w[w.reporting_mode.eq('consolidated')],'quarter')
        q=g[g.reporting_mode.eq('consolidated')].copy();q['_period']=pd.to_datetime(q.period_end)
        q=q[(date-q['_period']).dt.days.between(0,190)]
        if current:q=q[q['_period'].gt(current['period'])]
        if q.empty:continue
        target=q.sort_values(['_period','available_at_utc'],ascending=False).head(1).copy()
        target['_issuer_hash']=__import__('hashlib').sha256(symbol.encode()).hexdigest();chosen.append(target)
    if not chosen:return queue.iloc[:0].copy()
    return pd.concat(chosen).sort_values(['last_attempt','_issuer_hash']).head(budget).drop(columns=['_period','_issuer_hash'])

def enrich_evidence(frame,wide,root,asof):
    wide=balancing_quarters(wide)
    d=frame.copy();links=[];coverage=[];low,_=bounds(asof)
    for c in ('ttm_reported_PAT_INR','ttm_owner_attributable_PAT_INR','annual_reported_revenue_INR','latest_issued_shares','latest_diluted_shares','ttm_reported_eps','implied_reported_PE','required_eps_at_current_PE_for_2x','scenario_base_multiple','scenario_net_multiple','historical_sector_relative_PE','total_PAT_divided_by_latest_quarter_weighted_shares_proxy'):
        d[c]=np.nan
    d['financial_complete']=False;d['financial_unknown_fields']='';d['evidence_status']='MISSING_FINANCIAL_OR_EXECUTION'
    d['financial_quality_pass']=False;d['financial_quality_status']='UNKNOWN_FINANCIAL_INPUTS'
    d['earnings_scenario_status']='UNKNOWN_FORWARD_EARNINGS_AND_SECTOR_VALUATION'
    d['current_owner_profit_status']='UNKNOWN_MISSING_PREFERRED_BASIS_QUARTER'
    for i,r in d.iterrows():
        date=pd.Timestamp(r.date)
        from v13_information_clock import cutoff
        cut=cutoff(r)
        w=wide[wide.security_isin.eq(r['isin'])&wide.available.le(cut)&wide.period.le(date)&wide.period.ge(low)] if not wide.empty else pd.DataFrame()
        f,used=features(w,date) if not w.empty else ({},[])
        quality=financial_quality(f);unknown=quality['unknown_fields']
        readiness=financial_readiness(w,date,f)
        d.at[i,'financial_complete']=not unknown;d.at[i,'financial_unknown_fields']='|'.join(unknown)
        d.at[i,'financial_quality_pass']=quality['quality_pass'];d.at[i,'financial_quality_status']=quality['status']
        coverage.append({'isin':r['isin'],'symbol':r['symbol'],'source_asof':str(date.date()),'unknown_fields':unknown,'financial_complete':not unknown,
          'financial_quality_status':quality['status'],'financial_quality_pass':quality['quality_pass'],'failed_checks':quality['failed_checks'],
          'expanded_unknown_fields':[k for k in (*FINANCIAL,*UNSUPPORTED) if not np.isfinite(f.get(k,np.nan))],**readiness})
        for z in used:
            item={'isin':r['isin'],'source_asof':str(date.date()),'period_end':z['period_end'],
              'reporting_mode':z['reporting_mode'],'available_at_utc':z['available_at_utc'],
              'source_sha256':z['source_sha256'],'source_url':z.get('source_url'),
              'derived_balancing_quarter':bool(z.get('derived_balancing_quarter') is True),
              'derived_quarter_method':z.get('derived_quarter_method'),
              'input_source_sha256':z.get('input_source_sha256'),'input_source_urls':z.get('input_source_urls')}
            links.append({k:None if isinstance(v,float) and not np.isfinite(v) else v for k,v in item.items()})
        if w.empty:continue
        basis=preferred_reporting_basis(w,date)
        w=w[w.reporting_mode.eq(basis)];q=pick(w,'quarter')
        if not q or (date-q['period']).days>190:continue
        d.at[i,'current_owner_profit_status']=owner_profit_status(q)
        annual=pick(w,'annual')
        if annual and (date-annual['period']).days<=450:d.at[i,'annual_reported_revenue_INR']=number(annual,'revenue')
        capital=number(q,'paidup_equity_capital');face=number(q,'face_value')
        if capital>0 and face>0:d.at[i,'latest_issued_shares']=capital/face
        quarters=[pick(w,'quarter',q['period']-pd.offsets.MonthEnd(k)) for k in (0,3,6,9)]
        pats=[number(z,'pat') for z in quarters];shares=number(q,'weighted_diluted_shares')
        owner_pats=[owner_pat(z) for z in quarters]
        if all(np.isfinite(x) for x in owner_pats):d.at[i,'ttm_owner_attributable_PAT_INR']=sum(owner_pats)
        if all(np.isfinite(x) for x in pats):d.at[i,'ttm_reported_PAT_INR']=sum(pats)
        if all(np.isfinite(x) for x in pats) and shares>0:
            pat=sum(pats);eps=pat/shares
            d.at[i,'latest_diluted_shares']=shares
            d.at[i,'total_PAT_divided_by_latest_quarter_weighted_shares_proxy']=eps
            # Total consolidated PAT may include minority interests, and one
            # quarter's weighted shares are not a verified TTM denominator.
            # This arithmetic is not reported diluted EPS or a certified PE.
            d.at[i,'earnings_scenario_status']='TOTAL_PAT_SHARE_PROXY_NOT_REPORTED_TTM_EPS_OR_VERIFIED_PE'
        # Latest sector memberships or guessed growth are never applied to
        # history. A reviewed scenario must bind every input to dated sources.
    out=Path(root)/'evidence';out.mkdir(exist_ok=True)
    (out/'financial_completeness.json').write_text(json.dumps({'asof':asof,'rows':len(d),
      'complete_rows':int(d.financial_complete.sum()),'quality_pass_rows':int(d.financial_quality_pass.sum()),
      'underlying_required_facts_complete_rows':sum(z['underlying_required_facts_complete'] for z in coverage),
      'latest_reported_diluted_EPS_available_rows':sum(z['latest_reported_diluted_EPS_available'] for z in coverage),
      'known_failure_rows':int(d.financial_quality_status.eq('KNOWN_FINANCIAL_FAILURE').sum()),
      'expanded_field_nonmissing':{k:sum(k not in z['expanded_unknown_fields'] for z in coverage) for k in (*FINANCIAL,*UNSUPPORTED)},
      'required_checks':list(REQUIRED),'quality_is_not_future_profitability':True,'issuer_backlog':coverage,
      'reported_EPS_is_not_recurring_EPS':True},indent=2))
    (out/'financial_source_links.json').write_text(json.dumps(links,indent=2,default=str))
    # Missing ratios can reflect known losses/zero denominators rather than a
    # missing document. Expose both causes so downloads target actual gaps.
    from collections import Counter
    from v13_capture_policy import atomic_json
    reasons=Counter(reason for row in coverage for reason in row.get('unknown_reasons',{}).values())
    missing=Counter(field for row in coverage for field in row['unknown_fields'])
    atomic_json(out/'financial_capture_gaps.json',{'source_asof':asof,'current_market_rows':len(d),
      'complete_seven_check_rows':int(d.financial_complete.sum()),
      'raw_required_facts_complete_rows':sum(row['underlying_required_facts_complete'] for row in coverage),
      'missing_check_counts':dict(missing),'unknown_reason_counts':dict(reasons),
      'counts_are_fields_not_unique_issuers':True,'nonpositive_denominators_are_not_missing_downloads':True,
      'missing_fields_never_filled_with_zero':True})
    # Reverse arithmetic, explicitly hypothetical; it does not assign a PE,
    # forward margin or earnings growth to any company.
    target=2*1.005/.995
    stress=[{'assumed_PE_multiple':pe,'assumed_forward_share_multiple':shares,
             'required_PAT_growth':target*shares/pe-1} for pe in (.8,1.,1.2,1.4) for shares in (1.,1.1,1.25)]
    (out/'earnings_feasibility.json').write_text(json.dumps({'source_asof':asof,'status':'REVERSE_SCENARIO_ARITHMETIC_NOT_PREDICTION',
      'assumptions_not_source_verified':True,'net_return_target':2,'required_growth_grid':stress,
      'reported_TTM_PAT_available':int(d.ttm_reported_PAT_INR.notna().sum()),
      'issued_share_count_is_not_diluted_weighted_share_count':True,
      'total_PAT_share_proxy_is_not_reported_EPS_or_verified_PE':True,
      'independently_supported_forward_PAT_and_sector_PE_required':True},indent=2))
    # A per-stock EPS hurdle is helpful even when forward earnings and peer
    # valuations are undisclosed. Every PE below is a hypothetical sensitivity,
    # never an asserted sector multiple or a support flag used by selection.
    hurdles=[{'isin':z['isin'],'symbol':z['symbol'],'source_asof':str(pd.Timestamp(z['date']).date()),
      'source_close_INR':float(z['close']),'price_basis':'SOURCE_CLOSE_NOT_REGISTERED_NEXT_SESSION_ENTRY',
      'status':'HYPOTHETICAL_EPS_HURDLES_NOT_PREDICTION','source_sha256':z.get('official_source_sha256'),
      'assumed_PE_to_required_forward_EPS_INR':{str(pe):float(z['close'])*target/pe for pe in (10,15,20,30)},
      'assumptions_are_not_source_verified':True} for z in d.to_dict('records') if np.isfinite(z.get('close',np.nan)) and z['close']>0]
    (out/'earnings_hurdles.json').write_text(json.dumps(hurdles,indent=2))
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
            from v13_source_quality import bound_number
            value=r.get({'forward_PAT':'forward_PAT_INR','forward_diluted_shares':'forward_diluted_shares','sector_PE':'sector_PE'}[name])
            if not bound_number(value,excerpt,currency=name=='forward_PAT'):bound=False
            # A historical PAT number is not proof that forward earnings were
            # established. Reviewed scenario assumptions still are not facts.
            if name=='forward_PAT' and not re.search(r'forward|forecast|projected|expected|guidance|scenario',excerpt,re.I):bound=False
        if not bound:
            audit.append({'isin':r.get('isin'),'status':'UNKNOWN_UNBOUND_SCENARIO_EXCERPTS'});continue
        if r.get('forward_PAT_basis')!='OWNERS_ATTRIBUTABLE' or not re.search(r'owners|shareholders|attributable',excerpts['forward_PAT'],re.I):
            audit.append({'isin':r.get('isin'),'status':'UNKNOWN_FORWARD_PAT_OWNER_ATTRIBUTION'});continue
        cut=max(cut,reviewed_at)
        for i,row in d[d['isin'].eq(r['isin'])].iterrows():
            from v13_information_clock import cutoff
            decision=cutoff(row)
            if cut>decision:continue
            from v13_upgrade_diagnostics import scenario_horizon
            if not scenario_horizon(r,row.date):
                audit.append({'isin':r['isin'],'status':'UNKNOWN_UNBOUND_ANNUAL_PERIOD_HORIZON_OR_DILUTED_SHARE_BASIS'});continue
            z=earnings_scenario(row.close,row.ttm_reported_PAT_INR,row.latest_diluted_shares,r.get('forward_PAT_INR'),r.get('forward_diluted_shares'),r.get('sector_PE'))
            if 'earnings_growth' in z:
                owner=row.get('ttm_owner_attributable_PAT_INR',np.nan)
                z['earnings_growth']=r['forward_PAT_INR']/owner-1 if np.isfinite(owner) and owner>0 else None
                z['earnings_growth_status']='SOURCE_OWNER_PAT_BASELINE_NOT_RECURRING_EARNINGS' if z['earnings_growth'] is not None else 'UNKNOWN_CURRENT_OWNER_ATTRIBUTABLE_BASELINE'
            audit.append({'isin':r['isin'],'source_asof':str(pd.Timestamp(row.date).date()),'available_at_utc':cut.isoformat(),'input_source_sha256':hashes,
              'horizon_months':r['horizon_months'],'valuation_target_date':r['valuation_target_date'],
              'forecast_period_start':r['forecast_period_start'],'forecast_period_end':r['forecast_period_end'],**z})
            d.at[i,'earnings_scenario_status']=z['status']
            if 'scenario_multiple' in z:
                d.at[i,'scenario_base_multiple']=z['scenario_multiple'];d.at[i,'scenario_net_multiple']=z['net_multiple_after_registered_costs']
    (root/'evidence'/'earnings_scenario_audit.json').write_text(json.dumps(audit,indent=2));return d
