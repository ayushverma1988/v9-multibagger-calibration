"""Dated financial completeness, explicit earnings feasibility and source links."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from v12_5_features import features,pick,number
from v12_public_history import bounds

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
    d=frame.copy();links=[];coverage=[];low,_=bounds(asof)
    for c in ('ttm_reported_PAT_INR','latest_diluted_shares','ttm_reported_eps','implied_reported_PE','required_eps_at_current_PE_for_2x','scenario_base_multiple','scenario_net_multiple','historical_sector_relative_PE'):
        if c not in d:d[c]=np.nan
    d['financial_complete']=False;d['financial_unknown_fields']='';d['evidence_status']='MISSING_FINANCIAL_OR_EXECUTION'
    d['earnings_scenario_status']='UNKNOWN_FORWARD_EARNINGS_AND_SECTOR_VALUATION'
    for i,r in d.iterrows():
        date=pd.Timestamp(r.date);cut=(date.tz_localize('Asia/Kolkata')+pd.Timedelta(hours=15,minutes=30)).tz_convert('UTC')
        w=wide[wide.security_isin.eq(r['isin'])&wide.available.le(cut)&wide.period.le(date)&wide.period.ge(low)] if not wide.empty else pd.DataFrame()
        f,used=features(w,date) if not w.empty else ({},[])
        required=['four_profitable_quarters','quarter_sales_yoy','quarter_pat_yoy','quarter_pat_margin','annual_CFO_PAT','dated_debt_equity','annual_EBIT_proxy_interest_coverage']
        unknown=[k for k in required if not np.isfinite(f.get(k,np.nan))]
        d.at[i,'financial_complete']=not unknown;d.at[i,'financial_unknown_fields']='|'.join(unknown)
        coverage.append({'isin':r['isin'],'symbol':r['symbol'],'source_asof':str(date.date()),'unknown_fields':unknown,'financial_complete':not unknown})
        for z in used:
            links.append({'isin':r['isin'],'source_asof':str(date.date()),'period_end':z['period_end'],
              'reporting_mode':z['reporting_mode'],'available_at_utc':z['available_at_utc'],
              'source_sha256':z['source_sha256'],'source_url':z.get('source_url')})
        if w.empty:continue
        recent=w[w.period_kind.eq('quarter')&(date-w.period).dt.days.between(0,190)]
        basis='consolidated' if recent.reporting_mode.eq('consolidated').any() else 'standalone'
        w=w[w.reporting_mode.eq(basis)];q=pick(w,'quarter')
        if not q or (date-q['period']).days>190:continue
        quarters=[pick(w,'quarter',q['period']-pd.offsets.MonthEnd(k)) for k in (0,3,6,9)]
        pats=[number(z,'pat') for z in quarters];shares=number(q,'weighted_diluted_shares')
        if all(np.isfinite(x) for x in pats) and shares>0:
            pat=sum(pats);eps=pat/shares
            d.at[i,'ttm_reported_PAT_INR']=pat;d.at[i,'latest_diluted_shares']=shares;d.at[i,'ttm_reported_eps']=eps
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
    originals={__import__('hashlib').sha256(p.read_bytes()).hexdigest() for p in root.rglob('*.bin')}
    for r in json.loads(path.read_text()):
        cut=pd.Timestamp(r['available_at_utc'])
        hashes=r.get('input_source_sha256',{})
        if cut.tzinfo is None or set(hashes)!={'forward_PAT','forward_diluted_shares','sector_PE'} or not set(hashes.values()).issubset(originals):
            audit.append({'isin':r.get('isin'),'status':'UNKNOWN_UNBOUND_SCENARIO_INPUTS'});continue
        for i,row in d[d['isin'].eq(r['isin'])].iterrows():
            decision=pd.Timestamp(row.date).tz_localize('Asia/Kolkata')+pd.Timedelta(hours=15,minutes=30)
            if cut>decision:continue
            z=earnings_scenario(row.close,row.ttm_reported_PAT_INR,row.latest_diluted_shares,r.get('forward_PAT_INR'),r.get('forward_diluted_shares'),r.get('sector_PE'))
            audit.append({'isin':r['isin'],'source_asof':str(pd.Timestamp(row.date).date()),'available_at_utc':cut.isoformat(),'input_source_sha256':hashes,**z})
            d.at[i,'earnings_scenario_status']=z['status']
            if 'scenario_multiple' in z:
                d.at[i,'scenario_base_multiple']=z['scenario_multiple'];d.at[i,'scenario_net_multiple']=z['net_multiple_after_registered_costs']
    (root/'evidence'/'earnings_scenario_audit.json').write_text(json.dumps(audit,indent=2));return d
