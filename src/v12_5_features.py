"""Dated expanded features; source definitions are part of the experiment.

Reported EPS is not recurring EPS. PBT-plus-finance-cost is an EBIT proxy,
not operating profit. Net capital employed is assets less current liabilities.
All ratios require positive, matched-period denominators on one report basis.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd

FINANCIAL=('sales_yoy_previous_quarter','sales_growth_acceleration','pat_growth_acceleration','owner_PAT_yoy','owner_PAT_margin',
 'pat_margin_yoy_change','reported_diluted_EPS_yoy_same_document','annual_cash_conversion_change',
 'accruals_to_average_assets','annual_EBIT_proxy_interest_coverage','annual_ROCE_proxy',
 'incremental_ROCE_proxy','reported_share_count_yoy','exceptional_items_to_abs_PBT',
 'other_income_to_revenue','receivables_to_revenue','annual_CFO_to_revenue')
CATALYST=('commissioned_document_count90','utilisation_disclosed_pct','disclosed_order_value_due12_to_sales',
 'verified_incremental_profit12_to_PAT','verified_funding_sufficient')
UNSUPPORTED=('verified_recurring_diluted_EPS_growth','historical_sector_relative_PE')
MARKET=('breakout_close_vs_prior63_high','liquidity_volume_interaction')
BASE_FIELDS=('quarter_sales_yoy','quarter_pat_yoy','quarter_pat_margin','annual_CFO_PAT','dated_debt_equity')

def number(row,key):
    x=row.get(key,np.nan)
    return float(x) if pd.notna(x) and np.isfinite(float(x)) else np.nan

def ratio(a,b):return a/b if np.isfinite(a) and np.isfinite(b) and b>0 else np.nan
def growth(a,b):return ratio(a,b)-1

def owner_profit_status(row):
    owner=number(row,'pat_owners');minority=number(row,'pat_noncontrolling');total=number(row,'pat')
    if not all(np.isfinite(x) for x in (owner,minority,total)):
        return 'UNKNOWN_MISSING_OWNER_MINORITY_OR_TOTAL_PAT'
    if not np.isclose(owner+minority,total,rtol=1e-8,atol=1.):
        return 'UNKNOWN_OWNER_PLUS_MINORITY_DOES_NOT_RECONCILE_TOTAL_PAT'
    return 'ORIGINAL_OWNER_MINORITY_TOTAL_PAT_RECONCILED_NOT_RECURRING_EPS'

def owner_pat(row):
    # Some original filings put zero in both owner/minority fields alongside
    # positive total PAT. Preserve raw facts, but never score those placeholders
    # as verified zero owner profits or infer owner PAT by subtraction.
    return number(row,'pat_owners') if owner_profit_status(row).startswith('ORIGINAL_') else np.nan

def source_wide(base,extra):
    b=base.rename(columns={'value_INR':'value'})
    x=pd.concat([b,extra],ignore_index=True)
    keys=['symbol','security_isin','reporting_mode','period_kind','period_end','available_at_utc','source_sha256']
    # Differing numeric facts for an exact source/period are not resolved by mean.
    counts=x.groupby(keys+['metric'],dropna=False).value.nunique()
    if (counts>1).any():raise ValueError('Conflicting same-source numeric facts')
    x=x.drop_duplicates(keys+['metric'])
    w=x.pivot(index=keys,columns='metric',values='value').reset_index()
    w['period']=pd.to_datetime(w.period_end);w['available']=pd.to_datetime(w.available_at_utc,utc=True,format='mixed')
    return w

def pick(d,kind,period=None):
    mask=d.period_kind.to_numpy()==kind
    periods=d.period.to_numpy()
    if period is not None:mask&=periods==np.datetime64(period)
    ix=np.flatnonzero(mask)
    if not len(ix):return {}
    ix=ix[periods[ix]==periods[ix].max()]
    available=d.available.to_numpy();ix=ix[available[ix]==available[ix].max()]
    if len(ix)!=1:return {}
    return d.iloc[int(ix[0])].to_dict()

def preferred_reporting_basis(known,date):
    """Keep consolidated continuity when its newest quarter is missing.

    Availability-filtered observations only. A recent annual/comparative
    consolidated statement establishes the group basis even when a newer
    standalone filing is present. A missing group quarter is not evidence of
    falling standalone earnings. Explicit issuer basis changes need review.
    """
    if known.empty:return 'unknown'
    recent=known[(date-known.period).dt.days.between(0,450)]
    if recent.reporting_mode.eq('consolidated').any():return 'consolidated'
    if recent.reporting_mode.eq('standalone').any():return 'standalone'
    return 'unknown'

def reporting_basis_status(known,date):
    basis=preferred_reporting_basis(known,date)
    q=pick(known[known.reporting_mode.eq(basis)],'quarter') if not known.empty else {}
    quarters=known[known.period_kind.eq('quarter')] if not known.empty else known
    fresh=bool(q and 0<=(date-q['period']).days<=190)
    return {'preferred_reporting_basis':basis,
      'preferred_quarter_end':str(q['period'].date()) if q else None,
      'preferred_quarter_fresh':fresh,
      'latest_any_basis_quarter_end':str(quarters.period.max().date()) if not quarters.empty else None,
      'reporting_basis_status':'FRESH_PREFERRED_BASIS' if fresh else
        'MISSING_RECENT_CONSOLIDATED_QUARTER_NO_STANDALONE_SUBSTITUTION' if basis=='consolidated' else
        'MISSING_RECENT_PREFERRED_BASIS_QUARTER'}

def features(known,date):
    out={c:np.nan for c in (*FINANCIAL,*UNSUPPORTED,*BASE_FIELDS,'four_profitable_quarters')}
    basis=preferred_reporting_basis(known,date)
    d=known[known.reporting_mode.eq(basis)];q=pick(d,'quarter')
    if not q or (date-q['period']).days>190:return out,[]
    sources=[q];period=q['period'];old=pick(d,'quarter',period-pd.DateOffset(years=1))
    prev_period=period-pd.offsets.MonthEnd(3);prev=pick(d,'quarter',prev_period)
    prev_old=pick(d,'quarter',prev_period-pd.DateOffset(years=1))
    if old:sources.append(old)
    if prev:sources.append(prev)
    if prev_old:sources.append(prev_old)
    sales=growth(number(q,'revenue'),number(old,'revenue'))
    out['quarter_sales_yoy']=sales
    out['quarter_pat_yoy']=growth(number(q,'pat'),number(old,'pat'))
    out['quarter_pat_margin']=ratio(number(q,'pat'),number(q,'revenue'))
    out['owner_PAT_yoy']=growth(owner_pat(q),owner_pat(old))
    out['owner_PAT_margin']=ratio(owner_pat(q),number(q,'revenue'))
    quarters=[pick(d,'quarter',period-pd.offsets.MonthEnd(k)) for k in [0,3,6,9]]
    profits=[number(z,'pat') for z in quarters]
    if all(np.isfinite(v) for v in profits):out['four_profitable_quarters']=float(all(v>0 for v in profits))
    sources.extend([z for z in quarters if z])
    prev_sales=growth(number(prev,'revenue'),number(prev_old,'revenue'))
    out['sales_yoy_previous_quarter']=prev_sales
    out['sales_growth_acceleration']=sales-prev_sales
    out['pat_growth_acceleration']=growth(number(q,'pat'),number(old,'pat'))-growth(number(prev,'pat'),number(prev_old,'pat'))
    out['pat_margin_yoy_change']=ratio(number(q,'pat'),number(q,'revenue'))-ratio(number(old,'pat'),number(old,'revenue'))
    # Use an issuer's SAME filing version for comparative EPS, avoiding a splice
    # of old per-share bases from before a split/bonus and a newer EPS basis.
    same=d[d.source_sha256.eq(q['source_sha256'])]
    ep=pick(same,'quarter',period-pd.DateOffset(years=1))
    out['reported_diluted_EPS_yoy_same_document']=growth(number(q,'eps_diluted_continuing'),number(ep,'eps_diluted_continuing'))
    if ep:sources.append(ep)
    cap=number(q,'paidup_equity_capital');face=number(q,'face_value')
    # Share-count change is observable across dated filings. It includes bonus
    # issues and other capital changes, so it is NOT labelled dilution.
    oldcap=number(old,'paidup_equity_capital');oldface=number(old,'face_value')
    if face==oldface and face>0:
        out['reported_share_count_yoy']=growth(ratio(cap,face),ratio(oldcap,oldface))
    out['exceptional_items_to_abs_PBT']=ratio(abs(number(q,'exceptional_items')),abs(number(q,'pbt')))
    out['other_income_to_revenue']=ratio(number(q,'other_income'),number(q,'revenue'))
    a=pick(d,'annual')
    bal=pick(d,'instant')
    if bal and (date-bal['period']).days<=450:
        eq=number(bal,'equity');bc=number(bal,'borrowings_current');bnc=number(bal,'borrowings_noncurrent')
        if bc>=0 and bnc>=0:out['dated_debt_equity']=ratio(bc+bnc,eq)
        sources.append(bal)
    if a and (date-a['period']).days<=450:
        sources.append(a);ap=pick(d,'annual',a['period']-pd.DateOffset(years=1))
        bn=pick(d,'instant',a['period']);bo=pick(d,'instant',a['period']-pd.DateOffset(years=1))
        for z in [ap,bn,bo]:
            if z:sources.append(z)
        pat=number(a,'pat');cfo=number(a,'cfo');rev=number(a,'revenue')
        out['annual_CFO_PAT']=ratio(cfo,pat)
        out['annual_CFO_to_revenue']=ratio(cfo,rev)
        out['annual_cash_conversion_change']=ratio(cfo,pat)-ratio(number(ap,'cfo'),number(ap,'pat'))
        an=number(bn,'assets');ao=number(bo,'assets')
        if an>0 and ao>0:out['accruals_to_average_assets']=ratio(pat-cfo,(an+ao)/2)
        cost=number(a,'finance_cost');ebit=number(a,'pbt')+cost
        out['annual_EBIT_proxy_interest_coverage']=ratio(ebit,cost)
        cn=an-number(bn,'current_liabilities');co=ao-number(bo,'current_liabilities')
        if cn>0 and co>0:
            out['annual_ROCE_proxy']=ratio(ebit,(cn+co)/2)
            # A tiny capital change produces unstable incremental ratios.
            if cn-co>max(abs(co)*.05,1):
                out['incremental_ROCE_proxy']=ratio(ebit-(number(ap,'pbt')+number(ap,'finance_cost')),cn-co)
        out['receivables_to_revenue']=ratio(number(bn,'receivables_current')+number(bn,'receivables_noncurrent'),rev)
    return out,sources

def build(panel_path,matrix_path,base_path,extra_path,out_path,history_path=None):
    out=Path(out_path);out.mkdir(parents=True,exist_ok=True)
    panel=pd.read_parquet(panel_path);base=pd.read_parquet(base_path);extra=pd.read_parquet(extra_path)
    w=source_wide(base,extra);groups={k:g for k,g in w.groupby('security_isin')};records=[];links=[]
    empty={c:np.nan for c in (*FINANCIAL,*UNSUPPORTED,*BASE_FIELDS,'four_profitable_quarters')}
    for i,r in enumerate(panel[['date','symbol','isin']].to_dict('records'),1):
        date=pd.Timestamp(r['date']);cut=(date.tz_localize('Asia/Kolkata')+pd.Timedelta(hours=15,minutes=30)).tz_convert('UTC')
        d=groups.get(r['isin']);f=dict(empty);s=[]
        if d is not None:
            known=d[d.available.le(cut)&d.period.le(date)]
            if len(known):f,s=features(known,date)
        records.append({**r,**f,'expanded_financial_available_at':max([x['available'] for x in s]).isoformat() if s else None})
        for x in s:links.append({**r,'source_sha256':x['source_sha256'],'period_end':x['period_end'],'available_at_utc':x['available_at_utc'],'reporting_mode':x['reporting_mode']})
        if i%10000==0:print('expanded feature rows',i,flush=True)
    f=pd.DataFrame(records)
    old=pd.read_parquet(matrix_path)
    d=panel.merge(old.drop(columns=[c for c in old if c in panel and c not in ['date','symbol','isin']]),on=['date','symbol','isin'],validate='1:1')
    d=d.drop(columns=[c for c in BASE_FIELDS if c in d]).merge(f,on=['date','symbol','isin'],validate='1:1')
    # off_high63 includes the current observation; label it accordingly rather
    # than claiming to know an uncomputed prior-only high.
    d['distance_to_observed63_high']=d.off_high63
    d['liquidity_volume_interaction']=d.volume_confirmation*d.log_liquidity63
    d['breakout_close_vs_prior63_high']=np.nan
    if history_path:
        market=[]
        for p in sorted((Path(history_path)/'securities').glob('*.parquet')):
            h=pd.read_parquet(p).sort_values('date')
            if h.empty:continue
            adj_high=h.high*h.adj_close/h.close
            h['breakout_close_vs_prior63_high']=h.adj_close/adj_high.shift(1).rolling(63,min_periods=63).max()-1
            market.append(h[['date','isin','breakout_close_vs_prior63_high']])
        if market:
            z=pd.concat(market,ignore_index=True)
            if z.duplicated(['date','isin']).any():raise ValueError('Ambiguous daily market identity')
            d=d.drop(columns='breakout_close_vs_prior63_high').merge(z,on=['date','isin'],how='left',validate='m:1')
    for c in CATALYST:d[c]=np.nan
    d.to_parquet(out/'expanded_training_inputs.parquet',index=False)
    pd.DataFrame(links).drop_duplicates().to_parquet(out/'expanded_source_links.parquet',index=False)
    summary={'rows':len(d),'financial_features_nonmissing':d[list(FINANCIAL)].notna().sum().to_dict(),
             'unverified_features_not_fabricated':list(UNSUPPORTED),'current_financial_nonmissing':d[d.date.eq(d.date.max())][list(FINANCIAL)].notna().sum().to_dict()}
    (out/'coverage.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--panel',required=True);p.add_argument('--matrix',required=True);p.add_argument('--base',required=True);p.add_argument('--extra',required=True);p.add_argument('--output',required=True);p.add_argument('--history')
    a=p.parse_args();build(a.panel,a.matrix,a.base,a.extra,a.output,a.history)
