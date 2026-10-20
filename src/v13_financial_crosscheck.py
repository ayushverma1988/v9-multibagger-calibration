"""Comparable public financial observations, without voting source facts true.

NSE originals remain canonical. Provider rounded displays can corroborate an
amount; they never establish recurring EPS, future growth, or revised history.
"""
import hashlib,json,re
from pathlib import Path
from urllib.parse import urlparse,parse_qs,quote
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from v12_5_features import pick,number

METRICS={'Sales':'revenue','Revenue':'revenue','Net Profit':'provider_net_profit',
 'Cash from Operating Activity':'cfo'}

def display_number(text,scale):
    text=text.strip().replace(',','').replace('\u2212','-')
    if not re.fullmatch(r'-?\d+(?:\.\d+)?',text):return None
    decimals=len(text.split('.')[1]) if '.' in text else 0
    return float(text)*scale,.5*scale*10**(-decimals)

def parse_screener(raw,symbol,isin,receipt):
    soup=BeautifulSoup(raw,'html.parser');company=soup.select_one('#company-info')
    if company is None:raise ValueError('Missing source issuer identity')
    matches=set()
    for a in soup.select('a[href]'):
        u=urlparse(a['href'])
        if u.hostname in {'www.nseindia.com','nseindia.com'}:
            matches.update(parse_qs(u.query).get('symbol',[]))
    if matches!={symbol}:raise ValueError('Provider/source NSE symbol identity mismatch')
    basis='consolidated' if company.get('data-consolidated')=='true' else 'standalone'
    rows=[]
    for sid,kind in [('quarters','quarter'),('profit-loss','annual'),('cash-flow','annual')]:
        sec=soup.find(id=sid)
        if sec is None:continue
        text=' '.join(sec.get_text(' ',strip=True).split())
        if 'Figures in Rs. Crores' not in text:raise ValueError('Provider unit not explicitly INR crore')
        table=sec.select_one('table')
        if table is None:continue
        trs=table.select('tr');headers=[c.get_text(' ',strip=True) for c in trs[0].select('th,td')][1:]
        for tr in trs[1:]:
            cells=[c.get_text(' ',strip=True) for c in tr.select('th,td')]
            if not cells:continue
            label=cells[0].replace('+','').strip();metric=METRICS.get(label)
            if metric is None:continue
            for period,display in zip(headers,cells[1:]):
                if not re.fullmatch(r'[A-Z][a-z]{2} 20\d\d',period):continue
                end=pd.to_datetime(period,format='%b %Y')+pd.offsets.MonthEnd(0)
                value=display_number(display,10_000_000)
                if value is None:continue
                amount,tolerance=value
                rows.append({'symbol':symbol,'isin':isin,'reporting_mode':basis,'period_kind':kind,
                  'period_end':str(end.date()),'metric':metric,'value_INR':amount,
                  'rounding_half_width_INR':tolerance,'display_value':display,'display_unit':'INR_CRORE',
                  'publisher':'SCREENER','provider_family':'SCREENER','source_url':receipt['requested_url'],
                  'source_sha256':receipt['sha256'],'first_observed_utc':receipt['first_retrieved_utc'],
                  'identity_proof':'PROVIDER_EXACT_NSE_SYMBOL_LINK_JOINED_CURRENT_PUBLIC_ISIN',
                  'profit_definition':'UNVERIFIED_TOTAL_VERSUS_PARENT' if metric=='provider_net_profit' else 'NAMED_REVENUE_OR_OPERATING_CASH_FLOW'})
    return rows

def known_at(wide,asof):
    if wide.empty:return wide.copy()
    date=pd.Timestamp(asof)
    cut=date.tz_localize('Asia/Kolkata')+pd.Timedelta(hours=22)
    return wide[pd.to_datetime(wide.available,utc=True,format='mixed').le(cut)&pd.to_datetime(wide.period).le(date)].copy()

def compare(wide,observations,asof):
    date=pd.Timestamp(asof);results=[]
    wide=known_at(wide,asof)
    groups=dict(tuple(wide.groupby('security_isin'))) if not wide.empty else {}
    for r in observations:
        if pd.Timestamp(r['period_end'])>date or pd.Timestamp(r['period_end'])<date-pd.DateOffset(years=4):continue
        w=groups.get(r['isin'],pd.DataFrame())
        row={}
        if not w.empty:
            w=w[w.reporting_mode.eq(r['reporting_mode'])]
            row=pick(w,r['period_kind'],pd.Timestamp(r['period_end']))
        metric=r['metric'];v=number(row,metric)
        status='PRIMARY_FACT_MISSING_OR_AMBIGUOUS'
        if metric=='provider_net_profit':status='PROFIT_DEFINITION_REVIEW_REQUIRED'
        elif np.isfinite(v):
            status='NUMERIC_AGREEMENT_WITHIN_DISPLAY_ROUNDING' if abs(v-r['value_INR'])<=r['rounding_half_width_INR']+1 else 'NUMERIC_CONFLICT_REVIEW_REQUIRED'
        results.append({**r,'primary_value_INR':float(v) if np.isfinite(v) else None,
          'primary_source_sha256':row.get('source_sha256'),'primary_source_url':row.get('source_url'),
          'comparison_status':status,'canonical_fact_overwritten':False})
    keys=['isin','period_kind','period_end','reporting_mode','metric']
    groups={}
    for r in results:groups.setdefault(tuple(r[k] for k in keys),[]).append(r)
    triples=[]
    for key,rs in groups.items():
        by_provider={}
        for r in rs:
            if r['comparison_status']=='NUMERIC_AGREEMENT_WITHIN_DISPLAY_ROUNDING':by_provider[r['provider_family']]=r
        # Mirror sites or repeated observations of one provider count once.
        if len(by_provider)<2:continue
        vals=list(by_provider.values());low=max(r['value_INR']-r['rounding_half_width_INR'] for r in vals)
        high=min(r['value_INR']+r['rounding_half_width_INR'] for r in vals)
        triples.append({**dict(zip(keys,key)),'symbol':rs[0]['symbol'],'status':
          'PRIMARY_PLUS_TWO_PROVIDER_AGREEMENT_WITHIN_ROUNDING' if low<=high+1 else 'PROVIDER_ROUNDING_INTERVAL_CONFLICT',
          'primary_value_INR':rs[0]['primary_value_INR'],'additional_provider_families':sorted(by_provider),
          'sources':[{'url':r['source_url'],'first_observed_utc':r['first_observed_utc']} for r in vals],
          'common_underlying_issuer_origin_possible':True,'truth_certified':False})
    return results,triples

def tag_current_reviews(records,wide,comparison,asof,basis_by_isin):
    """Add review tiers without changing model probabilities or membership."""
    wide=known_at(wide,asof);tagged=[]
    for record in records:
        r=dict(record);isin=r['isin'];basis=basis_by_isin.get(isin)
        q=wide[wide.security_isin.eq(isin)&wide.reporting_mode.eq(basis)&wide.period_kind.eq('annual')] if not wide.empty and basis else pd.DataFrame()
        last=str(pd.Timestamp(q.period.max()).date()) if not q.empty else None
        rows=[x for x in comparison.get('rows',[]) if x['isin']==isin and x['reporting_mode']==basis and x['period_kind']=='annual' and x['period_end']==last and x['metric'] in {'revenue','cfo'}]
        conflicts=sorted({x['metric'] for x in rows if x['comparison_status']=='NUMERIC_CONFLICT_REVIEW_REQUIRED'})
        agreements={m:{x['provider_family'] for x in rows if x['metric']==m and x['comparison_status']=='NUMERIC_AGREEMENT_WITHIN_DISPLAY_ROUNDING'} for m in ('revenue','cfo')}
        tier=('SECTOR_RULES_PENDING_REVIEW' if r.get('sector_status')=='UNKNOWN_SECTOR_RULES_OR_CLASSIFICATION' else
          'ANNUAL_SOURCE_CONFLICT_REVIEW' if conflicts else
          'ANNUAL_REVENUE_CFO_TWO_ADDITIONAL_PROVIDERS_AGREE' if all(len(v)>=2 for v in agreements.values()) else
          'LIMITED_ANNUAL_COMPARISON' if rows else 'NO_CURRENT_ANNUAL_COMPARISON')
        r.update(source_comparison_tier=tier,source_review_period_end=last,source_conflicting_metrics=conflicts,
          source_review_scope='ANNUAL_REVENUE_AND_CFO_ONLY_NOT_ALL_FINANCIAL_FIELDS')
        tagged.append(r)
    return tagged

def export(root,wide,asof,extra_observations=()):
    root=Path(root);observations=list(extra_observations);folder=root/'financial_crosscheck';folder.mkdir(exist_ok=True)
    for path in (folder/'raw').glob('*_screener.html'):
        receipt=path.with_suffix(path.suffix+'.receipt.json')
        if not receipt.exists():continue
        proof=json.loads(receipt.read_text());raw=path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=proof['sha256']:raise ValueError('Financial comparison original hash mismatch')
        observations.extend(parse_screener(raw,proof['symbol'],proof['isin'],proof))
    seed=Path(__file__).resolve().parents[1]/'v13_evidence_seed'/'financial_provider_review_20261020.json'
    if seed.exists():observations.extend(json.loads(seed.read_text())['observations'])
    reviewed=seed.with_name('financial_screener_review_20261020.json')
    if reviewed.exists():observations.extend(json.loads(reviewed.read_text())['observations'])
    results,triples=compare(wide,observations,asof)
    result={'source_asof':asof,'observations_compared':len(results),
      'issuers_compared':len({r['isin'] for r in results}),
      'status_counts':pd.Series([r['comparison_status'] for r in results],dtype=object).value_counts().to_dict(),
      'primary_plus_two_provider_agreements':sum(r['status']=='PRIMARY_PLUS_TWO_PROVIDER_AGREEMENT_WITHIN_ROUNDING' for r in triples),
      'truth_certified':False,'definition_and_period_matching_required':True,'canonical_facts_overwritten':False,
      'rows':results,'three_source_checks':triples}
    (folder/'latest.json').write_text(json.dumps(result,indent=2,default=str))
    return {k:v for k,v in result.items() if k not in {'rows','three_source_checks'}}

def collect(root,asof,budget=160):
    """Rotate a whole-public-issuer queue, unrelated to predictions or ranks."""
    import requests,time
    from datetime import datetime,timezone
    root=Path(root);u=pd.read_parquet(root/'universe.parquet')
    folder=root/'financial_crosscheck';folder.mkdir(exist_ok=True)
    prior=folder/'attempts.json';attempts=json.loads(prior.read_text()) if prior.exists() else {}
    rows=sorted(u.to_dict('records'),key=lambda r:(attempts.get(r['isin'],'0000'),hashlib.sha256(r['isin'].encode()).hexdigest()))[:budget]
    errors=[];accepted=0
    for r in rows:
        s=r['nse_current_symbol'];url='https://www.screener.in/company/'+quote(s,safe='')+'/consolidated/'
        now=datetime.now(timezone.utc).isoformat()
        try:
            response=requests.get(url,headers={'User-Agent':'Mozilla/5.0'},timeout=(8,25))
            raw=response.content;response.raise_for_status()
            proof={'requested_url':url,'final_url':response.url,'sha256':hashlib.sha256(raw).hexdigest(),
              'first_retrieved_utc':now,'status':response.status_code,'symbol':s,'isin':r['isin']}
            parse_screener(raw,s,r['isin'],proof)
            out=folder/'raw';out.mkdir(exist_ok=True);p=out/(s+'_screener.html')
            # Keep source vintages; a thinner later page cannot erase receipts.
            archive=out/'versions'/proof['sha256'];archive.mkdir(parents=True,exist_ok=True)
            (archive/p.name).write_bytes(raw);(archive/(p.name+'.receipt.json')).write_text(json.dumps(proof,indent=2))
            p.write_bytes(raw);p.with_suffix(p.suffix+'.receipt.json').write_text(json.dumps(proof,indent=2));accepted+=1
        except Exception as exc:
            message=str(exc)[:160];errors.append({'symbol':s,'error':message})
            if '429' in message or '403' in message:break
        attempts[r['isin']]=now;time.sleep(.3)
    prior.write_text(json.dumps(attempts,indent=2))
    wide=pd.read_parquet(root/'financial_wide.parquet')
    return {'pages_accepted':accepted,'errors':errors,**export(root,wide,asof)}
