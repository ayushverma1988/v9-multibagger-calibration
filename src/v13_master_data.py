"""Dated master inputs and additional-source evidence, independent of stock ranks.

Original observations, corroborated observations and unresolved gaps are kept
separate. Neither provider voting nor a changed watchlist overwrites an NSE fact.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import time
from urllib.parse import quote

from bs4 import BeautifulSoup
import numpy as np
import pandas as pd

from v12_5_features import pick, preferred_reporting_basis
from v13_financial_crosscheck import parse_screener, display_number

PRIOR_TWENTY = ('SINGERIND','PAUSHAKLTD','GULFPETRO','SWANDEF','SONAMLTD',
    'NINSYS','CUPID','DHANBANK','UNITEDPOLY','ALICON','THELEELA','ONWARDTEC',
    'ORIENTELEC','VINEETLAB','ONEPOINT','JAYNECOIND','SHYAMMETL','GODIGIT','LTTS','COFORGE')
RAW_FIELDS = ('revenue','pat','pat_owners','pat_noncontrolling','pbt','finance_cost',
    'depreciation','other_income','exceptional_items','eps_diluted_continuing',
    'paidup_equity_capital','face_value','cfo','assets','equity',
    'borrowings_current','borrowings_noncurrent','cash','current_liabilities',
    'ppe','receivables_current','receivables_noncurrent')
SECTIONS = {
    'quarters': ('quarter', {'Sales':'revenue','Revenue':'revenue',
        'Net Profit':'provider_net_profit','Profit before tax':'pbt',
        'Interest':'finance_cost','Depreciation':'depreciation',
        'Other Income':'other_income','EPS in Rs':'provider_eps_unspecified'}),
    'profit-loss': ('annual', {'Sales':'revenue','Revenue':'revenue',
        'Net Profit':'provider_net_profit','Profit before tax':'pbt',
        'Interest':'finance_cost','Depreciation':'depreciation','Other Income':'other_income'}),
    'balance-sheet': ('instant', {'Equity Capital':'paidup_equity_capital',
        'Reserves':'reserves','Borrowings':'provider_borrowings_unspecified',
        'Total Assets':'assets','Fixed Assets':'provider_fixed_assets','CWIP':'cwip'}),
    'cash-flow': ('annual', {'Cash from Operating Activity':'cfo',
        'Cash from Investing Activity':'cfi','Cash from Financing Activity':'cff'})}
AMBIGUOUS = {'provider_net_profit','provider_eps_unspecified',
             'provider_borrowings_unspecified','provider_fixed_assets'}

def dump(path, data):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,allow_nan=False,default=str))

def clean_records(frame):
    return json.loads(frame.to_json(orient='records',date_format='iso',double_precision=15))

def parse_screener_inputs(raw, symbol, isin, receipt):
    # The established parser verifies the exact exchange symbol and explicit
    # INR-crore units. A redirected standalone response keeps its own basis.
    parse_screener(raw,symbol,isin,receipt)
    soup=BeautifulSoup(raw,'html.parser')
    basis='consolidated' if soup.select_one('#company-info').get('data-consolidated')=='true' else 'standalone'
    rows=[]
    for sid,(kind,metrics) in SECTIONS.items():
        section=soup.find(id=sid)
        if section is None: continue
        if 'Figures in Rs. Crores' not in ' '.join(section.get_text(' ',strip=True).split()):
            raise ValueError('Missing explicit section monetary units')
        table=section.select_one('table')
        if table is None: continue
        trs=table.select('tr')
        headers=[c.get_text(' ',strip=True) for c in trs[0].select('th,td')][1:]
        for tr in trs[1:]:
            cells=[c.get_text(' ',strip=True) for c in tr.select('th,td')]
            if not cells: continue
            label=re.sub(r'\s+',' ',cells[0].replace('+','')).strip()
            metric=metrics.get(label)
            if metric is None: continue
            for period,display in zip(headers,cells[1:]):
                if not re.fullmatch(r'[A-Z][a-z]{2} 20\d\d',period): continue
                end=pd.to_datetime(period,format='%b %Y')+pd.offsets.MonthEnd(0)
                scale=1 if metric=='provider_eps_unspecified' else 10_000_000
                value=display_number(display,scale)
                if value is None: continue
                rows.append({'symbol':symbol,'isin':isin,'reporting_mode':basis,
                    'period_kind':kind,'period_end':str(end.date()),'metric':metric,
                    'value_INR':value[0],'rounding_half_width_INR':value[1],
                    'display_value':display,'display_unit':'INR_PER_SHARE' if scale==1 else 'INR_CRORE',
                    'publisher':'SCREENER','provider_family':'SCREENER',
                    'source_url':receipt['requested_url'],'source_sha256':receipt['sha256'],
                    'first_observed_utc':receipt['first_retrieved_utc'],
                    'identity_proof':'EXACT_NSE_SYMBOL_JOINED_PUBLIC_ISIN',
                    'definition_status':'UNRESOLVED' if metric in AMBIGUOUS else 'NAMED_FINANCIAL_LINE'})
    # Keep ownership observations separate from financial-statement amounts.
    section=soup.find(id='shareholding')
    if section:
        table=section.select_one('table')
        if table:
            trs=table.select('tr');heads=[c.get_text(' ',strip=True) for c in trs[0].select('th,td')][1:]
            for tr in trs[1:]:
                cells=[c.get_text(' ',strip=True) for c in tr.select('th,td')]
                if not cells or cells[0].replace('+','').strip()!='Promoters':continue
                for period,value in zip(heads,cells[1:]):
                    if not re.fullmatch(r'[A-Z][a-z]{2} 20\d\d',period):continue
                    v=display_number(value.rstrip('%'),1)
                    if v:
                        rows.append({'symbol':symbol,'isin':isin,'reporting_mode':'issuer',
                            'period_kind':'ownership','period_end':str((pd.to_datetime(period,format='%b %Y')+pd.offsets.MonthEnd(0)).date()),
                            'metric':'promoter_holding_pct','value_INR':v[0],
                            'rounding_half_width_INR':v[1],'display_unit':'PERCENTAGE_POINTS',
                            'display_value':value,'publisher':'SCREENER','provider_family':'SCREENER',
                            'source_url':receipt['requested_url'],'source_sha256':receipt['sha256'],
                            'first_observed_utc':receipt['first_retrieved_utc'],
                            'identity_proof':'EXACT_NSE_SYMBOL_JOINED_PUBLIC_ISIN','definition_status':'NAMED_OWNERSHIP_LINE'})
    return rows

KEYS=('isin','reporting_mode','period_kind','period_end','metric')

def reconcile(primary, observations, asof, review_clock):
    """Match identity, period, basis and definitions before comparing numbers.

    Multiple URLs from one provider family count once. Conflicts within that
    family disqualify it. No percentage tolerance hides a rounding mismatch.
    """
    date=pd.Timestamp(asof);clock=pd.Timestamp(review_clock)
    if clock.tzinfo is None:raise ValueError('Review clock must be timezone aware')
    groups={};rejected=[]
    for r in observations:
        try:
            if pd.Timestamp(r['period_end'])>date:continue
            if pd.Timestamp(r['first_observed_utc'])>clock:continue
            if r.get('definition_status')=='UNRESOLVED' or r['metric'] in AMBIGUOUS:continue
            if not r.get('identity_proof') or r['reporting_mode'] not in {'standalone','consolidated','issuer'}:continue
            if not r.get('source_url','').startswith('https://'):continue
            if not np.isfinite(r['value_INR']) or not np.isfinite(r['rounding_half_width_INR']) or r['rounding_half_width_INR']<0:continue
            key=tuple(r[k] for k in KEYS);groups.setdefault(key,[]).append(r)
        except (KeyError,TypeError,ValueError):rejected.append(r.get('source_url'))
    out=[]
    for key,rs in groups.items():
        p=primary.get(key);families={}
        for r in rs:families.setdefault(r['provider_family'],[]).append(r)
        usable={};family_conflicts=[]
        for family,items in families.items():
            # Compare the latest observation vintage, not a mix of stale and
            # revised pages. Equal-time conflicting displays stay unresolved.
            latest=max(x['first_observed_utc'] for x in items)
            items=[x for x in items if x['first_observed_utc']==latest]
            low=max(x['value_INR']-x['rounding_half_width_INR'] for x in items)
            high=min(x['value_INR']+x['rounding_half_width_INR'] for x in items)
            if low>high+1e-6:family_conflicts.append(family)
            else:usable[family]=(low,high,items[-1])
        matches={k:v for k,v in usable.items() if p is not None and v[0]-1e-6<=p['value_INR']<=v[1]+1e-6}
        low=max((v[0] for v in matches.values()),default=np.nan)
        high=min((v[1] for v in matches.values()),default=np.nan)
        triple=p is not None and len(matches)>=2 and low<=high+1e-6
        status=('PRIMARY_PLUS_TWO_ADDITIONAL_PROVIDER_FAMILIES_AGREE' if triple else
            'PRIMARY_VALUE_MISSING_OR_AMBIGUOUS' if p is None else
            'NUMERIC_CONFLICT_REVIEW' if any(k not in matches for k in usable) else
            'FEWER_THAN_TWO_ADDITIONAL_PROVIDERS')
        out.append({**dict(zip(KEYS,key)),'symbol':rs[0]['symbol'],
            'primary_value_INR':None if p is None else p['value_INR'],
            'primary_source_url':None if p is None else p['source_url'],
            'primary_source_sha256':None if p is None else p['source_sha256'],
            'matching_provider_families':sorted(matches),'within_family_conflicts':family_conflicts,
            'status':status,'validated_for_new_master_input':bool(triple),
            'canonical_original_overwritten':False,'provider_observations':rs})
    return out,rejected

def primary_index(wide,asof,clock):
    d=wide[wide.period.le(pd.Timestamp(asof))&wide.available.le(pd.Timestamp(clock))]
    out={}
    for (isin,basis,kind,end),g in d.groupby(['security_isin','reporting_mode','period_kind','period_end']):
        row=pick(g,kind,pd.Timestamp(end))
        if not row:continue
        for field in RAW_FIELDS:
            value=row.get(field)
            if value is None or not np.isfinite(value):continue
            out[(isin,basis,kind,str(pd.Timestamp(end).date()),field)]={
                'value_INR':float(value),'source_url':row.get('source_url'),
                'source_sha256':row.get('source_sha256'),
                'available_at_utc':str(row.get('available_at_utc'))}
    return out

def collect_providers(root,asof,budget=3200):
    """Whole-public-universe rotation. Access denial stops the source host."""
    import requests
    root=Path(root);folder=root/'master_data';folder.mkdir(exist_ok=True)
    u=pd.read_parquet(root/'universe.parquet')
    prior=folder/'provider_attempts.json';attempts=json.loads(prior.read_text()) if prior.exists() else {}
    queue=sorted(u.to_dict('records'),key=lambda r:(attempts.get(r['isin'],'0000'),hashlib.sha256(r['isin'].encode()).hexdigest()))[:budget]
    accepted=0;errors=[];start=time.monotonic()
    for i,r in enumerate(queue):
        if time.monotonic()-start>1800:break
        symbol=r['nse_current_symbol'];now=datetime.now(timezone.utc).isoformat()
        url='https://www.screener.in/company/'+quote(symbol,safe='')+'/consolidated/'
        try:
            response=requests.get(url,headers={'User-Agent':'Mozilla/5.0'},timeout=(8,25));response.raise_for_status()
            raw=response.content;proof={'requested_url':url,'final_url':response.url,
                'sha256':hashlib.sha256(raw).hexdigest(),'first_retrieved_utc':now,'status':response.status_code}
            rows=parse_screener_inputs(raw,symbol,r['isin'],proof)
            vintage=folder/'raw'/proof['sha256'];vintage.mkdir(parents=True,exist_ok=True)
            (vintage/'screener.html').write_bytes(raw);dump(vintage/'receipt.json',proof)
            dump(folder/'provider_inputs'/(r['isin']+'.json'),rows);accepted+=1
        except Exception as exc:
            error=str(exc)[:180];errors.append({'symbol':symbol,'url':url,'error':error})
            if '403' in error or '429' in error:break
        attempts[r['isin']]=now
        if (i+1)%100==0:
            dump(prior,attempts);print('Master source pages',i+1,'accepted',accepted,flush=True)
        time.sleep(.30)
    dump(prior,attempts)
    result={'attempted':accepted+len(errors),'accepted':accepted,'errors':errors,
        'public_universe':len(u),'private_ranks_used_in_requests':False,
        'pending_issuers':max(0,len(u)-len(attempts))}
    dump(folder/'collection.json',result);return result

def export(root,asof):
    root=Path(root);folder=root/'master_data';folder.mkdir(exist_ok=True)
    clock=datetime.now(timezone.utc).isoformat()
    wide=pd.read_parquet(root/'financial_wide.parquet')
    primary=primary_index(wide,asof,clock);observations=[]
    low=str((pd.Timestamp(asof)-pd.DateOffset(years=3)).date())
    for p in sorted((folder/'provider_inputs').glob('*.json')):
        observations.extend(r for r in json.loads(p.read_text()) if r.get('period_end','')>=low)
    # Existing manually reviewed provider observations keep their explicit
    # review clocks. A web transcription hash is never an original PDF hash.
    review=root/'financial_crosscheck/latest.json'
    if review.exists():observations.extend(json.loads(review.read_text()).get('rows',[]))
    seeds=Path(__file__).resolve().parents[1]/'v13_evidence_seed/master_source_reviews_20261020.json'
    if seeds.exists():observations.extend(json.loads(seeds.read_text()).get('observations',[]))
    comparisons,rejected=reconcile(primary,observations,asof,clock)
    matched={tuple(r[k] for k in KEYS):r for r in comparisons}
    coverage=json.loads((root/'evidence/universe_financial_coverage.json').read_text())
    latest=root/'redevelopment/latest.json'
    forecasts=json.loads(latest.read_text()).get('records',[]) if latest.exists() else []
    by_isin={r['isin']:r for r in forecasts}
    groups=dict(tuple(wide[wide.period.le(pd.Timestamp(asof))&wide.available.le(pd.Timestamp(clock))].groupby('security_isin')))
    rows=[];cells=[]
    specs=[('quarter',f) for f in RAW_FIELDS if f not in {'cfo','assets','equity','borrowings_current','borrowings_noncurrent','cash','current_liabilities','ppe','receivables_current','receivables_noncurrent'}]
    specs += [('annual',f) for f in ('revenue','pat','cfo','pbt','finance_cost','depreciation')]
    specs += [('instant',f) for f in ('assets','equity','borrowings_current','borrowings_noncurrent','cash','current_liabilities','ppe','receivables_current','receivables_noncurrent')]
    for a in coverage['rows']:
        w=groups.get(a['isin'],pd.DataFrame());basis=a['preferred_reporting_basis']
        w=w[w.reporting_mode.eq(basis)] if not w.empty else w
        r={k:a.get(k) for k in ('symbol','isin','preferred_reporting_basis','financial_complete','financial_quality_status','preferred_quarter_end','owner_earnings_source_status')}
        r.update(market_asof=asof,review_clock_utc=clock,usable_current_market_inputs=a['isin'] in by_isin,
            missing_financial_checks=', '.join(a['unknown_fields']),known_failed_checks=', '.join(a['failed_checks']))
        current=by_isin.get(a['isin'],{})
        for field in ('close','rsi14','distance_ma50','distance_ma200'):r[field]=current.get(field)
        for kind,field in specs:
            row=pick(w,kind) if not w.empty else {};period=str(row['period'].date()) if row else None
            key=(a['isin'],basis,kind,period,field);p=primary.get(key);proof=matched.get(key,{})
            name=kind+'_'+field;r[name]=p['value_INR'] if p else None
            r['validated_'+name]=r[name] if proof.get('validated_for_new_master_input',False) else None
            cells.append({'symbol':a['symbol'],'isin':a['isin'],'column':name,
                'reporting_mode':basis,'period_kind':kind,'period_end':period,
                'metric':field,'value':r[name],
                'source_url':p['source_url'] if p else None,'source_sha256':p['source_sha256'] if p else None,
                'source_status':'ORIGINAL_OBSERVED' if p else 'MISSING_OR_AMBIGUOUS_ORIGINAL',
                'two_additional_sources_agree':proof.get('validated_for_new_master_input',False),
                'source_review':proof.get('status','NO_ADDITIONAL_SOURCE_COMPARISON'),
                'matching_provider_families':', '.join(proof.get('matching_provider_families',[]))})
        rows.append(r)
    master=pd.DataFrame(rows);long=pd.DataFrame(cells)
    master.to_parquet(folder/'master_inputs.parquet',index=False)
    master.to_csv(folder/'master_inputs.csv',index=False)
    long.to_parquet(folder/'master_input_sources.parquet',index=False)
    long.to_csv(folder/'master_input_sources.csv',index=False)
    dump(folder/'master_inputs.json',clean_records(master))
    dump(folder/'source_comparisons.json',comparisons)
    pending=long[long.value.isna()|~long.two_additional_sources_agree].copy()
    pending.to_parquet(folder/'pending_input_validation.parquet',index=False)
    counts=[]
    for column,g in long.groupby('column',sort=False):
        counts.append({'column':column,'issuers':len(g),'original_observed':int(g.value.notna().sum()),
            'missing_or_ambiguous_original':int(g.value.isna().sum()),
            'primary_plus_two_additional_sources_agree':int(g.two_additional_sources_agree.sum())})
    dump(folder/'column_coverage.json',counts)
    from v13_model import FEATURES
    model_columns=[{'column':c,'scored_issuers':len(forecasts),
        'observed':sum(r.get(c) is not None and isinstance(r.get(c),(int,float)) and np.isfinite(r[c]) for r in forecasts),
        'missing':sum(r.get(c) is None or not isinstance(r.get(c),(int,float)) or not np.isfinite(r[c]) for r in forecasts)} for c in FEATURES]
    dump(folder/'model_column_coverage.json',model_columns)
    old_doc=Path(__file__).resolve().parents[1]/'docs/v13_two_x_research_current_20261018.md'
    historic=[]
    for line in old_doc.read_text().splitlines():
        c=[s.strip() for s in line.split('|')[1:-1]]
        if len(c)>=7 and c[1] in PRIOR_TWENTY:
            historic.append({'rank':int(c[0]),'symbol':c[1],'close':float(c[2].replace(',','')),
                'p6':float(c[3].rstrip('%'))/100,'p12':float(c[4].rstrip('%'))/100,
                'loss12':float(c[5].rstrip('%'))/100,'financial_tier':c[6]})
    if len(historic)!=20:raise ValueError('Earlier twenty-name source does not reproduce exactly')
    by_symbol={r['symbol']:r for r in forecasts};prior=[]
    for old in historic:
        now=by_symbol.get(old['symbol'],{})
        prior.append({'symbol':old['symbol'],'historical_price_date':'2026-10-16',
            'historical':old,'current_market_input_available':bool(now),
            'current_close':now.get('close'),'current_p6':now.get('p6'),'current_p12':now.get('p12'),
            'basic_selected':bool(now.get('research_selected',False)),
            'basic_rank':now.get('research_rank'),'current_rsi14':now.get('rsi14'),
            'financial_quality_status':now.get('financial_quality_status'),
            'missing_financial_checks':next((r['missing_financial_checks'] for r in rows if r['symbol']==old['symbol']),None)})
    dump(folder/'earlier_twenty_comparison.json',prior)
    basic=sorted((r for r in forecasts if r.get('research_selected')),key=lambda r:r['research_rank'])
    dump(folder/'latest_completed_basic_twenty.json',basic)
    lines=['# Broader research list from the last completed model run','',
        'Prices: '+asof+'. This preserves the available-data screen separately from the refined size/sector screen. It is not a new prediction after this collection.','',
        '| Rank | Share | Close INR | Estimated 2x by 6m | Estimated 2x by 12m | Financial inputs |',
        '| ---: | --- | ---: | ---: | ---: | --- |']
    for r in basic:lines.append(f"| {int(r['research_rank'])} | {r['symbol']} | {r['close']:.2f} | {r['p6']:.2%} | {r['p12']:.2%} | {r.get('financial_quality_status')} |")
    (folder/'latest_completed_basic_twenty.md').write_text('\n'.join(lines)+'\n')
    result={'market_asof':asof,'review_clock_utc':clock,'issuers':len(master),
        'master_input_columns':len(specs),'required_input_cells':len(long),
        'original_observed_cells':int(long.value.notna().sum()),
        'missing_or_ambiguous_original_cells':int(long.value.isna().sum()),
        'cells_with_primary_plus_two_additional_sources':int(long.two_additional_sources_agree.sum()),
        'provider_observations':len(observations),'comparison_groups':len(comparisons),
        'rejected_observations':len(rejected),'column_coverage':counts,
        'all_missing_data_collected':False,'analysis_ready_under_three_source_requirement':bool(long.two_additional_sources_agree.all()),
        'new_prediction_run':False,'frozen_forecasts_unchanged':True,
        'prior_twenty_price_date':'2026-10-16','prior_twenty_current_market_rows':sum(r['current_market_input_available'] for r in prior)}
    dump(folder/'summary.json',result)
    return result

def run(root,passes=3,provider_budget=3200,incoming=None):
    from v13_checkpoint import restore, save
    from v13_collection import collect_financials
    from v13_universe_financial_audit import audit
    from v13_frozen_integrity import FrozenGuard
    root=Path(root)
    if not restore(root):raise RuntimeError('Existing checkpoint required; no fresh state substituted')
    guard=FrozenGuard(root)
    if incoming:
        incoming=Path(incoming)
        # Only source observations enter the fresh checkpoint. Never replace
        # an older restored financial matrix, model, registry or forecast.
        for source in incoming.rglob('*'):
            if not source.is_file() or source.is_symlink() or source.suffix not in {'.json','.gz'}:continue
            relative=source.relative_to(incoming)
            if relative.parts[0] not in {'provider_inputs','sector_nse','sector_finology','sector_screener','raw'} and relative.name not in {'alternate_attempts.json','alternate_collection.json'}:continue
            target=root/'master_data'/relative;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,target)
    previous=json.loads((root/'latest_run.json').read_text());asof=previous['asof']
    root.joinpath('master_data').mkdir(exist_ok=True)
    baseline=audit(pd.read_parquet(root/'universe.parquet'),pd.read_parquet(root/'financial_wide.parquet'),asof,root)
    if not (root/'master_data/before_collection.json').exists():dump(root/'master_data/before_collection.json',baseline)
    runs=[]
    for i in range(passes):
        report=collect_financials(root,asof,budget=2400);runs.append(report)
        dump(root/'master_data/original_collection_passes.json',runs)
        print('Master original collection pass',i+1,json.dumps({k:v for k,v in report.items() if k!='errors'}),flush=True)
    if provider_budget:collect_providers(root,asof,provider_budget)
    from v13_alternate_sources import sector_master
    sector_master(root)
    after=audit(pd.read_parquet(root/'universe.parquet'),pd.read_parquet(root/'financial_wide.parquet'),asof,root)
    dump(root/'master_data/after_collection.json',after)
    result=export(root,asof)
    print('MASTER_COLLECTION_SUMMARY',json.dumps(result),flush=True)
    guard.check(pd.read_parquet(root/'prospective_observations.parquet'))
    receipt=save(root);dump(root.parent/'master_archive_receipt.json',receipt)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--state',default='v13_state');p.add_argument('--passes',type=int,default=3);p.add_argument('--provider-budget',type=int,default=3200);p.add_argument('--merge-source-dir')
    args=p.parse_args();run(args.state,args.passes,args.provider_budget,args.merge_source_dir)
