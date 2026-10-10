"""Run resumable alternative-source recovery; aggregate logs, private evidence."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import gzip
import io
import json
import os
from pathlib import Path
from urllib.parse import quote

import pandas as pd

from v11_4_free_source_fallbacks import (
    PublicEvidenceStore, parse_interoperability_master, parse_finology,
    parse_yahoo_bse, reconcile_provider_facts)
from v11_4_exchange_reference_universe import parse_nse_master


def deterministic_order(frame, salt):
    x = frame.copy()
    x['_hash'] = x['isin'].map(lambda v: hashlib.sha256((salt+v).encode()).hexdigest())
    return x.sort_values('_hash').drop(columns='_hash')


def build_recovery_universe(reference, exclusive, market):
    if reference['isin'].duplicated().any() or exclusive['isin'].duplicated().any():
        raise ValueError("Ambiguous ISIN identities")
    cols = ['isin','company_name','bse_reference_symbol','nse_current_symbol',
            'present_NSE_current_master','amfi_size_category','amfi_period_end']
    x = reference[cols].merge(exclusive, on='isin', how='outer', validate='1:1')
    x['official_current_reference_present'] = x.present_NSE_current_master.eq(True) | x.bse_exclusive_official_interop_reference.eq(True)
    x['quote_request_symbol'] = x.bse_reference_symbol
    # A truncated $-suffixed outage alias is not automatically a Yahoo ticker.
    x['BSE_full_symbol_mapping_available'] = x.quote_request_symbol.fillna('').astype(str).str.strip().ne('')
    x['current_official_NSE_close'] = x['isin'].map(market.drop_duplicates('isin').set_index('isin').close)
    return x


def get_master(store, asof):
    stamp = pd.Timestamp(asof).strftime('%d%m%Y')
    url = f'https://nsearchives.nseindia.com/content/cm/interop/NSE_CM_security_{stamp}.csv.gz'
    raw, receipt = store.get(url)
    exclusive, master = parse_interoperability_master(raw, asof)
    master.attrs['unfiltered_official_master_rows']=len(pd.read_csv(io.BytesIO(gzip.decompress(raw)),low_memory=False))
    return exclusive, master, receipt


def financial_items(public_master, max_companies=0):
    """Outbound symbols depend only on the complete official public listings.

    Private features, watchlists, holdings, model ranks and chosen reporting
    bases never select a provider URL. Reconciliation with private source
    checkpoints takes place locally after public-source recovery.
    """
    x=public_master.rename(columns={'nse_current_symbol':'symbol',
                                     'nse_current_name':'company_name'}).copy()
    if x['symbol'].duplicated().any() or x['isin'].duplicated().any():
        raise ValueError('Ambiguous public NSE financial request identity')
    x=x.sort_values('symbol')
    if max_companies:x=x.head(max_companies)
    x['preferred_mode']='consolidated'
    return x[['symbol','isin','company_name','preferred_mode']].to_dict('records')


def get_public_financial_master(store):
    urls=['https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv',
          'https://nsearchives.nseindia.com/emerge/corporates/content/SME_EQUITY_L.csv']
    frames,receipts=[],[]
    for url in urls:
        raw,receipt=store.get(url);frames.append(parse_nse_master(raw));receipts.append(receipt)
    master=pd.concat(frames,ignore_index=True).drop_duplicates(['isin','nse_current_symbol'])
    financial_items(master)  # Validate conflicts before any provider request.
    return master,receipts


def run_financials(source_recovery, output, max_companies=0, workers=12):
    out = Path(output);out.mkdir(parents=True, exist_ok=True)
    store = PublicEvidenceStore(out/'raw_PRIVATE')
    public_master,public_master_receipts=get_public_financial_master(store)
    items=financial_items(public_master,max_companies)
    public_master.to_parquet(out/'public_NSE_financial_request_universe.parquet',index=False)
    (out/'public_request_universe_receipts.json').write_text(json.dumps(public_master_receipts,indent=2))
    results, errors, audits = [], [], []
    original_facts = pd.read_parquet(Path(source_recovery)/'financial_facts_exact_context_PRIVATE.parquet').to_dict('records')
    originals = {}
    for f in original_facts:
        originals.setdefault(f['symbol'], []).append(f)
    item_dir = out/'financial_companies_PRIVATE';item_dir.mkdir(exist_ok=True)

    def one(item):
        try:
            url = 'https://ticker.finology.in/company/'+quote(item['symbol'],safe='')
            if item['preferred_mode'] == 'consolidated':url += '?mode=C'
            raw, rec = store.get(url)
            data = parse_finology(raw, rec['final_url'], item['symbol'],item['company_name'],rec['first_retrieved_utc'])
            data['isin'] = item['isin']
            data['reconciliation'] = reconcile_provider_facts(data['facts'], originals.get(item['symbol'], []))
            # Provider ratios are not independently verified by agreement of
            # two other fields. Each observation retains its own evidence tier.
            path = item_dir/(hashlib.sha256(item['isin'].encode()).hexdigest()+'.json')
            temp = path.with_suffix('.json.tmp')
            temp.write_text(json.dumps(data,indent=2))
            os.replace(temp, path)
            return data, None
        except Exception as e:
            return None, {'symbol':item['symbol'],'isin':item['isin'],'stage':'free_financial_source','error':type(e).__name__+': '+str(e)[:180]}

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futures=[ex.submit(one,item) for item in items]
        for n, future in enumerate(concurrent.futures.as_completed(futures), 1):
            data,error=future.result()
            if data:
                results.append(data);audits.extend(data['reconciliation'])
            if error:errors.append(error)
            if n % 25 == 0 or n == len(items):
                print(json.dumps({'financial_progress':n,'planned':len(items),'pages_usable':len(results),'errors':len(errors)}),flush=True)
    facts = [f for r in results for f in r['facts']]
    pd.DataFrame(facts).to_parquet(out/'free_financial_facts_PRIVATE.parquet',index=False)
    pd.DataFrame(audits).to_csv(out/'financial_original_reconciliation_PRIVATE.csv',index=False)
    (out/'free_financial_errors_PRIVATE.json').write_text(json.dumps(errors,indent=2))
    summary = {'scope':'CURRENT_PUBLIC_PROVIDER_FINANCIAL_RECOVERY_NOT_HISTORICAL_PIT_TRAINING',
        'companies_requested':len(items),'financial_pages_usable':len(results),'errors':len(errors),
        'request_universe':'COMPLETE_OFFICIAL_PUBLIC_NSE_MAIN_AND_SME_LISTINGS',
        'private_cohort_watchlists_or_model_ranks_sent_to_provider':False,
        'source_fact_observations':len(facts),
        'original_filings_comparisons':sum(a['status']!='NO_UNIQUE_ORIGINAL_MATCH' for a in audits),
        'original_fact_agreements':sum(a['status']=='AGREES_WITH_ORIGINAL' for a in audits),
        'original_fact_disagreements':sum(a['status']=='DISAGREES_WITH_ORIGINAL' for a in audits),
        'provider_reported_ratios_companies':{k:sum(k in r['reported_ratios'] for r in results)
            for k in ['sales_growth_3y','sales_growth_5y','profit_growth_5y','roe_3y_avg','roe_5y_avg','roce_3y_avg']},
        'cashflow_companies':sum(any(f['metric']=='cfo' for f in r['facts']) for r in results),
        'latest_annual_source_periods':pd.Series([max((f['period_end'] for f in r['facts'] if f['period_kind']=='annual'),default='unknown') for r in results]).value_counts().to_dict(),
        'five_rows_never_treated_as_five_year_CAGR_observations':True,
        'original_rules_and_predictor_scores_unchanged':True,
        'source_qualified_full_model':False,'production_approved':False}
    (out/'free_financial_recovery_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)
    return summary


def run_bse(source_recovery, output, asof, max_quotes=0, history_companies=64, workers=20):
    out = Path(output);out.mkdir(parents=True,exist_ok=True)
    store = PublicEvidenceStore(out/'raw_PRIVATE')
    reference = pd.read_parquet(Path(source_recovery)/'NSE_BSE_AMFI_reference_universe_PRIVATE.parquet')
    market = pd.read_parquet(Path(source_recovery)/'same_day_official_NSE_market_identity_PRIVATE.parquet')
    exclusive, master, receipt = get_master(store,asof)
    universe = build_recovery_universe(reference,exclusive,market)
    universe.to_parquet(out/'NSE_BSE_current_and_dated_reference_PRIVATE.parquet',index=False)
    exclusive.to_parquet(out/'official_BSE_exclusive_interop_identities_PRIVATE.parquet',index=False)
    (out/'interoperability_master_receipt.json').write_text(json.dumps(receipt,indent=2))
    # Conflicting current/different-period aliases need corporate-action evidence.
    mapped = universe[universe.BSE_full_symbol_mapping_available].copy()
    ambiguous = mapped.quote_request_symbol.duplicated(keep=False)
    mapped = mapped[~ambiguous].copy()
    mapped = deterministic_order(mapped, 'BSE-free-quote-source-20261010')
    mapped['priority'] = (~mapped.bse_exclusive_official_interop_reference.eq(True)).astype(int)
    mapped = mapped.sort_values('priority',kind='stable')
    if max_quotes:mapped=mapped.head(max_quotes)
    items=mapped.to_dict('records')
    quote_rows, errors = [], []

    def one(item, range_='5d'):
        ticker = item['quote_request_symbol']+'.BO'
        url = 'https://query1.finance.yahoo.com/v8/finance/chart/'+quote(ticker,safe='')+f'?interval=1d&range={range_}&events=div%2Csplits'
        try:
            raw,rec=store.get(url)
            info,bars,bad=parse_yahoo_bse(raw,ticker,rec['first_retrieved_utc'],asof,item.get('company_name') or '')
            info.update({'isin':item['isin'],'official_current_reference_present':bool(item['official_current_reference_present']),
                'bse_exclusive_official_interop_reference':bool(item.get('bse_exclusive_official_interop_reference') is True),
                'amfi_size_category':item.get('amfi_size_category'),
                'bse_quote_verified_current_identity_and_name':bool(item['official_current_reference_present'] and info['official_reference_name_matches']),
                'source_url':url})
            return info,bars,bad,None
        except Exception as e:
            return None,[],[],{'isin':item['isin'],'ticker':ticker,'stage':'BSE_'+range_,'error':type(e).__name__+': '+str(e)[:160]}

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futures=[ex.submit(one,item) for item in items]
        for n,future in enumerate(concurrent.futures.as_completed(futures),1):
            info,bars,bad,error=future.result()
            if info:quote_rows.append(info)
            if error:errors.append(error)
            if n % 100==0 or n==len(items):
                print(json.dumps({'BSE_quote_progress':n,'planned':len(items),'provider_equity_quotes':len(quote_rows),'errors':len(errors)}),flush=True)
    pd.DataFrame(quote_rows).to_parquet(out/'BSE_latest_quotes_source_audit_PRIVATE.parquet',index=False)
    # History controls are chosen by the predeclared hash order, without future
    # outcomes or the predictor's probability ordering. Missing rows stay visible.
    eligible={r['isin'] for r in quote_rows if r['bse_quote_verified_current_identity_and_name'] and r['same_asof_trade_day']}
    selected=[x for x in items if x['isin'] in eligible][:history_companies]
    history, history_info, bar_errors=[],[],[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futures=[ex.submit(one,item,'10y') for item in selected]
        for future in concurrent.futures.as_completed(futures):
            info,bars,bad,error=future.result()
            if info:
                history_info.append(info)
                history.extend([{**b,'isin':info['isin']} for b in bars])
                bar_errors.extend([{**b,'isin':info['isin']} for b in bad])
            if error:errors.append(error)
    pd.DataFrame(history).to_parquet(out/'BSE_10year_history_source_PRIVATE.parquet',index=False)
    (out/'BSE_history_security_audit_PRIVATE.json').write_text(json.dumps(history_info,indent=2))
    (out/'BSE_source_errors_PRIVATE.json').write_text(json.dumps(errors,indent=2))
    (out/'BSE_bar_errors_PRIVATE.json').write_text(json.dumps(bar_errors,indent=2))
    summary={'scope':'PUBLIC_BSE_PRICE_HISTORY_FALLBACK_AND_OFFICIAL_IDENTITY_RECOVERY',
        'asof':asof,'official_BSE_exclusive_security_ISINs':len(exclusive),
        'official_interop_master_raw_rows':master.attrs['unfiltered_official_master_rows'],
        'official_interop_usable_equity_instrument_rows':len(master),
        'current_official_NSE_and_BSE_exclusive_reference_ISINs':int(universe.official_current_reference_present.sum()),
        'BSE_full_reference_tickers_available':int(universe.BSE_full_symbol_mapping_available.sum()),
        'BSE_ambiguous_duplicate_tickers_excluded':int(ambiguous.sum()),
        'quotes_requested':len(items),'provider_BSE_equity_quotes':len(quote_rows),
        'same_asof_trade_day_quotes':sum(r['same_asof_trade_day'] for r in quote_rows),
        'current_identity_and_official_name_match_quotes':sum(r['bse_quote_verified_current_identity_and_name'] for r in quote_rows),
        'fresh_current_identity_name_verified_quotes':sum(r['bse_quote_verified_current_identity_and_name'] and r['same_asof_trade_day'] for r in quote_rows),
        'BSE_exclusive_provider_quotes':sum(r['bse_exclusive_official_interop_reference'] for r in quote_rows),
        'BSE_exclusive_fresh_identity_name_verified_quotes':sum(r['bse_exclusive_official_interop_reference'] and r['same_asof_trade_day'] and r['bse_quote_verified_current_identity_and_name'] for r in quote_rows),
        'ten_year_history_companies_requested':len(selected),'ten_year_history_companies_available':len(history_info),
        'usable_daily_history_bars':len(history),'invalid_or_incomplete_history_bars':len(bar_errors),
        'source_errors':len(errors),
        'guaranteed_live_market_latency_proven':False,
        'corporate_action_adjustments_independently_verified':False,
        'all_current_BSE_active_quotes_histories_and_financials_complete':False,
        'NSE_model_automatically_applied_to_BSE':False,
        'original_model_scores_and_evaluation_freeze_unchanged':True,'production_approved':False}
    (out/'free_BSE_recovery_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)
    return summary


def run_bse_financials(bse_recovery,output):
    """Public, hash-selected history controls; no private ranks or outcomes."""
    bse=Path(bse_recovery);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    controls=json.loads((bse/'BSE_history_security_audit_PRIVATE.json').read_text())
    reference=pd.read_parquet(bse/'NSE_BSE_current_and_dated_reference_PRIVATE.parquet').set_index('isin')
    store=PublicEvidenceStore(out/'raw_PRIVATE');results,errors=[],[]
    for item in controls:
        try:
            if not item['bse_quote_verified_current_identity_and_name'] or not item['same_asof_trade_day']:
                raise ValueError('Public BSE control has unresolved identity or stale quote')
            official=reference.loc[item['isin']]
            symbol=official['quote_request_symbol']
            if item['provider_symbol']!=symbol+'.BO':raise ValueError('Public BSE mnemonic identity changed')
            url='https://ticker.finology.in/company/'+quote(symbol,safe='')+'?mode=C'
            raw,receipt=store.get(url)
            data=parse_finology(raw,receipt['final_url'],symbol,official['company_name'],receipt['first_retrieved_utc'])
            data.update({'isin':item['isin'],'public_BSE_history_control':True,
                         'control_selection':'PREDECLARED_PUBLIC_HASH_ORDER_NO_PRIVATE_MODEL_RANKS'})
            results.append(data)
        except Exception as e:
            errors.append({'isin':item['isin'],'error':type(e).__name__+': '+str(e)[:180]})
        if (len(results)+len(errors))%10==0:
            print(json.dumps({'BSE_public_financial_controls_processed':len(results)+len(errors),'planned':len(controls)}),flush=True)
    (out/'BSE_public_financial_controls_PRIVATE.json').write_text(json.dumps(results,indent=2))
    (out/'BSE_public_financial_errors_PRIVATE.json').write_text(json.dumps(errors,indent=2))
    summary={'scope':'PUBLIC_BSE_HASH_SELECTED_SOURCE_CONTROLS_NOT_MODEL_PREDICTIONS',
        'requested':len(controls),'usable_financial_pages':len(results),'errors':len(errors),
        'financial_fact_observations':sum(len(x['facts']) for x in results),
        'cashflow_companies':sum(any(f['metric']=='cfo' for f in x['facts']) for x in results),
        'full_BSE_financial_coverage_complete':False,'private_model_ranks_used_to_select_provider_requests':False,
        'production_approved':False}
    (out/'BSE_public_financial_controls_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)
    return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-recovery',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--mode',choices=['financial','bse','bse-financial'],required=True)
    p.add_argument('--asof',default='2026-10-09');p.add_argument('--limit',type=int,default=0)
    p.add_argument('--history-companies',type=int,default=64)
    a=p.parse_args()
    if a.mode=='financial':
        run_financials(a.source_recovery,a.output,a.limit)
    elif a.mode=='bse':run_bse(a.source_recovery,a.output,a.asof,a.limit,a.history_companies)
    else:run_bse_financials(a.source_recovery,a.output)


if __name__=='__main__':main()
