"""Current-only input recovery. Never rewrites the independent prediction freeze.

The exchange issued-capital field is treated as a share count only after its
unit is corroborated for that security. A provider's NSE quote cannot verify a
BSE market capitalization. Reported multi-year ratios stay unverified until
their original observations can be reproduced.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import re
from pathlib import Path

import pandas as pd

from v11_4_free_source_fallbacks import finite_number
from v11_4_direct_exchange_pe_valuation import normalize_company_pe


def verify_receipt_date(raw, receipt, asof, date_format):
    url=receipt.get('requested_url',receipt.get('url',''))
    expected=pd.Timestamp(asof).strftime(date_format)
    if receipt.get('status') != 200 or receipt.get('sha256') != hashlib.sha256(raw).hexdigest():
        raise ValueError('Source receipt does not identify the original successful bytes')
    if not re.search(r'(?:_|/)' + re.escape(expected) + r'\.(?:csv|csv.gz)$', url):
        raise ValueError('Source receipt trading date differs from requested current audit date')


def verify_current_market_cap(company, official_market, master, asof, bse_quotes=None):
    q = company.get('quote') or {}
    result = {'status': 'UNKNOWN', 'market_cap_verified_crore': None,
              'unit_independently_corroborated': False}
    symbol, isin = company['identity']['symbol'], company['isin']
    if q.get('venue') not in ['NSE','BSE'] or q.get('displayed_quote_day_IST') != str(pd.Timestamp(asof).date()):
        return {**result, 'reason': 'MISSING_OR_DIFFERENT_VENUE_OR_QUOTE_DATE'}
    m = official_market[official_market['symbol'].eq(symbol) & official_market['isin'].eq(isin)
                        & pd.to_datetime(official_market['date']).eq(pd.Timestamp(asof))]
    if len(m) != 1:
        return {**result, 'reason': 'NO_UNIQUE_SAME_DATE_OFFICIAL_PRICE_IDENTITY'}
    candidates = master[master.TckrSymb.eq(symbol) & master.ISIN.eq(isin)
                        & master.SctySrs.isin(['EQ', 'BE', 'BZ', 'SM', 'ST'])
                        & master.DelFlg.eq('N') & master.PrtdToTrad.eq(0)]
    counts = {finite_number(x) for x in candidates.IssdCptl}
    if len(counts) != 1 or None in counts or next(iter(counts)) <= 0:
        return {**result, 'reason': 'MISSING_OR_CONFLICTING_OFFICIAL_ISSUED_CAPITAL'}
    count = next(iter(counts))
    close, price, provider_count, provider_cap = [finite_number(v) for v in
        [m.iloc[0]['close'], q.get('latest_price_INR'), q.get('reported_shares_crore'), q.get('reported_market_cap_crore')]]
    if any(v is None or v <= 0 for v in [close, price, provider_count, provider_cap]):
        return {**result, 'reason': 'MISSING_POSITIVE_VALUATION_OPERANDS'}
    if not q.get('cap_shares_quote_arithmetic_agrees'):
        return {**result, 'reason': 'PROVIDER_CAP_SHARE_PRICE_ARITHMETIC_DISAGREES'}
    price_reference=close
    if q.get('venue')=='BSE':
        if bse_quotes is None:
            return {**result,'reason':'MISSING_INDEPENDENT_BSE_PRICE_FOR_PROVIDER_CAP_BASIS'}
        b=bse_quotes[bse_quotes['isin'].eq(isin)
            & bse_quotes.same_asof_trade_day.eq(True)
            & bse_quotes.bse_quote_verified_current_identity_and_name.eq(True)]
        if len(b)!=1:
            return {**result,'reason':'NO_UNIQUE_FRESH_REFERENCE_MATCHED_BSE_PROVIDER_QUOTE'}
        price_reference=finite_number(b.iloc[0]['latest_price_INR'])
        if price_reference is None or price_reference<=0:
            return {**result,'reason':'MISSING_POSITIVE_INDEPENDENT_BSE_PRICE'}
    if abs(price_reference-price) > .011:
        return {**result, 'reason': 'PROVIDER_QUOTE_DIFFERS_FROM_INDEPENDENT_SAME_VENUE_PRICE'}
    if abs(count/1e7-provider_count) > .0051:
        return {**result, 'reason': 'OFFICIAL_CAPITAL_NOT_CORROBORATED_AS_CURRENT_SHARE_COUNT'}
    cap = count*close/1e7
    provider_venue_cap=count*price_reference/1e7
    if abs(provider_venue_cap-provider_cap) > .0051*price_reference+.015:
        return {**result, 'reason': 'OFFICIAL_PRICE_TIMES_SHARES_DIFFERS_FROM_PROVIDER_CAP'}
    return {**result, 'status': 'CURRENT_THREE_WAY_CORROBORATED',
            'reason': 'EXACT_NSE_PRICE_ISIN_SHARE_UNIT_AND_PROVIDER_SAME_VENUE_CAP_RECONCILE',
            'market_cap_verified_crore': cap, 'unit_independently_corroborated': True,
            'official_close_INR': close, 'official_corroborated_shares': count,
            'provider_displayed_market_cap_crore': provider_cap,
            'market_cap_verified_venue':'NSE',
            'provider_cap_corroboration_venue':q['venue'],
            'provider_same_venue_cap_corroborated_crore':provider_venue_cap,
            'provider_display_rounding_tolerance_crore': .0051*price_reference+.015}


def current_financial_metrics(company, asof):
    facts = company['facts']
    def value(metric, period, kind):
        values = {f['value_INR'] for f in facts if f['metric']==metric
                  and f['period_end']==period and f['period_kind']==kind}
        return next(iter(values)) if len(values)==1 else None
    annual = sorted({f['period_end'] for f in facts if f['period_kind']=='annual'
                     and f['metric']=='revenue'})
    quarter = sorted({f['period_end'] for f in facts if f['period_kind']=='quarter'
                      and f['metric']=='revenue'})
    result = {'latest_annual_revenue_period': annual[-1] if annual else None,
              'latest_quarter_revenue_period': quarter[-1] if quarter else None,
              'annual_period_fresh_within_550_days': bool(annual and
                 0 <= (pd.Timestamp(asof)-pd.Timestamp(annual[-1])).days <= 550),
              'quarter_period_fresh_within_190_days':bool(quarter and
                 0 <= (pd.Timestamp(asof)-pd.Timestamp(quarter[-1])).days <= 190),
              'latest_annual_cfo_pat_ratio': None, 'debt_equity': None,
              'opm_annual': None, 'opm_5_observation_mean': None,
              'sales_growth_5y_reproduced': None, 'profit_growth_5y_reproduced': None,
              'sales_growth_7y_reproduced': None, 'profit_growth_7y_reproduced': None,
              'quarterly_sales_yoy_growth': None, 'quarterly_profit_yoy_growth': None,
              'negative_or_missing_equity': None, 'exceptional_items_in_latest_quarter': None,
              'four_quarter_PBT':None,'four_quarter_reported_exceptional_items':None,
              'PBT_components_reconcile_all_four_quarters':False,
              'four_quarter_PBT_excluding_reported_exceptional_items':None,
              'full_original_four_screener_inputs_completed': False,
              'historical_PIT_training_eligible': False}
    if annual:
        latest = annual[-1]
        cfo, pat = value('cfo',latest,'annual'), value('pat',latest,'annual')
        if cfo is not None and pat is not None and pat > 0:
            result['latest_annual_cfo_pat_ratio'] = cfo/pat
        equity_parts = [value('share_capital',latest,'instant'),value('reserves',latest,'instant')]
        equity = sum(equity_parts) if all(x is not None for x in equity_parts) else None
        debt = value('borrowings_provider',latest,'instant')
        result['negative_or_missing_equity'] = equity is None or equity <= 0
        if equity is not None and equity > 0 and debt is not None and debt >= 0:
            result['debt_equity'] = debt/equity
        opms=[]
        for period in annual[-5:]:
            sales, profit=value('revenue',period,'annual'),value('operating_profit_provider',period,'annual')
            if sales is not None and sales > 0 and profit is not None:
                opms.append((period,profit/sales))
        if opms and opms[-1][0]==latest:result['opm_annual']=opms[-1][1]
        expected={str((pd.Timestamp(latest)-pd.DateOffset(years=y)).date()) for y in range(5)}
        if len(opms)==5 and {x[0] for x in opms}==expected:
            result['opm_5_observation_mean']=sum(x[1] for x in opms)/5
        for years in [5,7]:
            past = str((pd.Timestamp(latest)-pd.DateOffset(years=years)).date())
            for label,metric in [('sales','revenue'),('profit','pat')]:
                a,b=value(metric,past,'annual'),value(metric,latest,'annual')
                if a is not None and b is not None and a > 0 and b > 0:
                    result[f'{label}_growth_{years}y_reproduced']=(b/a)**(1/years)-1
    if quarter:
        latest=quarter[-1];past=str((pd.Timestamp(latest)-pd.DateOffset(years=1)).date())
        for label,metric in [('sales','revenue'),('profit','pat')]:
            a,b=value(metric,past,'quarter'),value(metric,latest,'quarter')
            if a is not None and b is not None and a > 0:
                result[f'quarterly_{label}_yoy_growth']=b/a-1
        result['exceptional_items_in_latest_quarter']=value('exceptional_items_provider',latest,'quarter')
        ends=[str((pd.Timestamp(latest)-pd.DateOffset(months=3*i)+pd.offsets.MonthEnd(0)).date()) for i in range(4)]
        totals=[]
        for end in ends:
            metrics=['pbt','operating_profit_provider','depreciation','finance_cost',
                     'other_income_provider','exceptional_items_provider']
            amounts=[value(metric,end,'quarter') for metric in metrics]
            if any(x is None for x in amounts):break
            pbt,op,dep,interest,other,exceptional=amounts
            tolerance=sum(f.get('rendered_display_rounding_INR',0) for f in facts
                if f['metric'] in metrics and f['period_kind']=='quarter' and f['period_end']==end)+.01
            if abs(pbt-(op-dep-interest+other+exceptional))>tolerance:break
            totals.append((pbt,exceptional))
        if len(totals)==4:
            pbt=sum(t[0] for t in totals);exceptional=sum(t[1] for t in totals)
            result.update({'four_quarter_PBT':pbt,'four_quarter_reported_exceptional_items':exceptional,
                'PBT_components_reconcile_all_four_quarters':True,
                'four_quarter_PBT_excluding_reported_exceptional_items':pbt-exceptional})
    # Preserve provider ratios separately; agreement of other facts does not
    # certify their bases, endpoints or averaging convention.
    result.update({'provider_reported_'+k:v for k,v in company['reported_ratios'].items()})
    return result


def run(financial_folder, source_recovery, master_file, pe_file, output, asof,
        master_receipt, pe_receipt, bse_quotes=None):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    master_raw=Path(master_file).read_bytes()
    master_proof=json.loads(Path(master_receipt).read_text())
    verify_receipt_date(master_raw,master_proof,asof,'%d%m%Y')
    master=pd.read_csv(io.BytesIO(gzip.decompress(master_raw)),low_memory=False)
    market=pd.read_parquet(Path(source_recovery)/'same_day_official_NSE_market_identity_PRIVATE.parquet')
    pe=pd.read_parquet(pe_file).set_index('symbol')
    pe_proof=json.loads(Path(pe_receipt).read_text())
    pe_raw=(Path(pe_receipt).parent/'raw_PRIVATE'/pe_proof['body_file']).read_bytes()
    verify_receipt_date(pe_raw,pe_proof,asof,'%d%m%y')
    original_pe=normalize_company_pe(pd.read_csv(io.BytesIO(pe_raw))).set_index('symbol')
    pd.testing.assert_frame_equal(pe.sort_index(),original_pe.sort_index())
    if pe.index.duplicated().any():raise ValueError('Company PE symbol is not unique')
    bse=pd.read_parquet(bse_quotes) if bse_quotes else None
    rows=[]
    for path in sorted((Path(financial_folder)/'financial_companies_PRIVATE').glob('*.json')):
        company=json.loads(path.read_text());symbol=company['identity']['symbol']
        row={'symbol':symbol,'isin':company['isin'],**company['identity'],
             **verify_current_market_cap(company,market,master,asof,bse),
             **current_financial_metrics(company,asof),
             'official_company_PE':None,'official_adjusted_company_PE':None,
             'official_PE_provider_relative_difference':None}
        if symbol in pe.index:
            for source,target in [('company_pe','official_company_PE'),
                                  ('company_pe_adjusted_exchange','official_adjusted_company_PE')]:
                x=finite_number(pe.loc[symbol,source])
                if x is not None and x > 0:row[target]=x
        q=company.get('quote') or {};vendor=finite_number(q.get('reported_PE'))
        if row['official_company_PE'] and vendor and vendor > 0:
            row['official_PE_provider_relative_difference']=vendor/row['official_company_PE']-1
        row['provider_market_cap_quote_venue']=q.get('venue')
        row['original_fact_agreements']=sum(a['status']=='AGREES_WITH_ORIGINAL' for a in company['reconciliation'])
        row['original_fact_disagreements']=sum(a['status']=='DISAGREES_WITH_ORIGINAL' for a in company['reconciliation'])
        rows.append(row)
    frame=pd.DataFrame(rows)
    frame.to_parquet(out/'current_free_financial_input_audit_PRIVATE.parquet',index=False)
    fields=['market_cap_verified_crore','official_company_PE','latest_annual_cfo_pat_ratio',
            'debt_equity','opm_annual','opm_5_observation_mean','quarterly_sales_yoy_growth',
            'quarterly_profit_yoy_growth','sales_growth_5y_reproduced','profit_growth_5y_reproduced',
            'sales_growth_7y_reproduced','profit_growth_7y_reproduced']
    summary={'scope':'CURRENT_INPUT_RECOVERY_SOURCE_TIERS_NOT_RETRAINING',
        'companies':len(frame),'current_valuation_statuses':frame.status.value_counts().to_dict(),
        'valuation_unknown_reasons':frame[frame.status.eq('UNKNOWN')].reason.value_counts().to_dict(),
        'available_metrics':{field:int(frame[field].notna().sum()) for field in fields},
        'fresh_annual_available_metrics':{field:int(frame.loc[frame.annual_period_fresh_within_550_days,field].notna().sum())
            for field in fields if field not in ['market_cap_verified_crore','official_company_PE',
                                                 'quarterly_sales_yoy_growth','quarterly_profit_yoy_growth']},
        'fresh_quarterly_sales_yoy_companies':int(frame.loc[frame.quarter_period_fresh_within_190_days,'quarterly_sales_yoy_growth'].notna().sum()),
        'four_quarter_PBT_components_reconcile':int(frame.PBT_components_reconcile_all_four_quarters.sum()),
        'latest_annual_period_fresh_within_550_days':int(frame.annual_period_fresh_within_550_days.sum()),
        'official_master_sha256':hashlib.sha256(master_raw).hexdigest(),
        'official_master_url':master_proof.get('requested_url',master_proof.get('url')),
        'official_PE_original_bytes_sha256':hashlib.sha256(pe_raw).hexdigest(),
        'official_PE_url':pe_proof['requested_url'],
        'official_PE_file_sha256':hashlib.sha256(Path(pe_file).read_bytes()).hexdigest(),
        'seven_year_conditions_not_replaced_with_three_year_ratios':True,
        'provider_reported_ratios_not_automatically_primary_verified':True,
        'original_rules_model_and_scores_unchanged':True,
        'full_source_qualified_model':False,'production_approved':False}
    (out/'current_free_financial_audit_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)
    return summary


def main():
    p=argparse.ArgumentParser()
    for name in ['financial-folder','source-recovery','master-file','pe-file','master-receipt','pe-receipt','output']:p.add_argument('--'+name,required=True)
    p.add_argument('--bse-quotes')
    p.add_argument('--asof',default='2026-10-09');a=p.parse_args();run(**vars(a))


if __name__=='__main__':main()
