"""Whole-market valuation and public-cohort catalyst source collection.

Sector/PE downloaded today are current companions, never historical training
inputs. Announcement extraction is not verification of incremental earnings.
"""
from __future__ import annotations
import argparse,io,json
from datetime import datetime,timezone
from pathlib import Path
import pandas as pd
import requests
from v12_dated_source_collection import sha
from v12_dated_training_inputs import dated_events
from v12_3_catalyst_evidence import collect
from v11_4_direct_exchange_pe_valuation import fetch_original_company_pe
from v11_4_nse_valuation_peer_size_proxies import peer_reference

def valuation(output,market_path,asof='2026-10-09'):
    out=Path(output);out.mkdir(parents=True,exist_ok=True);proof=[];errors=[]
    try:
        raw,url,pe,attempts=fetch_original_company_pe(asof)
        (out/'official_PE.csv').write_bytes(raw)
        proof.append({'url':url,'sha256':sha(raw),'retrieved_at_utc':datetime.now(timezone.utc).isoformat(),'scope':'TRADE_DATE_REPORT_NOT_PROVEN_AVAILABLE_AT_1530'})
        url='https://nsearchives.nseindia.com/content/indices/ind_niftytotalmarket_list.csv'
        r=requests.get(url,headers={'User-Agent':'Mozilla/5.0'},timeout=40);r.raise_for_status()
        (out/'current_sector_members.csv').write_bytes(r.content)
        proof.append({'url':url,'sha256':sha(r.content),'retrieved_at_utc':datetime.now(timezone.utc).isoformat(),'scope':'CURRENT_MEMBERSHIP_NOT_HISTORICAL'})
        mem=pd.read_csv(io.BytesIO(r.content)).rename(columns={'Symbol':'symbol','ISIN Code':'isin','Industry':'industry'})
        market=pd.read_parquet(market_path).sort_values('series').drop_duplicates(['symbol','isin'])
        market=market[['symbol','isin','close']].merge(pe,on='symbol',validate='m:1')
        v=peer_reference(market,mem,min_peers=5)
        v['sector_relative_PE']=v.company_pe/v.industry_peer_median_PE_proxy
        # Algebraic scenario, not predicted earnings: EPS factor * PE factor.
        v['EPS_growth_required_for_2x_at_unchanged_PE']=1.
        v['EPS_growth_required_for_2x_at_peer_PE']=2*v.sector_relative_PE-1
        v['historical_training_eligible']=False;v['forecast_probability']=False
        v.to_parquet(out/'current_valuation_scenarios.parquet',index=False)
        summary={'same_date_company_PE_rows':len(v),'positive_PE_rows':int(v.company_pe.notna().sum()),'sector_peer_rows':int(v.sector_relative_PE.notna().sum()),'historical_training_eligible':False,'scenario_is_forecast':False}
    except Exception as e:errors.append(str(e));summary={'historical_training_eligible':False,'error':str(e)}
    (out/'proof.json').write_text(json.dumps(proof,indent=2));(out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)

def catalysts(catalog_path,financial_cohort,output):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    # Public source-defined cohort; no model ranking or outcomes determine requests.
    cohort=pd.read_parquet(financial_cohort)
    symbols=set(cohort.symbol)
    raw=json.loads(Path(catalog_path).read_text())
    raw=[r for r in raw if str(r['record'].get('symbol','')).upper() in symbols]
    events=dated_events(raw,'2026-10-09')
    events=events[events.symbol.isin(cohort.symbol)]
    events.to_parquet(out/'cohort_dated_events.parquet',index=False)
    collect(out/'cohort_dated_events.parquet',out/'documents',per_year=64)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['valuation','catalysts'],required=True)
    p.add_argument('--output',required=True);p.add_argument('--market');p.add_argument('--catalog');p.add_argument('--cohort');a=p.parse_args()
    if a.kind=='valuation':valuation(a.output,a.market)
    else:catalysts(a.catalog,a.cohort,a.output)
