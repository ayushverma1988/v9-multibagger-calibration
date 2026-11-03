"""Public BSE directory and standardized financial fallback on public controls.

Provider directory codes are candidate identity mappings, not an official
active-listing master. Private model cohorts never choose the requests.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import json
import re
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

from v11_4_free_source_fallbacks import PublicEvidenceStore, normalize_name, parse_stockanalysis_financials


def parse_directory(raw):
    s=BeautifulSoup(raw,'html.parser')
    if not any(e.get_text(' ',strip=True)=='Bombay Stock Exchange Stocks' for e in s.select('h1')):
        raise ValueError('Public BSE provider directory unavailable')
    rows=[]
    for table in s.select('table'):
        for tr in table.select('tr'):
            cells=tr.find_all('td',recursive=False)
            if len(cells)<3:continue
            link=cells[1].find('a',href=True)
            code=cells[1].get_text(' ',strip=True)
            if not link or not re.fullmatch(r'\d{6}',code):continue
            if link['href']!='/quote/bom/'+code+'/':
                raise ValueError('Directory displayed code disagrees with linked BSE instrument')
            name=cells[2].get_text(' ',strip=True)
            if name:rows.append({'bse_provider_code':code,'provider_directory_name':name})
    if not rows:raise ValueError('No BSE directory identities parsed')
    if len({r['bse_provider_code'] for r in rows})!=len(rows):raise ValueError('Duplicate directory code')
    return rows


def unique_name_map(directory):
    d={}
    for row in directory:
        key=normalize_name(row['provider_directory_name'])
        d.setdefault(key,{})[row['bse_provider_code']]=row
    return {key:next(iter(rows.values())) for key,rows in d.items() if len(rows)==1}


def run(bse_recovery,output):
    out=Path(output);out.mkdir(parents=True,exist_ok=True);store=PublicEvidenceStore(out/'raw_PRIVATE')
    directory,receipts=[],[]
    for page in range(1,11):
        url='https://stockanalysis.com/list/bse-india/'+(f'?page={page}' if page>1 else '')
        raw,receipt=store.get(url);directory.extend(parse_directory(raw));receipts.append(receipt)
    directory=pd.DataFrame(directory).drop_duplicates(['bse_provider_code','provider_directory_name'])
    if directory.bse_provider_code.duplicated().any():raise ValueError('Directory pages contain conflicting code names')
    directory.to_parquet(out/'public_BSE_provider_code_directory.parquet',index=False)
    (out/'public_BSE_directory_receipts.json').write_text(json.dumps(receipts,indent=2))
    names=unique_name_map(directory.to_dict('records'))
    controls=json.loads((Path(bse_recovery)/'BSE_history_security_audit_PRIVATE.json').read_text())
    mapping,results,errors=[],[],[]
    for item in controls:
        match=names.get(normalize_name(item['provider_long_name']))
        if not match:
            errors.append({'isin':item['isin'],'stage':'BSE_DIRECTORY_IDENTITY','reason':'NO_UNIQUE_EXACT_NORMALIZED_NAME_MATCH'})
            continue
        mapping.append({**match,'isin':item['isin'],'provider_symbol':item['provider_symbol'],
            'official_BSE_code_and_ISIN_binding_verified':False})
    def one(item,kind):
        url='https://stockanalysis.com/quote/bom/'+item['bse_provider_code']+'/financials/'+kind+'/'
        try:
            raw,receipt=store.get(url)
            data=parse_stockanalysis_financials(raw,item['bse_provider_code'],item['provider_directory_name'],receipt['first_retrieved_utc'])
            data.update({'source_url':url,'isin_candidate_reference':item['isin'],'provider_symbol':item['provider_symbol'],
                'source_kind':kind,'official_code_ISIN_binding_verified':False})
            return data,None
        except Exception as e:
            return None,{'isin':item['isin'],'stage':kind,'error':type(e).__name__+': '+str(e)[:180]}
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
        tasks=[executor.submit(one,item,kind) for item in mapping for kind in ['income-statement','cash-flow-statement']]
        for n,future in enumerate(concurrent.futures.as_completed(tasks),1):
            data,error=future.result()
            if data:results.append(data)
            if error:errors.append(error)
            if n%20==0:print(json.dumps({'secondary_BSE_financial_pages_processed':n,'planned':len(tasks)}),flush=True)
    (out/'BSE_secondary_financial_identity_mapping_PRIVATE.json').write_text(json.dumps(mapping,indent=2))
    (out/'BSE_secondary_standardized_financials_PRIVATE.json').write_text(json.dumps(results,indent=2))
    (out/'BSE_secondary_financial_errors_PRIVATE.json').write_text(json.dumps(errors,indent=2))
    summary={'scope':'PUBLIC_BSE_DIRECTORY_AND_STANDARDIZED_FINANCIAL_CONTROL_RECOVERY',
        'provider_directory_codes':len(directory),'public_history_controls':len(controls),
        'unique_name_matched_candidate_codes':len(mapping),'usable_source_pages':len(results),
        'income_statement_companies':len({x['isin_candidate_reference'] for x in results if x['source_kind']=='income-statement'}),
        'cashflow_statement_companies':len({x['isin_candidate_reference'] for x in results if x['source_kind']=='cash-flow-statement'}),
        'source_fact_observations':sum(len(x['facts']) for x in results),'source_errors':len(errors),
        'official_BSE_code_ISIN_binding_or_full_coverage_claimed':False,
        'standardized_PAT_and_OPM_definitions_not_silently_mixed':True,
        'private_model_ranks_used_to_select_provider_requests':False,'production_approved':False}
    (out/'BSE_secondary_financial_recovery_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True);return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--bse-recovery',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(a.bse_recovery,a.output)


if __name__=='__main__':main()
