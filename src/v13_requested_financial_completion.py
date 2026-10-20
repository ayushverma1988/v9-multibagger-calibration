"""Complete missing original periods for the user's explicit earlier watchlist.

This is a labelled source audit, not a way to force forecast selections. The
regular worker's independent public-universe completion queue is unchanged.
"""
import json
from pathlib import Path
import pandas as pd
from v13_collection import financial_wide,merge_financial_versions
from v13_financial_evidence import balancing_quarters,completion_queue,owner_completion_queue
from v13_2x_report import PRIOR_NO_PRICE_BAND
from v13_symbol_identity import collect as collect_identity,parse_mapped
from v12_dated_source_collection import Store,sha
from v13_parallel_collection import bounded_map


def collect(root,asof):
    root=Path(root);wide=pd.read_parquet(root/'financial_wide.parquet');cat=pd.read_parquet(root/'catalog.parquet')
    master=pd.read_parquet(root/'universe.parquet');isins=dict(zip(master.nse_current_symbol,master['isin']))
    q=cat[cat.symbol.isin(PRIOR_NO_PRICE_BAND)&cat.reporting_mode.ne('unknown')].copy()
    q['last_attempt']='0000';q['version_key']=q.source_url+'|'+q.available_at_utc
    expanded=balancing_quarters(wide)
    basic=completion_queue(q,expanded,asof,100)
    owner=owner_completion_queue(q,expanded,asof,70)
    targets=pd.concat([basic,owner],ignore_index=True).drop_duplicates('version_key')
    changes=collect_identity(root);facts=[];errors=[];attempted=0
    hashes={(url,clock):set(g.source_sha256) for (url,clock),g in wide[wide.source_url.notna()].groupby(['source_url','available_at_utc'])}
    def fetch(r):
        try:
            raw,proof=Store(root/'filing_raw'/sha(r['available_at_utc'].encode())).get(r['source_url'])
            expected=hashes.get((r['source_url'],r['available_at_utc']))
            if expected and proof['sha256'] not in expected:raise ValueError('Previously observed original changed; revision review required')
            fs=parse_mapped(raw,r,asof,isins[r['symbol']],changes)
            if not fs:raise ValueError('No usable same-identity original facts')
            return {'facts':[{**r,**f,'source_sha256':proof['sha256'],'first_retrieved_utc':proof['first_retrieved_utc']} for f in fs]}
        except Exception as exc:return {'error':str(exc)[:160]}
    for r,result in bounded_map(targets.to_dict('records'),fetch,workers=4,seconds=300):
        attempted+=1
        if result.get('facts'):facts.extend(result['facts'])
        else:errors.append({'symbol':r['symbol'],'source_url':r['source_url'],'error':result['error']})
    if facts:
        merged=merge_financial_versions(wide,financial_wide(pd.DataFrame(facts)))
        merged.to_parquet(root/'financial_wide.parquet',index=False)
    return {'requested':attempted,'new_facts':len(facts),'errors':errors,
      'scope':'USER_REQUESTED_EARLIER_ELEVEN_NAMES_MISSING_ORIGINAL_PERIODS',
      'basic_comparison_filings':len(basic),'owner_comparison_filings':len(owner),
      'names_forced_into_selection':False}
