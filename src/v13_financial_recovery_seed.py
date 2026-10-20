"""Replay verified original source bytes; never seed desired stock rankings."""
import base64,gzip,hashlib,json
from pathlib import Path
import pandas as pd
from v13_symbol_identity import parse_mapped
from v13_collection import financial_wide
from v13_financial_snapshot_union import union
from v12_dated_source_collection import sha


def apply(root,path,asof):
    root=Path(root);path=Path(path)
    if not path.exists():return {'status':'NO_RECOVERY_SOURCE_PACK'}
    digest=hashlib.sha256(path.read_bytes()).hexdigest();marker=root/'financial_primary_seed_recovery.json'
    pack=json.loads(path.read_text());master=pd.read_parquet(root/'universe.parquet');isins=dict(zip(master.nse_current_symbol,master['isin']))
    reference=hashlib.sha256(json.dumps(isins,sort_keys=True).encode()).hexdigest()
    if marker.exists():
        prior=json.loads(marker.read_text())
        if prior.get('pack_sha256')==digest and prior.get('master_identity_sha256')==reference:return prior
    identities=root/'identity/symbol_changes.json';changes=json.loads(identities.read_text()) if identities.exists() else []
    facts=[];errors=[];versions=[]
    for doc in pack['documents']:
        r=doc['record'];proof=doc['receipt']
        try:
            if r['symbol'] not in isins or pd.Timestamp(r['period_end'])>pd.Timestamp(asof):continue
            raw=gzip.decompress(base64.b64decode(doc['raw_gzip_base64']))
            if hashlib.sha256(raw).hexdigest()!=proof['sha256'] or proof['requested_url']!=r['source_url'] or proof['status']!=200:raise ValueError('Recovery original proof mismatch')
            fs=parse_mapped(raw,r,asof,isins[r['symbol']],changes)
            if not fs:continue
            out=root/'filing_raw'/sha(r['available_at_utc'].encode());out.mkdir(parents=True,exist_ok=True)
            p=out/(sha(r['source_url'].encode())+'.bin')
            if p.exists() and p.read_bytes()!=raw:raise ValueError('Existing original differs; revision review required')
            p.write_bytes(raw);p.with_suffix('.json').write_text(json.dumps(proof,indent=2))
            facts.extend({**r,**f,'source_sha256':proof['sha256'],'first_retrieved_utc':proof['first_retrieved_utc']} for f in fs)
            versions.append(r['source_url']+'|'+r['available_at_utc'])
        except Exception as exc:errors.append({'symbol':r['symbol'],'error':str(exc)[:160]})
    conflicts=[]
    if facts:
        current=root/'financial_wide.parquet';fresh=financial_wide(pd.DataFrame(facts))
        if current.exists():combined,conflicts=union(pd.read_parquet(current),fresh)
        else:combined=fresh
        combined.to_parquet(root/'financial_wide.parquet',index=False)
        for name in ['accepted_filings.json','financial_parser_owner_v2.json']:
            p=root/name;prior=json.loads(p.read_text()) if p.exists() else []
            p.write_text(json.dumps(sorted(set(prior)|set(versions))))
    result={'pack_sha256':digest,'master_identity_sha256':reference,'original_documents_replayed':len(versions),'extracted_fact_observations':len(facts),
      'same_original_numeric_conflicts':conflicts,'errors':errors,'frozen_forecasts_or_rankings_changed':False}
    marker.write_text(json.dumps(result,indent=2));return result
