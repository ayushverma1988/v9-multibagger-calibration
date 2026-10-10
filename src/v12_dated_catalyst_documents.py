"""Read a fixed public hash sample of dated primary catalyst attachments.

Disclosed quantities are candidates; margin, conditions and execution still
require verification before treating them as incremental earnings catalysts.
"""
from __future__ import annotations
import argparse,json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from v12_dated_training_inputs import dated_events
from v12_dated_source_collection import Store,sha
from v11_4_primary_document_catalysts import extract_pdf,classify_document

def collect(catalogs,output,asof='2026-10-09'):
    out=Path(output);out.mkdir(parents=True,exist_ok=True);store=Store(out/'raw')
    x=dated_events(json.loads((Path(catalogs)/'announcement_catalog.json').read_text()),asof)
    x=x[(x.order_disclosure|x.capacity_disclosure)&x.attachment_url.str.match(r'https://nsearchives\.nseindia\.com/.*\.pdf$',case=False)].copy()
    x['year']=x.available_at_utc.str[:4];x['hash']=x.attachment_url.map(lambda s:sha(s.encode()));x=x.sort_values('hash').groupby('year').head(16)
    x.to_parquet(out/'public_hash_dated_catalyst_request_universe.parquet',index=False)
    rows=[];errors=[]
    def read(r):
        try:
            raw,proof=store.get(r['attachment_url']);text=extract_pdf(raw);info=classify_document(text,r['symbol'],r['issuer_name'])
            return {**r,**info,'source_sha256':proof['sha256'],'first_retrieved_utc':proof['first_retrieved_utc'],'text_characters':len(text)},None
        except Exception as e:return None,{'symbol':r['symbol'],'attachment_url':r['attachment_url'],'available_at_utc':r['available_at_utc'],'error':str(e)[:200]}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for info,error in pool.map(read,x.to_dict('records')):
            if info:rows.append(info)
            if error:errors.append(error)
    (out/'dated_primary_catalyst_document_candidates.json').write_text(json.dumps(rows,indent=2));(out/'errors.json').write_text(json.dumps(errors,indent=2))
    summary={'requested':len(x),'documents_read':len(rows),'issuer_matched':sum(r['issuer_identity_read'] for r in rows),'quantified_candidates':sum(len(r['quantified_evidence']) for r in rows),'complete_verified_earnings_causal_chains':0,'errors':len(errors),'sample_only_not_full_universe':True,'publication_clocks_retained':True,'production_approved':False}
    (out/'summary.json').write_text(json.dumps(summary,indent=2));return summary

def main():
    p=argparse.ArgumentParser();p.add_argument('--catalogs',required=True);p.add_argument('--output',required=True);p.add_argument('--asof',default='2026-10-09');a=p.parse_args();print(json.dumps(collect(a.catalogs,a.output,a.asof),indent=2))
if __name__=='__main__':main()
