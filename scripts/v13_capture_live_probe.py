"""Bounded original-source probe in an isolated copy; never changes forecasts."""
import argparse,json
from pathlib import Path
import pandas as pd
from v13_capture_policy import atomic_json


def run(root,asof,filings=200,documents=20):
    from v13_collection import current_catalogs,collect_financials
    from v13_universe_financial_audit import audit
    from v13_catalyst_field_capture import capture_priority,export
    from v11_4_primary_document_catalysts import extract_pdf
    from v12_dated_source_collection import Store,sha
    from v13_parallel_collection import bounded_map
    root=Path(root);master=pd.read_parquet(root/'master_data/master_inputs.parquet')
    u=master[['symbol','isin','company_name']].rename(columns={'symbol':'nse_current_symbol','company_name':'nse_current_name'})
    u.to_parquet(root/'universe.parquet',index=False)
    old=pd.read_parquet(root/'financial_wide.parquet')
    before=audit(u,old,asof,root)
    catalogs=current_catalogs(root,asof)
    collected=collect_financials(root,asof,filings)
    after=audit(u,pd.read_parquet(root/'financial_wide.parquet'),asof,root)
    folder=root/'catalyst_evidence';docs=json.loads((folder/'document_evidence.json').read_text())
    ordered=sorted([r for r in docs if r.get('issuer_identity_read') and r.get('isin')],
        key=lambda r:(capture_priority(r),-pd.Timestamp(r['available_at_utc']).value))
    unique={}
    for r in ordered:unique.setdefault((r['isin'],r['source_sha256']),r)
    store=Store(folder/'raw');errors=[];accepted=[]
    def read(r):
        try:
            raw,proof=store.get(r['attachment_url'])
            if proof['sha256']!=r['source_sha256']:raise ValueError('Original changed since baseline; review required')
            text=extract_pdf(raw)
            if sha(text.encode())!=r['text_sha256']:raise ValueError('Original extraction differs; review required')
            (folder/(r['source_sha256']+'.txt')).write_text(text)
            return {'proof':proof,'symbol':r['symbol']}
        except Exception as exc:return {'error':str(exc)[:200],'symbol':r['symbol']}
    for _,result in bounded_map(list(unique.values())[:documents],read,workers=4,seconds=180):
        (errors if result.get('error') else accepted).append(result)
    fields=export(root,asof)
    result={'scope':'ISOLATED_CURRENT_PUBLIC_UNIVERSE_SOURCE_CAPTURE_PROBE',
        'source_asof':asof,'catalogs':catalogs,'financials':collected,
        'financial_before':before,'financial_after':after,
        'original_documents_hash_reverified':accepted,'original_document_errors':errors,
        'catalyst_field_capture':fields,'registered_forecasts_or_probabilities_changed':False}
    atomic_json(root/'capture_live_probe.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--state',required=True);p.add_argument('--asof',required=True)
    p.add_argument('--filings',type=int,default=200);p.add_argument('--documents',type=int,default=20)
    a=p.parse_args();run(a.state,a.asof,a.filings,a.documents)
