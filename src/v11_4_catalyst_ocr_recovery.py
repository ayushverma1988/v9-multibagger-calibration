"""Bounded local OCR of source-preserved original exchange PDFs.

OCR enables evidence review; it does not certify quantities, market purchases,
materiality or earnings impact. Confidence and page boundaries are retained.
"""
from __future__ import annotations

import argparse
import io
import json
import subprocess
from pathlib import Path

import pandas as pd
from pypdf import PdfReader

from v11_4_source_blocker_recovery import EvidenceStore
from v11_4_primary_document_catalysts import classify_document


def recover_pdf(raw, folder, max_pages=12):
    if not raw.startswith(b'%PDF'):
        raise ValueError('Original attachment is not a PDF')
    root=Path(folder);root.mkdir(parents=True,exist_ok=True)
    path=root/'original.pdf';path.write_bytes(raw)
    doc=PdfReader(io.BytesIO(raw));total=len(doc.pages)
    selected=list(range(min(max_pages,total)))
    text_rows=[]
    for index in selected:
        text=doc.pages[index].extract_text() or ''
        row={'page':index+1,'method':'ORIGINAL_PDF_TEXT','text':text,'OCR_mean_word_confidence':None}
        if len(text.strip()) < 80:
            prefix=root/f'page_{index+1:03d}'
            subprocess.run(['pdftoppm','-f',str(index+1),'-l',str(index+1),'-singlefile','-r','180','-png',str(path),str(prefix)],
                           check=True,capture_output=True,timeout=45)
            subprocess.run(['tesseract',str(prefix)+'.png',str(prefix),'-l','eng','--psm','6','txt','tsv'],
                           check=True,capture_output=True,timeout=45)
            text=Path(str(prefix)+'.txt').read_text()
            words=pd.read_csv(str(prefix)+'.tsv',sep='\t',keep_default_na=False)
            conf=pd.to_numeric(words['conf'],errors='coerce')
            valid=conf.ge(0)&words['text'].astype(str).str.strip().ne('')
            row.update({'text':text,'method':'LOCAL_OCR_REQUIRES_VISUAL_VERIFICATION',
                'OCR_mean_word_confidence':float(conf[valid].mean()) if valid.any() else None,
                'rendered_page':str(prefix)+'.png'})
        text_rows.append(row)
    if sum(len(r['text'].strip()) for r in text_rows)<100:
        raise ValueError('Bounded OCR did not recover enough document text')
    return text_rows, {'total_document_pages':total,'pages_read':len(selected),
        'entire_document_read':len(selected)==total,'any_OCR_used':any(r['method'].startswith('LOCAL_OCR') for r in text_rows),
        'quantified_OCR_evidence_visually_verified':False,'complete_causal_chain_verified':False}


def run(source_recovery,output):
    src=Path(source_recovery);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    store=EvidenceStore(src/'raw_PRIVATE')
    failures=json.loads((src/'source_errors_PRIVATE.json').read_text())
    companies={c['symbol']:c for c in json.loads((src/'company_enrichment_PRIVATE.json').read_text())}
    reference=pd.read_parquet(src/'NSE_BSE_AMFI_reference_universe_PRIVATE.parquet')
    names=reference.dropna(subset=['nse_current_symbol']).set_index('nse_current_symbol')['nse_current_name'].to_dict()
    target=[r for r in failures if 'OCR' in r.get('error','') or 'bounded event-document' in r.get('error','')]
    results,errors=[],[]
    for n,r in enumerate(target,1):
        try:
            raw,receipt=store.get(r['source_url'])
            pages,proof=recover_pdf(raw,out/'rendered_PRIVATE'/receipt['sha256'])
            flat='\n'.join(p['text'] for p in pages)
            classification=classify_document(flat,r['symbol'],names.get(r['symbol'],r['symbol']))
            row={**r,**proof,**classification,'source_sha256':receipt['sha256'],
                 'first_retrieved_utc':receipt['retrieved_at_utc'],
                 'source_available_utc':r.get('source_available_utc'),'pages':pages}
            row.pop('error',None)
            for fact in row['quantified_evidence']:
                fact['OCR_visual_verification_required']=proof['any_OCR_used']
                fact['complete_causal_chain_verified']=False
            results.append(row)
        except Exception as e:
            errors.append({**r,'OCR_error':type(e).__name__+': '+str(e)[:180]})
        print(json.dumps({'document_text_recovery_progress':n,'planned':len(target),'read':len(results),'errors':len(errors)}),flush=True)
    (out/'recovered_catalyst_documents_PRIVATE.json').write_text(json.dumps(results,indent=2))
    (out/'document_recovery_errors_PRIVATE.json').write_text(json.dumps(errors,indent=2))
    summary={'source_documents_attempted':len(target),'documents_text_recovered':len(results),
        'issuer_matched_recovered_documents':sum(r['issuer_identity_read'] for r in results),
        'OCR_documents':sum(r['any_OCR_used'] for r in results),
        'fully_read_documents':sum(r['entire_document_read'] for r in results),
        'partial_large_documents':sum(not r['entire_document_read'] for r in results),
        'additional_quantified_candidates':sum(len(r['quantified_evidence']) for r in results),
        'OCR_quantities_automatically_certified':False,'complete_verified_causal_chains':0,
        'source_errors':len(errors),'original_frozen_model_unchanged':True,'production_approved':False}
    (out/'OCR_catalyst_recovery_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)
    return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-recovery',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(a.source_recovery,a.output)


if __name__=='__main__':main()
