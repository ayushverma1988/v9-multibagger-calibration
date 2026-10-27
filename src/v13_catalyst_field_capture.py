"""Extract source-bound field candidates from cached original company PDFs.

These are review packets, not completed execution chains. In particular an
issuer margin cannot silently become a project margin or twelve-month profit.
"""
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from v12_dated_source_collection import sha
from v13_capture_policy import atomic_json

VERSION = 'V13_SOURCE_BOUND_CATALYST_FIELDS_20261028'
INTENT = re.compile(r'\b(?:will|plans?|planned|proposed|expected|target|intend|subject to|pending)\b', re.I)


def capture(text, document, clock):
    if not re.fullmatch(r'[0-9a-f]{64}',str(document.get('source_sha256',''))):
        raise ValueError('Invalid original source hash')
    if sha(text.encode()) != document.get('text_sha256'):
        raise ValueError('Cached catalyst text hash mismatch')
    if document.get('issuer_identity_read') is not True or not re.fullmatch(r'INE[A-Z0-9]{9}', str(document.get('isin', ''))):
        raise ValueError('Original issuer identity unresolved')
    now = pd.Timestamp(clock)
    publication = pd.Timestamp(document['available_at_utc'])
    retrieval = pd.Timestamp(document['first_retrieved_utc'])
    if any(x.tzinfo is None for x in (now, publication, retrieval)) or max(publication, retrieval) > now:
        raise ValueError('Unusable source observation clock')
    flat = re.sub(r'\s+', ' ', text)
    candidates = []
    patterns = {
        'commercial_operation_statement': r'(?:has|have)\s+(?:successfully\s+)?commenced\s+commercial\s+(?:production|operations)|commercial\s+(?:production|operations)\s+(?:has|have)\s+(?:been\s+)?commenced',
        'commissioning_statement': r'(?:has|have)\s+(?:been\s+)?(?:successfully\s+)?commissioned',
        'capacity_quantity': r'capacity\b[^.;]{0,65}?([\d,]+(?:\.\d+)?)\s*(MWh|MW|GW|MTPA|TPA|TPD|TCD|tonnes? per annum|units per annum)\b',
        'utilisation_pct': r'(?:capacity\s+)?utili[sz]ation(?:\s+(?:of|at|stood at|was))?\s*[:\-]?\s*([\d,]+(?:\.\d+)?)\s*%',
        'reported_margin_pct': r'\b(PAT|EBITDA|EBIT)\s+margin(?:\s+(?:of|at|stood at|was))?\s*[:\-]?\s*([\d,]+(?:\.\d+)?)\s*%',
        'funding_statement': r'(?:financial\s+closure\s+(?:has\s+been\s+)?achieved|(?:project|expansion)\s+(?:is|has been)\s+fully\s+funded|funding\s+(?:has\s+been\s+)?secured)',
        'operating_approval_statement': r'(?:all|required|requisite)\s+(?:regulatory|statutory|operating)\s+approvals\s+(?:have\s+been\s+)?(?:received|obtained|granted)',
        'firm_order_statement': r'(?:has|have)\s+(?:been\s+)?(?:received|awarded|secured|signed)\b[^.;]{0,100}?\b(?:order|contract|letter of award)\b',
        'execution_deadline': r'\b(?:completion|execution|delivery)\s+(?:by|on or before)\s+(\d{1,2}\s+[A-Za-z]+\s+20\d{2}|20\d{2}-\d{2}-\d{2})',
    }
    for field, pattern in patterns.items():
        for match in re.finditer(pattern, flat, re.I):
            excerpt = flat[max(0, match.start()-100):min(len(flat), match.end()+100)]
            value = match.group(0)
            unit = None
            if field == 'capacity_quantity':
                value=float(match.group(1).replace(',', ''));unit=match.group(2)
            elif field == 'utilisation_pct':
                value=float(match.group(1).replace(',', ''));unit='PERCENT'
                if not 0 <= value <= 100:continue
            elif field == 'reported_margin_pct':
                value=float(match.group(2).replace(',', ''));unit=match.group(1).upper()+'_PERCENT'
                if not -100 <= value <= 100:continue
            elif field == 'execution_deadline':
                try:value=str(pd.to_datetime(match.group(1)).date())
                except ValueError:continue
            conditional=bool(INTENT.search(excerpt))
            legal=bool(re.search(r'tax|court|tribunal|penalt|adjudicat',excerpt,re.I))
            if field=='firm_order_statement' and (legal or re.search(r'letter of intent|\bLOI\b',excerpt,re.I)):
                conditional=True
            candidates.append({'field':field,'value':value,'unit':unit,'source_excerpt':excerpt,
                'status':'CONDITIONAL_OR_CONTEXT_REVIEW' if conditional else 'SOURCE_TEXT_CANDIDATE',
                'project_binding_requires_review':True})
    unique={json.dumps(x,sort_keys=True):x for x in candidates}
    return {k:document.get(k) for k in ('symbol','isin','attachment_url','source_sha256','text_sha256','available_at_utc','first_retrieved_utc')} | {
        'capture_version':VERSION,'captured_at_utc':now.isoformat(),
        'available_for_new_analysis_at_utc':max(now,publication,retrieval).isoformat(),
        'fields':list(unique.values()),'project_or_order_id_verified':False,
        'full_execution_chain_verified':False,'automatic_future_profit_claims':0,
        'reported_margin_is_not_expected_project_margin':True}


def export(root, asof, max_documents=5000):
    root=Path(root);out=root/'catalyst_evidence'
    dp=out/'document_evidence.json'
    documents=json.loads(dp.read_text()) if dp.exists() else []
    clock=datetime.now(timezone.utc).isoformat();low=pd.Timestamp(asof)-pd.DateOffset(years=4)
    rows=[];errors=[];seen=set();text_verified=0
    for document in sorted(documents,key=lambda d:d.get('available_at_utc',''),reverse=True):
        if len(seen)>=max_documents:break
        key=(document.get('isin'),document.get('source_sha256'))
        if key in seen:continue
        seen.add(key)
        try:
            if pd.Timestamp(document['available_at_utc']) < low.tz_localize('Asia/Kolkata'):continue
            digest=document.get('source_sha256','')
            if not re.fullmatch(r'[0-9a-f]{64}',str(digest)):raise ValueError('Invalid original source hash')
            path=out/(digest+'.txt')
            if not path.exists():continue
            result=capture(path.read_text(),document,clock);text_verified+=1
            if result['fields']:rows.append(result)
        except (ValueError,KeyError,TypeError) as exc:
            errors.append({'source_sha256':document.get('source_sha256'),'error':str(exc)[:160]})
    counts=Counter(f['field'] for r in rows for f in r['fields'])
    summary={'source_asof':asof,'captured_at_utc':clock,'capture_version':VERSION,
        'unique_document_identities_considered':len(seen),'identity_and_text_hash_verified_documents':text_verified,
        'documents_with_field_candidates':len(rows),'field_candidate_counts':dict(counts),
        'reviewed_execution_chains_added':0,'original_downloads_added':0,
        'existing_cached_text_used_without_redownload':True,'frozen_forecasts_changed':False,
        'automatic_future_profit_claims':0,'errors':errors}
    atomic_json(out/'field_capture_candidates.json',rows)
    atomic_json(out/'field_capture_summary.json',summary)
    return summary


def capture_priority(record):
    """Business disclosures ahead of administrative boilerplate, plus rotation."""
    title=str(record.get('headline',''))
    text=title+' '+str(record.get('text',''))
    if re.search(r'bagging|receiving of orders|capacity addition|commercial production|commissioning',title,re.I):return 0
    if re.search(r'capacity|commenced commercial|utili[sz]ation|funding|financial closure',text,re.I):return 1
    if re.search(r'investor presentation|earnings.*presentation',text,re.I):return 2
    return 3
