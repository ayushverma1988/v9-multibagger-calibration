"""Source-bound manual transcription of scanned original financial tables.

The original PDF, embedded image, master identity, economic columns and units
are checked before importing a missing quarter. Actual review/retrieval times
are preserved; existing facts and frozen forecasts are never overwritten.
"""
from datetime import datetime, timezone
import hashlib
import io
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pdfplumber

from v12_dated_source_collection import Store
from v12_dated_training_inputs import exchange_clock
from v12_public_history import bounds
from v13_capture_policy import atomic_json
from v13_pdf_financial_reviews import amount, flat, original_get

LABELS = {
    'revenue': 'revenuefromoperations', 'pat': 'profitaftertax',
    'pat_owners': 'ownersofthecompany', 'pat_noncontrolling': 'noncontrollinginterest',
    'pbt': 'profitbeforetax', 'finance_cost': 'financialcosts',
    'depreciation': 'depreciationandamortization', 'other_income': 'otherincome',
    'eps_diluted_continuing': 'diluted',
}


def validate_table(record, raw, asof, now=None):
    now = pd.Timestamp(now or datetime.now(timezone.utc))
    reviewed = pd.Timestamp(record['first_reviewed_utc'])
    published = pd.Timestamp(record['published_at_utc'])
    if any(t.tzinfo is None for t in (now, reviewed, published)) or not published <= reviewed <= now:
        raise ValueError('Invalid actual image-review clocks')
    if record.get('review_status') != 'MANUALLY_REVIEWED_ORIGINAL_IMAGE_TABLE':
        raise ValueError('Explicit manual image-table review is required')
    if hashlib.sha256(raw).hexdigest() != record['source_sha256']:
        raise ValueError('Reviewed PDF bytes changed')
    if not re.fullmatch(r'https://nsearchives\.nseindia\.com/corporate/[^?]+\.pdf', record['source_url']):
        raise ValueError('Unapproved original financial PDF URL')
    with pdfplumber.open(io.BytesIO(raw)) as pdf:
        front = flat(pdf.pages[0].extract_text() or '').upper()
        if any(flat(record[k]).upper() not in front for k in ('isin', 'symbol', 'cin')):
            raise ValueError('Original PDF/current issuer identity conflict')
        index = record['page_number']
        if isinstance(index, bool) or not isinstance(index, int) or not 1 <= index <= len(pdf.pages):
            raise ValueError('Invalid reviewed image page')
        images = pdf.pages[index-1].images
        if len(images) != 1 or hashlib.sha256(images[0]['stream'].get_rawdata()).hexdigest() != record['page_image_sha256']:
            raise ValueError('Reviewed embedded page image changed')
    if record['reporting_mode'] not in ('standalone', 'consolidated') or record['reporting_mode'] not in record['basis_excerpt'].lower():
        raise ValueError('Reviewed financial basis mismatch')
    if record['scale_INR'] not in (100000, 1000000) or ('lakh' if record['scale_INR'] == 100000 else 'million') not in record['units_excerpt'].lower():
        raise ValueError('Reviewed image currency scale mismatch')
    count = len(record['column_dates'])
    if len(record['column_kinds']) != count or len(record['period_starts']) != count:
        raise ValueError('Financial date columns disagree')
    low, high = bounds(asof)
    for start, end, kind in zip(record['period_starts'], record['column_dates'], record['column_kinds']):
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        days = (end-start).days
        if not low <= start <= end <= high or not (kind == 'quarter' and 60 <= days <= 120 or kind == 'ytd' and 150 <= days <= 310 or kind == 'annual' and 330 <= days <= 400):
            raise ValueError('Invalid reviewed economic period or trailing window')
    current, old = record['current_column'], record['comparative_column']
    if any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < count for i in (current, old)):
        raise ValueError('Invalid comparative column index')
    if record['column_kinds'][current] != 'quarter' or record['column_kinds'][old] != 'quarter' or pd.Timestamp(record['column_dates'][current])-pd.DateOffset(years=1) != pd.Timestamp(record['column_dates'][old]):
        raise ValueError('Comparison must be the same quarter a year apart')
    values = {}
    for field in record['fields']:
        metric, cells = field['metric'], field['reported_cells']
        label = re.sub('[^a-z]', '', field['label'].lower())
        if metric not in LABELS or LABELS[metric] not in label or metric in values or len(cells) != count:
            raise ValueError('Missing, duplicated or mismatched reviewed image row')
        per_share = metric == 'eps_diluted_continuing'
        if field['unit'] != ('INR/SHARES' if per_share else 'INR'):
            raise ValueError('EPS and monetary units cannot be interchanged')
        values[metric] = [amount(cell)*(1 if per_share else record['scale_INR']) for cell in cells]
    if not {'revenue', 'pat', 'pat_owners', 'pat_noncontrolling', 'eps_diluted_continuing'}.issubset(values):
        raise ValueError('Explicit profit ownership and diluted EPS are required')
    for i in (current, old):
        if abs(values['pat'][i]-values['pat_owners'][i]-values['pat_noncontrolling'][i]) > record['scale_INR']*.02:
            raise ValueError('Owner/minority/total profit does not reconcile')
    return values


def collect(root, seed_path, asof):
    root, seed_path = Path(root), Path(seed_path)
    if not seed_path.exists():
        return {'status': 'NO_REVIEWED_IMAGE_FINANCIAL_SEED'}
    master = pd.read_parquet(root/'universe.parquet')
    identities = dict(zip(master.nse_current_symbol, master['isin']))
    signature = hashlib.sha256(seed_path.read_bytes()+json.dumps(identities, sort_keys=True).encode()).hexdigest()
    marker = root/'evidence/image_financial_review_receipt.json'
    if marker.exists():
        prior = json.loads(marker.read_text())
        if prior.get('seed_identity_sha256') == signature and prior.get('complete'):
            return prior
    store = Store(root/'filing_raw/reviewed_image_originals')
    path = root/'financial_wide.parquet';wide = pd.read_parquet(path)
    documents, errors, added = [], [], []
    eps_path = root/'statement_eps/accepted.json'
    eps_rows = json.loads(eps_path.read_text()) if eps_path.exists() else []
    for record in json.loads(seed_path.read_text())['documents']:
        try:
            if identities.get(record['symbol']) != record['isin']:
                raise ValueError('Reviewed image/current master ISIN conflict')
            catalogue, catalogue_proof = original_get(store, record['catalogue_url'])
            matches = [r for r in json.loads(catalogue) if r.get('symbol') == record['symbol'] and r.get('attchmntFile') == record['source_url']]
            if len(matches) != 1 or exchange_clock(matches[0].get('exchdisstime')) != pd.Timestamp(record['published_at_utc']):
                raise ValueError('Original exchange attachment/publication binding failed')
            raw, proof = original_get(store, record['source_url'])
            if hashlib.sha256(raw).hexdigest() != proof['sha256']:
                raise ValueError('Original image financial receipt mismatch')
            values = validate_table(record, raw, asof)
            retrieved = pd.Timestamp(proof['first_retrieved_utc'])
            if retrieved.tzinfo is None or not pd.Timestamp(record['published_at_utc']) <= retrieved <= pd.Timestamp.now(tz='UTC'):
                raise ValueError('Invalid actual original retrieval clock')
            seen = max(pd.Timestamp(record['published_at_utc']), pd.Timestamp(record['first_reviewed_utc']), retrieved)
            current, old = record['current_column'], record['comparative_column']
            anchor = wide[wide.security_isin.eq(record['isin']) & wide.reporting_mode.eq(record['reporting_mode']) & wide.period_kind.eq('quarter') & wide.period_end.eq(record['column_dates'][current])]
            if anchor.empty:raise ValueError('Current original financial anchor unavailable')
            anchor = anchor.sort_values('available').iloc[-1]
            for metric in ('revenue', 'pat', 'pat_owners', 'pat_noncontrolling', 'eps_diluted_continuing'):
                tolerance = .00001 if metric == 'eps_diluted_continuing' else record['scale_INR']*.02
                if not np.isfinite(anchor[metric]) or abs(float(anchor[metric])-values[metric][current]) > tolerance:
                    raise ValueError('Image/current original numeric conflict: '+metric)
            quarter_end = record['column_dates'][old]
            same = wide[wide.security_isin.eq(record['isin']) & wide.reporting_mode.eq(record['reporting_mode']) & wide.period_kind.eq('quarter') & wide.period_end.eq(quarter_end)]
            facts = []
            if not same.empty:
                previous = same.sort_values('available').iloc[-1]
                for metric, cells in values.items():
                    prior = previous.get(metric, np.nan)
                    tolerance = .00001 if metric == 'eps_diluted_continuing' else record['scale_INR']*.02
                    if np.isfinite(prior) and abs(prior-cells[old]) > tolerance:
                        raise ValueError('Existing comparative requires source-conflict review: '+metric)
            if same.empty:
                for metric, cells in values.items():
                    facts.append({'symbol': record['symbol'], 'security_isin': record['isin'], 'reporting_mode': record['reporting_mode'], 'period_kind': 'quarter',
                        'period_start': record['period_starts'][old], 'period_end': quarter_end, 'available_at_utc': seen.isoformat(),
                        'original_publication_utc': record['published_at_utc'], 'first_retrieved_utc': proof['first_retrieved_utc'],
                        'source_url': record['source_url'], 'source_sha256': record['source_sha256'], 'metric': metric, 'value': cells[old]})
                from v13_collection import financial_wide
                wide = pd.concat([wide, financial_wide(pd.DataFrame(facts))], ignore_index=True);added.extend(facts)
            review_sha = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
            comparison = {'isin': record['isin'], 'symbol': record['symbol'], 'current_period_end': record['column_dates'][current], 'comparative_period_end': quarter_end,
                'current_reported_diluted_EPS': values['eps_diluted_continuing'][current], 'comparative_reported_diluted_EPS': values['eps_diluted_continuing'][old],
                'reported_diluted_EPS_yoy_same_document': values['eps_diluted_continuing'][current]/values['eps_diluted_continuing'][old]-1 if values['eps_diluted_continuing'][old] > 0 else None,
                'statement_sha256': record['source_sha256'], 'anchor_sha256': anchor.source_sha256, 'first_reviewed_utc': record['first_reviewed_utc'],
                'available_at_utc': seen.isoformat(), 'source_publication_utc': record['published_at_utc'], 'review_sha256': review_sha,
                'recurring_EPS_verified': False, 'method': 'MANUALLY_REVIEWED_HASH_BOUND_IMAGE_TABLE'}
            eps_rows = [r for r in eps_rows if r.get('review_sha256') != review_sha]+[comparison]
            documents.append({'symbol': record['symbol'], 'source_sha256': record['source_sha256'], 'catalogue_sha256': catalogue_proof['sha256'],
                'catalogue_isin': matches[0].get('sm_isin'), 'identity_source': 'ORIGINAL_PDF_FRONT_PAGE_AND_CURRENT_MASTER_ISIN',
                'page_image_sha256': record['page_image_sha256'], 'facts_added': len(facts), 'existing_quarters_overwritten': False,
                'EPS_comparison': comparison, 'available_at_utc': seen.isoformat()})
        except Exception as exc:errors.append({'symbol': record.get('symbol'), 'error': str(exc)[:220]})
    if added:wide.to_parquet(path, index=False, compression='zstd', compression_level=9)
    if documents:
        eps_path.parent.mkdir(exist_ok=True);atomic_json(eps_path, eps_rows)
    result = {'seed_identity_sha256': signature, 'complete': not errors, 'documents': documents, 'errors': errors,
        'missing_comparative_fact_observations_added': len(added), 'existing_quarters_overwritten': False,
        'new_reviews_never_backdated': True, 'frozen_forecasts_or_probabilities_changed': False}
    atomic_json(marker, result)
    return result
