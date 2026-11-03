"""Import manually reviewed missing quarters from original NSE company PDFs.

No guessed table cells, owner-profit subtraction, EPS reconstruction or old
cohort rescoring. Review and retrieval clocks remain actual, not publication.
"""
from datetime import datetime, timezone
import hashlib
import io
import json
import re
import time
from pathlib import Path

import pandas as pd
import pdfplumber

from v12_public_history import bounds
from v12_dated_source_collection import Store
from v12_dated_training_inputs import exchange_clock
from v13_capture_policy import atomic_json

LABELS = {'revenue': 'revenuefromoperations', 'pat': 'profitfortheperiod',
    'pat_owners': 'ownersofthecompany', 'pat_noncontrolling': 'noncontrollinginterest',
    'pbt': 'profitbeforetax', 'finance_cost': 'financecosts',
    'depreciation': 'depreciationandamortisation', 'other_income': 'otherincome'}


def original_get(store, url):
    """Reuse checked bytes; retry transient failures with preserved vintages."""
    from v13_collection import _transient_universe_fetch_error
    for attempt in range(3):
        try:
            raw, proof = store.get(url)
            return raw, {**proof, 'bounded_fetch_attempts': attempt + 1}
        except Exception as exc:
            if attempt == 2 or not _transient_universe_fetch_error(exc):
                raise
            time.sleep(attempt + 1)
            store = Store(store.folder / ('retry_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')))


def flat(text):
    return re.sub(r'\s+', '', text).casefold()


def amount(token):
    """Require a numeric printed cell; a dash/nil is never a verified zero."""
    value = token.strip()
    negative = value.startswith('(') and value.endswith(')')
    if negative:
        value = value[1:-1]
    if not re.fullmatch(r'\d[\d,.]*(?:\.\d+)?', value):
        raise ValueError('Missing or ambiguous printed numeric cell')
    # Some embedded PDF fonts extract a thousands comma as a dot. Only exact
    # three-digit grouping plus a final decimal group is accepted.
    if value.count('.') > 1:
        parts = value.split('.')
        if any(len(p) != 3 for p in parts[1:-1]) or len(parts[-1]) > 2:
            raise ValueError('Ambiguous printed numeric grouping')
        value = ''.join(parts[:-1]) + '.' + parts[-1]
    result = float(value.replace(',', ''))
    return -result if negative else result


def reviewed_facts(record, text, retrieved_at, asof, clock=None):
    now = pd.Timestamp(clock or datetime.now(timezone.utc))
    published = pd.Timestamp(record['published_at_utc'])
    reviewed = pd.Timestamp(record['first_reviewed_utc'])
    retrieved = pd.Timestamp(retrieved_at)
    if any(t.tzinfo is None or t > now for t in (published, reviewed, retrieved)) or reviewed < published:
        raise ValueError('Unusable original PDF observation clocks')
    if record.get('review_status') != 'MANUALLY_REVIEWED_ORIGINAL_TABLE':
        raise ValueError('Financial table has not been reviewed')
    if not re.fullmatch('[0-9a-f]{64}', str(record.get('source_sha256', ''))):
        raise ValueError('Invalid reviewed original source hash')
    if record['reporting_mode'] not in ('standalone', 'consolidated'):
        raise ValueError('Unknown PDF financial reporting basis')
    normalized = flat(text)
    for name in ('issuer_excerpt', 'basis_excerpt', 'units_excerpt', 'header_excerpt'):
        if not record.get(name) or flat(record[name]) not in normalized:
            raise ValueError('PDF review excerpt missing: ' + name)
    if record['reporting_mode'] not in record['basis_excerpt'].lower():
        raise ValueError('PDF table does not explicitly state reviewed basis')
    scale = record['scale_INR']
    if scale not in (100000, 1000000) or ('lakh' if scale == 100000 else 'million') not in record['units_excerpt'].lower():
        raise ValueError('PDF currency scale is not bound to its table')
    index = record['column_index']
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(record['column_dates']):
        raise ValueError('Invalid reviewed PDF column')
    if len(record['column_kinds']) != len(record['column_dates']) or record['column_kinds'][index] != 'quarter':
        raise ValueError('An annual PDF column cannot become a quarterly observation')
    start = pd.Timestamp(record['period_start']); end = pd.Timestamp(record['column_dates'][index])
    low, high = bounds(asof)
    if not low <= start <= end <= high or not 60 <= (end - start).days <= 120:
        raise ValueError('Reviewed PDF quarter outside trailing window')
    facts = []
    for item in record['fields']:
        metric = item['metric']; excerpt = item['row_excerpt']; cells = item['reported_cells']
        letters = re.sub('[^a-z]', '', excerpt.lower())
        if metric not in LABELS or LABELS[metric] not in letters or flat(excerpt) not in normalized:
            raise ValueError('PDF metric label/excerpt is not source bound')
        if len(cells) != len(record['column_dates']):
            raise ValueError('PDF cells and reviewed date columns disagree')
        position = 0
        for token in cells:
            found = flat(excerpt).find(flat(token), position)
            if found < 0:
                raise ValueError('Reviewed PDF cells do not match printed row order')
            position = found + len(flat(token))
        value = amount(cells[index]) * scale
        facts.append({'symbol': record['symbol'], 'security_isin': record['isin'],
            'reporting_mode': record['reporting_mode'], 'period_kind': 'quarter',
            'period_start': str(start.date()), 'period_end': str(end.date()),
            'available_at_utc': max(published, reviewed, retrieved).isoformat(),
            'original_publication_utc': published.isoformat(),
            'first_retrieved_utc': retrieved.isoformat(),
            'source_url': record['source_url'], 'source_sha256': record['source_sha256'],
            'metric': metric, 'value': value})
    if len({f['metric'] for f in facts}) != len(facts) or not {'revenue', 'pat'}.issubset({f['metric'] for f in facts}):
        raise ValueError('Reviewed PDF quarter requires unique explicit revenue and total PAT')
    values = {f['metric']: f['value'] for f in facts}
    if {'pat', 'pat_owners', 'pat_noncontrolling'}.issubset(values) and abs(
            values['pat'] - values['pat_owners'] - values['pat_noncontrolling']) > scale * .02:
        raise ValueError('Reviewed owner/minority/total PAT do not reconcile')
    return facts


def collect(root, seed_path, asof):
    root = Path(root); seed_path = Path(seed_path)
    if not seed_path.exists():
        return {'status': 'NO_REVIEWED_PDF_FINANCIAL_SEED'}
    master = pd.read_parquet(root / 'universe.parquet')
    identities = dict(zip(master.nse_current_symbol, master['isin']))
    signature = hashlib.sha256(seed_path.read_bytes() + json.dumps(identities, sort_keys=True).encode()).hexdigest()
    target = root / 'evidence/pdf_financial_review_receipt.json'
    if target.exists():
        prior = json.loads(target.read_text())
        if prior.get('seed_identity_sha256') == signature and prior.get('complete'):
            return prior
    store = Store(root / 'filing_raw/reviewed_pdf_originals')
    path = root / 'financial_wide.parquet'
    wide = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    documents = []; errors = []; added = []
    for record in json.loads(seed_path.read_text())['documents']:
        try:
            if identities.get(record['symbol']) != record['isin']:
                raise ValueError('Reviewed PDF/current master identity mismatch')
            raw_catalog, catalog_proof = original_get(store, record['catalogue_url'])
            matches = [r for r in json.loads(raw_catalog) if r.get('symbol') == record['symbol']
                and r.get('sm_isin') == record['isin'] and r.get('attchmntFile') == record['source_url']]
            if len(matches) != 1 or exchange_clock(matches[0].get('exchdisstime')) != pd.Timestamp(record['published_at_utc']):
                raise ValueError('PDF attachment identity/publication is not bound to original NSE catalogue')
            raw, proof = original_get(store, record['source_url'])
            if proof['sha256'] != record['source_sha256'] or hashlib.sha256(raw).hexdigest() != record['source_sha256']:
                raise ValueError('Reviewed original PDF changed; new review required')
            with pdfplumber.open(io.BytesIO(raw)) as pdf:
                if not isinstance(record['page_number'], int) or not 1 <= record['page_number'] <= len(pdf.pages):
                    raise ValueError('Invalid reviewed PDF page')
                text = pdf.pages[record['page_number'] - 1].extract_text() or ''
            facts = reviewed_facts(record, text, proof['first_retrieved_utc'], asof)
            end = facts[0]['period_end']
            same = wide[wide.security_isin.eq(record['isin']) & wide.reporting_mode.eq(record['reporting_mode'])
                & wide.period_kind.eq('quarter') & wide.period_end.eq(end)] if not wide.empty else wide
            status = 'ORIGINAL_QUARTER_ALREADY_PRESENT_REVIEW_RETAINED_NO_OVERWRITE'
            if same.empty:
                from v13_collection import financial_wide
                fresh = financial_wide(pd.DataFrame(facts))
                wide = pd.concat([wide, fresh], ignore_index=True)
                added.extend(facts); status = 'ADDED_MISSING_REVIEWED_QUARTER_ACTUAL_OBSERVATION_CLOCK'
            documents.append({'symbol': record['symbol'], 'isin': record['isin'], 'quarter_end': end,
                'status': status, 'source_sha256': proof['sha256'],
                'catalogue_sha256': catalog_proof['sha256'], 'facts': facts,
                'reviewed_record': record})
        except Exception as exc:
            message = str(exc)[:220]; errors.append({'symbol': record.get('symbol'), 'error': message})
            if any(s in message.lower() for s in ('403', '429', 'forbidden', 'too many requests')):
                break
    if added:
        wide.to_parquet(path, index=False,compression='zstd',compression_level=9)
    result = {'seed_identity_sha256': signature, 'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
        'complete': not errors, 'documents': documents, 'errors': errors,
        'missing_quarter_fact_observations_added': len(added),
        'existing_quarters_overwritten': False, 'recurring_EPS_inferred': False,
        'old_forecasts_or_probabilities_changed': False}
    atomic_json(target, result)
    return result
