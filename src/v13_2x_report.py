"""Human-readable 2x research output from the frozen-model daily selection.

Apply the existing 17 October discovery selector to fresh frozen probabilities.
The basic selector remains available as a diagnostic, not the headline list.
"""
import base64
import gzip
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


PRIOR_NO_PRICE_BAND = (
    'MASTEK', 'SHREEJISPG', 'TANLA', 'CUPID', 'HEXT', 'THELEELA',
    'LTTS', 'PERSISTENT', 'ZENSARTECH', 'SUNTV', 'MPHASIS',
)


def _pct(value):
    return '—' if value is None else f'{100 * value:.2f}%'


def _money(value):
    return '—' if value is None else f'{value:,.2f}'


def _seed_classifications(target, failure, seed_path=None):
    """Reuse verified dated classifications after a transport failure, with clocks intact."""
    path = Path(seed_path) if seed_path else Path(__file__).resolve().parents[1] / 'v13_evidence_seed' / 'discovery_classifications_20261016.json.gz.b64'
    bundle = json.loads(gzip.decompress(base64.b64decode(path.read_text())))
    if pd.Timestamp.now(tz='UTC') > pd.Timestamp(bundle['reuse_not_after_utc']):
        raise ValueError('Stored classification snapshot expired; fresh sources required')
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    for name, content in bundle['files'].items():
        (target / name).write_text(content)
    receipts = json.loads((target / 'receipt.json').read_text())
    for receipt in receipts:
        receipt['refresh_attempt_error'] = failure[:200]
        receipt['reuse_policy'] = 'VERIFIED_DATED_SNAPSHOT_CLOCKS_PRESERVED'
    (target / 'receipt.json').write_text(json.dumps(receipts, indent=2))
    return target


def refine(rows, root, classification_dir=None, review_asof=None):
    """Reuse the selector that produced the user's 11-stock no-price-band list."""
    from v13_discovery_current import classify_current, watchlists
    root = Path(root)
    now = review_asof or datetime.now(timezone.utc).isoformat()
    if classification_dir is None:
        from v13_discovery_sources import run as collect_classifications
        classification_dir = root / 'two_x_research' / 'classifications' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        try:
            collect_classifications(classification_dir)
        except Exception as exc:
            classification_dir = _seed_classifications(root / 'two_x_research' / 'classifications' / 'verified_20261016', str(exc))
    snapshot = json.loads((root / 'evidence' / 'valuation_snapshot.json').read_text())
    members = pd.DataFrame(json.loads((root / 'evidence' / 'valuation_research_hurdles.json').read_text()))[['isin', 'symbol', 'industry']]
    members['available_at_utc'] = snapshot['membership_first_observed_utc']
    members['source_sha256'] = snapshot['membership_sha256']
    original = pd.DataFrame(rows)
    current = classify_current(original, classification_dir, members, now)
    documents = json.loads((root / 'catalyst_evidence' / 'document_evidence.json').read_text())
    current = watchlists(current, documents, now)
    columns = ['p6', 'p12', 'loss12', 'joint12']
    if not current[columns].equals(original[columns]):
        raise ValueError('Discovery reporting changed frozen probabilities')
    current['two_x_exclusion_reasons'] = current.apply(_reasons, axis=1)
    receipts = json.loads((Path(classification_dir) / 'receipt.json').read_text())
    return current, receipts


def _reasons(r):
    if bool(r['rsi70_research_eligible']):
        return []
    reasons = []
    if r['size_status'] != 'SMALL_MID_VERIFIED': reasons.append(r['size_status'])
    if r['sector_sector_kind'] == 'UNKNOWN': reasons.append('UNKNOWN_SECTOR')
    if not r['rsi14'] > 70: reasons.append('RSI_NOT_STRICTLY_ABOVE_70')
    if not r['distance_ma50'] > 0: reasons.append('NOT_ABOVE_50DMA')
    if r['log_ret63'] > np.log(2.5): reasons.append('EXCESSIVE_63_SESSION_EXTENSION')
    if r['sector_status'] == 'KNOWN_FINANCIAL_FAILURE': reasons.append('OBSERVED_SECTOR_FINANCIAL_FAILURE')
    owner = any(r.get(k) is not None and np.isfinite(r[k]) and
                (r[k] < 0 if k.endswith('yoy') else r[k] <= 0)
                for k in ['owner_PAT_yoy', 'owner_PAT_margin'])
    if owner or r.get('current_owner_profit_status') == 'UNKNOWN_OWNER_PLUS_MINORITY_DOES_NOT_RECONCILE_TOTAL_PAT':
        reasons.append('OBSERVED_OWNER_EARNINGS_FAILURE_OR_CONFLICT')
    return reasons or ['OTHER_DISCOVERY_ELIGIBILITY_CHECK']


def export(root, replay_receipt, classification_dir=None, review_asof=None):
    root = Path(root)
    selection = json.loads((root / 'research_selection' / 'latest.json').read_text())
    redevelopment = json.loads((root / 'redevelopment' / 'latest.json').read_text())
    if not replay_receipt.get('replay_verified'):
        raise ValueError('2x report requires verified model replay')
    if selection['prediction_replay'] != replay_receipt or redevelopment['prediction_replay'] != replay_receipt:
        raise ValueError('2x report inputs must share the same model replay')
    if selection['source_asof'] != redevelopment['source_date']:
        raise ValueError('2x report inputs must have the same source date')

    rows = redevelopment['records']
    if replay_receipt['rows'] != len(rows):
        raise ValueError('2x report row count differs from model replay')
    current, receipts = refine(rows, root, classification_dir, review_asof)
    records = json.loads(current.to_json(orient='records', date_format='iso', double_precision=15))
    ranked = sorted((r for r in records if r['rsi70_research_rank'] is not None), key=lambda r: r['rsi70_research_rank'])
    by_symbol = {r['symbol']: r for r in records}
    previous = []
    for symbol in PRIOR_NO_PRICE_BAND:
        r = by_symbol.get(symbol)
        if r is None:
            previous.append({'symbol': symbol, 'status': 'NOT_IN_CURRENT_SCORED_FRAME', 'reasons': []})
        else:
            reasons = r['two_x_exclusion_reasons']
            status = ('TOP_20' if r['rsi70_research_rank'] is not None else
                      'ELIGIBLE_BELOW_TOP_20' if r['rsi70_research_eligible'] else
                      'FAILED_CURRENT_SCREEN')
            previous.append({'symbol': symbol, 'status': status, 'reasons': reasons,
                             'close': r['close'], 'p6': r['p6'], 'p12': r['p12'],
                             'rsi14': r['rsi14'], 'rank': r['rsi70_research_rank']})

    summary = {'source_date': selection['source_asof'],
               'model_replay_verified': True, 'model_retrained': False,
               'ranked_count': len(ranked), 'prior_cohort_size': len(previous),
               'selector': 'V13_DISCOVERY_CURRENT_RSI70_NO_PRICE_BAND_20261017',
               'ranking_weights': {'p6_percentile': .70, 'price_discipline': .25, 'optional_DMA_band': .05},
               'classification_receipts': receipts, 'records': ranked,
               'basic_selector_ranked_count': sum(r['research_selected'] for r in rows),
               'scored_rows': len(rows),
               'previous': previous}
    out = root / 'two_x_research'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'latest.json').write_text(json.dumps(summary, indent=2))
    lines = [f"# Separate 2× research shortlist — {summary['source_date']}", '',
             'Frozen-model estimates for doubling by 6 or 12 months. These are research estimates, not confirmed returns or trade instructions.',
             'Existing refined selector: verified small/mid caps, known sector, strict RSI >70, financial/owner checks; 70% p6 percentile, 25% price discipline, 5% optional 200DMA band.', '',
             '| Rank | Symbol | Close ₹ | P(2× 6m) | P(2× 12m) | 30% loss event | Financial tier |',
             '| ---: | --- | ---: | ---: | ---: | ---: | --- |']
    for r in ranked:
        lines.append(f"| {int(r['rsi70_research_rank'])} | {r['symbol']} | {_money(r['close'])} | {_pct(r['p6'])} | {_pct(r['p12'])} | {_pct(r['loss12'])} | {r['sector_status']} |")
    lines += ['', '## Earlier no-price-band shortlist (17 October report, 15 October source prices)', '',
              'Prior membership is tracked for explanation only; it does not influence the daily selection.', '',
              '| Symbol | Current result | Reason if excluded | P(2× 6m) | P(2× 12m) |',
              '| --- | --- | --- | ---: | ---: |']
    for r in previous:
        lines.append(f"| {r['symbol']} | {r['status']} | {', '.join(r['reasons']) or '—'} | {_pct(r.get('p6'))} | {_pct(r.get('p12'))} |")
    lines += ['', f"Scored frame: {len(rows)} shares. Current source date: {summary['source_date']}.",
              'Classification source observation dates: ' + ', '.join(sorted({r['first_observed_utc'][:10] for r in receipts})) + '.',
              'A missing name cannot be reassessed from this snapshot. No 50–100% forecast is inferred from this 2× model.', '']
    (out / 'latest.md').write_text('\n'.join(lines))
    return {'source_date': summary['source_date'], 'ranked_count': len(ranked),
            'prior_cohort_size': len(previous), 'status': 'RESEARCH_ONLY_ACCURACY_UNVALIDATED'}
