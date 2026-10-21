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
from v13_research_selection import RSI_MIN


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
    from v13_alternate_sources import apply_sector_master
    current = apply_sector_master(current,root,now)
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
    if not r['rsi14'] >= RSI_MIN: reasons.append('RSI_BELOW_50_OR_MISSING')
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
    from v13_financial_crosscheck import tag_current_reviews
    financial_path=root/'financial_wide.parquet';comparison_path=root/'financial_crosscheck/latest.json';audit_path=root/'evidence/universe_financial_coverage.json'
    wide=pd.read_parquet(financial_path) if financial_path.exists() else pd.DataFrame()
    comparison=json.loads(comparison_path.read_text()) if comparison_path.exists() else {}
    audited=json.loads(audit_path.read_text()).get('rows',[]) if audit_path.exists() else []
    basis={r['isin']:r.get('preferred_reporting_basis') for r in audited}
    records=tag_current_reviews(records,wide,comparison,selection['source_asof'],basis)
    ranked = sorted((r for r in records if r['rsi70_research_rank'] is not None), key=lambda r: r['rsi70_research_rank'])
    by_symbol = {r['symbol']: r for r in records}
    previous = []
    seed_path=Path(__file__).resolve().parents[1]/'v13_evidence_seed'/'prior_no_price_band_20261017.json'
    prior_data=json.loads(seed_path.read_text()) if seed_path.exists() else {'records':[]}
    original_by_symbol={r['symbol']:r for r in prior_data['records']}
    backlog_path=root/'early_potential'/'coverage_queue.json'
    backlog={r['symbol']:r for r in json.loads(backlog_path.read_text())['records']} if backlog_path.exists() else {}
    for symbol in PRIOR_NO_PRICE_BAND:
        r = by_symbol.get(symbol)
        old=original_by_symbol.get(symbol)
        historical={'prior_source_date':prior_data.get('source_date'),
          'prior_close':old.get('close') if old else None,'prior_p6':old.get('p6') if old else None,
          'prior_p12':old.get('p12') if old else None,'prior_loss12':old.get('loss12') if old else None,
          'prior_estimates_are_not_current':True}
        if r is None:
            gap=backlog.get(symbol,{})
            previous.append({'symbol': symbol, 'status': 'AWAITING_CURRENT_DATA_RETAINED_HISTORICAL_WATCH',
              'reasons': [gap.get('history_status','NO_CURRENT_VERIFIED_MODEL_INPUTS')],**historical})
        else:
            reasons = r['two_x_exclusion_reasons']
            status = ('TOP_20' if r['rsi70_research_rank'] is not None else
                      'ELIGIBLE_BELOW_TOP_20' if r['rsi70_research_eligible'] else
                      'FAILED_CURRENT_SCREEN')
            previous.append({'symbol': symbol, 'status': status, 'reasons': reasons,
                             'close': r['close'], 'p6': r['p6'], 'p12': r['p12'],
                             'rsi14': r['rsi14'], 'rank': r['rsi70_research_rank'],**historical})

    universe_path=root/'universe.parquet'
    public_count=len(pd.read_parquet(universe_path)) if universe_path.exists() else len(rows)+len(backlog)
    coverage_fraction=len(rows)/public_count if public_count else None
    coverage_status='PARTIAL_DATA_DO_NOT_REPLACE_FULL_WATCHLIST' if coverage_fraction is None or coverage_fraction<.95 else 'BROAD_PUBLIC_UNIVERSE_COVERAGE'

    summary = {'source_date': selection['source_asof'],
               'model_replay_verified': True, 'model_retrained': False,
               'ranked_count': len(ranked), 'prior_cohort_size': len(previous),
               'selector': 'V13_DISCOVERY_CURRENT_RSI_GE50_NO_PRICE_BAND_20261021',
               'ranking_weights': {'p6_percentile': .70, 'price_discipline': .25, 'optional_DMA_band': .05},
               'classification_receipts': receipts, 'records': ranked,
               'basic_selector_ranked_count': sum(r['research_selected'] for r in rows),
               'scored_rows': len(rows),
               'public_universe_rows':public_count,'coverage_fraction':coverage_fraction,'coverage_status':coverage_status,
               'earlier_watchlist_preserved_independent_of_current_selection':True,
               'source_review_groups':{tier:[r['symbol'] for r in ranked if r['source_comparison_tier']==tier] for tier in sorted({r['source_comparison_tier'] for r in ranked})},
               'previous': previous}
    # Preserve the broader screen requested in the user's 20-stock screenshot.
    # Its definitions and estimates remain separate from the refined list.
    basic=sorted((r for r in rows if r.get('research_selected')),key=lambda r:r['research_rank'])
    summary['recovered_missing_sector_rows']=sum(bool(r.get('sector_recovered_from_three_sources')) for r in records)
    out = root / 'two_x_research'
    out.mkdir(parents=True, exist_ok=True)
    (out/'basic_latest.json').write_text(json.dumps({'source_date':selection['source_asof'],
        'policy':selection.get('policy'),'records':basic,'model_retrained':False,
        'scope':'BROADER_AVAILABLE_DATA_SCREEN_SEPARATE_FROM_REFINED_SIZE_SECTOR_SCREEN',
        'missing_financial_inputs_are_disclosed':True},indent=2))
    basic_lines=[f"# Broader 2x research shortlist — {selection['source_asof']}",'',
        'Available-data screen retained separately from the refined sector/size list. Estimates are unvalidated; incomplete financials are not verified passes.','',
        '| Rank | Share | Close INR | 2x by 6m | 2x by 12m | Financial inputs |',
        '| ---: | --- | ---: | ---: | ---: | --- |']
    for r in basic:basic_lines.append(f"| {int(r['research_rank'])} | {r['symbol']} | {_money(r['close'])} | {_pct(r['p6'])} | {_pct(r['p12'])} | {r.get('research_financial_tier','UNKNOWN')} |")
    (out/'basic_latest.md').write_text('\n'.join(basic_lines)+'\n')
    (out / 'latest.json').write_text(json.dumps(summary, indent=2))
    lines = [f"# Separate 2× research shortlist — {summary['source_date']}", '',
             'Frozen-model estimates for doubling by 6 or 12 months. These are research estimates, not confirmed returns or trade instructions.',
             'Existing refined selector: verified small/mid caps, known sector, RSI >=50 (inclusive), financial/owner checks; 70% p6 percentile, 25% price discipline, 5% optional 200DMA band.', '',
             f"Coverage: {len(rows):,}/{public_count:,} public shares. {coverage_status}. Missing data is not a failed investment thesis.", '',
             '| Rank | Symbol | Close ₹ | P(2× 6m) | P(2× 12m) | 30% loss event | Financial tier | Source review |',
             '| ---: | --- | ---: | ---: | ---: | ---: | --- | --- |']
    for r in ranked:
        lines.append(f"| {int(r['rsi70_research_rank'])} | {r['symbol']} | {_money(r['close'])} | {_pct(r['p6'])} | {_pct(r['p12'])} | {_pct(r['loss12'])} | {r['sector_status']} | {r['source_comparison_tier']} |")
    lines += ['', '## Earlier no-price-band shortlist (17 October report, 15 October source prices)', '',
              'Prior membership is tracked for explanation only; it does not influence the daily selection.', '',
              '| Symbol | Current result | Reason if excluded | Current P(2× 6m) | Current P(2× 12m) | 15 Oct P(2× 6m) | 15 Oct P(2× 12m) |',
              '| --- | --- | --- | ---: | ---: | ---: | ---: |']
    for r in previous:
        lines.append(f"| {r['symbol']} | {r['status']} | {', '.join(r['reasons']) or '—'} | {_pct(r.get('p6'))} | {_pct(r.get('p12'))} | {_pct(r.get('prior_p6'))} | {_pct(r.get('prior_p12'))} |")
    lines += ['', f"Scored frame: {len(rows)} shares. Current source date: {summary['source_date']}.",
              'Classification source observation dates: ' + ', '.join(sorted({r['first_observed_utc'][:10] for r in receipts})) + '.',
              'A missing name cannot be reassessed from this snapshot. No 50–100% forecast is inferred from this 2× model.', '']
    (out / 'latest.md').write_text('\n'.join(lines))
    return {'source_date': summary['source_date'], 'ranked_count': len(ranked),
            'prior_cohort_size': len(previous), 'status': 'RESEARCH_ONLY_ACCURACY_UNVALIDATED'}
