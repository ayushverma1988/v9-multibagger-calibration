"""Human-readable 2x research output from the frozen-model daily selection.

This is a presentation and continuity report. It never changes the model,
selector, forecasts, or saved historical cohorts.
"""
import json
from pathlib import Path


PRIOR_NO_PRICE_BAND = (
    'MASTEK', 'SHREEJISPG', 'TANLA', 'CUPID', 'HEXT', 'THELEELA',
    'LTTS', 'PERSISTENT', 'ZENSARTECH', 'SUNTV', 'MPHASIS',
)


def _pct(value):
    return '—' if value is None else f'{100 * value:.2f}%'


def _money(value):
    return '—' if value is None else f'{value:,.2f}'


def export(root, replay_receipt):
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
    ranked = sorted((r for r in rows if r['research_selected']), key=lambda r: r['research_rank'])
    by_symbol = {r['symbol']: r for r in rows}
    previous = []
    for symbol in PRIOR_NO_PRICE_BAND:
        r = by_symbol.get(symbol)
        if r is None:
            previous.append({'symbol': symbol, 'status': 'NOT_IN_CURRENT_SCORED_FRAME', 'reasons': []})
        else:
            reasons = json.loads(r['research_exclusion_reasons'])
            status = ('TOP_20' if r['research_selected'] else
                      'ELIGIBLE_BELOW_TOP_20' if r['research_eligible'] else
                      'FAILED_CURRENT_SCREEN')
            previous.append({'symbol': symbol, 'status': status, 'reasons': reasons,
                             'close': r['close'], 'p6': r['p6'], 'p12': r['p12'],
                             'rsi14': r['rsi14'], 'rank': r['research_rank']})

    summary = {'source_date': selection['source_asof'],
               'model_replay_verified': True, 'model_retrained': False,
               'ranked_count': len(ranked), 'prior_cohort_size': len(previous),
               'previous': previous}
    out = root / 'two_x_research'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'latest.json').write_text(json.dumps(summary, indent=2))
    lines = [f"# Separate 2× research shortlist — {summary['source_date']}", '',
             'Frozen-model estimates for doubling by 6 or 12 months. These are research estimates, not confirmed returns or trade instructions.',
             'The ranking also uses evidence and risk preferences; it is not sorted solely by P(2×).', '',
             '| Rank | Symbol | Close ₹ | P(2× 6m) | P(2× 12m) | 30% loss event | Financial tier |',
             '| ---: | --- | ---: | ---: | ---: | ---: | --- |']
    for r in ranked:
        lines.append(f"| {int(r['research_rank'])} | {r['symbol']} | {_money(r['close'])} | {_pct(r['p6'])} | {_pct(r['p12'])} | {_pct(r['loss12'])} | {r['research_financial_tier']} |")
    lines += ['', '## Earlier no-price-band shortlist (17 October report, 15 October source prices)', '',
              'Prior membership is tracked for explanation only; it does not influence the daily selection.', '',
              '| Symbol | Current result | Reason if excluded | P(2× 6m) | P(2× 12m) |',
              '| --- | --- | --- | ---: | ---: |']
    for r in previous:
        lines.append(f"| {r['symbol']} | {r['status']} | {', '.join(r['reasons']) or '—'} | {_pct(r.get('p6'))} | {_pct(r.get('p12'))} |")
    lines += ['', f"Scored frame: {len(rows)} shares. Current source date: {summary['source_date']}.",
              'A missing name cannot be reassessed from this snapshot. No 50–100% forecast is inferred from this 2× model.', '']
    (out / 'latest.md').write_text('\n'.join(lines))
    return {'source_date': summary['source_date'], 'ranked_count': len(ranked),
            'prior_cohort_size': len(previous), 'status': 'RESEARCH_ONLY_ACCURACY_UNVALIDATED'}
