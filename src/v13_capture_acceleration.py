"""Source-bound capture priorities and fixed-cohort recovery measurements.

This module never reads selections, probabilities, model heads or outcomes.
An undefined ratio with an observed nonpositive base is not a download gap.
"""
from collections import Counter
from datetime import datetime, timezone
import pandas as pd

from v12_public_history import bounds
from v12_5_features import features
from v13_financial_evidence import financial_quality, financial_readiness
from v13_information_clock import research_cutoffs, cutoff

NONPOSITIVE = 'OBSERVED_NONPOSITIVE_DENOMINATOR_NOT_MISSING_SOURCE'


def financial_profile(wide, master, symbols, asof):
    """Classify the same unranked public price cohort before and after capture."""
    day = pd.Timestamp(asof)
    low, _ = bounds(asof)
    cut = cutoff(research_cutoffs(pd.DataFrame({'date': [day]}),
                                 datetime.now(timezone.utc)).iloc[0])
    identities = dict(zip(master.nse_current_symbol, master['isin']))
    groups = {} if wide.empty else dict(tuple(wide.groupby('security_isin')))
    rows = []
    for symbol in sorted(set(symbols)):
        if symbol not in identities:
            continue
        isin = identities[symbol]
        known = groups.get(isin, pd.DataFrame())
        if not known.empty:
            known = known[known.available.le(cut) & known.period.between(low, day)]
        f, _ = features(known, day) if not known.empty else ({}, [])
        quality = financial_quality(f)
        ready = financial_readiness(known, day, f)
        reasons = ready['unknown_reasons']
        missing = [k for k, reason in reasons.items() if reason != NONPOSITIVE]
        nonpositive = [k for k, reason in reasons.items() if reason == NONPOSITIVE]
        if quality['complete']:
            status = 'SEVEN_CHECK_COMPLETE'
        elif missing:
            status = ('MISSING_FACTS_AND_NONPOSITIVE_BASE' if nonpositive
                      else 'MISSING_OR_AMBIGUOUS_FACTS')
        else:
            status = 'OBSERVED_NONPOSITIVE_BASE_ONLY'
        rows.append({'symbol': symbol, 'isin': isin, 'status': status,
            'missing_fields': missing, 'observed_nonpositive_fields': nonpositive,
            'unknown_reasons': reasons, 'failed_checks': quality['failed_checks'],
            'financial_complete': quality['complete'],
            'underlying_required_facts_complete': ready['underlying_required_facts_complete'],
            'preferred_reporting_basis': ready.get('preferred_reporting_basis'),
            'preferred_quarter_end': ready.get('preferred_quarter_end'),
            'priority': 0 if missing and not quality['failed_checks'] else 1 if missing else 2})
    counts = Counter(r['status'] for r in rows)
    return {'source_asof': asof, 'information_cutoff_utc': cut.isoformat(),
        'scope': 'UNRANKED_CURRENT_PUBLIC_PRICE_ELIGIBLE_IDENTITIES',
        'rows': rows, 'cohort_rows': len(rows), 'status_counts': dict(counts),
        'complete_rows': sum(r['financial_complete'] for r in rows),
        'raw_fact_complete_rows': sum(r['underlying_required_facts_complete'] for r in rows),
        'recoverable_missing_fact_rows': sum(bool(r['missing_fields']) for r in rows),
        'observed_nonpositive_ratios_are_not_missing_downloads': True}


def fixed_cohort_comparison(before, after):
    old = {(r['symbol'], r['isin']): r for r in before['rows']}
    new = {(r['symbol'], r['isin']): r for r in after['rows']}
    if set(old) != set(new) or before['source_asof'] != after['source_asof']:
        raise ValueError('Capture comparison requires identical dated public identities')
    gained = [s for s in sorted(old) if not old[s]['financial_complete'] and new[s]['financial_complete']]
    lost = [s for s in sorted(old) if old[s]['financial_complete'] and not new[s]['financial_complete']]
    fields_before = sum(len(r['missing_fields']) for r in old.values())
    fields_after = sum(len(r['missing_fields']) for r in new.values())
    return {'source_asof': before['source_asof'], 'fixed_cohort_rows': len(old),
        'complete_before': before['complete_rows'], 'complete_after': after['complete_rows'],
        'raw_fact_complete_before': before['raw_fact_complete_rows'],
        'raw_fact_complete_after': after['raw_fact_complete_rows'],
        'missing_field_positions_before': fields_before,
        'missing_field_positions_after': fields_after,
        'newly_complete': [{'symbol': s, 'isin': i} for s, i in gained],
        'newly_incomplete': [{'symbol': s, 'isin': i} for s, i in lost],
        'same_cohort_not_expanding_denominator': True,
        'forecast_changes_or_prediction_accuracy_claimed': False}


def terminal_history_error(error):
    """Variants cannot repair a definite instrument/identity/absence response.

    Session omission and short/stale histories still use the existing variants.
    Access errors remain a global bounded-map stop, never a retry bypass.
    """
    text = str(error).lower()
    return any(x in text for x in (
        'wrong symbol, venue, currency or instrument',
        '404 client error',
        'history identity changed'))
