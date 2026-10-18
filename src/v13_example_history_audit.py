"""Dated coverage and gate audit for user-supplied chart examples.

This diagnostic never supplies training labels or changes ranking. The stored
price history is a rolling four-year window, not a listing-to-date archive.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from v12_dated_source_collection import sha
from v12_public_history import bounds
from v13_features import daily_features, MIN_TURNOVER
from v13_research_selection import screen
from v13_redevelopment import daily as redevelopment_daily


EXAMPLES = (
    ('eMudhra', 'EMUDHRA', 'INE01QM01018'),
    ('Dynacons Systems & Solutions', 'DSSL', 'INE417B01040'),
    ('Corona Remedies', 'CORONA', 'INE02ZQ01018'),
    ('Paragon Fine and Speciality Chemical', 'PARAGON', 'INE0N4G01012'),
    ('Ashapuri Gold Ornament', 'AGOL', 'INE05FR01029'),
    ('Raj Rayon Industries', 'RAJRILTD', 'INE533D01032'),
    ('R Systems International', 'RSYSTEMS', 'INE411H01032'),
    ('Chemcrux Enterprises', 'CHEMCRUX', 'INE298W01016'),
)


def _number(value):
    try:
        value = float(value)
        return round(value, 6) if np.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def _date(value):
    return str(pd.Timestamp(value).date())


def audit(root, asof, scored=None, collection_report=None):
    root = Path(root)
    asof = pd.Timestamp(asof)
    low, _ = bounds(_date(asof))
    universe = pd.read_parquet(root / 'universe.parquet')
    attempt_path = root / 'history_attempts.json'
    attempts = json.loads(attempt_path.read_text()) if attempt_path.exists() else {}
    errors = {e['symbol']: e['error'] for e in (collection_report or {}).get('errors', [])}
    scored = scored if scored is not None else pd.DataFrame()
    screened, _ = screen(scored) if not scored.empty else (pd.DataFrame(), {})
    rows = []
    for name, symbol, isin in EXAMPLES:
        members = universe[universe['isin'].eq(isin)]
        row = {'company': name, 'requested_symbol': symbol, 'isin': isin,
               'source_asof': _date(asof), 'rolling_window_start': _date(low),
               'universe_member': len(members) == 1,
               'current_symbol': str(members.iloc[0].nse_current_symbol) if len(members) == 1 else None,
               'nse_series': str(members.iloc[0].get('nse_current_series')) if len(members) == 1 else None,
               'nse_listing_date': (_date(members.iloc[0].get('nse_listing_date'))
                                    if len(members) == 1 and pd.notna(members.iloc[0].get('nse_listing_date')) else None),
               'history_status': None, 'current_status': None}
        if len(members) != 1:
            row['history_status'] = 'NOT_IN_CURRENT_NSE_UNIVERSE'
            row['current_status'] = 'OUTSIDE_NSE_COLLECTION'
            rows.append(row)
            continue
        current_symbol = row['current_symbol']
        row['last_public_history_attempt_utc'] = attempts.get(current_symbol)
        row['current_run_public_history_error'] = errors.get(current_symbol)
        path = root / 'history' / 'securities' / (sha(current_symbol.encode()) + '.parquet')
        if not path.exists():
            row['history_status'] = 'NO_STORED_HISTORY_FOR_CURRENT_SYMBOL'
            row['current_status'] = 'NOT_SCORABLE_MISSING_HISTORY'
            rows.append(row)
            continue
        bars = pd.read_parquet(path)
        bars = bars[bars['isin'].eq(isin)].copy()
        bars.date = pd.to_datetime(bars.date)
        bars = bars[bars.date.ge(low) & bars.date.le(asof)].sort_values('date')
        if bars.empty:
            row['history_status'] = 'NO_EXACT_ISIN_BARS_IN_ROLLING_WINDOW'
            row['current_status'] = 'NOT_SCORABLE_MISSING_HISTORY'
            rows.append(row)
            continue
        if bars.date.duplicated().any():
            row['history_status'] = 'DUPLICATE_DATES_REVIEW_REQUIRED'
            row['current_status'] = 'NOT_SCORABLE_AMBIGUOUS_HISTORY'
            rows.append(row)
            continue
        d = daily_features(bars, _date(asof))
        if d.empty:
            row['history_status'] = 'NO_USABLE_POSITIVE_VOLUME_BARS'
            row['current_status'] = 'NOT_SCORABLE_INVALID_HISTORY'
            rows.append(row)
            continue
        pattern = d[['date']].merge(redevelopment_daily(bars)[[
            'date', 'rd_peak_drawdown756', 'rd_base_range30',
            'rd_base_return30', 'rd_low_distance30']],
            on='date', how='left', validate='one_to_one', sort=False)
        last = d.iloc[-1]
        price = pd.to_numeric(bars.adj_close, errors='coerce')
        peaks = price.cummax()
        drawdown = price / peaks - 1
        row.update(history_status='STORED_ROLLING_HISTORY', first_bar=_date(bars.date.min()),
                   last_bar=_date(bars.date.max()), observed_bars=len(bars),
                   first_adjusted_close=_number(price.iloc[0]),
                   last_adjusted_close=_number(price.iloc[-1]),
                   highest_adjusted_close=_number(price.max()),
                   lowest_adjusted_close=_number(price.min()),
                   largest_observed_peak_to_trough_drawdown=_number(drawdown.min()),
                   latest_drawdown_from_observed_peak=_number(drawdown.iloc[-1]),
                   last_feature_date=_date(last.date),
                   feature_eligible_dates=int(d.feature_eligible.fillna(False).sum()),
                   first_feature_eligible_date=(_date(d.loc[d.feature_eligible, 'date'].min())
                                                if d.feature_eligible.any() else None),
                   latest_feature_eligible=bool(last.feature_eligible),
                   feature_eligible_and_ma50_positive_sessions=int((d.feature_eligible & d.distance_ma50.gt(0)).sum()),
                   feature_eligible_and_rsi_above70_sessions=int((d.feature_eligible & d.rsi14.gt(70)).sum()),
                   feature_eligible_and_both_legacy_trend_rules_sessions=int((
                       d.feature_eligible & d.distance_ma50.gt(0) & d.rsi14.gt(70)).sum()),
                   latest_median_turnover63_INR=_number(last.get('median_turnover63')),
                   latest_gap_max63=_number(last.get('gap_max63')),
                   latest_jump_max63=_number(last.get('jump_max63')),
                   latest_observed_sessions35d=_number(last.get('observed_sessions35d')),
                   latest_rsi14=_number(last.get('rsi14')),
                   latest_distance_ma50=_number(last.get('distance_ma50')),
                   latest_distance_ma200=_number(last.get('distance_ma200')),
                   rolling_history_only=True,
                   corporate_actions_independently_verified=False)
        base = (pattern.rd_peak_drawdown756.between(-.80, -.25) &
                pattern.rd_base_range30.between(0, .27) &
                pattern.rd_base_return30.between(-.18, .20) &
                pattern.rd_low_distance30.between(0, .45))
        observed_base = base & d.feature_eligible
        row['observed_eligible_base_sessions'] = int(observed_base.sum())
        row['first_observed_eligible_base_date'] = (_date(d.loc[observed_base, 'date'].min())
                                                    if observed_base.any() else None)
        row['latest_peak_drawdown756'] = _number(pattern.rd_peak_drawdown756.iloc[-1])
        row['latest_base_range30'] = _number(pattern.rd_base_range30.iloc[-1])
        row['latest_feature_gate_failures'] = [k for k, fails in (
            ('TURNOVER_63_BELOW_25_LAKH', pd.notna(last.get('median_turnover63')) and last.median_turnover63 < MIN_TURNOVER),
            ('TURNOVER_20_BELOW_25_LAKH', pd.notna(last.get('median_turnover20')) and last.median_turnover20 < MIN_TURNOVER),
            ('GAP_OVER_10_SESSIONS', pd.notna(last.get('gap_max63')) and last.gap_max63 > 10),
            ('JUMP_OVER_40_PERCENT', pd.notna(last.get('jump_max63')) and last.jump_max63 > np.log(1.4)),
            ('FEWER_THAN_18_SESSIONS_IN_35_DAYS', pd.notna(last.get('observed_sessions35d')) and last.observed_sessions35d < 18),
            ('NO_HISTORY_WARMUP', pd.isna(last.get('history_start_used'))),
        ) if fails]
        if _date(last.date) != _date(asof):
            row['current_status'] = 'NO_SOURCE_DATE_BAR'
        elif not bool(last.feature_eligible):
            row['current_status'] = 'SOURCE_DATE_FEATURE_INELIGIBLE'
        else:
            selected = screened[screened['isin'].eq(isin)] if not screened.empty else pd.DataFrame()
            if len(selected) == 1:
                item = selected.iloc[0]
                row.update(current_status='SCORED', p6=_number(item.get('p6')),
                           p12=_number(item.get('p12')), loss12=_number(item.get('loss12')),
                           research_exclusion_reasons=json.loads(item.research_exclusion_reasons),
                           research_selected=bool(item.research_selected),
                           research_financial_tier=str(item.research_financial_tier),
                           formal_selected=bool(item.get('selected_verified', False) or item.get('selected_discovery', False)))
            else:
                row['current_status'] = 'FEATURE_ELIGIBLE_BUT_NOT_SCORED_CHECK_OFFICIAL_RECONCILIATION'
        rows.append(row)
    summary = {'source_asof': _date(asof), 'scope': 'USER_SUPPLIED_DIAGNOSTIC_NOT_TRAINING_OR_SELECTION',
               'rolling_window_start': _date(low), 'listing_to_date_history_not_claimed': True,
               'current_status_counts': pd.Series([r['current_status'] for r in rows]).value_counts().to_dict(),
               'records': rows}
    output = root / 'evidence' / 'example_history_audit.json'
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(summary, indent=2))
    return {k: v for k, v in summary.items() if k != 'records'}
