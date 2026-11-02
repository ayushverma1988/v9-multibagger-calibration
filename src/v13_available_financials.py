"""Source-dated available-financial research eligibility, never a head change.

Missing facts, undefined arithmetic, observed weakness and a valid N/A are
different states. Raw facts/ratios, strict evidence bars, registered forecasts
and calibrated probabilities are never filled or rewritten by this overlay.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
from urllib.parse import urlsplit

import numpy as np
import pandas as pd

from v12_public_history import bounds
from v12_5_features import (features, number, owner_profit_status, pick,
                           preferred_reporting_basis)
from v13_financial_evidence import (REQUIRED, balancing_quarters,
                                    financial_quality, financial_readiness)
from v13_information_clock import cutoff
from v13_price_policy import valid_price
import v13_model as model

POLICY = 'V13_AVAILABLE_FINANCIALS_SECTOR_AWARE_RSI_GT70_20261103'
RSI_MIN = 70
WEIGHTS = {'joint12_percentile': .75, 'financial_evidence': .10,
           'complete_execution': .05, 'source_supported_net_2x': .05,
           'within_200dma_band': .05}
CORE = ('current_revenue', 'current_PAT', 'annual_CFO', 'balance_cash',
        'balance_equity', 'current_borrowings', 'noncurrent_borrowings')
QUALIFIED = {'AVAILABLE_SEVEN_CHECK_PASS_RESEARCH',
             'AVAILABLE_CURRENT_FINANCIALS_RESEARCH'}
TURNAROUND = 'TURNAROUND_RESEARCH'
DISCOVERY = 'INCOMPLETE_FINANCIAL_DISCOVERY'


def bound_source(row):
    """Require a canonical original identity/hash/link, not extra providers."""
    url = urlsplit(str(row.get('source_url', '')))
    return bool(re.fullmatch(r'[0-9a-f]{64}', str(row.get('source_sha256', '')))
                and url.scheme in ('http', 'https') and url.netloc
                and not url.username and not url.password)


def _latest(d, kind, date, max_age, period=None):
    q = pick(d, kind, period)
    return q if q and 0 <= (date-q['period']).days <= max_age else {}


def _ambiguous_latest(d, kind, date, max_age):
    q = d[d.period_kind.eq(kind) & (date-d.period).dt.days.between(0, max_age)]
    if q.empty:
        return False
    q = q[q.period.eq(q.period.max())]
    return int(q.available.eq(q.available.max()).sum()) > 1


def assess(known, date, sector='UNKNOWN'):
    """One issuer, one cutoff-filtered basis. Never infer owner PAT or EPS."""
    date = pd.Timestamp(date)
    if sector=='UNKNOWN' and not known.empty and 'source_url' in known:
        if known.source_url.fillna('').map(lambda u:urlsplit(str(u)).hostname in {'nsearchives.nseindia.com','www.nseindia.com'} and 'INTEGRATED_FILING_BANKING' in str(u).upper()).any():
            sector='BANK'
    basis = preferred_reporting_basis(known, date)
    d = known[known.reporting_mode.eq(basis)] if not known.empty else known
    f, _ = features(known, date) if not known.empty else ({}, [])
    quality = financial_quality(f)
    ready = financial_readiness(known, date, f)
    q = _latest(d, 'quarter', date, 190) if not d.empty else {}
    a = _latest(d, 'annual', date, 450) if not d.empty else {}
    b = _latest(d, 'instant', date, 450) if not d.empty else {}
    old = pick(d, 'quarter', q['period']-pd.DateOffset(years=1)) if q else {}
    prev = pick(d, 'quarter', q['period']-pd.offsets.MonthEnd(3)) if q else {}
    annual_balance = pick(d, 'instant', a['period']) if a else {}
    values = {'current_revenue': number(q, 'revenue'),
              'current_PAT': number(q, 'pat'), 'annual_CFO': number(a, 'cfo'),
              'balance_cash': number(b, 'cash'), 'balance_equity': number(b, 'equity'),
              'current_borrowings': number(b, 'borrowings_current'),
              'noncurrent_borrowings': number(b, 'borrowings_noncurrent'),
              'prior_year_quarter_PAT': number(old, 'pat'),
              'annual_PAT': number(a, 'pat'), 'annual_finance_cost': number(a, 'finance_cost')}
    core_missing = [k for k in CORE if not np.isfinite(values[k])]
    failures = []
    for key in ('current_revenue', 'current_PAT', 'balance_equity'):
        if np.isfinite(values[key]) and values[key] <= 0:
            failures.append('OBSERVED_NONPOSITIVE_'+key.upper())
    for key in (('balance_cash',) if sector in {'BANK','INSURER','OTHER_FINANCIAL'} else ('annual_CFO','balance_cash')):
        if np.isfinite(values[key]) and values[key] < 0:
            failures.append('OBSERVED_NEGATIVE_'+key.upper())
    if sector not in {'BANK','INSURER','OTHER_FINANCIAL'} and f.get('dated_debt_equity', np.nan) > 1.5:
        failures.append('OBSERVED_DEBT_EQUITY_ABOVE_1_5')
    if sector not in {'BANK','INSURER','OTHER_FINANCIAL'} and f.get('annual_EBIT_proxy_interest_coverage', np.nan) < 1.5:
        failures.append('OBSERVED_INTEREST_COVER_BELOW_1_5')
    owner_status = owner_profit_status(q)
    conflict = owner_status == 'UNKNOWN_OWNER_PLUS_MINORITY_DOES_NOT_RECONCILE_TOTAL_PAT'
    if owner_status.startswith('ORIGINAL_') and number(q, 'pat_owners') <= 0:
        failures.append('OBSERVED_NONPOSITIVE_RECONCILED_OWNER_PAT')
    ambiguous = any(_ambiguous_latest(d, kind, date, age)
                    for kind, age in [('quarter', 190), ('annual', 450), ('instant', 450)]) if not d.empty else False
    invalid = any(np.isfinite(values[k]) and values[k] < 0 for k in
                  ('current_borrowings', 'noncurrent_borrowings', 'annual_finance_cost'))
    conflict = conflict or ambiguous or invalid
    zero_interest = (values['annual_finance_cost'] == 0 and number(a, 'pbt') >= 0
                     and values['current_borrowings'] == 0 and values['noncurrent_borrowings'] == 0
                     and number(annual_balance, 'borrowings_current') == 0
                     and number(annual_balance, 'borrowings_noncurrent') == 0)
    turnaround = (values['current_PAT'] > 0 and
                  (values['prior_year_quarter_PAT'] <= 0 or values['annual_PAT'] <= 0))
    states = {}
    for key in REQUIRED:
        value = f.get(key, np.nan)
        if np.isfinite(value):
            state = 'OBSERVED_WEAKNESS' if key in quality['failed_checks'] else 'OBSERVED_PASS'
        elif key == 'annual_EBIT_proxy_interest_coverage' and zero_interest:
            state = 'NOT_APPLICABLE_ZERO_INTEREST_AND_NO_BORROWINGS'
        elif key == 'quarter_pat_yoy' and values['current_PAT'] > 0 and values['prior_year_quarter_PAT'] <= 0:
            state = 'TURNAROUND_COMPARISON_NOT_PERCENT_GROWTH'
        elif ready['unknown_reasons'].get(key) == 'OBSERVED_NONPOSITIVE_DENOMINATOR_NOT_MISSING_SOURCE':
            state = 'OBSERVED_NONPOSITIVE_BASE_REQUIRES_INTERPRETATION'
        else:
            state = 'MISSING_OR_AMBIGUOUS_INPUT'
        states[key] = state
    interest_ready = np.isfinite(f.get('annual_EBIT_proxy_interest_coverage', np.nan)) or zero_interest
    sector_states={};sector_score=None
    if sector in {'BANK','INSURER','OTHER_FINANCIAL'}:
        tests={'bank_roa':lambda x:x>0,'bank_capital_buffer_pp':lambda x:x>=0,
          'bank_net_npa_change_pp':lambda x:x<=0} if sector=='BANK' else {
          'insurer_solvency_buffer_pp':lambda x:x>=0,'insurer_underwriting_or_vnb_margin':lambda x:x>0} if sector=='INSURER' else {}
        core_missing=[k for k in ('current_revenue','current_PAT','balance_equity') if not np.isfinite(values[k])]
        for key,test in tests.items():
            value=number(q,key);values[key]=value
            sector_states[key]='MISSING_OR_AMBIGUOUS_INPUT' if not np.isfinite(value) else 'OBSERVED_PASS' if test(value) else 'OBSERVED_WEAKNESS'
            if not np.isfinite(value):core_missing.append(key)
            elif not test(value):failures.append('OBSERVED_SECTOR_WEAKNESS_'+key.upper())
        if not tests:core_missing.append('dated_financial_sector_safety_rules')
        for key in ('annual_CFO_PAT','dated_debt_equity','annual_EBIT_proxy_interest_coverage'):
            states[key]='NOT_APPLICABLE_INDUSTRIAL_RATIO_FOR_FINANCIAL_SECTOR'
        interest_ready=True
        sector_score=(sum(v=='OBSERVED_PASS' for v in sector_states.values())+sum(np.isfinite(values[k]) and values[k]>0 for k in ('current_revenue','current_PAT','balance_equity')))/(len(tests)+3) if tests else 0.
    if conflict:
        tier = 'QUARANTINED_SOURCE_CONFLICT'
    elif failures:
        tier = 'OBSERVED_CURRENT_FINANCIAL_WEAKNESS'
    elif core_missing or not interest_ready:
        tier = DISCOVERY
    elif turnaround:
        tier = TURNAROUND
    elif quality['quality_pass'] and sector not in {'BANK','INSURER','OTHER_FINANCIAL'}:
        tier = 'AVAILABLE_SEVEN_CHECK_PASS_RESEARCH'
    else:
        tier = 'AVAILABLE_CURRENT_FINANCIALS_RESEARCH'
    confidence = ('LIMITED_MISSING_ESSENTIAL_INPUTS' if tier == DISCOVERY else
                  'LIMITED_TURNAROUND_HISTORY' if tier == TURNAROUND else
                  'SOURCE_CONFLICT_UNRANKED' if conflict else
                  'OBSERVED_WEAKNESS_UNRANKED' if failures else
                  'CURRENT_FACTS_OBSERVED_ACCURACY_UNVALIDATED')
    sources = []
    for role, item in [('current_quarter', q), ('year_ago_quarter', old),
                       ('previous_quarter', prev), ('annual', a),
                       ('current_balance', b), ('annual_matched_balance', annual_balance)]:
        if item:
            sources.append({'role': role, **{k: item.get(k) for k in
                ('period_end', 'period_kind', 'reporting_mode', 'available_at_utc',
                 'source_url', 'source_sha256', 'derived_quarter_method',
                 'input_source_sha256', 'input_source_urls')}})
    # Differences across a loss base are monetary changes, never % PAT growth.
    alternatives = {'PAT_change_INR_year_on_year': values['current_PAT']-values['prior_year_quarter_PAT'],
                    'sales_qoq': values['current_revenue']/number(prev, 'revenue')-1
                        if number(prev, 'revenue') > 0 else np.nan,
                    'PAT_change_INR_quarter_on_quarter': values['current_PAT']-number(prev, 'pat')}
    return {'tier': tier, 'evidence_confidence': confidence,
            'missing_financial_risk_increment': None,
            'missing_financial_risk_increment_is_not_calibrated': True,
            'preferred_reporting_basis': basis,
            'strict_financial_complete': quality['complete'],
            'strict_financial_quality_pass': quality['quality_pass'],
            'strict_failed_checks': quality['failed_checks'],
            'core_missing_fields': core_missing,
            'current_safety_failures': failures, 'owner_profit_status': owner_status,
            'source_conflict': conflict, 'check_states': states,
            'genuine_download_gap_fields': [k for k, v in states.items() if v == 'MISSING_OR_AMBIGUOUS_INPUT'],
            'raw_required_facts_complete': ready['underlying_required_facts_complete'],
            'zero_interest_no_borrowings_NA': bool(zero_interest),
            'observed_turnaround': bool(turnaround), 'raw_observations': values,
            'alternatives': alternatives, 'sources': sources,
            'annual_loss_to_current_quarter_profit_is_not_same_period_growth': True,
            'QoQ_is_not_seasonally_adjusted': True,
            'owner_PAT_is_not_recurring_EPS': True,
            'single_source_allowed': True, 'sector_kind':sector, 'sector_check_states':sector_states,
            'sector_evidence_score':sector_score}


def enrich(scored, wide):
    """Append an interpretation overlay; retain every original value exactly."""
    d = scored.copy()
    if d['isin'].duplicated().any() or not d.index.is_unique:
        raise ValueError('Available-financial research needs unique current identities')
    groups = {} if wide.empty else dict(tuple(wide.groupby('security_isin')))
    rows = []
    for _, row in d.iterrows():
        date = pd.Timestamp(row.date)
        if pd.isna(row.get('information_cutoff_utc')):
            raise ValueError('Missing actual information cutoff cannot be retrofitted')
        cut = cutoff(row)
        low, _ = bounds(str(date.date()))
        known = groups.get(row['isin'], pd.DataFrame())
        if not known.empty:
            known = known[known.available.le(cut) & known.period.between(low, date)].copy()
            known = known[known.apply(bound_source, axis=1)]
            if not known.empty:
                # This is a current policy review at this explicit cutoff,
                # not a reconstruction of all historical filing vintages.
                # Retain tied latest versions as ambiguous; never average.
                # Bound quarter arithmetic to the versions actually usable
                # now, rather than deriving every past version combination.
                latest = known.groupby(['reporting_mode', 'period_kind', 'period'])['available'].transform('max')
                known = balancing_quarters(known[known.available.eq(latest)])
        assessment = assess(known, date, row.get('available_sector_kind', 'UNKNOWN'))
        assessment.update(isin=row['isin'], symbol=row['symbol'], source_asof=str(date.date()),
                          information_cutoff_utc=cut.isoformat())
        rows.append(assessment)
    overlay = pd.DataFrame(index=d.index)
    overlay['available_financial_tier'] = [r['tier'] for r in rows]
    overlay['available_financial_core_observed_fraction'] = [1-len(r['core_missing_fields'])/len(CORE) for r in rows]
    overlay['available_financial_evidence_score'] = [r['sector_evidence_score'] if r.get('sector_evidence_score') is not None else sum(v in ('OBSERVED_PASS',
        'NOT_APPLICABLE_ZERO_INTEREST_AND_NO_BORROWINGS') for v in r['check_states'].values())/len(REQUIRED) for r in rows]
    overlay['available_financial_turnaround'] = [r['observed_turnaround'] for r in rows]
    overlay['available_financial_zero_interest_NA'] = [r['zero_interest_no_borrowings_NA'] for r in rows]
    overlay['available_financial_unknown_count'] = [len(r['genuine_download_gap_fields']) for r in rows]
    overlap = set(overlay) & set(d)
    if overlap:
        raise ValueError('Financial interpretation overlay cannot overwrite existing columns')
    return pd.concat([d, overlay], axis=1), rows


def screen(scored, wide):
    d, rows = enrich(scored, wide)
    def col(key):
        return pd.to_numeric(d.get(key, pd.Series(np.nan, index=d.index)), errors='coerce')
    market = valid_price(col('close')) & col('distance_ma50').gt(0) & col('log_ret63').le(np.log(2.5))
    market &= col('rsi14').gt(RSI_MIN)
    identity = d.get('source_status', pd.Series('', index=d.index)).eq('VALID_POSITIVE_VOLUME_SESSION')
    identity &= d.get('official_source_sha256', pd.Series('', index=d.index)).astype(str).str.fullmatch(r'[0-9a-f]{64}')
    identity &= d.get('symbol_official', pd.Series('', index=d.index)).eq(d['symbol'])
    probability_ok = (col('p6').between(0, 1) & col('p12').between(0, 1)
        & col('joint12').between(0, 1) & col('loss12').between(0, model.RISK_CEILING)
        & col('p6').le(col('p12')+1e-12) & col('joint12').le(col('p12')+1e-12)
        & (col('joint12')+col('loss12')).le(1+1e-12))
    permitted = d.available_financial_tier.isin(QUALIFIED | {TURNAROUND, DISCOVERY})
    eligible = market & identity & probability_ok & permitted
    execution = d.get('evidence_status', pd.Series('', index=d.index)).eq('FINANCIAL_AND_EXECUTION_SUPPORTED')
    valuation = (d.get('earnings_scenario_status', pd.Series('', index=d.index)).eq('SCENARIO_NOT_PROBABILITY')
                 & col('scenario_net_multiple').ge(2))
    dma = col('distance_ma200'); within = np.isfinite(dma) & dma.abs().le(.10+1e-12)
    components = {'joint12_percentile': col('joint12').where(eligible).rank(pct=True),
                  'financial_evidence': d.available_financial_evidence_score,
                  'complete_execution': execution, 'source_supported_net_2x': valuation,
                  'within_200dma_band': within}
    d['available_research_eligible'] = eligible
    d['available_research_score'] = sum(WEIGHTS[k]*v for k, v in components.items()).where(eligible)
    d['available_research_selected'] = False
    d['available_research_rank'] = np.nan
    d['available_research_group'] = np.where(d.available_financial_tier.isin(QUALIFIED), 'available_current_financials',
        np.where(d.available_financial_tier.eq(TURNAROUND), 'turnaround', 'incomplete_discovery'))
    for group in ('available_current_financials', 'turnaround', 'incomplete_discovery'):
        pool = d[eligible & d.available_research_group.eq(group)].sort_values(
            ['available_research_score', 'joint12', 'isin'], ascending=[False, False, True]).head(model.TOP_K)
        d.loc[pool.index, 'available_research_selected'] = True
        d.loc[pool.index, 'available_research_rank'] = np.arange(1, len(pool)+1)
    for item, (_, row) in zip(rows, d.iterrows()):
        item.update(eligible=bool(row.available_research_eligible),
                    selected=bool(row.available_research_selected), group=row.available_research_group,
                    rank=row.available_research_rank, research_score=row.available_research_score,
                    exclusion_reasons=[name for name, mask in
                        [('MARKET_OR_RSI_GT70_REQUIRED', market), ('OFFICIAL_CURRENT_IDENTITY_REQUIRED', identity),
                         ('VALID_FROZEN_PROBABILITIES_AND_RISK_REQUIRED', probability_ok),
                         ('OBSERVED_WEAKNESS_OR_SOURCE_CONFLICT', permitted)] if not bool(mask.loc[row.name])],
                    frozen_replay_probabilities={k: row[k] for k in ('p6', 'p12', 'loss12', 'joint12')})
    from v13_research_selection import screen as legacy_screen
    legacy, _ = legacy_screen(scored)
    reference = legacy.research_eligible & col('rsi14').gt(RSI_MIN)
    states = Counter(v for r in rows for v in r['check_states'].values())
    return d, rows, {'policy': POLICY, 'rows': len(d), 'RSI_rule': 'strict >70; simple rolling 14 sessions',
        'price_band': None, 'DMA_band_optional': True, 'single_source_allowed': True,
        'supplementary_financial_catalyst_valuation_inputs_optional': True,
        'tiers': dict(Counter(r['tier'] for r in rows)), 'check_states': dict(states),
        'classified_check_positions': sum(states.values()), 'total_required_check_positions': len(d)*len(REQUIRED),
        'classification_is_not_fact_completeness_or_accuracy': True,
        'eligible': int(eligible.sum()), 'selected': int(d.available_research_selected.sum()),
        'eligible_by_group': d.loc[eligible, 'available_research_group'].value_counts().to_dict(),
        'selected_by_group': d.loc[d.available_research_selected, 'available_research_group'].value_counts().to_dict(),
        'legacy_eligible_at_same_RSI_gt70': int(reference.sum()),
        'newly_eligible_at_same_RSI_gt70': d.loc[eligible & ~reference, 'symbol'].tolist(),
        'removed_at_same_RSI_gt70': d.loc[~eligible & reference, 'symbol'].tolist(),
        'ranking_weights': WEIGHTS.copy(), 'ranking_score_is_probability': False,
        'weights_fitted_to_outcomes': False, 'accuracy_approved': False,
        'strict_financial_bars_changed': False, 'formal_forecasts_changed': False,
        'existing_model_missing_value_handling': 'TRAIN_ONLY_MEDIAN_PLUS_EXPLICIT_MISSING_INDICATOR_FOR_EVERY_FEATURE',
        'new_head_trained_or_promoted': False}


def export(scored, wide, root, replay_receipt):
    """Immutable policy review including unselected names, linked to exact replay."""
    if not replay_receipt.get('replay_verified') or replay_receipt.get('rows') != len(scored):
        raise ValueError('Available financial selection needs matching exact replay')
    root = Path(root)
    replay_dir = root/'prediction_replay'/replay_receipt['input_parquet_sha256']
    receipt_path = replay_dir/('champion_'+replay_receipt['receipt_sha256']+'.json')
    raw = receipt_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != replay_receipt['receipt_sha256']:
        raise ValueError('Exact replay receipt hash mismatch')
    recorded = json.loads(raw)
    if recorded['model_sha256'] != replay_receipt['model_sha256']:
        raise ValueError('Exact replay head mismatch')
    input_raw = (replay_dir/'inputs.parquet').read_bytes()
    if hashlib.sha256(input_raw).hexdigest() != replay_receipt['input_parquet_sha256']:
        raise ValueError('Exact replay input bytes mismatch')
    original = pd.read_parquet(io.BytesIO(input_raw))
    try:
        pd.testing.assert_frame_equal(original.sort_values('isin').reset_index(drop=True),
            scored[original.columns].sort_values('isin').reset_index(drop=True),
            check_dtype=False, check_exact=True)
    except (AssertionError, KeyError) as exc:
        raise ValueError('Available financial policy must preserve exact replay inputs') from exc
    if set(recorded['isins']) != set(scored['isin']) or len(recorded['isins']) != len(scored):
        raise ValueError('Exact replay cohort mismatch')
    ordered = scored.set_index('isin').loc[recorded['isins']]
    for key in ('p6', 'p12', 'loss12', 'joint12'):
        if not np.allclose(ordered[key], recorded['probabilities'][key], rtol=0, atol=1e-12, equal_nan=False):
            raise ValueError('Available financial policy must preserve exact replay probabilities')
    d, rows, audit = screen(scored, wide)
    now = datetime.now(timezone.utc).isoformat()
    obj = {**audit, 'source_asof': str(pd.Timestamp(d.date.max()).date()),
           'reviewed_at_utc': now, 'prediction_replay': replay_receipt,
           'scope': 'RESEARCH_ELIGIBILITY_REVIEW_NOT_NEW_REGISTERED_MODEL_FORECAST',
           'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           'all_names': rows, 'unselected_included': True,
           'later_policy_review_never_backdated_into_frozen_forecasts': True}
    # Dataframe serialization emits JSON null for unknown values, not zero.
    obj['all_names'] = json.loads(pd.DataFrame(rows).to_json(orient='records', date_format='iso', double_precision=15))
    raw = json.dumps(obj, sort_keys=True, indent=2, allow_nan=False).encode()
    key = hashlib.sha256(raw).hexdigest()
    out = root/'available_financial_research';out.mkdir(exist_ok=True)
    path = out/(key+'.json')
    if path.exists() and path.read_bytes() != raw:
        raise ValueError('Immutable available-financial review changed')
    if not path.exists():
        path.write_bytes(raw)
    selected = [r for r in obj['all_names'] if r['selected']]
    latest = {**{k: v for k, v in obj.items() if k != 'all_names'},
              'snapshot_sha256': key, 'complete_all_names_snapshot': path.name, 'records': selected}
    from v13_capture_policy import atomic_json
    atomic_json(out/'latest.json', latest)
    return {**audit, 'snapshot_sha256': key, 'source_asof': obj['source_asof'], 'reviewed_at_utc': now}


def review_saved(root, output=None):
    """Review an existing exact replay without loading/executing model weights."""
    import shutil
    root = Path(root);target = Path(output) if output is not None else root
    receipt = json.loads((root/'latest_run.json').read_text())['exact_prediction_replay']['champion']
    folder = Path('prediction_replay')/receipt['input_parquet_sha256']
    name = 'champion_'+receipt['receipt_sha256']+'.json'
    raw = (root/folder/name).read_bytes()
    if hashlib.sha256(raw).hexdigest() != receipt['receipt_sha256']:
        raise ValueError('Saved replay receipt mismatch')
    original = (root/folder/'inputs.parquet').read_bytes()
    if hashlib.sha256(original).hexdigest() != receipt['input_parquet_sha256']:
        raise ValueError('Saved replay input mismatch')
    frame = pd.read_parquet(io.BytesIO(original));recorded = json.loads(raw)
    if frame['isin'].tolist() != recorded['isins']:
        raise ValueError('Saved replay input order mismatch')
    for key in ('p6', 'p12', 'loss12', 'joint12'):
        frame[key] = recorded['probabilities'][key]
    if target.resolve() != root.resolve():
        (target/folder).mkdir(parents=True, exist_ok=True)
        for item in (name, 'inputs.parquet'):
            shutil.copyfile(root/folder/item, target/folder/item)
    return export(frame, pd.read_parquet(root/'financial_wide.parquet'), target, receipt)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', required=True)
    parser.add_argument('--output')
    args = parser.parse_args()
    print(json.dumps(review_saved(args.state, args.output), indent=2, allow_nan=False))
