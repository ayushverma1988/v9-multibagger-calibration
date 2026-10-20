"""Universe-wide, pre-breakout research queue for 6–12 month upside ideas.

Ranks are declared heuristics, never 50%/2x probabilities. User examples are
diagnostics only: no symbol/ISIN enters a selection rule or training label.
"""
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from v13_price_policy import valid_price
from v12_dated_source_collection import sha

VERSION = 'V13_EARLY_POTENTIAL_SHADOW_20261018'
MAX_MARKET_CAP_CRORE = 50_000  # declared small/mid discovery scope


def select(scored):
    d = scored.copy()
    def col(name):
        return pd.to_numeric(d.get(name, pd.Series(np.nan, index=d.index)), errors='coerce')

    # Existing official reconciliation, liquidity and history-quality checks
    # live in feature_eligible. No nominal rupee-price, RSI or 200-DMA gate.
    market = (valid_price(d.close) &
        d.get('feature_eligible', pd.Series(False, index=d.index)).fillna(False).astype(bool))
    deep = (col('rd_peak_drawdown756').between(-.80, -.20) &
        col('rd_base_range30').between(.06, .45) &
        col('rd_base_return30').between(-.25, .25) &
        col('rd_low_distance30').between(0, .55))
    young = (col('rd_observed_sessions').between(126, 251) &
        col('rd_peak_drawdown126').between(-.40, -.05) &
        col('rd_base_range30').between(.06, .30) &
        col('rd_base_return30').between(-.20, .20))
    retest = (col('rd_peak_drawdown756').between(-.80, -.20) &
        col('rd_surge_retest_age').between(1, 45) &
        col('rd_surge_peak_retrace').between(-.35, -.08) &
        col('rd_surge_base_lift').between(.08, .90))
    pattern = deep | young | retest
    d['early_pattern'] = np.select([retest, young, deep],
        ['SURGE_PULLBACK', 'SHORT_HISTORY_BASE', 'DEEP_RESET_BASE'],
        default='NO_OBSERVED_SETUP')

    volume = col('rd_recent_volume_burst5').ge(1.5)
    sales = col('quarter_sales_yoy').gt(.05)
    owner = col('owner_PAT_yoy').gt(0) & col('owner_PAT_margin').gt(0)
    # A dated filing count is a cue to read the document, not proof of a
    # commissioned line, realised order margin, or recurring EPS growth.
    filing = (col('hd_order_90d').gt(0) | col('hd_capacity_90d').gt(0) |
        col('hd_commissioning_language_90d').gt(0) |
        col('hd_approval_language_90d').gt(0) |
        col('order_disclosure_count90').gt(0) |
        col('capacity_disclosure_count90').gt(0) |
        col('commissioned_document_count90').gt(0))
    adverse = (col('hd_adverse_project_90d').gt(0) |
        col('hd_dilution_90d').gt(0) |
        col('current_adverse_project_documents').gt(0))
    financial_failure = (d.get('financial_quality_status', pd.Series('', index=d.index))
        .eq('KNOWN_FINANCIAL_FAILURE') | col('owner_PAT_yoy').lt(0) |
        col('owner_PAT_margin').le(0) |
        d.get('current_owner_profit_status', pd.Series('', index=d.index)).eq(
            'UNKNOWN_OWNER_PLUS_MINORITY_DOES_NOT_RECONCILE_TOTAL_PAT'))
    chart = market & pattern
    market_cap = col('latest_issued_shares') * col('close') / 1e7
    d['early_market_cap_crore_proxy'] = market_cap
    in_scope = market_cap.between(0, MAX_MARKET_CAP_CRORE, inclusive='right')
    # Capture the chart hypothesis even when fundamentals disagree, but keep
    # it in an explicit conflict/review tier rather than a potential list.
    d['early_state'] = np.select(
        [~market, ~pattern, ~in_scope & market_cap.notna(),
         market_cap.isna(), adverse, financial_failure,
         (sales & owner) | (volume & (sales | filing))],
        ['NO_VALID_CURRENT_MARKET', 'NO_OBSERVED_SETUP', 'OUTSIDE_SIZE_SCOPE',
         'SIZE_UNKNOWN_REVIEW', 'ADVERSE_NEWS_REVIEW',
         'FINANCIAL_CONFLICT_REVIEW', 'EARLY_POTENTIAL'], default='CHART_WATCH')
    d['early_volume_cue'] = volume
    d['early_filing_cue'] = filing
    d['early_sales_growth_cue'] = sales
    d['early_owner_profit_cue'] = owner
    d['early_financial_conflict'] = financial_failure
    d['early_eligible'] = d.early_state.eq('EARLY_POTENTIAL')

    # Score only observed cues and price shape. Frozen 2x probabilities are
    # displayed for context, never used as 50% return probabilities/rank.
    compression = (1 - col('rd_base_range30') / .45).clip(0, 1).fillna(0)
    low_distance = (1 - col('rd_low_distance30') / .55).clip(0, 1).fillna(0)
    volume_strength = (col('rd_recent_volume_burst5') / 4).clip(0, 1).fillna(0)
    d['early_score'] = (
        .18 * compression + .12 * low_distance + .14 * volume_strength +
        .14 * sales.astype(float) + .12 * owner.astype(float) +
        .12 * filing.astype(float) + .10 * retest.astype(float) +
        .08 * young.astype(float)).where(chart)
    pool = d[d.early_eligible].sort_values(['early_score', 'isin'], ascending=[False, True])
    watch = d[d.early_state.eq('CHART_WATCH')].sort_values(
        ['early_score', 'isin'], ascending=[False, True])
    d['early_rank'] = d['isin'].map({v: i + 1 for i, v in enumerate(pool['isin'])})
    d['early_watch_rank'] = d['isin'].map({v: i + 1 for i, v in enumerate(watch['isin'])})
    summary = {'version': VERSION, 'source_date': str(pd.Timestamp(d.date.max()).date()),
        'scored_rows': len(d), 'observed_chart_setups': int(chart.sum()),
        'early_potentials': len(pool), 'chart_watch': len(watch),
        'financial_conflicts': int(d.early_state.eq('FINANCIAL_CONFLICT_REVIEW').sum()),
        'short_history_setups': int((chart & young).sum()),
        'size_ceiling_crore': MAX_MARKET_CAP_CRORE,
        'size_unknown_review': int(d.early_state.eq('SIZE_UNKNOWN_REVIEW').sum()),
        'return_target': 'UNTRAINED_50_TO_100_PERCENT_6_TO_12_MONTH_RESEARCH',
        'weights_are_probabilities': False, 'frozen_model_changed': False,
        'promotion_approved': False}
    return d, summary


def export(attached, root, replay_receipt):
    d, summary = select(attached)
    cols = ['date', 'symbol', 'isin', 'close', 'early_pattern', 'early_state',
        'early_score', 'early_rank', 'early_watch_rank', 'early_volume_cue',
        'early_filing_cue', 'early_sales_growth_cue', 'early_owner_profit_cue',
        'early_financial_conflict', 'early_market_cap_crore_proxy',
        'rd_peak_drawdown756', 'rd_peak_drawdown126',
        'rd_observed_sessions', 'rd_base_range30', 'rd_recent_volume_burst5',
        'quarter_sales_yoy', 'owner_PAT_yoy', 'financial_quality_status',
        'p6', 'p12', 'loss12']
    for c in cols:
        if c not in d:d[c] = np.nan
    queue = d[d.early_state.isin(['EARLY_POTENTIAL', 'CHART_WATCH',
        'FINANCIAL_CONFLICT_REVIEW', 'ADVERSE_NEWS_REVIEW',
        'SIZE_UNKNOWN_REVIEW'])].sort_values(
            ['early_state', 'early_score', 'isin'], ascending=[True, False, True])
    summary['candidates'] = json.loads(queue[cols].to_json(orient='records', date_format='iso'))
    summary['scored_states'] = json.loads(d[[
        'isin', 'early_pattern', 'early_state', 'early_rank',
        'early_watch_rank', 'early_score']].to_json(orient='records'))
    path = Path(root) / 'early_potential';path.mkdir(exist_ok=True)
    universe_path = Path(root) / 'universe.parquet'
    if universe_path.exists():
        universe = pd.read_parquet(universe_path)
        missing = universe[~universe['isin'].isin(d['isin'])].copy()
        attempts_path = Path(root) / 'history_attempts.json'
        attempts = json.loads(attempts_path.read_text()) if attempts_path.exists() else {}
        queue_rows=[]
        for r in missing.to_dict('records'):
            symbol=r['nse_current_symbol']
            has_history=(Path(root)/'history'/'securities'/(
                sha(symbol.encode())+'.parquet')).exists()
            queue_rows.append({'symbol':symbol, 'isin':r['isin'],
                'nse_series':r.get('nse_current_series'),
                'history_status':('STORED_BUT_NOT_SCORED' if has_history else
                    'NO_STORED_HISTORY'),
                'last_public_history_attempt_utc':attempts.get(symbol)})
        coverage={'source_date':summary['source_date'], 'public_nse_universe':len(universe),
            'unscored':len(queue_rows), 'unscored_are_not_potential_labels':True,
            'records':sorted(queue_rows,key=lambda x:(x['history_status'],x['symbol']))}
        (path/'coverage_queue.json').write_text(json.dumps(coverage,indent=2))
        summary['public_universe']=len(universe)
        summary['unscored_coverage_queue']=len(queue_rows)
    summary['prediction_replay'] = replay_receipt
    (path / 'latest.json').write_text(json.dumps(summary, indent=2))
    # Content-addressed, immutable prospective cohort: keep repeated refreshes
    # of the same market date distinct instead of overwriting earlier evidence.
    cohort = {'version': VERSION, 'source_date': summary['source_date'],
        'labels': ['hit1p5_6m', 'hit1p5_12m', 'hit2_6m', 'hit2_12m'],
        'outcomes_matured': False,
        'corporate_actions_independently_verified': False,
        'rows': json.loads(d[d.early_state.isin(['EARLY_POTENTIAL', 'CHART_WATCH'])][[
            'date', 'isin', 'symbol', 'close', 'early_state', 'early_pattern',
            'early_score', 'early_rank', 'early_watch_rank',
            'early_market_cap_crore_proxy']].sort_values(
                ['early_state', 'isin']).to_json(orient='records', date_format='iso'))}
    raw = json.dumps(cohort, sort_keys=True, separators=(',', ':')).encode()
    cohorts = path / 'cohorts';cohorts.mkdir(exist_ok=True)
    cohort_file = cohorts / (summary['source_date'] + '_' + hashlib.sha256(raw).hexdigest() + '.json')
    if not cohort_file.exists():cohort_file.write_bytes(raw)
    leaders = d[d.early_rank.between(1, 20)].sort_values('early_rank')
    lines = [f'# Early potential research queue — {summary["source_date"]}', '',
        'Untrained 6–12 month upside hypotheses. Rank is not a return probability.', '',
        '| Rank | Symbol | Price ₹ | Pattern | Volume | Sales | Owner profit | Filing cue |',
        '| ---: | --- | ---: | --- | --- | --- | --- | --- |']
    for r in leaders.itertuples():
        lines.append(f'| {int(r.early_rank)} | {r.symbol} | {r.close:.2f} | {r.early_pattern} | '
            f'{"Yes" if r.early_volume_cue else "No"} | '
            f'{"Yes" if r.early_sales_growth_cue else "Unknown/No"} | '
            f'{"Yes" if r.early_owner_profit_cue else "Unknown/No"} | '
            f'{"Review" if r.early_filing_cue else "None observed"} |')
    for pattern, title, limit in [('SURGE_PULLBACK', 'Surge pullbacks', 30),
                                  ('SHORT_HISTORY_BASE', 'Short-history bases', 20)]:
        subset=d[d.early_eligible & d.early_pattern.eq(pattern)].sort_values(
            ['early_score','isin'],ascending=[False,True]).head(limit)
        lines += ['',f'## {title} (separate pattern queue)','',
            '| Pattern rank | Symbol | Price ₹ | Score | Volume cue | Filing cue |',
            '| ---: | --- | ---: | ---: | --- | --- |']
        for i,r in enumerate(subset.itertuples(),1):
            lines.append(f'| {i} | {r.symbol} | {r.close:.2f} | {r.early_score:.3f} | '
                f'{"Yes" if r.early_volume_cue else "No"} | '
                f'{"Review" if r.early_filing_cue else "None observed"} |')
    lines += ['', f'{summary["chart_watch"]} more chart watches and '
        f'{summary["financial_conflicts"]} financial conflicts appear in latest.json.']
    (path / 'latest.md').write_text('\n'.join(lines) + '\n')
    result = {k: v for k, v in summary.items() if k not in (
        'candidates', 'scored_states', 'prediction_replay')}
    result['prospective_cohort'] = str(cohort_file.relative_to(root))
    return result
