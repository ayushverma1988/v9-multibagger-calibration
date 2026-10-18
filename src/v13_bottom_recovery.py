"""Shadow research watchlist for post-drawdown bases and early recovery.

These declared thresholds are discovery preferences, not trained forecasts.
Missing price history or company evidence is never silently treated as a pass.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import v13_model as model
from v13_financial_evidence import REQUIRED, financial_quality
from v13_price_policy import valid_price

VERSION = 'V13_BOTTOM_RECOVERY_SHADOW_20261018'


def select(scored):
    d = scored.copy()
    def col(name):
        return pd.to_numeric(d.get(name, pd.Series(np.nan, index=d.index)), errors='coerce')

    financial = pd.DataFrame({k: col(k) for k in REQUIRED}, index=d.index)
    bad_financial = pd.Series([financial_quality(r)['status'] == 'KNOWN_FINANCIAL_FAILURE'
                               for r in financial.to_dict('records')], index=d.index)
    owner_bad = col('owner_PAT_yoy').lt(0) | col('owner_PAT_margin').le(0)
    known_owner_failure = d.get('current_owner_profit_status', pd.Series('', index=d.index)).eq(
        'UNKNOWN_OWNER_PLUS_MINORITY_DOES_NOT_RECONCILE_TOTAL_PAT')
    probability_ok = (col('p6').between(0, 1) & col('p12').between(0, 1) &
        col('loss12').between(0, model.RISK_CEILING) & col('joint12').between(0, 1) &
        col('p6').le(col('p12') + 1e-12) & col('joint12').le(col('p12') + 1e-12) &
        (col('joint12') + col('loss12')).le(1 + 1e-12))
    observed_market = valid_price(d.close) & d.get('feature_eligible', pd.Series(False, index=d.index)).fillna(False).astype(bool)
    market = observed_market & probability_ok & ~bad_financial & ~owner_bad & ~known_owner_failure
    # A bottom must be observed, not inferred from an old screenshot or a
    # missing 252-session history. A 200 DMA or RSI >70 is not mandatory.
    base = (col('rd_peak_drawdown252').between(-.80, -.25) &
        col('rd_base_range30').between(0, .27) &
        col('rd_base_return30').between(-.18, .20) &
        col('rd_low_distance30').between(0, .45))
    volume = col('rd_recent_volume_burst5').ge(1.5)
    turning = col('rd_recovery_turn').eq(1) | col('rd_return20').gt(0)
    early = base & volume & turning
    d['bottom_recovery_state'] = np.select(
        [~col('rd_peak_drawdown252').notna() | ~col('rd_base_range30').notna(),
         early, base],
        ['UNKNOWN_HISTORY', 'EARLY_RECOVERY', 'BASE_WATCH'], default='NO_OBSERVED_BASE')
    adverse = (col('hd_adverse_project_90d').gt(0) | col('hd_dilution_90d').gt(0) |
        col('current_adverse_project_documents').gt(0))
    catalyst = (col('hd_order_90d').gt(0) | col('hd_capacity_90d').gt(0) |
        col('hd_commissioning_language_90d').gt(0) | col('hd_approval_language_90d').gt(0) |
        col('order_disclosure_count90').gt(0) | col('capacity_disclosure_count90').gt(0) |
        col('commissioned_document_count90').gt(0))
    d['bottom_recovery_catalyst_status'] = np.where(catalyst,
        'RECENT_DISCLOSURE_CUE_REVIEW_REQUIRED', 'NO_OBSERVED_RECENT_POSITIVE_DISCLOSURE')
    d['bottom_recovery_eligible'] = market & base & ~adverse
    valid = d.bottom_recovery_eligible
    volume_credit = (col('rd_recent_volume_burst5') / 3).clip(0, 1).fillna(0)
    tightness = (1 - col('rd_base_range30') / .27).clip(0, 1).fillna(0)
    d['bottom_recovery_score'] = (
        .35 * col('p6').where(valid).rank(pct=True) +
        .20 * (-col('loss12')).where(valid).rank(pct=True) +
        .15 * volume_credit + .10 * tightness +
        .10 * early.astype(float) + .10 * catalyst.astype(float)).where(valid)
    pool = d[valid].sort_values(['bottom_recovery_score', 'isin'], ascending=[False, True]).head(20)
    d['bottom_recovery_rank'] = d['isin'].map({v: i + 1 for i, v in enumerate(pool['isin'])})
    return d, {'version': VERSION, 'source_date': str(pd.Timestamp(d.date.max()).date()),
        'scored_rows': len(d), 'observed_bases': int(base.sum()),
        'eligible': int(valid.sum()), 'early_recoveries': int(early[valid].sum()),
        'selected': len(pool), 'unknown_history': int(d.bottom_recovery_state.eq('UNKNOWN_HISTORY').sum()),
        'weights_are_probabilities': False, 'frozen_probabilities_recalibrated': False,
        'promotion_approved': False}


def export(attached, root, replay_receipt):
    d, audit = select(attached)
    cols = ['date', 'symbol', 'isin', 'close', 'p6', 'p12', 'loss12',
        'bottom_recovery_state', 'bottom_recovery_catalyst_status',
        'rd_peak_drawdown252', 'rd_base_range30', 'rd_recent_volume_burst5',
        'rsi14', 'distance_ma200', 'bottom_recovery_score', 'bottom_recovery_rank']
    audit['top20'] = json.loads(d[d.bottom_recovery_rank.notna()].sort_values(
        'bottom_recovery_rank')[cols].to_json(orient='records', date_format='iso'))
    audit['prediction_replay'] = replay_receipt
    path = Path(root) / 'bottom_recovery'; path.mkdir(exist_ok=True)
    (path / 'latest.json').write_text(json.dumps(audit, indent=2))
    return {k: v for k, v in audit.items() if k not in ('top20', 'prediction_replay')}
