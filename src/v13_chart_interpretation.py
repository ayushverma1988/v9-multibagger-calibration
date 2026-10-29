"""Causal chart diagnostics; experimental features, never doubling probabilities.

Daily and completed-week data only. Thresholds below are declared hypotheses,
not estimated success rates. Existing forecast values and selection are copied.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = 'V13_CHART_EVIDENCE_SHADOW_20261024'
FEATURES = [
    'chart_atr14', 'chart_atr_fraction', 'chart_bb_width20',
    'chart_bb_percentile126', 'chart_prior_squeeze5',
    'chart_resistance63', 'chart_volume_ratio20', 'chart_volume_dry10_50',
    'chart_gap_atr', 'chart_first_breakout', 'chart_breakout_age10',
    'chart_breakout_anchor', 'chart_retest_hold', 'chart_failed_breakout',
    'chart_daily_trend', 'chart_completed_week_trend',
    'chart_extension50_atr', 'chart_adjustment_jump63', 'chart_base_range63',
    'chart_history_close', 'chart_adjustment_factor',
]
SHAPE_FIELDS = ['chart_rounded_handle_proxy', 'chart_proxy_depth',
    'chart_proxy_handle_retrace', 'chart_proxy_bottom_dwell', 'chart_proxy_rim']
SETUPS = ['CONFIRMED_BREAKOUT', 'PULLBACK_HOLD', 'PRE_BREAKOUT_WATCH',
    'ROUNDED_HANDLE_WATCH', 'BREAKOUT_TREND_REVIEW']


def prepare(bars):
    d = bars.copy()
    required = ['date', 'isin', 'close', 'adj_close', 'high', 'low', 'volume']
    if not set(required).issubset(d):
        raise ValueError('Missing OHLCV/identity fields')
    d['date'] = pd.to_datetime(d.date)
    if d.empty:
        return d
    if d.date.isna().any() or d.date.duplicated().any() or d['isin'].nunique() != 1:
        raise ValueError('Ambiguous history date or issuer identity')
    for c in required[2:]:
        d[c] = pd.to_numeric(d[c], errors='coerce')
    numbers = d[required[2:]]
    if not np.isfinite(numbers.to_numpy()).all():
        raise ValueError('Non-finite OHLCV')
    if (d[['close', 'adj_close', 'high', 'low']] <= 0).any().any() or (d.volume < 0).any():
        raise ValueError('Non-positive price or negative volume')
    if (d.high < d.low).any() or (d.close > d.high + .011).any() or (d.close < d.low - .011).any():
        raise ValueError('Inconsistent OHLC')
    return d[d.volume.gt(0)].sort_values('date').reset_index(drop=True)


def wilder(values, n=14):
    """Exact arithmetic seed, followed by Wilder recursion; no warmup guesses."""
    a = np.asarray(values, dtype=float)
    out = np.full(len(a), np.nan)
    if len(a) >= n:
        out[n - 1] = a[:n].mean()
        for i in range(n, len(a)):
            out[i] = ((n - 1) * out[i - 1] + a[i]) / n
    return pd.Series(out, index=values.index)


def daily(bars):
    d = prepare(bars)
    if d.empty:
        for name in FEATURES:
            d[name] = pd.Series(dtype=float)
        return d
    p = d.adj_close
    factor = p / d.close
    hi, lo = d.high * factor, d.low * factor
    tr = pd.concat([hi - lo, (hi - p.shift()).abs(),
                    (lo - p.shift()).abs()], axis=1).max(axis=1)
    atr = wilder(tr)
    prior_atr = atr.shift().replace(0, np.nan)
    ma20 = p.rolling(20).mean()
    # Population standard deviation is a declared implementation convention.
    width = 4 * p.rolling(20).std(ddof=0) / ma20
    percentile = width.rolling(126).apply(
        lambda a: np.count_nonzero(a <= a[-1]) / len(a), raw=True)
    squeeze = percentile.shift().rolling(5).min().le(.20)
    squeeze = squeeze.astype(float).where(percentile.shift().rolling(5).count().eq(5))
    resistance = hi.shift().rolling(63).max()
    gap = (p - resistance) / prior_atr
    vol = d.volume / d.volume.shift().rolling(20).median().replace(0, np.nan)
    crossed = gap.gt(.25) & gap.shift().le(.25) & vol.ge(1.5)
    first = crossed & crossed.shift().rolling(20).sum().eq(0)
    known = gap.notna() & gap.shift().notna() & vol.notna()
    seq = pd.Series(np.arange(len(d)), index=d.index)
    age = seq - seq.where(first).ffill()
    recent = age.between(0, 10)
    anchor = resistance.where(first).ffill().where(recent)
    ma50, ma200 = p.rolling(50).mean(), p.rolling(200).mean()
    trend = p.gt(ma50) & ma50.gt(ma200) & ma200.gt(ma200.shift(20))
    weekly = pd.DataFrame({'date':d.date, 'p':p}).set_index('date').p.resample('W-FRI').last()
    w10, w40 = weekly.rolling(10).mean(), weekly.rolling(40).mean()
    wt = ((weekly > w10) & (w10 > w40) & (w40 > w40.shift(4))).astype(float)
    wt = wt.where(w40.shift(4).notna()).rename('chart_completed_week_trend')
    # Friday labels prevent Monday-Thursday bars from using the current week.
    closed_week = pd.merge_asof(d[['date']], wt.reset_index(), on='date', direction='backward')
    d['chart_atr14'] = atr
    d['chart_atr_fraction'] = atr / p
    d['chart_bb_width20'] = width
    d['chart_bb_percentile126'] = percentile
    d['chart_prior_squeeze5'] = squeeze
    d['chart_resistance63'] = resistance
    d['chart_volume_ratio20'] = vol
    d['chart_volume_dry10_50'] = (d.volume.shift().rolling(10).median() /
        d.volume.shift().rolling(50).median().replace(0, np.nan))
    d['chart_gap_atr'] = gap
    d['chart_first_breakout'] = first.astype(float).where(known)
    d['chart_breakout_age10'] = age.where(recent)
    d['chart_breakout_anchor'] = anchor
    d['chart_retest_hold'] = (age.between(1, 10) &
        lo.le(anchor + .5 * prior_atr) & p.ge(anchor) & p.ge(p.shift())).astype(float).where(anchor.notna())
    d['chart_failed_breakout'] = (age.between(1, 10) &
        p.lt(anchor - .5 * prior_atr)).astype(float).where(anchor.notna())
    d['chart_daily_trend'] = trend.astype(float).where(ma200.shift(20).notna())
    d['chart_completed_week_trend'] = closed_week.chart_completed_week_trend.to_numpy()
    d['chart_extension50_atr'] = (p - ma50) / atr.replace(0, np.nan)
    action = factor.pct_change().abs().gt(.10)
    d['chart_adjustment_jump63'] = action.astype(float).rolling(63).max()
    d['chart_base_range63'] = hi.shift().rolling(63).max() / lo.shift().rolling(63).min() - 1
    d['chart_history_close'] = d.close
    d['chart_adjustment_factor'] = factor
    return d.replace([np.inf, -np.inf], np.nan)


def rounded_handle_proxy(bars):
    """Conservative rounded-base/handle proxy, not a validated O'Neil detector.

    Three fixed cup windows and four fixed handle windows; no outcome tuning.
    Today's bar is excluded from shape construction, including the handle.
    """
    d = prepare(bars)
    result = {k:np.nan for k in SHAPE_FIELDS}
    if len(d) < 60 + 5 + 64:
        return result
    result['chart_rounded_handle_proxy'] = 0.
    p = d.adj_close.to_numpy()
    for length in (60, 90, 126):
        for handle in (5, 10, 15, 20):
            start = len(p) - 1 - handle - length
            if start < 63:
                continue
            cup = p[start:start + length]
            h = p[start + length:-1]
            edge = max(3, length // 5)
            left_i = int(np.argmax(cup[:edge]))
            bottom_i = int(np.argmin(cup))
            left, right, bottom = cup[left_i], np.max(cup[-edge:]), np.min(cup)
            if not length * .2 <= bottom_i <= length * .75:
                continue
            rim = min(left, right)
            depth = 1 - bottom / rim
            if not .12 <= depth <= .50 or not .92 <= right / left <= 1.08:
                continue
            if left / p[start + left_i - 63] - 1 < .15:
                continue
            dwell = float(np.mean(cup <= bottom + .15 * (rim - bottom)))
            retrace = (right - np.min(h)) / (right - bottom)
            if (dwell >= .25 and 0 <= retrace <= 1/3 and
                np.max(h) <= right * 1.03 and np.min(h) > (right + bottom) / 2):
                return dict(zip(SHAPE_FIELDS, [1., depth, retrace, dwell, right]))
    return result


def attach(frame, history):
    """Exact issuer/date alignment, clipping at each decision close; no ffill."""
    out = frame.drop(columns=[*FEATURES, *SHAPE_FIELDS, 'chart_history_status'], errors='ignore').copy()
    if out.empty:
        for name in FEATURES + SHAPE_FIELDS:
            out[name] = pd.Series(dtype=float)
        out['chart_history_status'] = pd.Series(dtype=str)
        return out
    out['date'] = pd.to_datetime(out.date)
    if out.duplicated(['date', 'isin']).any():
        raise ValueError('Duplicate scored issuer/date')
    records = []
    for isin, rows in out.groupby('isin', sort=False):
        for _, row in rows.iterrows():
            # The collector stores symbol-hashed filenames; ISIN/date inside
            # the file remain authoritative across renames and replacements.
            symbol_key = hashlib.sha256(str(row.get('symbol', '')).encode()).hexdigest()
            path = Path(history) / (symbol_key + '.parquet')
            if not path.exists():
                path = Path(history) / (str(isin) + '.parquet')
            rec = {'date':row.date, 'isin':isin, 'chart_history_status':'MISSING_EXACT_HISTORY'}
            if path.exists():
                try:
                    b = pd.read_parquet(path)
                    b['date'] = pd.to_datetime(b.date)
                    b = b[b.date.le(row.date)]
                    x = daily(b)
                    if len(x) and x['isin'].eq(isin).all() and x.iloc[-1].date == row.date:
                        last = x.iloc[-1]
                        if not np.isclose(last.close, row.close, rtol=.0005, atol=.01):
                            rec['chart_history_status'] = 'SOURCE_CLOSE_CONFLICT'
                        else:
                            rec.update(last[FEATURES].to_dict())
                            rec.update(rounded_handle_proxy(b))
                            rec['chart_history_status'] = 'EXACT_DATE_MATCH_PROVIDER_ACTIONS_UNVERIFIED'
                    else:
                        rec['chart_history_status'] = 'NO_TRADED_BAR_ON_DECISION_DATE'
                except (ValueError, KeyError, TypeError) as exc:
                    rec['chart_history_status'] = 'INVALID_HISTORY_' + type(exc).__name__
            records.append(rec)
    out = out.merge(pd.DataFrame(records), on=['date', 'isin'], how='left', validate='one_to_one')
    for name in FEATURES + SHAPE_FIELDS:
        if name not in out:
            out[name] = np.nan
    return out


def select(attached, rsi_min=50.):
    d = attached.copy()
    def col(name):
        return pd.to_numeric(d.get(name, pd.Series(np.nan, index=d.index)), errors='coerce')
    def yes(name):
        return col(name).eq(1)
    observed = d.get('chart_history_status', pd.Series('', index=d.index)).str.startswith('EXACT_DATE_MATCH')
    market = d.get('feature_eligible', pd.Series(False, index=d.index)).fillna(False).astype(bool)
    market &= observed & col('close').gt(0)
    rsi = col('rsi14').between(rsi_min, 100)
    weekly_trend = yes('chart_daily_trend') & yes('chart_completed_week_trend')
    extended = col('chart_extension50_atr').gt(5) | col('chart_gap_atr').gt(3)
    squeeze_watch = (yes('chart_prior_squeeze5') & col('chart_gap_atr').between(-2, .25) &
        col('chart_volume_dry10_50').le(1) & col('chart_base_range63').le(.35))
    rounded_watch = (yes('chart_rounded_handle_proxy') & col('chart_proxy_rim').gt(0) &
        ((col('adj_close') - col('chart_proxy_rim')) / col('chart_atr14')).between(-2, .25))
    known = col('chart_bb_percentile126').notna() & col('chart_gap_atr').notna()
    d['chart_state'] = np.select([
        ~market, ~rsi, yes('chart_adjustment_jump63'), ~known,
        yes('chart_failed_breakout'), extended,
        yes('chart_first_breakout') & weekly_trend,
        yes('chart_retest_hold') & weekly_trend,
        yes('chart_first_breakout'), rounded_watch, squeeze_watch, weekly_trend],
        ['UNKNOWN_CURRENT_MARKET', 'RSI_BELOW_50_OR_UNKNOWN', 'CORPORATE_ACTION_REVIEW',
         'INSUFFICIENT_HISTORY', 'FAILED_BREAKOUT', 'EXTENDED_REVIEW',
         'CONFIRMED_BREAKOUT', 'PULLBACK_HOLD', 'BREAKOUT_TREND_REVIEW',
         'ROUNDED_HANDLE_WATCH', 'PRE_BREAKOUT_WATCH', 'TREND_WITHOUT_NEW_SETUP'],
        default='NO_CONFIRMED_SETUP')
    financial_failure = d.get('financial_quality_status', pd.Series('', index=d.index)).eq('KNOWN_FINANCIAL_FAILURE')
    financial_failure |= col('owner_PAT_yoy').lt(0) | col('owner_PAT_margin').le(0)
    adverse = (col('hd_adverse_project_90d').gt(0) | col('hd_dilution_90d').gt(0) |
        col('current_adverse_project_documents').gt(0))
    d['chart_candidate'] = market & rsi & d.chart_state.isin(SETUPS) & ~financial_failure & ~adverse
    d['chart_financial_review'] = np.where(financial_failure, 'KNOWN_FAILURE',
        np.where(d.get('financial_quality_status', pd.Series('', index=d.index)).eq(
            'PASS_AVAILABLE_SEVEN_CHECKS'), 'SEVEN_CHECKS_PASS', 'INCOMPLETE'))
    d['chart_adverse_disclosure_observed'] = adverse
    # Raw currency volatility reference, not an order or guaranteed stop fill.
    atr_raw = col('chart_atr14') / col('chart_adjustment_factor')
    d['chart_3atr_below_close_INR'] = (col('close') - 3 * atr_raw).where(market & atr_raw.gt(0))
    d['chart_signal_is_2x_probability'] = False
    return d


def export(scored, root, replay_receipt, history=None, rsi_min=None):
    if rsi_min is None:
        from v13_research_selection import RSI_MIN
        rsi_min = RSI_MIN
    root = Path(root)
    d = select(attach(scored, history or root / 'history' / 'securities'), rsi_min)
    path = root / 'chart_interpretation'
    path.mkdir(parents=True, exist_ok=True)
    raw = json.loads(d.to_json(orient='records', date_format='iso', double_precision=15))
    summary = {'version':VERSION, 'rsi_min_inclusive':rsi_min, 'scored_rows':len(d),
        'source_date':str(d.date.max().date()) if len(d) else None,
        'state_counts':{str(k):int(v) for k,v in d.chart_state.value_counts().items()},
        'chart_candidates':int(d.chart_candidate.sum()),
        'history_status_counts':{str(k):int(v) for k,v in d.chart_history_status.value_counts().items()},
        'probabilities_recalibrated':False, 'forecast_selection_changed':False,
        'thresholds_empirically_validated':False, 'promotion_approved':False,
        'corporate_actions_independently_verified':False,
        'prediction_replay':replay_receipt, 'records':raw}
    payload = json.dumps(summary, indent=2, sort_keys=True).encode()
    key = hashlib.sha256(payload).hexdigest()
    cohort = path / 'cohorts' / (key + '.json')
    cohort.parent.mkdir(exist_ok=True)
    if not cohort.exists():
        cohort.write_bytes(payload)
    (path / 'latest.json').write_bytes(payload)
    d.to_csv(path / 'all_states.csv', index=False)
    candidates = d[d.chart_candidate].copy()
    if len(candidates):
        candidates['_financial_order'] = candidates.chart_financial_review.map({'SEVEN_CHECKS_PASS':0, 'INCOMPLETE':1}).fillna(2)
        candidates['_setup_order'] = candidates.chart_state.map({v:i for i,v in enumerate(SETUPS)})
        candidates.sort_values(['_financial_order','_setup_order','isin'], inplace=True)
        candidates.drop(columns=['_financial_order','_setup_order'], inplace=True)
    candidates.to_csv(path / 'candidates.csv', index=False)
    return {k:v for k,v in summary.items() if k not in ('records','prediction_replay')}
