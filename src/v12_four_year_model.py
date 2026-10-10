"""A separately frozen four-year research model for calendar 6/12-month 2x.

One predeclared ensemble, not a threshold/model search. All predictions,
calibration, tail bounds and stability omissions respect chronological label
maturity. Current financial checks are an explicit untrained selection gate;
they do not artificially increase the historical market-model probability.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from v12_public_history import bounds

VERSION = 'V12-4Y-calendar6-12M-open-data-research-20261010'
FEATURES = ('log_ret20', 'log_ret63', 'momentum_acceleration', 'rsi14',
            'distance_ma20', 'distance_ma50', 'off_high63', 'volatility20',
            'volatility63', 'volume_confirmation', 'up_down_volume63',
            'trend_consistency63', 'drawdown63', 'log_liquidity63',
            'relative_strength63', 'relative_strength20', 'market_breadth63',
            'market_median_ret63', 'market_dispersion63')
TARGETS = ('hit2_6', 'hit2_12', 'loss30_12')
STABILITY_SEEDS = (17, 29, 41, 53, 67, 83, 97, 109, 127, 139, 157, 173)
MIN_TURNOVER = 1e7
COST_PER_SIDE = .005
MIN_BASE_ROWS = 500
MIN_BASE_DATES = 2


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_recipe(output, history_path, asof):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    recipe = {'version': VERSION, 'registered_utc': datetime.now(timezone.utc).isoformat(),
        'asof': asof, 'four_year_start': str(bounds(asof)[0].date()),
        'historical_feature_lookback_max_sessions': 63,
        'RSI_definition': '14_session_Cutler_simple_gain_loss_means',
        'current_selector_lookback_max_sessions': 200,
        'calendar_target_months': [6, 12],
        'target': 'NEXT_SESSION_OPEN_TO_FUTURE_CLOSING_TOTAL_RETURN_PROXY_AT_LEAST_2X_NET_OF_COSTS',
        'entry_cost_per_side': COST_PER_SIDE, 'target_includes_dividend_adjustment': True,
        'price_only_2x_target': False, 'endpoint_return_is_separately_reported': True,
        'future_high_or_intraday_touch_is_not_a_success': True,
        'full_horizon_maturity_required_even_if_2x_already_hit': True,
        'features': list(FEATURES), 'targets': list(TARGETS),
        'algorithm': '0.7_regularized_logistic_plus_0.3_shallow_histogram_gradient_boosting',
        'logistic_C': .03, 'boosting': {'max_iter': 60, 'max_leaf_nodes': 7,
            'min_samples_leaf': 100, 'l2_regularization': 10, 'learning_rate': .05},
        'equal_date_weights_and_18month_training_half_life': True,
        'training_tail_quantiles': [.005, .995],
        'calibration': 'three_previous_fully_matured_months; positive-slope Platt or shrunk intercept',
        'training_labels_mature_before_first_calibration_decision': True,
        'calibration_labels_mature_before_forecast_decision': True,
        'new_recipe_variants': 1, 'hyperparameter_search': False,
        'current_financials_inserted_into_historical_training': False,
        'legacy_weights_predictions_or_rankings_used': False,
        'current_technical_gate': {'rsi': [50, 80], 'price_above_ma50': True,
            'price_above_ma200': True, 'max_63_session_prior_runup': .75,
            'median_20_and_63_session_turnover_INR_min': MIN_TURNOVER},
        'current_financial_gate': {'market_cap_crore': [100, 50000],
            'max_annual_age_days': 450, 'max_quarter_age_days': 190,
            'quarter_sales_yoy_min': .10, 'quarter_pat_yoy_min': .15,
            'OPM_min': .08, 'debt_to_equity_max': 1.5, 'CFO_PAT_min': .6,
            'missing_required_financials': 'UNKNOWN_not_pass',
            'reported_recurring_PBT_nonpositive_when_components_reconcile': 'REJECT'},
        'high_conviction_probability_floors': {'hit2_6': .10, 'hit2_12': .25,
                                                'loss30_12_max': .45},
        'do_not_force_ten_high_conviction_picks': True,
        'incoherent_independent_p6_above_p12': 'ABSTAIN_HIGH_CONVICTION_NEVER_INFLATE_OR_PROJECT_PROBABILITIES',
        'probabilities_are_market_model_estimates_not_financial_gate_calibrated': True,
        'research_readiness_gates': {'six_month_fully_matured_test_dates_min': 6,
            'twelve_month_fully_matured_test_dates_min': 6,
            'current_mean_jaccard_min': .8, 'current_p05_jaccard_min': .6,
            'heldout_selected_precision_min': .30,
            'Brier_skill_vs_past_calibration_prior_min': 0},
        'survivor_bias_current_public_listing_universe': True,
        'corporate_actions_independently_verified': False,
        'production_accuracy_verified': False,
        'history_directory': str(Path(history_path).resolve()),
        'method_sources': ['https://www.williamoneil.com/about-us/legal/oneil-proprietary-rating-and-rankings',
                          'https://scikit-learn.org/stable/modules/calibration.html']}
    path = out / 'frozen_recipe.json'
    with path.open('x') as f:
        json.dump(recipe, f, indent=2)
    return recipe


def rsi(series):
    change = series.diff()
    up = change.clip(lower=0).rolling(14).mean()
    down = (-change.clip(upper=0)).rolling(14).mean()
    result = 100 - 100 / (1 + up / down)
    return result.where(down.ne(0), 100).where(up.ne(0) | down.ne(0), 50)


def make_security_features(bars, asof):
    low, high = bounds(asof)
    d = bars.copy()
    d['date'] = pd.to_datetime(d.date).dt.normalize()
    if d.date.duplicated().any() or not d.date.is_monotonic_increasing:
        raise ValueError('Invalid ordered daily source')
    # Trim BEFORE any rolling/RSI operations. Never use pre-boundary warmup.
    d = d[d.date.between(low, high) & d.volume.gt(0)].reset_index(drop=True)
    price = d.adj_close.astype(float)
    ret = np.log(price).diff()
    turnover = d.close * d.volume
    d['log_ret20'] = np.log(price / price.shift(20))
    d['log_ret63'] = np.log(price / price.shift(63))
    d['momentum_acceleration'] = d.log_ret20 - d.log_ret63 * 20/63
    d['rsi14'] = rsi(price)
    d['distance_ma20'] = price / price.rolling(20).mean() - 1
    d['distance_ma50'] = price / price.rolling(50).mean() - 1
    d['distance_ma200'] = price / price.rolling(200).mean() - 1
    d['off_high63'] = price / price.rolling(63).max() - 1
    d['volatility20'] = ret.rolling(20).std() * np.sqrt(252)
    d['volatility63'] = ret.rolling(63).std() * np.sqrt(252)
    d['volume_confirmation'] = np.log(d.volume.rolling(20).mean() / d.volume.rolling(63).mean())
    up = d.volume.where(ret.gt(0), 0).rolling(63).sum()
    down = d.volume.where(ret.le(0), 0).rolling(63).sum()
    d['up_down_volume63'] = np.log((up + 1) / (down + 1))
    d['trend_consistency63'] = ret.gt(0).rolling(63).mean()
    d['drawdown63'] = price.rolling(63).apply(lambda a: float(np.min(a/np.maximum.accumulate(a))-1), raw=True)
    d['median_turnover63'] = turnover.rolling(63).median()
    d['median_turnover20'] = turnover.rolling(20).median()
    d['log_liquidity63'] = np.log1p(d.median_turnover63)
    d['history_start_used'] = d.date.shift(63)
    d['gap_max63'] = d.date.diff().dt.days.rolling(63).max()
    d['jump_max63'] = ret.abs().rolling(63).max()
    d['observed_sessions35d'] = pd.Series(1, index=pd.DatetimeIndex(d.date)).rolling('35D').sum().to_numpy()
    d['feature_eligible'] = (d.median_turnover63.ge(MIN_TURNOVER)
        & d.median_turnover20.ge(MIN_TURNOVER) & d.gap_max63.le(10)
        & d.jump_max63.le(np.log(1.4)) & d.observed_sessions35d.ge(18)
        & d.history_start_used.notna())
    return d


def forward_outcome(d, decision_index, months, asof):
    t = decision_index
    if t + 1 >= len(d):
        return None
    entry = d.iloc[t + 1]
    end = entry.date + pd.DateOffset(months=months)
    if pd.Timestamp(asof) < end or d.date.max() < end:
        return None
    f = d[(d.date >= entry.date) & (d.date <= end)]
    if len(f) < months * 16 or (end - f.date.iloc[-1]).days > 5:
        return None
    changes = np.log(f.adj_close).diff()
    if f.date.diff().dt.days.max() > 10 or changes.abs().max() > np.log(1.4):
        return None
    entry_adj_open = float(entry.open * entry.adj_close / entry.close)
    net_close = f.adj_close.to_numpy(float) / entry_adj_open * (1-COST_PER_SIDE)/(1+COST_PER_SIDE)
    adjusted_lows = f.low.to_numpy(float) * f.adj_close.to_numpy(float) / f.close.to_numpy(float)
    net_lows = adjusted_lows / entry_adj_open * (1-COST_PER_SIDE)/(1+COST_PER_SIDE)
    hits = np.flatnonzero(net_close >= 2)
    losses = np.flatnonzero(net_lows <= .7)
    return {'hit2': int(bool(len(hits))), 'endpoint_return': float(net_close[-1] - 1),
            'peak_return': float(net_close.max() - 1), 'loss30': int(bool(len(losses))),
            'hit2_before_loss30': int(bool(len(hits)) and (not len(losses) or hits[0] < losses[0])),
            'mature_date': str(end.date()), 'entry_date': str(entry.date.date())}


def build_panel(history_path, asof, output):
    paths = sorted((Path(history_path) / 'securities').glob('*.parquet'))
    sources = [pd.read_parquet(p) for p in paths]
    if not sources:
        raise ValueError('No usable public history')
    counts = pd.concat([x[['date']] for x in sources]).groupby('date').size()
    calendar = pd.DatetimeIndex(counts[counts >= 20].index).sort_values()
    month_dates = pd.Series(calendar, index=calendar).resample('ME').max().dropna()
    decisions = set(month_dates.tolist()) | {pd.Timestamp(asof)}
    records = []
    for source in sources:
        d = make_security_features(source, asof)
        for t in d.index[d.date.isin(decisions) & d.feature_eligible]:
            row = d.loc[t].to_dict()
            for months, name in ((6, '6'), (12, '12')):
                outcome = forward_outcome(d, t, months, asof)
                for field in ('hit2', 'endpoint_return', 'peak_return', 'loss30', 'hit2_before_loss30', 'mature_date'):
                    row[field + '_' + name] = outcome[field] if outcome else None
            records.append(row)
    x = pd.DataFrame(records)
    if x.empty or x.duplicated(['date', 'symbol']).any():
        raise ValueError('Empty or duplicate monthly feature panel')
    groups = x.groupby('date')
    x['relative_strength63'] = groups.log_ret63.rank(pct=True)
    x['relative_strength20'] = groups.log_ret20.rank(pct=True)
    x['market_breadth63'] = groups.log_ret63.transform(lambda s: s.gt(0).mean())
    x['market_median_ret63'] = groups.log_ret63.transform('median')
    x['market_dispersion63'] = groups.log_ret63.transform('std')
    if not np.isfinite(x[list(FEATURES)].to_numpy(float)).all():
        raise ValueError('Unusable derived predictor: no synthetic missing market values')
    for c in ('mature_date_6', 'mature_date_12'):
        x[c] = pd.to_datetime(x[c])
    x['loss30_12'] = x['loss30_12'].astype(float)
    x['technical_pass'] = (x.rsi14.between(50, 80) & x.distance_ma50.gt(0)
        & x.distance_ma200.gt(0) & x.log_ret63.le(np.log(1.75)))
    x.to_parquet(Path(output) / 'monthly_four_year_features_and_outcomes.parquet', index=False)
    future = [c for c in x if c.startswith(('hit2_', 'loss30_', 'mature_date_', 'endpoint_return_', 'peak_return_'))]
    x.loc[x.date.eq(pd.Timestamp(asof))].drop(columns=future).to_parquet(
        Path(output) / 'current_source_predictors.parquet', index=False)
    return x


def chronological_parts(panel, date, target):
    horizon = '6' if target == 'hit2_6' else '12'
    mature = 'mature_date_' + horizon
    known = panel[(panel.date < pd.Timestamp(date)) & panel[mature].lt(pd.Timestamp(date))
                  & panel[target].isin([0, 1])].copy()
    dates = sorted(known.date.unique())
    if len(dates) < 3:
        return None
    cal_dates = dates[-3:]
    train = known[known.date.lt(cal_dates[0]) & known[mature].lt(cal_dates[0])].copy()
    cal = known[known.date.isin(cal_dates)].copy()
    if (len(train) < MIN_BASE_ROWS or train.date.nunique() < MIN_BASE_DATES
            or train[target].sum() < 10 or train[target].eq(0).sum() < 10 or len(cal) < 300):
        return None
    assert train[mature].lt(min(cal_dates)).all()
    assert cal[mature].lt(pd.Timestamp(date)).all()
    return train, cal


def weights(frame):
    counts = frame.groupby('date').date.transform('size').to_numpy(float)
    age = (frame.date.max() - frame.date).dt.days.to_numpy(float)
    w = np.exp2(-age/(18*30.4375)) / counts
    return w / w.mean()


def calibrate(raw, y, train_y):
    raw = np.clip(np.asarray(raw, float), 1e-5, 1-1e-5)
    y = np.asarray(y, int)
    if min(int(y.sum()), int((y == 0).sum())) >= 12:
        lr = LogisticRegression(C=.1, max_iter=500).fit(logit(raw).reshape(-1, 1), y)
        if lr.coef_[0, 0] > 0:
            return {'method': 'positive_slope_Platt', 'slope': float(lr.coef_[0, 0]),
                    'intercept': float(lr.intercept_[0])}
    goal = (y.sum() + 200*np.asarray(train_y).mean())/(len(y)+200)
    lo, hi = -20., 20.
    for _ in range(60):
        mid = (lo+hi)/2
        if expit(logit(raw)+mid).mean() < goal:
            lo = mid
        else:
            hi = mid
    return {'method': 'shrunk_monotone_intercept', 'slope': 1., 'intercept': (lo+hi)/2}


def transform(package, data):
    a = data[list(FEATURES)].to_numpy(float)
    if not np.isfinite(a).all():
        raise ValueError('Nonfinite market predictors')
    return np.clip(a, package['lower'], package['upper'])


def raw_prediction(package, data):
    a = transform(package, data)
    return .7*package['linear'].predict_proba(a)[:, 1] + .3*package['nonlinear'].predict_proba(a)[:, 1]


def score(package, data):
    raw = np.clip(raw_prediction(package, data), 1e-5, 1-1e-5)
    cal = package['calibration']
    return expit(cal['slope']*logit(raw) + cal['intercept'])


def fit_head(train, cal, target):
    a = train[list(FEATURES)].to_numpy(float)
    lower, upper = np.quantile(a, [.005, .995], axis=0)
    a = np.clip(a, lower, upper)
    w = weights(train)
    linear = make_pipeline(StandardScaler(), LogisticRegression(C=.03, max_iter=1000, random_state=31))
    linear.fit(a, train[target].astype(int), logisticregression__sample_weight=w)
    nonlinear = HistGradientBoostingClassifier(max_iter=60, max_leaf_nodes=7,
        min_samples_leaf=100, l2_regularization=10, learning_rate=.05,
        early_stopping=False, random_state=31)
    nonlinear.fit(a, train[target].astype(int), sample_weight=w)
    p = {'linear': linear, 'nonlinear': nonlinear, 'lower': lower, 'upper': upper,
         'target': target, 'train_rows': len(train), 'train_dates': train.date.nunique(),
         'train_max_label_maturity': str(train['mature_date_' + ('6' if target == 'hit2_6' else '12')].max()),
         'calibration_first_date': str(cal.date.min()), 'calibration_last_date': str(cal.date.max()),
         'calibration_rows': len(cal), 'past_calibration_event_rate': float(cal[target].mean())}
    p['calibration'] = calibrate(raw_prediction(p, cal), cal[target], train[target])
    return p


def rank(frame, probability):
    return frame.sort_values([probability, 'symbol'], ascending=[False, True])


def reliability(frame, target, probability):
    known = frame[frame[target].isin([0, 1])].copy()
    rows = []
    if len(known):
        known['bin'] = pd.cut(known[probability], [0, .05, .10, .20, .30, .50, .75, 1], include_lowest=True)
        for bucket, g in known.groupby('bin', observed=True):
            rows.append({'bin': str(bucket), 'n': len(g), 'predicted': float(g[probability].mean()),
                         'observed': float(g[target].mean()), 'distinct_dates': g.date.nunique()})
    return rows


def evaluate(panel, asof, output):
    output = Path(output)
    results, score_frames = {}, []
    for target in TARGETS:
        folds, scored = [], []
        for date in sorted(panel.date.unique()):
            if pd.Timestamp(date) == pd.Timestamp(asof):
                continue
            parts = chronological_parts(panel, date, target)
            if parts is None:
                continue
            train, cal = parts
            test = panel[panel.date.eq(date)].copy()
            package = fit_head(train, cal, target)
            test['probability'] = score(package, test)
            test['target_name'] = target
            test['past_calibration_event_rate'] = package['past_calibration_event_rate']
            horizon = '6' if target == 'hit2_6' else '12'
            mature = 'mature_date_' + horizon
            known = test[test[mature].le(pd.Timestamp(asof)) & test[target].isin([0, 1])]
            eligible = test.loc[test.technical_pass] if target != 'loss30_12' else test
            selected = rank(eligible, 'probability').head(10)
            selected_known = selected[selected[mature].le(pd.Timestamp(asof)) & selected[target].isin([0, 1])]
            test['top10_selected'] = test.symbol.isin(selected.symbol)
            metric = {'date': str(pd.Timestamp(date).date()), 'train_rows': len(train),
                'train_dates': train.date.nunique(), 'calibration_rows': len(cal),
                'test_rows': len(test), 'known_outcomes': len(known),
                'selected_rows': len(selected), 'selected_known': len(selected_known),
                'selected_hits': int(selected_known[target].sum()),
                'selected_precision': float(selected_known[target].mean()) if len(selected_known) else None,
                'past_calibration_rate': package['past_calibration_event_rate']}
            if len(known) and known[target].nunique() == 2:
                metric.update({'Brier': brier_score_loss(known[target], known.probability),
                    'prior_Brier': brier_score_loss(known[target], np.repeat(package['past_calibration_event_rate'], len(known))),
                    'AUC': roc_auc_score(known[target], known.probability),
                    'average_precision': average_precision_score(known[target], known.probability)})
            folds.append(metric)
            scored.append(test)
        joined = pd.concat(scored, ignore_index=True) if scored else pd.DataFrame()
        full = [f for f in folds if f['selected_rows'] >= 10 and f['selected_known'] == f['selected_rows']]
        selections = sum(f['selected_known'] for f in full)
        hits = sum(f['selected_hits'] for f in full)
        briers = [f for f in folds if 'Brier' in f]
        model_loss = np.average([f['Brier'] for f in briers], weights=[f['known_outcomes'] for f in briers]) if briers else None
        prior_loss = np.average([f['prior_Brier'] for f in briers], weights=[f['known_outcomes'] for f in briers]) if briers else None
        results[target] = {'fully_matured_top10_dates': len(full), 'top10_selections': selections,
            'top10_hits': hits, 'selected_precision': hits/selections if selections else None,
            'Brier': float(model_loss) if model_loss is not None else None,
            'prior_Brier': float(prior_loss) if prior_loss is not None else None,
            'Brier_skill': float(1-model_loss/prior_loss) if prior_loss else None,
            'reliability': reliability(joined, target, 'probability') if len(joined) else [], 'folds': folds}
        if len(joined):
            joined.to_parquet(output / (target+'_chronological_scores.parquet'), index=False)
        print(json.dumps({'target_tested': target, 'matured_top10_dates': len(full),
                          'selected_hits': hits, 'selected_total': selections}), flush=True)
    (output/'chronological_evaluation.json').write_text(json.dumps(results, indent=2))
    return results


def current_financial_gate(current, audit, reference, official, asof):
    # Do not propagate legacy five/seven-year source ratios into V12 outputs.
    allowed = {'symbol', 'isin', 'company_name', 'reporting_mode', 'source_url',
        'source_sha256', 'first_retrieved_utc', 'issuer_name_matches', 'issuer_symbol_matches',
        'status', 'market_cap_verified_crore', 'latest_annual_revenue_period',
        'latest_quarter_revenue_period', 'quarterly_sales_yoy_growth',
        'quarterly_profit_yoy_growth', 'opm_annual', 'debt_equity', 'latest_annual_cfo_pat_ratio',
        'negative_or_missing_equity', 'PBT_components_reconcile_all_four_quarters',
        'four_quarter_PBT_excluding_reported_exceptional_items',
        'four_quarter_reported_exceptional_items', 'original_fact_disagreements',
        'original_fact_agreements', 'official_company_PE'}
    financial = audit[[c for c in audit if c in allowed]].copy()
    if financial.duplicated(['symbol', 'isin']).any():
        raise ValueError('Duplicate current financial identity')
    q = current.merge(financial, on=['symbol', 'isin'], how='left', validate='1:1', suffixes=('', '_financial'))
    refs = reference[['isin', 'amfi_size_category']].drop_duplicates('isin')
    q = q.merge(refs, on='isin', how='left', validate='m:1')
    market = official.sort_values('series').drop_duplicates(['symbol', 'isin'])
    q = q.merge(market[['symbol', 'isin', 'date', 'close']].rename(columns={'date': 'official_date', 'close': 'official_close'}),
                on=['symbol', 'isin'], how='left', validate='1:1')
    reasons, statuses = [], []
    for row in q.to_dict('records'):
        fail, unknown = [], []
        def check(key, fn, label):
            v = row.get(key)
            if v is None or pd.isna(v) or not np.isfinite(float(v)):
                unknown.append(label)
            elif not fn(float(v)):
                fail.append(label)
        if pd.isna(row.get('official_date')) or pd.Timestamp(row['official_date']) != pd.Timestamp(asof):
            unknown.append('same_date_official_NSE_price')
        elif not np.isclose(row['close'], row['official_close'], rtol=.001, atol=.06):
            fail.append('provider_vs_official_NSE_price')
        check('market_cap_verified_crore', lambda v: 100 <= v <= 50000, 'corroborated_size_band_100_to_50000_crore')
        if row.get('status') != 'CURRENT_THREE_WAY_CORROBORATED':
            unknown.append('corroborated_venue_and_share_units')
        for key, age, label in [('latest_annual_revenue_period', 450, 'annual_age'),
                                 ('latest_quarter_revenue_period', 190, 'quarter_age')]:
            date = pd.to_datetime(row.get(key), errors='coerce')
            if pd.isna(date):
                unknown.append(label)
            elif not 0 <= (pd.Timestamp(asof)-date).days <= age:
                fail.append(label)
        check('quarterly_sales_yoy_growth', lambda v: v >= .10, 'quarter_sales_growth_10pct')
        check('quarterly_profit_yoy_growth', lambda v: v >= .15, 'quarter_profit_growth_15pct')
        check('opm_annual', lambda v: v >= .08, 'OPM_8pct')
        check('debt_equity', lambda v: 0 <= v <= 1.5, 'debt_equity_1_5')
        check('latest_annual_cfo_pat_ratio', lambda v: v >= .6, 'CFO_PAT_0_6')
        if row.get('negative_or_missing_equity') is True:
            fail.append('nonpositive_equity')
        if row.get('PBT_components_reconcile_all_four_quarters') is True:
            check('four_quarter_PBT_excluding_reported_exceptional_items', lambda v: v > 0, 'recurring_PBT_positive')
        else:
            unknown.append('four_quarter_earnings_quality')
        if pd.notna(row.get('original_fact_disagreements')) and row['original_fact_disagreements'] > 0:
            fail.append('unresolved_original_filing_disagreement')
        # The primary row proves the exact source name even when Finology's
        # displayed exchange symbol is an alias; never use name alone for ISIN.
        if row.get('issuer_name_matches') is not True:
            unknown.append('provider_issuer_name')
        if not row.get('technical_pass', False):
            fail.append('RSI_trend_or_prior_runup')
        if not 20 <= row['close'] <= 2000:
            fail.append('current_price_20_to_2000')
        statuses.append('REJECT' if fail else 'UNKNOWN' if unknown else 'CURRENT_SOURCE_CHECKS_PASS')
        reasons.append(';'.join('FAIL:'+s for s in fail) + (';' if fail and unknown else '') + ';'.join('UNKNOWN:'+s for s in unknown))
    q['current_quality_status'] = statuses
    q['quality_reasons'] = reasons
    q['financial_gate_historically_calibrated'] = False
    return q


def issuer_omission(train, seed):
    symbols = np.array(sorted(train.symbol.unique()))
    rng = np.random.default_rng(seed)
    keep = set(rng.choice(symbols, size=max(1, int(np.ceil(.95*len(symbols)))), replace=False))
    return train[train.symbol.isin(keep)].copy()


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b)/len(a | b) if a or b else 1.


def fit_current(panel, asof, output, audit, reference, official):
    out = Path(output)
    current = panel[panel.date.eq(pd.Timestamp(asof))].copy()
    packages, parts_by_target = {}, {}
    for target in TARGETS:
        parts = chronological_parts(panel, pd.Timestamp(asof)+pd.Timedelta(days=1), target)
        if parts is None:
            raise ValueError('Insufficient honest four-year train/calibration data for '+target)
        train, cal = parts
        packages[target] = fit_head(train, cal, target)
        parts_by_target[target] = (train, cal)
        current['p_'+target] = score(packages[target], current)
    # Never improve-looking probabilities by projection. Inconsistent nested
    # head estimates are explicitly ineligible for the high-conviction view.
    current['probability_coherence'] = current.p_hit2_6.le(current.p_hit2_12)
    q = current_financial_gate(current, audit, reference, official, asof)
    q['high_conviction'] = (q.current_quality_status.eq('CURRENT_SOURCE_CHECKS_PASS')
        & q.p_hit2_6.ge(.10) & q.p_hit2_12.ge(.25) & q.p_loss30_12.le(.45)
        & q.probability_coherence)
    q = rank(q, 'p_hit2_12').reset_index(drop=True)
    # Stability assesses the complete calibrated twelve-month head on the
    # same currently source-checked candidate set, with issuer-cluster omission.
    pool = q[q.current_quality_status.eq('CURRENT_SOURCE_CHECKS_PASS')].copy()
    reference_top = rank(pool, 'p_hit2_12').head(10).symbol
    stabilities, draws = [], []
    train, cal = parts_by_target['hit2_12']
    for seed in STABILITY_SEEDS:
        alt = fit_head(issuer_omission(train, seed), cal, 'hit2_12')
        probability = score(alt, pool) if len(pool) else np.array([])
        draws.append(probability)
        top = rank(pool.assign(perturbed_probability=probability), 'perturbed_probability').head(10).symbol
        stabilities.append({'seed': seed, 'jaccard': jaccard(reference_top, top),
                            'source_checked_pool': len(pool), 'selected': len(top)})
    if len(pool):
        sampled = np.vstack(draws)
        intervals = pd.DataFrame({'symbol': pool.symbol.to_numpy(),
            'p12_perturbation_p05': np.quantile(sampled, .05, axis=0),
            'p12_perturbation_p95': np.quantile(sampled, .95, axis=0)})
        q = q.merge(intervals, on='symbol', how='left', validate='1:1')
    q['forecast_decision_utc'] = datetime.now(timezone.utc).isoformat()
    q['model_version'] = VERSION
    q['probability_scope'] = 'MARKET_MODEL_RESEARCH_ESTIMATE; CURRENT_FINANCIAL_GATE_NOT_HISTORICALLY_CALIBRATED'
    q['expected_return_guaranteed'] = False
    q['production_accuracy_verified'] = False
    q.to_parquet(out/'current_predictions_and_gate_audit.parquet', index=False)
    columns = ['symbol', 'isin', 'close', 'market_cap_verified_crore', 'amfi_size_category',
        'p_hit2_6', 'p_hit2_12', 'p_loss30_12', 'high_conviction', 'current_quality_status',
        'rsi14', 'quarterly_sales_yoy_growth', 'quarterly_profit_yoy_growth',
        'latest_annual_cfo_pat_ratio', 'debt_equity', 'quality_reasons']
    if 'p12_perturbation_p05' in q:
        columns += ['p12_perturbation_p05', 'p12_perturbation_p95']
    view = q[q.current_quality_status.eq('CURRENT_SOURCE_CHECKS_PASS')].head(10).copy()
    view[columns].to_csv(out/'source_checked_research_shortlist.csv', index=False)
    q[q.high_conviction][columns].head(10).to_csv(out/'high_conviction_shortlist.csv', index=False)
    pd.DataFrame(stabilities).to_csv(out/'current_issuer_omission_stability.csv', index=False)
    package = {'version': VERSION, 'asof': asof, 'features': FEATURES, 'heads': packages,
               'production_accuracy_verified': False, 'recipe_sha256': sha_file(out/'frozen_recipe.json'),
               'probability_projection': None,
               'incoherent_nested_heads_abstain_high_conviction': True}
    model_path = out/'frozen_model.joblib'
    if model_path.exists():
        raise ValueError('Refuse to overwrite frozen V12 weights')
    joblib.dump(package, model_path, compress=3)
    metrics = {'current_candidates': len(q), 'quality_status_counts': q.current_quality_status.value_counts().to_dict(),
        'source_checked_shortlist': len(view), 'high_conviction_count': int(q.high_conviction.sum()),
        'mean_top10_jaccard': float(np.mean([s['jaccard'] for s in stabilities])) if len(pool) >= 10 else None,
        'p05_top10_jaccard': float(np.quantile([s['jaccard'] for s in stabilities], .05)) if len(pool) >= 10 else None,
        'stability_pool_at_least_ten': len(pool) >= 10,
        'no_forced_high_conviction_picks': True,
        'current_unconstrained_probability_order_violations': int(current.p_hit2_6.gt(current.p_hit2_12).sum()),
        'model_sha256': sha_file(model_path), 'financial_gate_historically_calibrated': False,
        'no_verified_complete_earnings_catalyst_chain_required_or_invented': True}
    (out/'current_prediction_summary.json').write_text(json.dumps(metrics, indent=2))
    return metrics


def run(history_path, output, asof, financial_path, reference_path, official_path):
    out = Path(output)
    recipe = json.loads((out/'frozen_recipe.json').read_text())
    if recipe['version'] != VERSION or recipe['asof'] != asof:
        raise ValueError('Predeclared recipe changed')
    panel = build_panel(history_path, asof, out)
    print(json.dumps({'monthly_rows': len(panel), 'dates': panel.date.nunique(),
                      'earliest_predictor_date': str(panel.date.min().date())}), flush=True)
    evaluation = evaluate(panel, asof, out)
    current = fit_current(panel, asof, out, pd.read_parquet(financial_path),
        pd.read_parquet(reference_path), pd.read_parquet(official_path))
    gates = {
        'six_month_at_least_six_matured_test_dates': evaluation['hit2_6']['fully_matured_top10_dates'] >= 6,
        'twelve_month_at_least_six_matured_test_dates': evaluation['hit2_12']['fully_matured_top10_dates'] >= 6,
        'six_month_selected_precision_at_least_30pct': (evaluation['hit2_6']['selected_precision'] or 0) >= .30,
        'twelve_month_selected_precision_at_least_30pct': (evaluation['hit2_12']['selected_precision'] or 0) >= .30,
        'six_month_Brier_skill_positive': (evaluation['hit2_6']['Brier_skill'] or 0) > 0,
        'twelve_month_Brier_skill_positive': (evaluation['hit2_12']['Brier_skill'] or 0) > 0,
        'current_mean_jaccard_at_least_080': (current['mean_top10_jaccard'] or 0) >= .8,
        'current_p05_jaccard_at_least_060': (current['p05_top10_jaccard'] or 0) >= .6,
        'survivor_and_corporate_action_bias_resolved': False,
        'prospective_prediction_accuracy_verified': False,
        'current_financial_gate_historically_calibrated': False}
    result = {'version': VERSION, 'frozen_for_reproducible_research': True,
        'production_accuracy_verified': False, 'gates': gates,
        'all_gates_pass': all(gates.values()), 'current': current,
        'data_start': str(bounds(asof)[0].date()), 'data_end': asof,
        'monthly_feature_rows': len(panel), 'snapshot_dates': panel.date.nunique(),
        'legacy_independent_model_or_protocol_overwritten': False,
        'new_independent_blind_outcomes': 0,
        'evaluation': {k: {f: v for f, v in r.items() if f not in ('folds', 'reliability')} for k, r in evaluation.items()}}
    (out/'freeze_and_evaluation_summary.json').write_text(json.dumps(result, indent=2))
    manifest = {'version': VERSION, 'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
        'files_sha256': {p.name: sha_file(p) for p in sorted(out.iterdir()) if p.is_file()},
        'source_code_sha256': sha_file(__file__), 'recipe_immutable': True,
        'weights_immutable': True, 'production_accuracy_verified': False}
    (out/'freeze_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--history', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--asof', default='2026-10-09')
    p.add_argument('--financial', required=True)
    p.add_argument('--reference', required=True)
    p.add_argument('--official', required=True)
    a = p.parse_args()
    run(a.history, a.output, a.asof, a.financial, a.reference, a.official)
