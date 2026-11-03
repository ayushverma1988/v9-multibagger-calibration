"""Predeclared structural correction: nested, calibrated doubling hazards.

P(2x by 12 months)=P(2x by 6 months)+(1-P6)*P(late 2x | no early 2x).
The conditional target is used only to train the late head; a future early
outcome is never a predictor. Probabilities are not projected or inflated.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

import v12_four_year_model as core

VERSION = 'V12.1-4Y-coherent-calendar6-12M-research-20261010'
LATE = 'late_hit2_12'
LOSS = 'loss30_before_hit2_12'


def prepare(panel):
    p = panel.copy()
    p[LATE] = p.hit2_12.where(p.hit2_6.eq(0))
    p[LOSS] = ((p.loss30_12.eq(1)) & p.hit2_before_loss30_12.eq(0)).astype(float).where(p.loss30_12.notna())
    p['technical_pass'] = (p.rsi14.between(45, 90) & p.distance_ma50.gt(0)
                          & p.log_ret63.le(np.log(1.75)))
    p['entry_sleeve'] = np.where(p.distance_ma200.gt(0), 'CONFIRMED_GROWTH', 'EMERGING_TURNAROUND')
    return p


def union_probability(early, late):
    a, b = np.asarray(early, float), np.asarray(late, float)
    if (not np.isfinite(a).all() or not np.isfinite(b).all()
            or (a < 0).any() or (a > 1).any() or (b < 0).any() or (b > 1).any()):
        raise ValueError('Invalid conditional probability')
    return a + (1-a)*b


def fit_packages(panel, date):
    heads, splits = {}, {}
    for target in ('hit2_6', LATE, LOSS):
        parts = core.chronological_parts(panel, date, target)
        if parts is None:
            return None
        train, cal = parts
        heads[target] = core.fit_head(train, cal, target)
        splits[target] = parts
    return heads, splits


def probabilities(heads, frame):
    early = core.score(heads['hit2_6'], frame)
    late = core.score(heads[LATE], frame)
    return {'p_hit2_6': early, 'p_hit2_12': union_probability(early, late),
            'p_late_given_no_early': late, 'p_loss30_12': core.score(heads[LOSS], frame)}


def recipe(output, original_output, asof):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    original = json.loads((Path(original_output)/'frozen_recipe.json').read_text())
    r = dict(original)
    r.update({'version': VERSION, 'registered_utc': datetime.now(timezone.utc).isoformat(),
        'revision_reason': 'First candidate failed precision and produced incoherent independently calibrated nested probabilities',
        'first_candidate_preserved': True, 'revision_count': 1,
        'targets': ['hit2_6', LATE, LOSS],
        'twelve_month_probability': 'p6+(1-p6)*p_late_given_no_early',
        'conditional_early_outcome_used_only_as_training_target_filter_never_predictor': True,
        'risk_target': '30pct_loss_before_2x_closing_target_or_deadline',
        'entry_sleeves': ['EMERGING_TURNAROUND', 'CONFIRMED_GROWTH'],
        'current_technical_gate': {'rsi': [45, 90], 'price_above_ma50': True,
            'price_above_ma200': 'confirmed_sleeve_only_not_a_universal_requirement',
            'max_63_session_prior_runup': .75,
            'median_20_and_63_session_turnover_INR_min': core.MIN_TURNOVER},
        'accuracy_thresholds_not_relaxed_after_failed_first_test': True,
        'parameter_search': False, 'already_examined_dates_are_retrospective_not_new_blind_tests': True,
        'original_panel_sha256': core.sha_file(Path(original_output)/'monthly_four_year_features_and_outcomes.parquet')})
    with (out/'frozen_recipe.json').open('x') as f:
        json.dump(r, f, indent=2)
    return r


def evaluate(panel, asof, output):
    records, data = {'hit2_6': [], 'hit2_12': []}, []
    for date in sorted(panel.date.unique()):
        if pd.Timestamp(date) == pd.Timestamp(asof):
            continue
        early_split = core.chronological_parts(panel, date, 'hit2_6')
        if early_split is None:
            continue
        early_head = core.fit_head(*early_split, 'hit2_6')
        fitted = fit_packages(panel, date)
        test = panel[panel.date.eq(date)].copy()
        predictions = probabilities(fitted[0], test) if fitted else {'p_hit2_6': core.score(early_head, test)}
        for name, values in predictions.items():
            test[name] = values
        for target in ('hit2_6', 'hit2_12'):
            prob = 'p_'+target
            if prob not in test:
                continue
            selected = core.rank(test[test.technical_pass], prob).head(10)
            known = test[test[target].isin([0, 1])]
            selected_known = selected[selected[target].isin([0, 1])]
            if target == 'hit2_6':
                prior = early_head['past_calibration_event_rate']
            else:
                # A genuine earlier observed union-event prior; not the
                # conditional late event rate, and not test outcomes.
                first_cal = fitted[1][LATE][1].date.min()
                last_cal = fitted[1][LATE][1].date.max()
                prior_rows = panel[panel.date.between(first_cal, last_cal)
                    & panel.mature_date_12.lt(pd.Timestamp(date)) & panel.hit2_12.isin([0, 1])]
                prior = float(prior_rows.hit2_12.mean())
            row = {'date': str(pd.Timestamp(date).date()), 'selected_rows': len(selected),
                'selected_known': len(selected_known), 'hits': int(selected_known[target].sum()),
                'precision': float(selected_known[target].mean()) if len(selected_known) else None,
                'known_outcomes': len(known), 'past_calibration_prior': prior,
                'selected_symbols': selected.symbol.tolist(),
                'Brier': brier_score_loss(known[target], known[prob]) if len(known) else None,
                'prior_Brier': brier_score_loss(known[target], np.repeat(prior, len(known))) if len(known) else None}
            records[target].append(row)
            test['selected_'+target] = test.symbol.isin(selected.symbol)
        data.append(test)
    scored = pd.concat(data, ignore_index=True)
    scored.to_parquet(Path(output)/'coherent_chronological_scores.parquet', index=False)
    result = {}
    for target, folds in records.items():
        full = [r for r in folds if r['selected_rows'] == 10 and r['selected_known'] == 10]
        hits = sum(r['hits'] for r in full)
        labelled = [r for r in folds if r['Brier'] is not None]
        loss = np.average([r['Brier'] for r in labelled], weights=[r['known_outcomes'] for r in labelled])
        prior_loss = np.average([r['prior_Brier'] for r in labelled], weights=[r['known_outcomes'] for r in labelled])
        result[target] = {'fully_matured_top10_dates': len(full), 'top10_selections': len(full)*10,
            'top10_hits': hits, 'selected_precision': hits/(len(full)*10) if full else None,
            'Brier': float(loss), 'prior_Brier': float(prior_loss), 'Brier_skill': float(1-loss/prior_loss),
            'distinct_selected_issuers': len(set(s for r in full for s in r['selected_symbols'])),
            'reliability': core.reliability(scored, target, 'p_'+target), 'folds': folds}
        print(json.dumps({'target_tested': target, 'hits': hits, 'selections': len(full)*10}), flush=True)
    (Path(output)/'chronological_evaluation.json').write_text(json.dumps(result, indent=2))
    return result


def run(original_output, output, asof, financial, reference, official):
    out = Path(output)
    r = json.loads((out/'frozen_recipe.json').read_text())
    source = Path(original_output)/'monthly_four_year_features_and_outcomes.parquet'
    if r['version'] != VERSION or core.sha_file(source) != r['original_panel_sha256']:
        raise ValueError('Registered revision source/recipe changed')
    panel = prepare(pd.read_parquet(source))
    evaluation = evaluate(panel, asof, out)
    current = panel[panel.date.eq(pd.Timestamp(asof))].copy()
    heads, splits = fit_packages(panel, pd.Timestamp(asof)+pd.Timedelta(days=1))
    for name, values in probabilities(heads, current).items():
        current[name] = values
    q = core.current_financial_gate(current, pd.read_parquet(financial), pd.read_parquet(reference), pd.read_parquet(official), asof)
    q['probability_coherence'] = q.p_hit2_6.le(q.p_hit2_12)
    q['high_conviction'] = (q.current_quality_status.eq('CURRENT_SOURCE_CHECKS_PASS')
        & q.p_hit2_6.ge(.10) & q.p_hit2_12.ge(.25) & q.p_loss30_12.le(.45))
    q['approved_for_trading'] = False
    q['forecast_decision_utc'] = datetime.now(timezone.utc).isoformat()
    q['model_version'] = VERSION
    q['probability_scope'] = 'MARKET_MODEL_RESEARCH_ESTIMATE; FINANCIAL_GATE_NOT_HISTORICALLY_CALIBRATED'
    q = core.rank(q, 'p_hit2_12').reset_index(drop=True)
    q.to_parquet(out/'current_predictions_and_gate_audit.parquet', index=False)
    pool = q[q.current_quality_status.eq('CURRENT_SOURCE_CHECKS_PASS')].copy()
    columns = ['symbol', 'isin', 'close', 'market_cap_verified_crore', 'amfi_size_category',
        'entry_sleeve', 'p_hit2_6', 'p_hit2_12', 'p_loss30_12', 'high_conviction',
        'approved_for_trading', 'current_quality_status', 'rsi14',
        'quarterly_sales_yoy_growth', 'quarterly_profit_yoy_growth', 'latest_annual_cfo_pat_ratio',
        'debt_equity', 'quality_reasons']
    pool.head(10)[columns].to_csv(out/'source_checked_research_shortlist.csv', index=False)
    q[q.high_conviction].head(10)[columns].to_csv(out/'threshold_pass_research_shortlist.csv', index=False)
    draws, stability = [], []
    reference_top = core.rank(pool, 'p_hit2_12').head(10).symbol
    early_train, early_cal = splits['hit2_6']
    late_train, late_cal = splits[LATE]
    for seed in core.STABILITY_SEEDS:
        alternate = {'hit2_6': core.fit_head(core.issuer_omission(early_train, seed), early_cal, 'hit2_6'),
                     LATE: core.fit_head(core.issuer_omission(late_train, seed), late_cal, LATE)}
        p = union_probability(core.score(alternate['hit2_6'], pool), core.score(alternate[LATE], pool))
        draws.append(p)
        top = core.rank(pool.assign(probability=p), 'probability').head(10).symbol
        stability.append({'seed': seed, 'jaccard': core.jaccard(reference_top, top), 'candidate_pool': len(pool)})
    pd.DataFrame(stability).to_csv(out/'current_issuer_omission_stability.csv', index=False)
    if len(pool):
        sampled = np.vstack(draws)
        pd.DataFrame({'symbol': pool.symbol.to_numpy(), 'p12_p05': np.quantile(sampled, .05, axis=0),
            'p12_p95': np.quantile(sampled, .95, axis=0)}).to_csv(out/'probability_perturbation_ranges.csv', index=False)
    package = {'version': VERSION, 'asof': asof, 'features': core.FEATURES, 'heads': heads,
        'probability_model': 'conditional_hazards', 'production_accuracy_verified': False,
        'recipe_sha256': core.sha_file(out/'frozen_recipe.json'), 'probability_projection': None}
    with (out/'frozen_model.joblib').open('xb') as f:
        joblib.dump(package, f, compress=3)
    future = [c for c in current if c.startswith(('hit2_', 'late_hit2_', 'loss30_', 'mature_date_', 'endpoint_return_', 'peak_return_', 'p_'))]
    current.drop(columns=future).to_parquet(out/'current_source_predictors.parquet', index=False)
    current_metrics = {'current_candidates': len(q), 'quality_status_counts': q.current_quality_status.value_counts().to_dict(),
        'source_checked_shortlist': min(10, len(pool)), 'threshold_pass_count': int(q.high_conviction.sum()),
        'probability_order_violations': int(q.p_hit2_6.gt(q.p_hit2_12).sum()),
        'stability_candidate_pool': len(pool), 'stability_mean_jaccard': float(np.mean([s['jaccard'] for s in stability])) if len(pool)>10 else None,
        'stability_p05_jaccard': float(np.quantile([s['jaccard'] for s in stability], .05)) if len(pool)>10 else None,
        'stability_meaningful_pool_larger_than_top10': len(pool)>10,
        'weights_sha256': core.sha_file(out/'frozen_model.joblib')}
    gates = { 'six_month_matured_dates_at_least_six': evaluation['hit2_6']['fully_matured_top10_dates']>=6,
        'twelve_month_matured_dates_at_least_six': evaluation['hit2_12']['fully_matured_top10_dates']>=6,
        'six_month_selected_precision_at_least_30pct': (evaluation['hit2_6']['selected_precision'] or 0)>=.3,
        'twelve_month_selected_precision_at_least_30pct': (evaluation['hit2_12']['selected_precision'] or 0)>=.3,
        'six_month_Brier_skill_positive': evaluation['hit2_6']['Brier_skill']>0,
        'twelve_month_Brier_skill_positive': evaluation['hit2_12']['Brier_skill']>0,
        'probability_coherence': current_metrics['probability_order_violations']==0,
        'current_mean_jaccard_at_least_080': (current_metrics['stability_mean_jaccard'] or 0)>=.8,
        'current_p05_jaccard_at_least_060': (current_metrics['stability_p05_jaccard'] or 0)>=.6,
        'historical_financial_gate_and_catalysts_validated': False,
        'survivor_and_corporate_action_bias_resolved': False,
        'prospective_accuracy_verified': False}
    result = {'version': VERSION, 'frozen_for_reproducible_research': True,
        'production_accuracy_verified': False, 'data_start': r['four_year_start'], 'data_end': asof,
        'current': current_metrics, 'gates': gates, 'all_gates_pass': all(gates.values()),
        'new_independent_blind_outcomes': 0, 'first_candidate_preserved': True,
        'evaluation': {k:{a:b for a,b in v.items() if a not in ('folds','reliability')} for k,v in evaluation.items()}}
    (out/'freeze_and_evaluation_summary.json').write_text(json.dumps(result, indent=2))
    manifest = {'version': VERSION, 'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
        'source_code_sha256': core.sha_file(core.__file__),
        'additional_source_files_sha256': {Path(__file__).name: core.sha_file(__file__)},
        'files_sha256': {p.name:core.sha_file(p) for p in sorted(out.iterdir()) if p.is_file()},
        'weights_immutable': True, 'production_accuracy_verified': False}
    (out/'freeze_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('original', 'output', 'financial', 'reference', 'official'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--asof', default='2026-10-09')
    a = p.parse_args()
    run(a.original, a.output, a.asof, a.financial, a.reference, a.official)
