"""Apply recovered facts to the same current source cohort without reranking."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from v11_4_systematic_run import validate_source_only, screen_audit
from v11_4_four_family_live_screener import evaluate_family
from v11_4_robust_research_model import FEATURES
from v11_4_source_blocker_recovery import EvidenceStore
import v11_4_standalone_train_walkforward as core
from v11_4_forward_research_release import apply_frozen_calibration
import v11_4_live_nse_market_catalyst as exchange


def merge_verified_overlay(features, official_market, companies, universe):
    x = validate_source_only(features).copy()
    if official_market['symbol'].duplicated().any():
        raise ValueError('Ambiguous same-day market security identity')
    if 'isin' in x:
        raise ValueError('Source feature identity must be verified against original market archive')
    x = x.merge(official_market[['symbol', 'isin', 'close']].rename(columns={'close': 'identity_check_close'}),
                on='symbol', how='left', validate='1:1')
    if x['isin'].isna().any() or not (x['identity_check_close']-x['close']).abs().le(.10).all():
        raise ValueError('Recovered source close/ISIN disagrees with frozen current features')
    x = x.merge(universe[['isin', 'amfi_size_category', 'verified_small_or_mid_reference']],
                on='isin', how='left', validate='m:1')
    count, rejected = 0, 0
    if len({c['symbol'] for c in companies}) != len(companies):
        raise ValueError('Duplicate financial company evidence')
    for c in companies:
        mask = x['symbol'].eq(c['symbol'])
        if not mask.any():
            continue
        if not x.loc[mask, 'isin'].eq(c['isin']).all():
            rejected += 1
            continue
        for key, value in c['financial_metrics'].items():
            if key == 'latest_annual_period_end':
                continue
            if key in features:
                raise ValueError('Financial overlay attempted to overwrite an original predictor')
            if key not in x:
                x[key] = pd.Series([None] * len(x), dtype='object')
            x.loc[mask, key] = value
        count += 1
    # Every actual model input and original source order remains byte-level
    # equivalent in values. Newly recovered fields are an audit companion.
    if not x[list(FEATURES)].equals(features[list(FEATURES)].reset_index(drop=True)):
        raise ValueError('Enrichment changed tested predictor inputs')
    return x, {'source_company_rows_joined': count, 'mismatched_current_ISINs_rejected': rejected,
               'same_source_cohort_preserved': len(x) == len(features),
               'all_original_predictor_values_unchanged': True}


def run(features_path, recovery, config_path, output, model_path=None):
    p, out = Path(recovery), Path(output)
    out.mkdir(parents=True, exist_ok=True)
    store = EvidenceStore(p/'raw_PRIVATE')
    # Retain the exact original daily archive bytes as well as their proof.
    original_get = exchange.get_bytes
    exchange.get_bytes = lambda url, session=None: store.get(url)[0]
    try:
        market, proof = exchange.exchange_primary_for_day('2026-10-09')
    finally:
        exchange.get_bytes = original_get
    market.to_parquet(p/'same_day_official_NSE_market_identity_PRIVATE.parquet', index=False)
    (p/'official_market_identity_validation.json').write_text(json.dumps(proof, indent=2))
    (p/'market_identity_source_receipts_PRIVATE.json').write_text(json.dumps(store.receipts, indent=2))
    features = pd.read_parquet(features_path)
    companies = json.loads((p/'company_enrichment_PRIVATE.json').read_text())
    universe = pd.read_parquet(p/'NSE_BSE_AMFI_reference_universe_PRIVATE.parquet')
    enriched, joins = merge_verified_overlay(features, market, companies, universe)
    config = json.loads(Path(config_path).read_text())
    before_table, before = screen_audit(features, config, out/'before')
    after_table, after = screen_audit(enriched, config, out/'after')
    enriched.to_parquet(out/'same_original_cohort_enriched_source_PRIVATE.parquet', index=False)
    families = config['conditions']
    coverage = {}
    for key, condition in families.items():
        coverage[key] = {f: {'before': int(features[f].notna().sum()) if f in features else 0,
                              'after': int(enriched[f].notna().sum()) if f in enriched else 0}
                         for f, op, threshold in condition['hard_rules']}
    report = {'scope': 'SAME_FROZEN_SOURCE_COHORT_POST_DECISION_FINANCIAL_ENRICHMENT_AUDIT',
              'market_date': '2026-10-09', 'same_current_feature_rows': len(features),
              'source_features_sha256': hashlib.sha256(Path(features_path).read_bytes()).hexdigest(),
              'same_day_dual_NSE_close_validation': proof, 'joins': joins,
              'official_AMFI_size_category_available_in_current_cohort': int(enriched.amfi_size_category.notna().sum()),
              'official_AMFI_small_mid_reference_in_current_cohort': int(enriched.verified_small_or_mid_reference.fillna(False).sum()),
              'financial_metric_coverage_by_family': coverage,
              'additional_user_checks_before': before['additional_user_checks'],
              'additional_user_checks_after': after['additional_user_checks'],
              'four_independent_families_before': before['families'],
              'four_independent_families_after': after['families'],
              'AMFI_six_month_average_not_inserted_as_current_market_cap': True,
              'exact_quality_and_valuation_fields_still_incomplete': True,
              'no_ranking_weights_or_source_candidates_changed': True,
              'post_decision_recovery_not_used_for_new_blind_predictions': True,
              'production_approved': False}
    if model_path:
        package = joblib.load(model_path)
        original = core.safe_featureize(features)
        revised = core.safe_featureize(enriched)
        raw_before = package['model'].predict_proba(original[list(FEATURES)])[:, 1]
        raw_after = package['model'].predict_proba(revised[list(FEATURES)])[:, 1]
        calibrated_before = apply_frozen_calibration(raw_before, package['calibration'])
        calibrated_after = apply_frozen_calibration(raw_after, package['calibration'])
        if not np.array_equal(raw_before, raw_after) or not np.array_equal(calibrated_before, calibrated_after):
            raise ValueError('Supplemental fields changed actual existing model probabilities')
        report['actual_model_probability_arrays_identical_before_and_after'] = True
        report['actual_model_rows_scored_for_invariance'] = len(original)
        report['frozen_model_sha256'] = hashlib.sha256(Path(model_path).read_bytes()).hexdigest()
    (out/'enriched_current_source_audit_summary.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='financial_metric_coverage_by_family'}, indent=2))
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--features', required=True); p.add_argument('--recovery', required=True)
    p.add_argument('--config', required=True); p.add_argument('--output', required=True)
    p.add_argument('--model', default='')
    a=p.parse_args();run(a.features, a.recovery, a.config, a.output, a.model or None)


if __name__ == '__main__':main()
