"""Verify and replay frozen V12 inference. This module never fits weights."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from v12_four_year_model import FEATURES, TARGETS, VERSION, current_financial_gate, score, sha_file


def predict(frozen, predictors, financial, reference, official, asof, output):
    frozen = Path(frozen)
    manifest = json.loads((frozen/'freeze_manifest.json').read_text())
    if sha_file(Path(__file__).with_name('v12_four_year_model.py')) != manifest['source_code_sha256']:
        raise ValueError('Frozen feature/selector/calibration source code changed')
    for name, expected in manifest.get('additional_source_files_sha256', {}).items():
        if Path(name).name != name or sha_file(Path(__file__).with_name(name)) != expected:
            raise ValueError('Frozen conditional model source changed')
    for name in ('frozen_model.joblib', 'frozen_recipe.json'):
        if sha_file(frozen/name) != manifest['files_sha256'][name]:
            raise ValueError('Frozen model/recipe bytes changed: '+name)
    package = joblib.load(frozen/'frozen_model.joblib')
    if package['version'] not in (VERSION, 'V12.1-4Y-coherent-calendar6-12M-research-20261010') or tuple(package['features']) != FEATURES:
        raise ValueError('Wrong frozen model schema')
    x = pd.read_parquet(predictors)
    if any(c.startswith(('hit2_', 'late_hit2_', 'loss30_', 'mature_date_', 'endpoint_return_', 'peak_return_', 'p_')) for c in x):
        raise ValueError('Future outcomes or existing predictions inside source predictors')
    if x.duplicated(['date', 'symbol']).any() or not pd.to_datetime(x.date).eq(pd.Timestamp(asof)).all():
        raise ValueError('Duplicate or wrong forecast date')
    if pd.Timestamp(asof) < pd.Timestamp(package['asof']):
        raise ValueError('Frozen model cannot make retroactive forecasts before its freeze')
    if (pd.Timestamp(asof)-pd.Timestamp(package['asof'])).days > 190:
        raise ValueError('Frozen model review overdue; refuse unnoticed stale inference')
    if package.get('probability_model') == 'conditional_hazards':
        from v12_coherent_hazard_model import probabilities
        x['technical_pass'] = (x.rsi14.between(45, 90) & x.distance_ma50.gt(0)
                               & x.log_ret63.le(np.log(1.75)))
        x['entry_sleeve'] = np.where(x.distance_ma200.gt(0), 'CONFIRMED_GROWTH', 'EMERGING_TURNAROUND')
        for name, values in probabilities(package['heads'], x).items():
            x[name] = values
    else:
        x['technical_pass'] = (x.rsi14.between(50, 80) & x.distance_ma50.gt(0)
            & x.distance_ma200.gt(0) & x.log_ret63.le(np.log(1.75)))
        for target in TARGETS:
            x['p_'+target] = score(package['heads'][target], x)
    x['probability_coherence'] = x.p_hit2_6.le(x.p_hit2_12)
    q = current_financial_gate(x, pd.read_parquet(financial), pd.read_parquet(reference),
                              pd.read_parquet(official), asof)
    q['high_conviction'] = (q.current_quality_status.eq('CURRENT_SOURCE_CHECKS_PASS')
        & q.p_hit2_6.ge(.10) & q.p_hit2_12.ge(.25) & q.p_loss30_12.le(.45) & q.probability_coherence)
    q['model_version'] = package['version']
    q['probability_scope'] = 'MARKET_MODEL_RESEARCH_ESTIMATE; FINANCIAL_GATE_NOT_HISTORICALLY_CALIBRATED'
    q['production_accuracy_verified'] = False
    q = q.sort_values(['p_hit2_12', 'symbol'], ascending=[False, True])
    path = Path(output)
    if path.exists():
        raise ValueError('Refuse to overwrite an existing forecast observation')
    path.parent.mkdir(parents=True, exist_ok=True)
    q.to_parquet(path, index=False)
    return q


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('frozen', 'predictors', 'financial', 'reference', 'official', 'asof', 'output'):
        p.add_argument('--'+name, required=True)
    a = p.parse_args()
    q = predict(a.frozen, a.predictors, a.financial, a.reference, a.official, a.asof, a.output)
    print(json.dumps({'scored_rows': len(q), 'high_conviction_rows': int(q.high_conviction.sum()),
                      'weights_refit': False, 'production_accuracy_verified': False}))
