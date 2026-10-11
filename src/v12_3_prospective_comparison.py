"""Immutable future NSE comparison of all four predeclared research variants.

This is a separate research registration, not a replacement of V12.1.
It reuses the original return/entry adjudication protocol without changing it.
"""
from __future__ import annotations
import argparse,json,re
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
from v12_four_year_model import sha_file
from v12_prospective_registry import digest,validate_registry
from v12_3_selective_evaluation import VARIANTS,choose
from v12_3_point_in_time import FIELDS


def comparison_records(scores,asof,now):
    clock=pd.Timestamp(now)
    if clock.tzinfo is None:raise ValueError('Registration clock requires timezone')
    clock=clock.tz_convert('UTC');day=pd.Timestamp(asof)
    if day.tzinfo is not None:raise ValueError('As-of must be a date without timezone')
    if day.tz_localize('Asia/Kolkata')+pd.Timedelta(hours=15,minutes=30)>clock:
        raise ValueError('Unfinished source session')
    if (clock.tz_convert('Asia/Kolkata').tz_localize(None).normalize()-day).days>4:
        raise ValueError('Source snapshot too old')
    s=scores[pd.to_datetime(scores.date).eq(day)].copy()
    if s.empty or s.duplicated('isin').any():raise ValueError('Empty or duplicate current forecast identity')
    if not s['isin'].map(lambda v:bool(re.fullmatch(r'INE[A-Z0-9]{8}[0-9]',str(v)))).all():
        raise ValueError('Invalid security identity')
    cols=['p_hit2_6','p_hit2_12','p_loss30_12'];a=s[cols].to_numpy(float)
    if not np.isfinite(a).all() or (a<0).any() or (a>1).any() or s.p_hit2_6.gt(s.p_hit2_12).any():
        raise ValueError('Invalid forecast probabilities')
    selected={(v,h):set(choose(s,v,h)['isin']) for v in VARIANTS for h in (6,12)}
    records=[]
    for _,r in s.sort_values('isin').iterrows():
        records.append({'isin':r['isin'],'symbol':r['symbol'],'venue':'NSE','data_asof_date':str(day.date()),
            'decision_at_utc':clock.isoformat(),**{c:float(r[c]) for c in cols},
            'dated_financial_predictors':{c:float(r[c]) if pd.notna(r[c]) and np.isfinite(r[c]) else None for c in FIELDS},
            'variant_membership':{v:{str(h):r['isin'] in selected[(v,h)] for h in (6,12)} for v in VARIANTS},
            'entry_status':'PENDING_ACTUAL_VERIFIED_FIRST_TRADE','outcome6_status':'PENDING','outcome12_status':'PENDING',
            'full_original_financial_catalyst_gate_verified':False,'trading_approved':False})
    return records


def register(scores_path,model_path,run_receipt_path,protocol_path,original_registry,output,asof='2026-10-09',now=None):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    if (out/'registration.json').exists() or (out/'public_registration_receipt.json').exists():
        raise FileExistsError('Prospective comparison already registered')
    original,original_receipt=validate_registry(original_registry)
    protocol=json.loads(Path(protocol_path).read_text());run=json.loads(Path(run_receipt_path).read_text())
    if protocol.get('variants')!=list(VARIANTS) or run['protocol_sha256']!=sha_file(protocol_path):
        raise ValueError('Changed predeclared comparison protocol')
    if any(sha_file(Path(__file__).with_name(n))!=v for n,v in run['source_hashes'].items()):
        raise ValueError('Scoring source changed after fitting')
    clock=pd.Timestamp(now or datetime.now(timezone.utc))
    if clock<pd.Timestamp(run['run_started_at_utc']):raise ValueError('Backdated relative to model fit')
    records=comparison_records(pd.read_parquet(scores_path),asof,clock)
    reg={'schema':'V12_3_PROSPECTIVE_VARIANT_COMPARISON_1','registered_at_utc':clock.tz_convert('UTC').isoformat(),
        'status':'UNPROMOTED_RESEARCH','variants':list(VARIANTS),'venue':'NSE',
        'original_V12_1_registration_sha256':original_receipt['registration_sha256'],
        'model_sha256':sha_file(model_path),'score_file_sha256':sha_file(scores_path),
        'model_run_receipt_sha256':sha_file(run_receipt_path),'selection_protocol_sha256':sha_file(protocol_path),
        'registration_code_sha256':sha_file(Path(__file__)),
        'adjudication_protocol':original['protocol'],'adjudication_protocol_sha256':original['protocol_sha256'],
        'selection_code_sha256':sha_file(Path(__file__).with_name('v12_3_selective_evaluation.py')),
        'original_registry_modified':False,'historical_results_are_development_only':True,
        'production_approved':False,'unseen_matured_outcomes':0,'records':records}
    receipt={'schema':reg['schema'],'registered_at_utc':reg['registered_at_utc'],'registration_sha256':digest(reg),
        'model_sha256':reg['model_sha256'],'predictions':len(records),'venue':'NSE',
        'selection_counts':{v:{str(h):sum(r['variant_membership'][v][str(h)] for r in records) for h in (6,12)} for v in VARIANTS},
        'all_four_variants_registered':True,'unseen_matured_outcomes':0,'production_approved':False,
        'BSE_prospective_registration_included':False}
    with (out/'registration.json').open('x') as f:json.dump(reg,f,indent=2,allow_nan=False)
    with (out/'public_registration_receipt.json').open('x') as f:json.dump(receipt,f,indent=2,allow_nan=False)
    return receipt


def validate_comparison(output):
    root=Path(output);reg=json.loads((root/'registration.json').read_text());receipt=json.loads((root/'public_registration_receipt.json').read_text())
    if digest(reg)!=receipt['registration_sha256']:raise ValueError('Comparison registration altered')
    if reg['schema']!='V12_3_PROSPECTIVE_VARIANT_COMPARISON_1' or digest(reg['adjudication_protocol'])!=reg['adjudication_protocol_sha256']:
        raise ValueError('Changed outcome protocol')
    return reg,receipt


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scores',required=True);p.add_argument('--model',required=True)
    p.add_argument('--run-receipt',required=True);p.add_argument('--protocol',required=True)
    p.add_argument('--original-registry',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(register(a.scores,a.model,a.run_receipt,a.protocol,a.original_registry,a.output),indent=2))
