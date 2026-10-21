"""Compact, public numeric source receipts for the dated current input gaps.

Full page originals remain in the private source archive. This packet carries
numbers and provenance only, so independently collected evidence survives a
different runner's provider access limits.
"""
import argparse
import base64
import gzip
import hashlib
import json
from pathlib import Path
import pandas as pd
from v13_master_data import RAW_FIELDS, AMBIGUOUS

def pack(root,output,baseline=None):
    root=Path(root);master=pd.read_parquet(root/'master_data/master_inputs.parquet')
    current={r['isin']:r for r in master.to_dict('records')};observations={}
    columns=('symbol','isin','reporting_mode','period_kind','period_end','metric',
        'value_INR','rounding_half_width_INR','display_value','display_unit',
        'publisher','provider_family','source_url','source_sha256','first_observed_utc',
        'identity_proof','definition_status','upstream_field','identity_source_url',
        'identity_source_sha256','reporting_basis_proof','basis_anchor_sources',
        'reporting_basis_independently_confirmed','calculation','supporting_upstream_fields',
        'financial_definition_reference','cross_source_validation_required')
    for path in sorted((root/'master_data/provider_inputs').glob('*.json')):
        for r in json.loads(path.read_text()):
            issuer=current.get(r.get('isin'));kind=r.get('period_kind')
            if not issuer or r.get('metric') not in set(RAW_FIELDS)|AMBIGUOUS:continue
            if r.get('reporting_mode')!=issuer['preferred_reporting_basis']:continue
            if r.get('period_end')!=issuer.get(str(kind)+'_period_end'):continue
            key=tuple(r.get(k) for k in ('isin','reporting_mode','period_kind','period_end','metric','provider_family','source_sha256','first_observed_utc','value_INR'))
            observations[key]={k:r[k] for k in columns if k in r}
    if baseline:
        previous=json.loads(gzip.decompress(base64.b64decode(Path(baseline).read_text())))
        fields=('isin','reporting_mode','period_kind','period_end','metric','provider_family','source_sha256','first_observed_utc','value_INR')
        existing={tuple(r.get(k) for k in fields) for r in previous.get('observations',[])}
        observations={k:v for k,v in observations.items() if k not in existing}
    packet={'observations':list(observations.values()),'sector_sources':[],
        'scope':'CURRENT_DATED_INPUT_SECTIONS_NUMERIC_FACTS_AND_RECEIPTS',
        'frozen_predictions_modified':False,'required_additional_sources':1}
    raw=json.dumps(packet,separators=(',',':'),allow_nan=False).encode()
    path=Path(output);path.parent.mkdir(parents=True,exist_ok=True)
    encoded=base64.b64encode(gzip.compress(raw,mtime=0)).decode();path.write_text(encoded+'\n')
    return {'observations':len(observations),'compressed_base64_bytes':path.stat().st_size,
        'packet_sha256':hashlib.sha256(raw).hexdigest()}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--state',required=True);p.add_argument('--output',required=True);p.add_argument('--baseline');a=p.parse_args();print(json.dumps(pack(a.state,a.output,a.baseline)))
