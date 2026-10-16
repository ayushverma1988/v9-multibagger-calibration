"""Immutable exact inference inputs and outputs for new decisions only."""
import hashlib,io,json,platform,importlib.metadata,subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import v13_model as model

def snapshot(frame,scored,model_path,root,label):
    inputs=frame.sort_values('isin').reset_index(drop=True)
    if inputs['isin'].duplicated().any():raise ValueError('Replay input identity conflict')
    predicted=scored.set_index('isin').loc[inputs['isin']]
    model_path=Path(model_path);head_sha=hashlib.sha256(model_path.read_bytes()).hexdigest()
    buf=io.BytesIO();inputs.to_parquet(buf,index=False);raw=buf.getvalue()
    restored=pd.read_parquet(io.BytesIO(raw));heads=joblib.load(model_path)
    replay=model.probabilities(heads,restored)
    for name in ('p6','p12','loss12','joint12'):
        if name not in replay or not np.allclose(predicted[name].to_numpy(float),replay[name],atol=1e-12,rtol=0,equal_nan=False):
            raise ValueError('Exact inference replay failed: '+name)
    input_sha=hashlib.sha256(raw).hexdigest()
    out=Path(root)/'prediction_replay'/input_sha;out.mkdir(parents=True,exist_ok=True)
    def immutable(path,data):
        if path.exists() and path.read_bytes()!=data:raise ValueError('Replay snapshot overwrite blocked')
        if not path.exists():path.write_bytes(data)
    immutable(out/'inputs.parquet',raw)
    versions={k:importlib.metadata.version(k) for k in ('numpy','pandas','scipy','scikit-learn','joblib','pyarrow')}
    try:commit=subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()
    except Exception:commit='UNKNOWN'
    source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('v13*.py')}
    obj={'input_parquet_sha256':input_sha,'model_sha256':head_sha,'label':label,'rows':len(inputs),
      'source_asof':str(pd.Timestamp(inputs.date.max()).date()),'python':platform.python_version(),
      'runtime_versions':versions,'code_commit':commit,'source_code_sha256':source_hashes,
      'probabilities':{k:np.asarray(v).tolist() for k,v in replay.items()},'isins':inputs['isin'].tolist(),
      'maximum_replay_absolute_tolerance':1e-12,'accuracy_proven':False,
      'legacy_missing_inputs_or_unknown_cutoffs_reconstructed':False}
    data=json.dumps(obj,sort_keys=True,indent=2).encode();key=hashlib.sha256(data).hexdigest()
    immutable(out/(label+'_'+key+'.json'),data)
    return {'input_parquet_sha256':input_sha,'model_sha256':head_sha,'replay_verified':True,'rows':len(inputs),'receipt_sha256':key}
