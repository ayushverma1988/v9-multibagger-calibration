"""Preserve accepted history and recover source-bound entry vintages only."""
from pathlib import Path
import io,json,hashlib,re
import numpy as np
import pandas as pd
from v12_public_history import bounds

def validate_refresh(previous,fresh,asof,symbol,isin):
    if previous.empty:return
    if set(previous.symbol)!={symbol} or set(previous['isin'])!={isin}:
        raise ValueError('History identity changed; original retained for primary review')
    low,high=bounds(asof)
    old=pd.to_datetime(previous.date);new=pd.to_datetime(fresh.date)
    missing=set(old[old.between(low,high)])-set(new)
    if missing:raise ValueError('Provider refresh omitted '+str(len(missing))+' accepted sessions; original history retained')

def cache_digest(row):
    return hashlib.sha256(json.dumps(row,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def validated_cached_entry(cached,official,expected_date,symbol,isin):
    """Recheck a past source receipt; never certifies current returns/actions."""
    value=cached.get('evidence',{})
    try:
        if cache_digest(value)!=cached.get('evidence_sha256'):return None
    except (ValueError,TypeError):return None
    needed={'isin','symbol','entry_date','entry_open_INR','entry_official_sha256','entry_provider_sha256','entry_evidence_recorded_at_utc'}
    if not needed.issubset(value) or value['isin']!=isin or value['symbol']!=symbol or value['entry_date']!=str(pd.Timestamp(expected_date).date()):return None
    if not all(re.fullmatch('[0-9a-f]{64}',str(value[k])) for k in ('entry_official_sha256','entry_provider_sha256')):return None
    try:
        clock=pd.Timestamp(value['entry_evidence_recorded_at_utc'])
        if clock.tzinfo is None or clock<pd.Timestamp(expected_date).tz_localize('Asia/Kolkata')+pd.Timedelta(hours=9,minutes=15):return None
    except (ValueError,TypeError):return None
    q=official[pd.to_datetime(official.date).eq(pd.Timestamp(expected_date))&official['isin'].eq(isin)&official.symbol.eq(symbol)]
    if len(q)!=1:return None
    r=q.iloc[0];price=value['entry_open_INR']
    if isinstance(price,(bool,np.bool_)) or not isinstance(price,(int,float,np.number)):return None
    if r.source_status!='VALID_POSITIVE_VOLUME_SESSION' or r.source_sha256!=value['entry_official_sha256'] or not np.isfinite(price) or price<=0 or abs(float(r.open)-price)>.011:return None
    return value

def recover_known_entries(root,config,fetch):
    """One-time repair from a verified old private snapshot, no forecast writes.

    fetch takes only an approved immutable archive object path. Unit tests can
    inject an offline fetcher; operational access uses the existing worker.
    """
    from v13_checkpoint import validate_object
    root=Path(root);out=root/'action_audit';out.mkdir(exist_ok=True)
    marker=out/'entry_history_recovery.json'
    if marker.exists():
        prior=json.loads(marker.read_text())
        if prior.get('complete') and prior.get('manifest_sha256')==config['manifest_sha256']:return prior
    if not re.fullmatch('[A-Za-z0-9_-]+',str(config['snapshot'])) or not re.fullmatch('[0-9a-f]{64}',str(config['manifest_sha256'])):raise ValueError('Unsafe entry recovery snapshot')
    raw=Path(fetch('v13/state/snapshots/'+config['snapshot']+'.json')).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=config['manifest_sha256']:raise ValueError('Entry recovery manifest hash mismatch')
    manifest=json.loads(raw);items={z['path']:z for z in manifest['files']}
    def original(item):
        validate_object(item);data=Path(fetch(item['object_path'])).read_bytes()
        if len(data)!=item['bytes'] or hashlib.sha256(data).hexdigest()!=item['sha256']:raise ValueError('Entry recovery original mismatch')
        return data
    audits=sorted(k for k in items if re.fullmatch(r'action_audit/\d{4}-\d{2}-\d{2}_audit.json',k))
    if not audits:raise ValueError('Known entry checkpoint has no original audit')
    reference=json.loads(original(items[audits[-1]]));observations=pd.read_parquet(root/'prospective_observations.parquet')
    symbols=observations.set_index(['registry_sha256','isin']).symbol.to_dict()
    cache_path=out/'verified_entry_cache.json';cache=json.loads(cache_path.read_text()) if cache_path.exists() else {}
    official_paths=list((root/'official').glob('*_official_market.parquet'))
    official=pd.concat([pd.read_parquet(p) for p in official_paths],ignore_index=True) if official_paths else pd.DataFrame()
    restored=set();cached=0;eligible=0
    for proof in reference.get('records',[]):
        if proof.get('entry_verified') is not True:continue
        key=(proof['registry_sha256'],proof['isin']);symbol=symbols.get(key)
        if not symbol:continue
        eligible+=1
        evidence={k:proof[k] for k in ('isin','entry_date','entry_open_INR','entry_official_sha256','entry_provider_sha256','entry_evidence_recorded_at_utc')}
        evidence['symbol']=symbol
        item={'evidence':evidence,'evidence_sha256':cache_digest(evidence),'recovery_manifest_sha256':config['manifest_sha256']}
        if official.empty or validated_cached_entry(item,official,evidence['entry_date'],symbol,proof['isin']) is None:continue
        rel='history/securities/'+hashlib.sha256(symbol.encode()).hexdigest()+'.parquet'
        if rel not in items:continue
        current=root/rel;h=pd.read_parquet(current) if current.exists() else pd.DataFrame()
        q=h[pd.to_datetime(h.date).eq(pd.Timestamp(proof['entry_date']))] if not h.empty else h
        if q.empty or set(q['isin'])!={proof['isin']} or set(q.symbol)!={symbol} or abs(float(q.open.iloc[0])-proof['entry_open_INR'])>.011:
            old=original(items[rel]);d=pd.read_parquet(io.BytesIO(old));z=d[pd.to_datetime(d.date).eq(pd.Timestamp(proof['entry_date']))]
            if len(z)!=1 or z.source_sha256.iloc[0]!=proof['entry_provider_sha256'] or z['isin'].iloc[0]!=proof['isin'] or z.symbol.iloc[0]!=symbol or abs(float(z.open.iloc[0])-proof['entry_open_INR'])>.011:raise ValueError('Old entry history does not bind original proof')
            if current.exists():
                quarantine=root/'history_quarantine'/('history_'+hashlib.sha256(current.read_bytes()).hexdigest()+'.parquet')
                quarantine.parent.mkdir(exist_ok=True);quarantine.write_bytes(current.read_bytes())
            current.parent.mkdir(parents=True,exist_ok=True);current.write_bytes(old);restored.add(symbol)
        # The old full audit is hash matched, and current primary price is
        # rechecked. Cache only entry facts, never old outcome certifications.
        cache[proof['registry_sha256']+'|'+proof['isin']]=item;cached+=1
    cache_path.write_text(json.dumps(cache,indent=2))
    result={'complete':bool(eligible and cached==eligible),'source_snapshot':config['snapshot'],'manifest_sha256':config['manifest_sha256'],
      'reference_verified_receipts':eligible,
      'entry_receipts_recovered':cached,'history_files_restored':len(restored),'restored_public_symbols':sorted(restored),
      'original_forecasts_or_heads_changed':False,'outcomes_or_action_intervals_certified':False}
    marker.write_text(json.dumps(result,indent=2));return result

def recover_from_seed(root,seed):
    path=Path(seed)
    if not path.exists():return {'status':'NO_RECOVERY_CONFIGURATION'}
    if not (Path(root)/'prospective_observations.parquet').exists() or not list((Path(root)/'official').glob('*_official_market.parquet')):
        return {'status':'WAITING_FOR_EXISTING_LEDGER_AND_PRIMARY_PRICES','complete':False}
    import os
    from huggingface_hub import hf_hub_download
    from v13_checkpoint import ARCHIVE
    return recover_known_entries(root,json.loads(path.read_text()),lambda p:hf_hub_download(ARCHIVE,p,repo_type='dataset',token=os.environ.get('HF_ARCHIVE_TOKEN')))
