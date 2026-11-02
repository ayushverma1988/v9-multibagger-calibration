"""Prepare a read-only, hash-bound research dataset from preserved recovery bytes."""
import argparse,hashlib,json,os,tarfile,zipfile
from datetime import datetime,timezone
from pathlib import Path
import pandas as pd
from v13_recovery_restore import restore_delta

def run(archive,out):
    from huggingface_hub import HfApi,hf_hub_download
    from v13_checkpoint import ARCHIVE
    from v13_history_views import histories
    from v13_frozen_integrity import FrozenGuard
    from v13_information_clock import research_cutoffs
    import v13_features as features
    import v13_collection as collection
    import v13_redevelopment as rd
    from v13_available_financials import enrich as available_enrich
    from v13_runner import current_frame
    from v13_financial_evidence import enrich_evidence,apply_scenarios
    from v13_catalyst_execution import join
    import v13_statement_eps as eps
    out=Path(out);out.mkdir(parents=True,exist_ok=True);root=out/'state'
    token=os.environ['HF_ARCHIVE_TOKEN'];api=HfApi(token=token)
    if not api.repo_info(ARCHIVE,repo_type='dataset').private:raise ValueError('Private archive required')
    with zipfile.ZipFile(archive) as z:
        descriptor=json.loads(z.read('v13_checkpoint_recovery.json'))
        raw=z.read('v13_checkpoint_recovery.tar.gz')
        if len(raw)!=descriptor['bytes'] or hashlib.sha256(raw).hexdigest()!=descriptor['sha256']:raise ValueError('Preserved recovery bytes differ from receipt')
        bundle=out/'recovery.tar.gz';bundle.write_bytes(raw)
        recovery=restore_delta(bundle,root,lambda p:hf_hub_download(ARCHIVE,p,repo_type='dataset',token=token),descriptor['sha256'])
    guard=FrozenGuard(root)
    attempt=json.loads((root/'latest_run.json').read_text());asof=attempt['asof']
    universe=pd.read_parquet(root/'universe.parquet');wanted=set(zip(universe['isin'],universe['symbol']))
    chosen,audit=histories(root/'history/securities',wanted)
    view=out/'history_view/securities';view.mkdir(parents=True,exist_ok=True)
    for i,(path,bars) in enumerate(chosen.values()):os.link(path,view/(str(i)+'.parquet'))
    blank=out/'empty.parquet';pd.DataFrame(columns=['date','isin']).to_parquet(blank,index=False)
    market=features.build(view.parent,blank,out/'market',asof)
    market=research_cutoffs(market,datetime.now(timezone.utc))
    wide=pd.read_parquet(root/'financial_wide.parquet')
    matrix=collection.enrich(market,wide)
    matrix=collection.announcement_counts(matrix,root,asof)
    matrix=rd.attach(matrix,view)
    # Source-bound assessment at every historical row's own cutoff. This is
    # retrospective reconstruction, never an original prospective decision.
    pieces=[]
    for date,g in matrix.groupby('date',sort=True):
        enriched,_=available_enrich(g.reset_index(drop=True),wide);pieces.append(enriched)
        print('Dated financial assessment',str(date.date()),len(g),flush=True)
    matrix=pd.concat(pieces,ignore_index=True)
    # Validate the actual historical test sessions against primary NSE bars.
    # A failed/mismatched source cannot acquire a passing historical identity.
    from v12_3_official_monitor import collect as collect_official
    import time
    dates=sorted(matrix.loc[pd.to_datetime(matrix.mature_date_6).le(pd.Timestamp(asof)),'date'].unique())[-12:]
    matrix['historical_official_identity_match']=False
    primary_audit=[]
    for date in dates:
        day=str(pd.Timestamp(date).date())
        try:
            result=collect_official(day,out/'historical_official')
            primary=pd.read_parquet(out/'historical_official'/(day+'_official_market.parquet')).set_index('isin')
            g=matrix[matrix.date.eq(date)]
            accepted=0
            for i,row in g.iterrows():
                if row['isin'] not in primary.index:continue
                q=primary.loc[row['isin']]
                if isinstance(q,pd.DataFrame):continue
                good=q.symbol==row.symbol and q.source_status=='VALID_POSITIVE_VOLUME_SESSION' and q.volume==row.volume
                good=good and all(abs(q[c]-row[c])<=.011 for c in ['open','high','low','close'])
                if good:
                    matrix.at[i,'historical_official_identity_match']=True
                    matrix.at[i,'series']=q.series
                    accepted+=1
            primary_audit.append({'date':day,'status':'PRIMARY_SESSION_CHECKED','accepted':accepted,'rows':len(g),'source_sha256':result['source_sha256']})
        except Exception as exc:primary_audit.append({'date':day,'status':'SOURCE_UNAVAILABLE','error_type':type(exc).__name__})
        time.sleep(.25)
    matrix.to_parquet(out/'matrix.parquet',index=False)
    # Current scoring reads only the exact whole histories accepted above.
    original=root/'history/securities';temporary=root/'history/original_securities'
    original.rename(temporary);original.symlink_to(view.resolve(),target_is_directory=True)
    try:current,_=current_frame(root,asof)
    finally:original.unlink();temporary.rename(original)
    current=enrich_evidence(current,wide,root,asof);current=eps.enrich(current,root)
    current=apply_scenarios(join(current,root),root)
    current=rd.attach(current,view)
    current,_=available_enrich(current,wide)
    current.to_parquet(out/'current.parquet',index=False)
    observed=pd.read_parquet(root/'prospective_observations.parquet')
    integrity=guard.check(observed)
    summary={'source_date':asof,'prepared_at_utc':datetime.now(timezone.utc).isoformat(),
      'recovery':recovery,'current_rows':len(current),'historical_rows':len(matrix),
      'decision_dates':matrix.date.nunique(),'canonical_histories':len(chosen),
      'quarantined_histories':len(audit['quarantined']),'frozen_integrity':integrity,
      'current_listing_survivor_bias':True,'historical_price_vintage_limit':True,
      'historical_primary_session_checks':primary_audit,
      'historical_official_identity_independently_verified':False,
      'primary_check_scope':'Per-row flag is true only after exact NSE session identity/OHLC/volume match; full population and action intervals remain unverified',
      'historical_policy_is_reconstructed_not_prospective':True,
      'remote_checkpoint_written':False}
    (out/'prepared_summary.json').write_text(json.dumps(summary,indent=2,default=str))
    for name in ['active_heads.joblib','challenger/research_heads.joblib']:
        target=out/Path(name).name;target.write_bytes((root/name).read_bytes())
    with tarfile.open(out/'research_inputs.tar.gz','w:gz') as tar:
        for name in ['matrix.parquet','current.parquet','active_heads.joblib','research_heads.joblib','prepared_summary.json']:
            tar.add(out/name,arcname=name)
    print(json.dumps(summary,default=str),flush=True)
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',required=True);p.add_argument('--out',required=True);a=p.parse_args();run(a.archive,a.out)
