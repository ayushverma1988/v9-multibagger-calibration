"""Daily resumable research worker; independent of a ChatGPT token session.

The paper champion keeps a fixed head. Retraining makes a challenger and never
overwrites frozen predictions. Durable state is archived using the existing
private Hugging Face credential, with GitHub artifacts as a second checkpoint.
"""
import argparse,base64,hashlib,io,json,os,shutil,tarfile
from datetime import datetime,timezone
from pathlib import Path
import joblib,numpy as np,pandas as pd
from v12_3_official_monitor import collect as collect_official
import v13_collection as collection
import v13_features as features
import v13_model as model
from v13_lifecycle import register,adjudicate,promotion_gate,paired_observations,validate_registry

ARCHIVE='ayushverma1988/v10-multibagger-archive'

def restore(root):
    from v13_checkpoint import restore as restore_incremental
    return restore_incremental(root)

def verify_manifest(root):
    path=root/'state_manifest.json'
    if not path.exists():raise ValueError('Missing state integrity manifest')
    manifest=json.loads(path.read_text())
    for item in manifest['files']:
        p=root/item['path']
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=item['sha256']:raise ValueError('Checkpoint file integrity failure: '+item['path'])

def initialize(root,seed):
    root.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((seed/'manifest.json').read_text())
    for name,hash_ in manifest['files'].items():
        raw=base64.b64decode((seed/(name+'.b64')).read_text())
        if hashlib.sha256(raw).hexdigest()!=hash_:raise ValueError('Seed integrity failure')
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
    shutil.copy2(seed/'initial_registry.json',root/'initial_registry.json')
    reg=json.loads((root/'initial_registry.json').read_text());folder=root/'registries'/'champion';folder.mkdir(parents=True,exist_ok=True)
    from v13_lifecycle import digest
    (folder/(digest(reg)+'.json')).write_text(json.dumps(reg,indent=2))
    (root/'active_model.json').write_text(json.dumps({'mode':'PAPER_RESEARCH','accuracy_approved':False,'sha256':manifest['files']['active_heads.joblib'],'version':'V13','initialised_at_utc':datetime.now(timezone.utc).isoformat()},indent=2))

def save(root):
    from v13_checkpoint import save as save_incremental
    return save_incremental(root)

def last_completed_session(root,day):
    errors=[]
    for offset in range(8):
        candidate=pd.Timestamp(day)-pd.Timedelta(days=offset)
        if candidate.weekday()>4:continue
        try:
            report=collect_official(str(candidate.date()),root/'official')
            return str(candidate.date()),report,errors
        except Exception as e:errors.append({'date':str(candidate.date()),'error':str(e)[:160]})
    raise RuntimeError('No verified completed NSE session; '+json.dumps(errors))

def current_frame(root,asof):
    rows=[];history={};low,_=collection.bounds(asof)
    for path in sorted((root/'history'/'securities').glob('*.parquet')):
        bars=pd.read_parquet(path);bars.date=pd.to_datetime(bars.date);bars=bars[bars.date.ge(low)]
        if bars.empty:continue
        history[bars['isin'].iloc[0]]=bars
        d=features.daily_features(bars,asof)
        current=d[d.date.eq(pd.Timestamp(asof))&d.feature_eligible]
        if len(current)==1:rows.append(current.iloc[0].to_dict())
    if not rows:return pd.DataFrame(),history
    d=pd.DataFrame(rows)
    for source,target in [('log_ret63','relative_strength63'),('log_ret20','relative_strength20'),('momentum126','relative_strength126'),('momentum252_skip21','relative_strength252_skip21')]:d[target]=d[source].rank(pct=True)
    d['market_breadth63']=d.log_ret63.gt(0).mean();d['market_median_ret63']=d.log_ret63.median();d['market_dispersion63']=d.log_ret63.std()
    d['trend_regime_interaction']=d.relative_strength63*(d.market_breadth63-.5)
    d['liquidity_volume_interaction']=d.volume_confirmation*d.log_liquidity63
    d['earnings_market_confirmation']=np.nan
    if (root/'financial_wide.parquet').exists():d=collection.enrich(d,pd.read_parquet(root/'financial_wide.parquet'))
    d=collection.announcement_counts(d,root,asof)
    for c in model.FEATURES:
        if c not in d:d[c]=np.nan
    for c in ['four_profitable_quarters',*model.old.BASE_FIELDS]:
        if c not in d:d[c]=np.nan
    # Check raw latest OHLC and volume against whole-market official data.
    o=pd.read_parquet(root/'official'/(asof+'_official_market.parquet'))
    m=d.merge(o[['isin','symbol','open','high','low','close','volume','source_status']],on='isin',suffixes=('','_official'),validate='1:1')
    valid=m.symbol.eq(m.symbol_official)&m.source_status.eq('VALID_POSITIVE_VOLUME_SESSION')&m.volume.eq(m.volume_official)
    for c in ('open','high','low','close'):valid&=(m[c]-m[c+'_official']).abs().le(.011)
    d=m.loc[valid].copy();return d,history

def forecasts(heads,frame,root,label):
    d=frame.copy()
    for k,v in model.probabilities(heads,d).items():d[k]=v
    if not {'p6','p12','loss12','joint12'}.issubset(d):return None
    d['selected_verified']=d['isin'].isin(model.ranked(d,'verified')['isin'])
    d['selected_discovery']=d['isin'].isin(model.ranked(d,'discovery')['isin'])
    model_path=root/'active_heads.joblib' if label=='champion' else root/'challenger'/'research_heads.joblib'
    sha=hashlib.sha256(model_path.read_bytes()).hexdigest()
    for previous in sorted((root/'registries'/label).glob('*.json')):
        old=validate_registry(previous)
        if old['model_sha256']==sha and all(r['source_asof']==str(pd.Timestamp(frame.date.max()).date()) for r in old['records']):
            return previous.name
    path=register(d,sha,root/'registries'/label)
    d[d.selected_verified|d.selected_discovery].to_json(root/(label+'_latest_watchlist.json'),orient='records',date_format='iso',indent=2)
    return str(path.name)

def run(root,seed,day,history_budget=400,filing_budget=400,retrain=False,local=False):
    root=Path(root);seed=Path(seed)
    if not local:
        if not restore(root):initialize(root,seed)
    report={'started_at_utc':datetime.now(timezone.utc).isoformat(),'mode':'PAPER_RESEARCH','accuracy_approved':False,'steps':{},'errors':[]}
    try:
        asof,official,errors=last_completed_session(root,day);report['asof']=asof;report['steps']['official']=official
        # Every completed phase is checkpointed. Interrupted work resumes from
        # accepted filing versions and the public-history age queue.
        collection.universe(root,asof)
        report['steps']['catalogs']=collection.current_catalogs(root,asof)
        report['steps']['financials']=collection.collect_financials(root,asof,filing_budget)
        if not local:save(root)
        report['steps']['history']=collection.collect_histories(root,asof,history_budget)
        if not local:save(root)
        frame,history=current_frame(root,asof)
        report['current_verified_market_rows']=len(frame)
        # Partial coverage is disclosed, not mistaken for a full-universe scan.
        report['public_universe_rows']=len(pd.read_parquet(root/'universe.parquet'))
        if not frame.empty:
            heads=joblib.load(root/'active_heads.joblib');report['champion_registry']=forecasts(heads,frame,root,'champion')
            if (root/'challenger'/'research_heads.joblib').exists():report['challenger_registry']=forecasts(joblib.load(root/'challenger'/'research_heads.joblib'),frame,root,'challenger')
        originals=[pd.read_parquet(p) for p in sorted((root/'official').glob('*_official_market.parquet'))]
        official_all=pd.concat(originals,ignore_index=True);observations=[]
        for p in sorted((root/'registries').glob('*/*.json')):observations.extend(adjudicate(p,official_all,history,asof))
        if observations:pd.DataFrame(observations).to_parquet(root/'prospective_observations.parquet',index=False)
        report['prospective_status_counts']=pd.Series([r['status'] for r in observations],dtype=object).value_counts().to_dict()
        report['raw_price_2x_alerts']=sum(r['raw_price_2x_observed'] for r in observations)
        report['independently_verified_mature_outcomes']=0
        # Monthly refresh learns coefficients/blends automatically from the
        # trailing four-year, maturity-purged data. It is a new paper challenger.
        marker=root/'last_retrain_month.txt';month=asof[:7]
        if (retrain or not marker.exists() or marker.read_text()!=month) and len(history)>=500:
            blank=pd.DataFrame(columns=['date','isin']);blank.to_parquet(root/'empty_previous.parquet',index=False)
            market=features.build(root/'history',root/'empty_previous.parquet',root/'challenger_features',asof)
            matrix=collection.enrich(market,pd.read_parquet(root/'financial_wide.parquet'))
            for c in model.FEATURES:
                if c not in matrix:matrix[c]=np.nan
            matrix.to_parquet(root/'challenger_matrix.parquet',index=False)
            target=root/'challenger_runs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
            model.run(root/'challenger_matrix.parquet',target,asof,only_current=True)
            if (target/'research_heads.joblib').exists():
                (root/'challenger').mkdir(exist_ok=True);shutil.copy2(target/'research_heads.joblib',root/'challenger'/'research_heads.joblib')
            marker.write_text(month);report['challenger_refit']=str(target.relative_to(root))
        # Promotion requires independently verified unseen outcomes. Current
        # provider adjustment/action reconciliation is incomplete, so promotion
        # is deliberately closed rather than silently learning from guesses.
        challenger_model=root/'challenger'/'research_heads.joblib'
        if challenger_model.exists() and observations:
            challenger_sha=hashlib.sha256(challenger_model.read_bytes()).hexdigest()
            freeze_path=root/'challenger'/'freeze.json'
            freeze=json.loads(freeze_path.read_text()) if freeze_path.exists() else {'sha256':challenger_sha,'frozen_at_utc':datetime.now(timezone.utc).isoformat()}
            if freeze['sha256']!=challenger_sha:freeze={'sha256':challenger_sha,'frozen_at_utc':datetime.now(timezone.utc).isoformat()}
            freeze_path.write_text(json.dumps(freeze,indent=2))
            paired=paired_observations(pd.DataFrame(observations),challenger_sha)
            report['promotion']=promotion_gate(paired,freeze['frozen_at_utc'],source_complete=False)
            if report['promotion']['promote']:
                backup=root/'retired_models'/hashlib.sha256((root/'active_heads.joblib').read_bytes()).hexdigest()
                backup.parent.mkdir(exist_ok=True);shutil.copy2(root/'active_heads.joblib',backup)
                shutil.copy2(challenger_model,root/'active_heads.joblib')
        else:report['promotion']={'promote':False,'reasons':['independent_action_reconciliation_incomplete','insufficient_untouched_mature_paired_outcomes']}
        report['completed_at_utc']=datetime.now(timezone.utc).isoformat()
    except Exception as e:
        report['errors'].append({'phase':'worker','error':str(e)[:400]});raise
    finally:
        root.mkdir(parents=True,exist_ok=True);(root/'latest_run.json').write_text(json.dumps(report,indent=2))
        if not local:
            receipt=save(root);(root.parent/'archive_receipt.json').write_text(json.dumps(receipt,indent=2))
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--state',default='v13_state');p.add_argument('--seed',default='v13_seed');p.add_argument('--day');p.add_argument('--history-budget',type=int,default=400);p.add_argument('--filing-budget',type=int,default=400);p.add_argument('--retrain',action='store_true');p.add_argument('--local',action='store_true');a=p.parse_args()
    day=a.day or str(pd.Timestamp.now(tz='Asia/Kolkata').date())
    print(json.dumps(run(a.state,a.seed,day,a.history_budget,a.filing_budget,a.retrain,a.local),indent=2))
