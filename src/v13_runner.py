"""Daily resumable research worker; independent of a ChatGPT token session.

The paper champion keeps a fixed head. Retraining makes a challenger and never
overwrites frozen predictions. Durable state is archived using the existing
private Hugging Face credential, with GitHub artifacts as a second checkpoint.
"""
import argparse,base64,hashlib,io,json,os,shutil,tarfile,tempfile
from datetime import datetime,timezone
from pathlib import Path
import joblib,numpy as np,pandas as pd
from v12_3_official_monitor import collect as collect_official
import v13_collection as collection
import v13_features as features
import v13_model as model
from v13_lifecycle import register,adjudicate,promotion_gate,paired_observations,validate_registry
from v13_paired_forecasts import COMPARISON_CONTRACT,scored,register_pair,export_watchlists,promotion_coverage
import v13_financial_evidence as financial_evidence
import v13_catalyst_execution as catalyst_execution
import v13_action_audit as action_audit
import v13_research_evaluation as evaluation

ARCHIVE='ayushverma1988/v10-multibagger-archive'

def paired_registration_coverage_gate(root,current_rows):
    """Prevent a collapsed source-day scan from becoming a formal cohort.

    Recent committed cohorts define a relative breadth baseline. The guard is
    prospective only: it never edits, enlarges or deletes an existing cohort.
    """
    prior=[]
    for path in sorted((Path(root)/'paired_cohorts').glob('*.json')):
        obj=json.loads(path.read_text())
        if obj.get('comparison_contract')!=COMPARISON_CONTRACT:continue
        prior.append((pd.Timestamp(obj['registered_at_utc']),int(obj['rows_per_model'])))
    prior=[rows for _,rows in sorted(prior)[-3:]]
    if not prior:
        return {'passed':True,'current_rows':int(current_rows),'minimum_rows':None,
          'reference_recent_rows':[],'status':'FIRST_COHORT_BASELINE'}
    reference=float(np.median(prior));minimum=max(20,int(np.ceil(reference*.5)))
    passed=int(current_rows)>=minimum
    return {'passed':passed,'current_rows':int(current_rows),'minimum_rows':minimum,
      'reference_recent_rows':prior,'reference_median_rows':reference,
      'status':'PASS' if passed else 'BLOCKED_COLLAPSED_CURRENT_COVERAGE'}

def restore(root):
    if (Path(root)/'recovery_restore_receipt.json').exists():
        verify_manifest(Path(root))
        return True
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
    from v13_artifact_checkpoint import save_with_quota_fallback
    return save_with_quota_fallback(root,save_incremental)

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

def current_frame(root,asof,primary_volume_research=False,fallback_prior_dir=None):
    rows=[];history={};low,_=collection.bounds(asof)
    funnel={'history_files':0,'empty_after_window':0,'no_source_date_bar':0,'source_date_feature_ineligible':0,'feature_eligible_source_date':0}
    # The alternate source is strictly scoring-only: the returned `history`
    # remains the original provider data used by entry and outcome audits.
    import v13_breadth_fallback as breadth_fallback
    try:
        fallback, fallback_status=breadth_fallback.load(root,asof)
    except Exception as exc:
        fallback, fallback_status=None,{'status':'BLOCKED_OPTIONAL_FALLBACK_ERROR','reason':str(exc)[:180]}
    if fallback is not None:
        (root/'evidence').mkdir(exist_ok=True)
        receipt_path=root/'evidence'/('breadth_fallback_receipt_'+asof+'_'+fallback[1]['sha256'][:16]+'.json')
        if not receipt_path.exists():receipt_path.write_text(json.dumps(fallback[1],indent=2)+'\n')
    fallback_groups={k:g for k,g in fallback[0].groupby('symbol',sort=False)} if fallback is not None else {}
    # The newly relaxed liquidity path also needs an actually timely original
    # provider receipt, even when there is no public fallback snapshot.
    fallback_history_receipts=breadth_fallback.timely_history_receipts(root,asof)
    prior_index_path=Path(fallback_prior_dir)/'index.json' if fallback_prior_dir else None
    try:prior_index=json.loads(prior_index_path.read_text()) if prior_index_path and prior_index_path.exists() else {}
    except (ValueError,OSError):prior_index={};fallback_prior_dir=None
    fallback_official_cache={};fallback_blocked={};fallback_attempted=0;fallback_accepted=0
    single_attempted=0;single_accepted=0;single_new_identity=0;single_blocked={};low_liquidity_timing_blocked=0
    primary=pd.read_parquet(root/'official'/(asof+'_official_market.parquet')) if primary_volume_research else pd.DataFrame()
    volume_overlays=0
    from v13_history_views import histories
    wanted=None
    if (root/'universe.parquet').exists():
        u=pd.read_parquet(root/'universe.parquet')
        symbol_col='nse_current_symbol' if 'nse_current_symbol' in u else 'symbol'
        wanted=set(zip(u['isin'],u[symbol_col]))
    canonical,identity_audit=histories(root/'history'/'securities',wanted)
    funnel['history_files']=identity_audit['files']
    funnel['conflicting_history_identities']=len(identity_audit['quarantined'])
    funnel['redundant_history_files']=identity_audit['redundant_files']
    def single_source(group,symbol,isin,reference=None):
        terminal=pd.to_datetime(group.date).max()
        start=min(terminal+pd.Timedelta(days=1),pd.Timestamp(asof))
        key=str(start.date())
        if key not in fallback_official_cache:
            fallback_official_cache[key]=breadth_fallback._calendar_and_official(root,start,asof,fallback[1],fallback[2])
        sessions,originals=fallback_official_cache[key]
        return breadth_fallback.single_source_view(group,originals,sessions,asof,symbol,isin,
          fallback[1]['sha256'],fallback[1].get('source_day_action_symbols',()),reference)
    for path,bars in canonical.values():
        bars=bars.copy();bars.date=pd.to_datetime(bars.date);bars=bars[bars.date.ge(low)]
        if bars.empty:funnel['empty_after_window']+=1;continue
        history[bars['isin'].iloc[0]]=bars
        fallback_proof=None
        if fallback is not None and bars.date.max()<pd.Timestamp(asof):
            fallback_attempted+=1
            provider_clock=None; reference=None
            try:
                symbol=bars.symbol.iloc[0];isin=bars['isin'].iloc[0]
                if symbol not in fallback_groups:raise ValueError('Alternative symbol missing')
                source_hashes=set(bars.source_sha256)
                provider_hash=next(iter(source_hashes)) if len(source_hashes)==1 else None
                provider_clock=fallback_history_receipts.get((symbol,isin,provider_hash))
                if provider_clock is None:
                    prior=breadth_fallback.prior_history(fallback_prior_dir,isin,symbol,fallback_history_receipts,prior_index)
                    if prior is not None:
                        bars=prior;provider_hash=next(iter(set(bars.source_sha256)))
                        provider_clock=fallback_history_receipts.get((symbol,isin,provider_hash))
                last_close=bars.date.max().tz_localize('Asia/Kolkata')+pd.Timedelta(hours=15,minutes=30)
                if provider_clock is None or provider_clock<last_close.tz_convert('UTC'):
                    raise ValueError('No pre-cutoff original provider adjustment receipt')
                reference=bars
                start=bars.date.max()+pd.Timedelta(days=1);key=str(start.date())
                if key not in fallback_official_cache:
                    fallback_official_cache[key]=breadth_fallback._calendar_and_official(root,start,asof,fallback[1],fallback[2])
                sessions,originals=fallback_official_cache[key]
                bars,fallback_proof=breadth_fallback.extend(bars,fallback_groups[symbol],originals,sessions,asof,symbol,isin,fallback[1]['sha256'],fallback[1].get('source_day_action_symbols',()))
                fallback_proof['provider_source_sha256']=provider_hash
                fallback_proof['provider_first_retrieved_utc']=provider_clock.isoformat()
                fallback_accepted+=1
            except (ValueError,KeyError,TypeError,OSError) as exc:
                reason=str(exc)[:180];fallback_blocked[reason]=fallback_blocked.get(reason,0)+1
                # A single timely public adjusted history is useful for paper
                # scoring when a second provider is unavailable or has fewer
                # than 126 overlapping sessions. Known disagreements remain
                # quarantined; a later provider observation is never used.
                if reason in ('No pre-cutoff original provider adjustment receipt',
                              'Insufficient independent adjusted-close overlap') and symbol in fallback_groups:
                    single_attempted+=1
                    try:
                        bars,fallback_proof=single_source(fallback_groups[symbol],symbol,isin,reference)
                        fallback_proof['provider_source_sha256']=provider_hash if reference is not None else None
                        fallback_proof['provider_first_retrieved_utc']=provider_clock.isoformat() if reference is not None else None
                        single_accepted+=1;fallback_accepted+=1
                    except (ValueError,KeyError,TypeError,OSError) as single_exc:
                        detail=str(single_exc)[:180];single_blocked[detail]=single_blocked.get(detail,0)+1
        if primary_volume_research:
            from v13_primary_volume import reconcile
            bars,changed=reconcile(bars,primary,asof);volume_overlays+=int(changed)
        d=features.relax_scoring_liquidity(features.daily_features(bars,asof))
        current=d[d.date.eq(pd.Timestamp(asof))&d.feature_eligible]
        if len(current)==1:
            row=current.iloc[0].to_dict()
            original_receipt=None;original_hash=None
            if (row.get('scoring_liquidity_status')=='BELOW_25_LAKH_OUTSIDE_TRAINING_BAND'
                    and fallback_proof is None):
                hashes=set(bars.source_sha256) if 'source_sha256' in bars else set()
                original_hash=next(iter(hashes)) if len(hashes)==1 else None
                original_receipt=fallback_history_receipts.get((bars.symbol.iloc[0],bars['isin'].iloc[0],original_hash))
                close=(pd.Timestamp(asof).tz_localize('Asia/Kolkata')+pd.Timedelta(hours=15,minutes=30)).tz_convert('UTC')
                if original_receipt is None or original_receipt<close:
                    low_liquidity_timing_blocked+=1
                    continue
            row['scoring_price_history_source']=fallback_proof['source_status'] if fallback_proof else 'ORIGINAL_ADJUSTED_PROVIDER'
            row['scoring_fallback_source_sha256']=fallback_proof['source_sha256'] if fallback_proof else None
            row['scoring_fallback_action_interval_certified']=False if fallback_proof else None
            row['scoring_fallback_provider_sha256']=fallback_proof.get('provider_source_sha256') if fallback_proof else None
            row['scoring_fallback_provider_first_retrieved_utc']=fallback_proof.get('provider_first_retrieved_utc') if fallback_proof else None
            row['scoring_fallback_reference_status']=fallback_proof.get('reference_status') if fallback_proof else None
            row['scoring_fallback_overlap_sessions']=fallback_proof.get('overlap_sessions') if fallback_proof else None
            row['scoring_history_series_transition_unverified']=fallback_proof.get('historical_series_transition_unverified') if fallback_proof else None
            row['scoring_original_provider_sha256']=original_hash
            row['scoring_original_provider_first_retrieved_utc']=original_receipt.isoformat() if original_receipt is not None else None
            rows.append(row)
        elif not d.date.eq(pd.Timestamp(asof)).any():funnel['no_source_date_bar']+=1
        else:funnel['source_date_feature_ineligible']+=1
    # The original loop cannot visit listings with no stored provider file.
    # Add only current-universe identities from the timed adjusted snapshot;
    # do not write those histories or add them to outcome adjudication.
    if fallback is not None and wanted is not None:
        for isin,symbol in sorted(wanted-set(canonical)):
            if (isin,symbol) in identity_audit['quarantined'] or symbol not in fallback_groups:continue
            single_attempted+=1
            try:
                bars,proof=single_source(fallback_groups[symbol],symbol,isin)
                single_accepted+=1
                d=features.relax_scoring_liquidity(features.daily_features(bars,asof))
                current=d[d.date.eq(pd.Timestamp(asof))&d.feature_eligible]
                if len(current)!=1:
                    single_blocked['SOURCE_DATE_FEATURE_INELIGIBLE']=single_blocked.get('SOURCE_DATE_FEATURE_INELIGIBLE',0)+1
                    continue
                row=current.iloc[0].to_dict()
                row.update(scoring_price_history_source=proof['source_status'],
                  scoring_fallback_source_sha256=proof['source_sha256'],
                  scoring_fallback_action_interval_certified=False,
                  scoring_fallback_provider_sha256=None,scoring_fallback_provider_first_retrieved_utc=None,
                  scoring_fallback_reference_status=proof['reference_status'],
                  scoring_fallback_overlap_sessions=proof['overlap_sessions'],
                  scoring_history_series_transition_unverified=proof['historical_series_transition_unverified'])
                rows.append(row);single_new_identity+=1
            except (ValueError,KeyError,TypeError,OSError) as exc:
                detail=str(exc)[:180];single_blocked[detail]=single_blocked.get(detail,0)+1
    funnel['feature_eligible_source_date']=len(rows)
    out=root/('primary_volume_research' if primary_volume_research else 'evidence');out.mkdir(exist_ok=True)
    funnel['primary_volume_overlays']=volume_overlays
    funnel['independent_volume_agreement_required']=not primary_volume_research
    funnel['scoring_only_fallback']={**fallback_status,'attempted':fallback_attempted,'accepted_before_feature_gate':fallback_accepted,
      'blocked_reasons':fallback_blocked,'original_histories_unchanged':True,'entry_or_outcome_certified':False}
    funnel['single_source_scoring_only']={'attempted':single_attempted,'accepted_before_feature_gate':single_accepted,
      'new_identities_without_provider_history':single_new_identity,'blocked_reasons':single_blocked,
      'adjustment_independently_verified':False,'entry_or_outcome_certified':False}
    funnel['low_liquidity_original_timing_blocked']=low_liquidity_timing_blocked
    if not rows:
        (out/'market_coverage_funnel.json').write_text(json.dumps({'source_asof':asof,**funnel,'scored':0},indent=2))
        return pd.DataFrame(),history
    d=pd.DataFrame(rows)
    # Cross-sectional context is computed after official reconciliation below;
    # rejected provider rows cannot influence accepted names' ranks or breadth.
    d=market_context(d)
    d['liquidity_volume_interaction']=d.volume_confirmation*d.log_liquidity63
    d['earnings_market_confirmation']=np.nan
    from v13_information_clock import research_cutoffs
    d=research_cutoffs(d,datetime.now(timezone.utc))
    if (root/'financial_wide.parquet').exists():d=collection.enrich(d,pd.read_parquet(root/'financial_wide.parquet'))
    d=collection.announcement_counts(d,root,asof)
    for c in model.FEATURES:
        if c not in d:d[c]=np.nan
    for c in ['four_profitable_quarters',*model.old.BASE_FIELDS]:
        if c not in d:d[c]=np.nan
    # Check raw latest OHLC and volume against whole-market official data.
    o=pd.read_parquet(root/'official'/(asof+'_official_market.parquet'))
    m=d.merge(o[['isin','symbol','series','open','high','low','close','volume','source_status','source_sha256']],on='isin',suffixes=('','_official'),validate='1:1')
    funnel['official_identity_absent']=len(d)-len(m)
    funnel['official_invalid_session']=int(m.source_status.ne('VALID_POSITIVE_VOLUME_SESSION').sum())
    funnel['symbol_mismatch']=int(m.symbol.ne(m.symbol_official).sum())
    funnel['volume_mismatch']=int(m.volume.ne(m.volume_official).sum())
    funnel['OHLC_mismatch']=int(pd.concat([(m[c]-m[c+'_official']).abs().gt(.011) for c in ('open','high','low','close')],axis=1).any(axis=1).sum())
    valid=m.symbol.eq(m.symbol_official)&m.source_status.eq('VALID_POSITIVE_VOLUME_SESSION')&m.volume.eq(m.volume_official)
    for c in ('open','high','low','close'):valid&=(m[c]-m[c+'_official']).abs().le(.011)
    d=m.loc[valid].copy();d['source_close_INR']=d.close_official;d['official_source_sha256']=d.source_sha256_official
    funnel['single_source_scoring_only']['scored_after_official_reconciliation']=int(d.scoring_price_history_source.eq('SINGLE_SOURCE_ADJUSTED_SCORING_ONLY').sum())
    funnel['liquidity_research_scoring']={
      'below_25_lakh_outside_training_band':int(d.scoring_liquidity_status.eq('BELOW_25_LAKH_OUTSIDE_TRAINING_BAND').sum()),
      'standard_25_lakh_turnover':int(d.scoring_liquidity_status.eq('STANDARD_25_LAKH_TURNOVER').sum()),
      'original_provider_low_liquidity_blocked_without_timely_receipt':low_liquidity_timing_blocked,
      'original_training_and_frozen_forecasts_unchanged':True,'accuracy_approved':False}
    d=market_context(d)
    d['market_context_scope']='OFFICIAL_RECONCILED_CURRENT_PARTIAL_UNIVERSE'
    d['earnings_market_confirmation']=d.quarter_pat_yoy*d.relative_strength63
    (out/'market_coverage_funnel.json').write_text(json.dumps({'source_asof':asof,**funnel,'scored':len(d),
      'mismatch_counts_overlap':True,'stale_bars_not_advanced_or_filled':True},indent=2))
    return d,history

def market_context(d):
    d=d.copy()
    for source,target in [('log_ret63','relative_strength63'),('log_ret20','relative_strength20'),('momentum126','relative_strength126'),('momentum252_skip21','relative_strength252_skip21')]:d[target]=d[source].rank(pct=True)
    d['market_breadth63']=d.log_ret63.gt(0).mean();d['market_median_ret63']=d.log_ret63.median();d['market_dispersion63']=d.log_ret63.std()
    d['trend_regime_interaction']=d.relative_strength63*(d.market_breadth63-.5)
    d['market_context_rows']=len(d)
    d['market_context_scope']='CURRENT_PARTIAL_UNIVERSE'
    return d

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

def run(root,seed,day,history_budget=800,filing_budget=800,retrain=False,local=False,catalyst_budget=360,
        collection_workers=4,filing_seconds=900,catalyst_seconds=360):
    root=Path(root);seed=Path(seed)
    def phase(name):
        item={'workflow_run':os.environ.get('GITHUB_RUN_ID'),'phase':name,'updated_at_utc':datetime.now(timezone.utc).isoformat()}
        path=root.parent/'v13_worker_attempt.json';temp=path.with_suffix('.tmp');temp.write_text(json.dumps(item,indent=2));temp.replace(path)
        print('V13 worker phase',name,item['updated_at_utc'],flush=True)
    phase('RESTORING_CHECKPOINT' if not local else 'LOCAL_RESEARCH_START')
    if not local:
        if not restore(root):initialize(root,seed)
        from v13_checkpoint import restore_review_originals
        review_seed=seed.parent/'v13_evidence_seed'/'primary_execution_sources.json'
        seed_hashes=[r['source_sha256'] for r in json.loads(review_seed.read_text())] if review_seed.exists() else []
        restore_review_originals(root,additional_hashes=seed_hashes)
    report={'workflow_run':os.environ.get('GITHUB_RUN_ID'),'started_at_utc':datetime.now(timezone.utc).isoformat(),'mode':'PAPER_RESEARCH','accuracy_approved':False,'steps':{},'errors':[]}
    from v13_frozen_integrity import FrozenGuard
    guard=FrozenGuard(root)
    prior_cache=None
    try:
        if not local:
            from v13_history_integrity import recover_from_seed
            report['steps']['entry_history_recovery']=recover_from_seed(root,seed.parent/'v13_evidence_seed/entry_history_recovery.json')
        phase('OFFICIAL_SESSION_AND_PUBLIC_CATALOGUES')
        asof,official,errors=last_completed_session(root,day);report['asof']=asof;report['steps']['official']=official
        # Every completed phase is checkpointed. Interrupted work resumes from
        # accepted filing versions and the public-history age queue.
        collection.universe(root,asof)
        report['steps']['catalogs']=collection.current_catalogs(root,asof)
        report['steps']['catalogue_recovery']=collection.recover_public_catalogue(root,seed.parent/'v13_evidence_seed/catalogue_recovery',asof)
        # Fresh public prices must exist before financial completion chooses
        # the unranked price/history-eligible population for this source date.
        # Previously a new session always prioritised yesterday's histories.
        try:
            from v13_breadth_fallback import load as load_breadth,snapshot_prior_histories
            timely_snapshot,_=load_breadth(root,asof)
            if timely_snapshot is not None:
                prior_cache=tempfile.TemporaryDirectory(prefix='v13_breadth_prior_',dir=root.parent)
                report['steps']['breadth_prior_histories']=snapshot_prior_histories(root,asof,prior_cache.name)
            del timely_snapshot
        except Exception as exc:
            if prior_cache is not None:prior_cache.cleanup();prior_cache=None
            report['steps']['breadth_prior_histories']={'status':'BLOCKED_OPTIONAL_FALLBACK_ERROR','reason':str(exc)[:180]}
        phase('PUBLIC_PRICE_HISTORY_COLLECTION')
        report['steps']['history']=collection.collect_histories(root,asof,history_budget,workers=collection_workers)
        if not local:
            path=root/'prospective_observations.parquet'
            guard.check(pd.read_parquet(path) if path.exists() else pd.DataFrame())
            phase('SAVING_HISTORY_CHECKPOINT')
            save(root)
        phase('ORIGINAL_FINANCIAL_COLLECTION')
        from v13_financial_recovery_seed import apply as apply_financial_recovery_seed
        report['steps']['verified_original_financial_recovery']=apply_financial_recovery_seed(root,seed.parent/'v13_evidence_seed/financial_primary_recovery_20261020.json',asof)
        report['steps']['current_original_financial_recovery']=apply_financial_recovery_seed(
            root,seed.parent/'v13_evidence_seed/financial_primary_recovery_20261112.json',asof,
            marker_name='financial_primary_seed_recovery_20261112.json')
        from v13_pdf_financial_reviews import collect as collect_reviewed_financial_pdfs
        report['steps']['reviewed_PDF_financial_comparatives']=collect_reviewed_financial_pdfs(
          root,seed.parent/'v13_evidence_seed/financial_pdf_reviews_20261102.json',asof)
        from v13_image_financial_reviews import collect as collect_image_financials
        report['steps']['reviewed_image_financial_comparatives']=collect_image_financials(
            root,seed.parent/'v13_evidence_seed/image_financial_reviews_20261112.json',asof)
        report['steps']['financials']=collection.collect_financials(root,asof,filing_budget,
          workers=collection_workers,seconds=filing_seconds)
        try:
            from v13_alternate_sources import collect as collect_alternates,sector_master
            report['steps']['additional_master_sources']=collect_alternates(root,budget=160,seconds=60,sources=('NSE',))
            report['steps']['sector_master']=sector_master(root)
        except Exception as exc:
            report['steps']['additional_master_sources']={'status':'SOURCE_COLLECTION_INCOMPLETE','error':str(exc)[:200]}
        try:
            from v13_yahoo_financials import collect as collect_yahoo_financials
            report['steps']['yahoo_statement_sources']=collect_yahoo_financials(root,budget=480,seconds=300,asof=asof)
        except Exception as exc:
            report['steps']['yahoo_statement_sources']={'status':'SOURCE_COLLECTION_INCOMPLETE','error':str(exc)[:200]}
        try:
            from v13_web_missing_data import collect as collect_missing_web
            report['steps']['missing_web_inputs']=collect_missing_web(root,budget=160,seconds=60,google_budget=1)
        except Exception as exc:
            report['steps']['missing_web_inputs']={'status':'SOURCE_COLLECTION_INCOMPLETE','error':str(exc)[:200]}
        try:
            # Reuse source receipts; do not request the same host a second
            # time after the bounded collector has stopped on an access limit.
            from v13_financial_crosscheck import export as export_financial_crosschecks
            from v13_master_data import reviewed_public_packet
            observations=reviewed_public_packet().get('observations',[])
            for path in (root/'master_data/provider_inputs').glob('*.json'):
                observations.extend(json.loads(path.read_text()))
            observations=[r for r in observations if r['metric'] in {'revenue','cfo','provider_net_profit'}]
            report['steps']['financial_crosschecks']=export_financial_crosschecks(
                root,pd.read_parquet(root/'financial_wide.parquet'),asof,observations)
        except Exception as exc:
            report['steps']['financial_crosschecks']={'status':'COMPARISON_INCOMPLETE','error':str(exc)[:200]}
        import v13_statement_eps as statement_eps
        report['steps']['reviewed_statement_EPS']=statement_eps.collect(root,seed.parent/'v13_evidence_seed'/'statement_eps_reviews.json')
        report['steps']['historical_events']=collection.historical_event_backfill(root,asof,budget=2)
        report['steps']['catalysts']=catalyst_execution.collect(root,asof,budget=catalyst_budget,
          workers=collection_workers,seconds=catalyst_seconds)
        report['steps']['reviewed_primary_sources']=catalyst_execution.seed_primary_reviews(root,seed.parent/'v13_evidence_seed'/'primary_execution_sources.json')
        from v13_catalyst_field_capture import export as capture_catalyst_fields
        report['steps']['catalyst_field_capture']=capture_catalyst_fields(root,asof)
        report['steps']['entry_backfill']=action_audit.backfill(root,asof,budget=8)
        from v13_completion_work import primary_history_backfill
        report['steps']['primary_history_backfill']=primary_history_backfill(root,asof,budget=8)
        import v13_action_sources as action_sources
        report['steps']['action_notice_collection']=action_sources.collect(root,asof,budget=30)
        import v13_valuation_evidence as valuation_evidence
        report['steps']['valuation_snapshot']=valuation_evidence.collect(root,asof)
        phase('SAVE_MASTER_INPUTS_BEFORE_ANALYSIS')
        from v13_universe_financial_audit import audit as audit_financial_inputs
        report['universe_financial_coverage']=audit_financial_inputs(
            pd.read_parquet(root/'universe.parquet'),pd.read_parquet(root/'financial_wide.parquet'),asof,root)
        from v13_master_data import export as export_master_data
        report['steps']['master_data_before_analysis']=export_master_data(root,asof)
        if not local:
            path=root/'prospective_observations.parquet'
            guard.check(pd.read_parquet(path) if path.exists() else pd.DataFrame())
            phase('SAVING_FINANCIAL_CHECKPOINT')
            save(root)
        phase('CURRENT_SOURCE_ALIGNED_MODEL_INPUTS')
        frame,history=current_frame(root,asof,fallback_prior_dir=prior_cache.name if prior_cache is not None else None)
        report['current_verified_market_rows']=len(frame)
        # Partial coverage is disclosed, not mistaken for a full-universe scan.
        report['public_universe_rows']=len(pd.read_parquet(root/'universe.parquet'))
        # Monthly refresh learns coefficients/blends automatically from the
        # trailing four-year, maturity-purged data. It is a new paper challenger.
        marker=root/'last_retrain_month.txt';month=asof[:7]
        if (retrain or not marker.exists() or marker.read_text()!=month) and len(history)>=500:
            try:
                blank=pd.DataFrame(columns=['date','isin']);blank.to_parquet(root/'empty_previous.parquet',index=False,compression='zstd',compression_level=9)
                master=pd.read_parquet(root/'universe.parquet')
                symbol_col='nse_current_symbol' if 'nse_current_symbol' in master else 'symbol'
                market=features.build(root/'history',root/'empty_previous.parquet',root/'challenger_features',asof,
                    current_pairs=set(zip(master['isin'],master[symbol_col])))
                from v13_information_clock import research_cutoffs
                market=research_cutoffs(market,datetime.now(timezone.utc))
                matrix=collection.enrich(market,pd.read_parquet(root/'financial_wide.parquet'))
                matrix=collection.announcement_counts(matrix,root,asof)
                for c in model.FEATURES:
                    if c not in matrix:matrix[c]=np.nan
                matrix.to_parquet(root/'challenger_matrix.parquet',index=False,compression='zstd',compression_level=9)
                target=root/'challenger_runs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
                model.run(root/'challenger_matrix.parquet',target,asof,only_current=True)
                report['research_experiments']=str(target.relative_to(root)/'predeclared_experiments.json')
                evaluation.experiments(matrix,asof,target/'predeclared_experiments.json')
                if (target/'research_heads.joblib').exists():
                    if not set(model.TARGETS).issubset(joblib.load(target/'research_heads.joblib')):raise ValueError('Monthly refit missing conditional heads')
                    (root/'challenger').mkdir(exist_ok=True)
                    # Keep the formal challenger fixed while its unseen outcomes
                    # mature. Monthly learned heads are archived separately;
                    # replacing the formal head every month would prevent a
                    # valid twelve-month paired comparison from ever finishing.
                    if not (root/'challenger'/'research_heads.joblib').exists():
                        shutil.copy2(target/'research_heads.joblib',root/'challenger'/'research_heads.joblib')
                    shutil.copy2(target/'research_heads.joblib',root/'latest_learned_heads.joblib')
                marker.write_text(month);report['challenger_refit']=str(target.relative_to(root))
            except Exception as exc:
                report['errors'].append({'phase':'monthly_research','error':str(exc)[:300]})
                report['monthly_research_status']='FAILED_RETAINING_FIXED_MODELS'
        # Commit paired forecasts after initial refitting. Both models score
        # the same immutable inputs at the same actual registration timestamp.
        initial_experiments=root/'research_experiments_initial.json'
        if (root/'challenger_matrix.parquet').exists() and not initial_experiments.exists():
            try:
                diagnostic=evaluation.experiments(pd.read_parquet(root/'challenger_matrix.parquet'),asof,initial_experiments)
                report['initial_experiment_folds']=len(diagnostic['folds'])
            except Exception as exc:
                report['errors'].append({'phase':'initial_research_experiments','error':str(exc)[:300]})
        challenger_model=root/'challenger'/'research_heads.joblib'
        six_experiments=root/'evidence'/'research_experiments_six_area.json'
        if six_experiments.exists() and json.loads(six_experiments.read_text()).get('input_contract')!=evaluation.EXPERIMENT_INPUT_CONTRACT:
            previous=six_experiments.read_bytes()
            archive=six_experiments.with_name('research_experiments_six_area_previous_'+hashlib.sha256(previous).hexdigest()+'.json')
            if not archive.exists():archive.write_bytes(previous)
            six_experiments.unlink()
        if (root/'challenger_matrix.parquet').exists() and not six_experiments.exists():
            try:
                from v13_information_clock import research_cutoffs
                diagnostic=research_cutoffs(pd.read_parquet(root/'challenger_matrix.parquet'),datetime.now(timezone.utc))
                diagnostic=collection.enrich(diagnostic,pd.read_parquet(root/'financial_wide.parquet'))
                evaluation.experiments(diagnostic,asof,six_experiments)
                report['six_area_development_experiments']='COMPLETED_PREVIOUSLY_EXAMINED_NOT_PROMOTION_EVIDENCE'
            except Exception as exc:report['six_area_development_experiments']={'status':'BLOCKED_RETAINING_FIXED_MODELS','error':str(exc)[:200]}
        audited_scored=None
        phase('FULL_UNIVERSE_FINANCIAL_AUDIT_AND_MODEL_REPLAY')
        try:
            from v13_universe_financial_audit import audit as audit_universe_financials
            universe_wide=pd.read_parquet(root/'financial_wide.parquet') if (root/'financial_wide.parquet').exists() else pd.DataFrame()
            if report.get('universe_financial_coverage',{}).get('source_asof')!=asof:
                report['universe_financial_coverage']=audit_universe_financials(pd.read_parquet(root/'universe.parquet'),universe_wide,asof,root)
        except Exception as exc:
            report['universe_financial_coverage']={'status':'AUDIT_FAILED','error':str(exc)[:200]}
        from v13_priority_research import unavailable as unavailable_priority
        report['priority_research']=unavailable_priority(root,asof,
            'BLOCKED_NO_EXACT_CURRENT_REPLAY' if not frame.empty else 'NO_CURRENT_VERIFIED_INPUTS')
        if not frame.empty:
            wide=pd.read_parquet(root/'financial_wide.parquet') if (root/'financial_wide.parquet').exists() else pd.DataFrame()
            frame=financial_evidence.enrich_evidence(frame,wide,root,asof)
            from v13_dated_sector import attach as attach_dated_sector
            frame=attach_dated_sector(frame,root)
            frame=statement_eps.enrich(frame,root)
            frame=catalyst_execution.join(frame,root)
            frame=financial_evidence.apply_scenarios(frame,root)
            report['financial_complete_rows']=int(frame.financial_complete.sum())
            report['financial_quality_pass_rows']=int(frame.financial_quality_pass.sum())
            report['financial_known_failure_rows']=int(frame.financial_quality_status.eq('KNOWN_FINANCIAL_FAILURE').sum())
            report['execution_chain_supported_rows']=int(frame.evidence_status.eq('FINANCIAL_AND_EXECUTION_SUPPORTED').sum())
            report['source_bound_earnings_scenario_rows']=int(frame.earnings_scenario_status.eq('SCENARIO_NOT_PROBABILITY').sum())
            # A source-bound scenario below the 2x net target is not support
            # for doubling. Preserve total scenario coverage separately.
            report['source_bound_2x_scenario_rows']=int((frame.earnings_scenario_status.eq('SCENARIO_NOT_PROBABILITY') & frame.scenario_net_multiple.ge(2)).sum()) if 'scenario_net_multiple' in frame else 0
            baseline=evaluation.prior_rates(pd.read_parquet(root/'challenger_matrix.parquet'),asof) if (root/'challenger_matrix.parquet').exists() else {}
            for name,value in baseline.items():
                if value is not None:frame[name]=value
            if challenger_model.exists():
                sha=hashlib.sha256(challenger_model.read_bytes()).hexdigest()
                freeze_path=root/'challenger'/'freeze.json'
                freeze=json.loads(freeze_path.read_text()) if freeze_path.exists() else {'sha256':sha,'frozen_at_utc':datetime.now(timezone.utc).isoformat()}
                if freeze['sha256']!=sha:raise ValueError('Formal challenger changed during prospective evaluation')
                freeze_path.write_text(json.dumps(freeze,indent=2))
                active_meta=json.loads((root/'active_model.json').read_text()) if (root/'active_model.json').exists() else {}
                a=scored(joblib.load(root/'active_heads.joblib'),frame,policy=active_meta.get('selection_policy','legacy'))
                audited_scored=a
                b=scored(joblib.load(challenger_model),frame,policy='evidence')
                from v13_prediction_replay import snapshot as replay_snapshot
                report['exact_prediction_replay']={
                  'champion':replay_snapshot(frame,a,root/'active_heads.joblib',root,'champion'),
                  'challenger':replay_snapshot(frame,b,challenger_model,root,'challenger')}
                from v13_research_selection import export as export_research
                report['separate_research_selector']=export_research(a,root,report['exact_prediction_replay']['champion'])
                # Additional available-input policy is a separately dated
                # research overlay. It cannot alter heads, registered pairs,
                # original financial features or strict evidence coverage.
                try:
                    from v13_available_financials import export as export_available_financials
                    report['available_financial_research']=export_available_financials(
                        a,wide,root,report['exact_prediction_replay']['champion'])
                except Exception as exc:
                    report['available_financial_research']={'status':'BLOCKED_DIAGNOSTICS_ONLY','error':str(exc)[:200]}
                try:
                    from v13_priority_research import export as export_priority
                    report['priority_research']=export_priority(a,wide,root,report['exact_prediction_replay']['champion'])
                except Exception as exc:
                    report['priority_research']={**unavailable_priority(root,asof,'BLOCKED_DIAGNOSTICS_ONLY'),'error':str(exc)[:200]}
                from v13_redevelopment import export as export_redevelopment
                try:
                    report['redevelopment_diagnostics']=export_redevelopment(a,root,report['exact_prediction_replay']['champion'])
                except Exception as exc:
                    report['redevelopment_diagnostics']={'status':'BLOCKED_DIAGNOSTICS_ONLY','error':str(exc)[:200]}
                try:
                    from v13_2x_report import export as export_two_x
                    report['two_x_research']=export_two_x(root,report['exact_prediction_replay']['champion'])
                except Exception as exc:
                    report['two_x_research']={'status':'BLOCKED_DIAGNOSTICS_ONLY','error':str(exc)[:200]}
                coverage_gate=paired_registration_coverage_gate(root,len(frame))
                report['paired_registration_coverage_gate']=coverage_gate
                if coverage_gate['passed']:
                    pair=register_pair(frame,a,b,hashlib.sha256((root/'active_heads.joblib').read_bytes()).hexdigest(),sha,root)
                    export_watchlists(root,pair)
                    report['paired_registration']={k:v for k,v in pair.items() if k!='isins'}
                    report['champion_registry']=pair['champion_registry'];report['challenger_registry']=pair['challenger_registry']
                else:
                    report['paired_registration']={'status':'BLOCKED_LOW_CURRENT_COVERAGE',**coverage_gate}
            else:report['paired_registration']={'status':'BLOCKED_NO_COMPLETE_CHALLENGER'}
        try:
            from v13_example_history_audit import audit as audit_examples
            report['example_history_audit']=audit_examples(root,asof,audited_scored,report['steps']['history'])
        except Exception as exc:
            report['example_history_audit']={'status':'BLOCKED_DIAGNOSTICS_ONLY','error':str(exc)[:200]}
        try:
            from v13_master_data import export as export_master_data
            report['master_data']=export_master_data(root,asof)
        except Exception as exc:
            report['master_data']={'status':'MASTER_EXPORT_INCOMPLETE','error':str(exc)[:200]}
        originals=[pd.read_parquet(p) for p in sorted((root/'official').glob('*_official_market.parquet'))]
        official_all=pd.concat(originals,ignore_index=True)
        verification,report['steps']['independent_entry_actions']=action_audit.audit(root,official_all,history,asof)
        observations=[]
        committed={r for p in (root/'paired_cohorts').glob('*.json') for r in [json.loads(p.read_text())['champion_registry'],json.loads(p.read_text())['challenger_registry']]}
        for p in sorted((root/'registries').glob('*/*.json')):
            reg=validate_registry(p)
            if 'paired_context' in reg and p.name not in committed:continue
            observations.extend(adjudicate(p,official_all,history,asof,verification))
        observed=pd.DataFrame(observations)
        if observations:observed.to_parquet(root/'prospective_observations.parquet',index=False,compression='zstd',compression_level=9)
        report['prospective_status_counts']=pd.Series([r['status'] for r in observations],dtype=object).value_counts().to_dict()
        report['raw_price_2x_alerts']=sum(r['raw_price_2x_observed'] for r in observations)
        report['independently_verified_mature_outcomes']=sum(r['independent_outcome_verified'] and r['twelve_month_mature'] for r in observations)
        (root/'prospective_metrics.json').write_text(json.dumps(evaluation.prospective_report(observed),indent=2))
        (root/'paired_evaluation.json').write_text(json.dumps(evaluation.paired_evaluation_report(observed),indent=2))
        from v13_progress import snapshot
        snapshot(root,report,observed)
        from v13_completion_work import completion_plan
        completion_plan(root,report,observed)
        # Promotion requires independently verified unseen outcomes. Current
        # provider adjustment/action reconciliation is incomplete, so promotion
        # is deliberately closed rather than silently learning from guesses.
        challenger_model=root/'challenger'/'research_heads.joblib'
        if challenger_model.exists() and observations:
            challenger_sha=hashlib.sha256(challenger_model.read_bytes()).hexdigest()
            freeze_path=root/'challenger'/'freeze.json'
            freeze=json.loads(freeze_path.read_text()) if freeze_path.exists() else {'sha256':challenger_sha,'frozen_at_utc':datetime.now(timezone.utc).isoformat()}
            if freeze['sha256']!=challenger_sha:raise ValueError('Formal challenger freeze changed')
            freeze_path.write_text(json.dumps(freeze,indent=2))
            paired=paired_observations(pd.DataFrame(observations),challenger_sha,comparison_contract=COMPARISON_CONTRACT)
            paired.to_parquet(root/'paired_mature_observations.parquet',index=False,compression='zstd',compression_level=9)
            report['promotion']=promotion_gate(paired,freeze['frozen_at_utc'],source_complete=promotion_coverage(root,observed,challenger_sha),require_baseline=True)
            if report['promotion']['promote']:
                backup=root/'retired_models'/hashlib.sha256((root/'active_heads.joblib').read_bytes()).hexdigest()
                backup.parent.mkdir(exist_ok=True);shutil.copy2(root/'active_heads.joblib',backup)
                shutil.copy2(challenger_model,root/'active_heads.joblib')
                (root/'active_model.json').write_text(json.dumps({'mode':'PAPER_RESEARCH','version':'V13',
                  'sha256':challenger_sha,'accuracy_approved':False,'promotion_gate_passed':True,
                  'selection_policy':'evidence','comparison_contract':COMPARISON_CONTRACT,
                  'promoted_at_utc':datetime.now(timezone.utc).isoformat(),'promotion_evidence':report['promotion']},indent=2))
        else:report['promotion']={'promote':False,'reasons':['independent_action_reconciliation_incomplete','insufficient_untouched_mature_paired_outcomes']}
        report['steps']['frozen_integrity']=guard.check(observed,promote=report['promotion'].get('promote') is True)
        report['completed_at_utc']=datetime.now(timezone.utc).isoformat()
        phase('MODEL_AND_AUDITS_COMPLETED')
    except Exception as e:
        report['errors'].append({'phase':'worker','error':str(e)[:400]});raise
    finally:
        if prior_cache is not None:
            try:prior_cache.cleanup()
            except OSError:pass
        if 'frozen_integrity' not in report['steps']:
            try:
                old_path=root/'prospective_observations.parquet'
                state=pd.read_parquet(old_path) if old_path.exists() else pd.DataFrame()
                report['steps']['frozen_integrity']=guard.check(state,promote=report.get('promotion',{}).get('promote') is True)
            except Exception as exc:report['errors'].append({'phase':'frozen_integrity','error':str(exc)[:400]})
        root.mkdir(parents=True,exist_ok=True);(root/'latest_run.json').write_text(json.dumps(report,indent=2))
        if not local:
            phase('SAVING_VERIFIED_CHECKPOINT')
            receipt=save(root);(root.parent/'archive_receipt.json').write_text(json.dumps(receipt,indent=2))
        phase('COMPLETED' if not report['errors'] else 'COMPLETED_WITH_RECORDED_ERRORS')
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--state',default='v13_state');p.add_argument('--seed',default='v13_seed');p.add_argument('--day');p.add_argument('--history-budget',type=int,default=800);p.add_argument('--filing-budget',type=int,default=800);p.add_argument('--catalyst-budget',type=int,default=360);p.add_argument('--retrain',action='store_true');p.add_argument('--local',action='store_true')
    p.add_argument('--collection-workers',type=int,choices=range(1,9),default=4)
    p.add_argument('--filing-seconds',type=int,choices=range(1,1801),default=900)
    p.add_argument('--catalyst-seconds',type=int,choices=range(1,901),default=360)
    a=p.parse_args()
    day=a.day or str(pd.Timestamp.now(tz='Asia/Kolkata').date())
    print(json.dumps(run(a.state,a.seed,day,a.history_budget,a.filing_budget,a.retrain,a.local,a.catalyst_budget,
      a.collection_workers,a.filing_seconds,a.catalyst_seconds),indent=2))
