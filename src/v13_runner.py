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
from v13_paired_forecasts import scored,register_pair,export_watchlists,promotion_coverage
import v13_financial_evidence as financial_evidence
import v13_catalyst_execution as catalyst_execution
import v13_action_audit as action_audit
import v13_research_evaluation as evaluation

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
    funnel={'history_files':0,'empty_after_window':0,'no_source_date_bar':0,'source_date_feature_ineligible':0,'feature_eligible_source_date':0}
    for path in sorted((root/'history'/'securities').glob('*.parquet')):
        funnel['history_files']+=1
        bars=pd.read_parquet(path);bars.date=pd.to_datetime(bars.date);bars=bars[bars.date.ge(low)]
        if bars.empty:funnel['empty_after_window']+=1;continue
        history[bars['isin'].iloc[0]]=bars
        d=features.daily_features(bars,asof)
        current=d[d.date.eq(pd.Timestamp(asof))&d.feature_eligible]
        if len(current)==1:rows.append(current.iloc[0].to_dict())
        elif not d.date.eq(pd.Timestamp(asof)).any():funnel['no_source_date_bar']+=1
        else:funnel['source_date_feature_ineligible']+=1
    funnel['feature_eligible_source_date']=len(rows)
    out=root/'evidence';out.mkdir(exist_ok=True)
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

def run(root,seed,day,history_budget=800,filing_budget=800,retrain=False,local=False):
    root=Path(root);seed=Path(seed)
    if not local:
        if not restore(root):initialize(root,seed)
        from v13_checkpoint import restore_review_originals
        review_seed=seed.parent/'v13_evidence_seed'/'primary_execution_sources.json'
        seed_hashes=[r['source_sha256'] for r in json.loads(review_seed.read_text())] if review_seed.exists() else []
        restore_review_originals(root,additional_hashes=seed_hashes)
    report={'started_at_utc':datetime.now(timezone.utc).isoformat(),'mode':'PAPER_RESEARCH','accuracy_approved':False,'steps':{},'errors':[]}
    from v13_frozen_integrity import FrozenGuard
    guard=FrozenGuard(root)
    try:
        if not local:
            from v13_history_integrity import recover_from_seed
            report['steps']['entry_history_recovery']=recover_from_seed(root,seed.parent/'v13_evidence_seed/entry_history_recovery.json')
        asof,official,errors=last_completed_session(root,day);report['asof']=asof;report['steps']['official']=official
        # Every completed phase is checkpointed. Interrupted work resumes from
        # accepted filing versions and the public-history age queue.
        collection.universe(root,asof)
        report['steps']['catalogs']=collection.current_catalogs(root,asof)
        report['steps']['catalogue_recovery']=collection.recover_public_catalogue(root,seed.parent/'v13_evidence_seed/catalogue_recovery',asof)
        # Fresh public prices must exist before financial completion chooses
        # the unranked price/history-eligible population for this source date.
        # Previously a new session always prioritised yesterday's histories.
        report['steps']['history']=collection.collect_histories(root,asof,history_budget)
        if not local:
            path=root/'prospective_observations.parquet'
            guard.check(pd.read_parquet(path) if path.exists() else pd.DataFrame())
            save(root)
        report['steps']['financials']=collection.collect_financials(root,asof,filing_budget)
        import v13_statement_eps as statement_eps
        report['steps']['reviewed_statement_EPS']=statement_eps.collect(root,seed.parent/'v13_evidence_seed'/'statement_eps_reviews.json')
        report['steps']['historical_events']=collection.historical_event_backfill(root,asof,budget=2)
        report['steps']['catalysts']=catalyst_execution.collect(root,asof,budget=120)
        report['steps']['reviewed_primary_sources']=catalyst_execution.seed_primary_reviews(root,seed.parent/'v13_evidence_seed'/'primary_execution_sources.json')
        report['steps']['entry_backfill']=action_audit.backfill(root,asof,budget=8)
        from v13_completion_work import primary_history_backfill
        report['steps']['primary_history_backfill']=primary_history_backfill(root,asof,budget=8)
        import v13_action_sources as action_sources
        report['steps']['action_notice_collection']=action_sources.collect(root,asof,budget=30)
        import v13_valuation_evidence as valuation_evidence
        report['steps']['valuation_snapshot']=valuation_evidence.collect(root,asof)
        if not local:
            path=root/'prospective_observations.parquet'
            guard.check(pd.read_parquet(path) if path.exists() else pd.DataFrame())
            save(root)
        frame,history=current_frame(root,asof)
        report['current_verified_market_rows']=len(frame)
        # Partial coverage is disclosed, not mistaken for a full-universe scan.
        report['public_universe_rows']=len(pd.read_parquet(root/'universe.parquet'))
        # Monthly refresh learns coefficients/blends automatically from the
        # trailing four-year, maturity-purged data. It is a new paper challenger.
        marker=root/'last_retrain_month.txt';month=asof[:7]
        if (retrain or not marker.exists() or marker.read_text()!=month) and len(history)>=500:
            try:
                blank=pd.DataFrame(columns=['date','isin']);blank.to_parquet(root/'empty_previous.parquet',index=False)
                market=features.build(root/'history',root/'empty_previous.parquet',root/'challenger_features',asof)
                from v13_information_clock import research_cutoffs
                market=research_cutoffs(market,datetime.now(timezone.utc))
                matrix=collection.enrich(market,pd.read_parquet(root/'financial_wide.parquet'))
                matrix=collection.announcement_counts(matrix,root,asof)
                for c in model.FEATURES:
                    if c not in matrix:matrix[c]=np.nan
                matrix.to_parquet(root/'challenger_matrix.parquet',index=False)
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
        if not frame.empty:
            wide=pd.read_parquet(root/'financial_wide.parquet') if (root/'financial_wide.parquet').exists() else pd.DataFrame()
            frame=financial_evidence.enrich_evidence(frame,wide,root,asof)
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
                b=scored(joblib.load(challenger_model),frame,policy='evidence')
                from v13_prediction_replay import snapshot as replay_snapshot
                report['exact_prediction_replay']={
                  'champion':replay_snapshot(frame,a,root/'active_heads.joblib',root,'champion'),
                  'challenger':replay_snapshot(frame,b,challenger_model,root,'challenger')}
                from v13_research_selection import export as export_research
                report['separate_research_selector']=export_research(a,root,report['exact_prediction_replay']['champion'])
                pair=register_pair(frame,a,b,hashlib.sha256((root/'active_heads.joblib').read_bytes()).hexdigest(),sha,root)
                export_watchlists(root,pair)
                report['paired_registration']={k:v for k,v in pair.items() if k!='isins'}
                report['champion_registry']=pair['champion_registry'];report['challenger_registry']=pair['challenger_registry']
            else:report['paired_registration']={'status':'BLOCKED_NO_COMPLETE_CHALLENGER'}
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
        if observations:observed.to_parquet(root/'prospective_observations.parquet',index=False)
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
            from v13_paired_forecasts import COMPARISON_CONTRACT
            paired=paired_observations(pd.DataFrame(observations),challenger_sha,comparison_contract=COMPARISON_CONTRACT)
            paired.to_parquet(root/'paired_mature_observations.parquet',index=False)
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
    except Exception as e:
        report['errors'].append({'phase':'worker','error':str(e)[:400]});raise
    finally:
        if 'frozen_integrity' not in report['steps']:
            try:
                old_path=root/'prospective_observations.parquet'
                state=pd.read_parquet(old_path) if old_path.exists() else pd.DataFrame()
                report['steps']['frozen_integrity']=guard.check(state,promote=report.get('promotion',{}).get('promote') is True)
            except Exception as exc:report['errors'].append({'phase':'frozen_integrity','error':str(exc)[:400]})
        root.mkdir(parents=True,exist_ok=True);(root/'latest_run.json').write_text(json.dumps(report,indent=2))
        if not local:
            receipt=save(root);(root.parent/'archive_receipt.json').write_text(json.dumps(receipt,indent=2))
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--state',default='v13_state');p.add_argument('--seed',default='v13_seed');p.add_argument('--day');p.add_argument('--history-budget',type=int,default=800);p.add_argument('--filing-budget',type=int,default=800);p.add_argument('--retrain',action='store_true');p.add_argument('--local',action='store_true');a=p.parse_args()
    day=a.day or str(pd.Timestamp.now(tz='Asia/Kolkata').date())
    print(json.dumps(run(a.state,a.seed,day,a.history_budget,a.filing_budget,a.retrain,a.local),indent=2))
