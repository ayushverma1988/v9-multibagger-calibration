"""Six-area evidence tasks and bounded whole-market primary-price backfill.

No forecasts, probabilities, heads, selections or outcomes are modified here.
Queue-sized lower bounds are not promises of evidence or calibration dates.
"""
from pathlib import Path
from datetime import datetime,timezone
import json,math,time
import pandas as pd
from v12_public_history import bounds

def load(path,default):
    return json.loads(path.read_text()) if path.exists() else default

def primary_history_backfill(root,asof,budget=8,goal_sessions=127,seconds=150):
    from v12_3_official_monitor import collect
    from v13_action_audit import expected_sessions
    root=Path(root);out=root/'evidence';out.mkdir(exist_ok=True)
    low,day=bounds(asof)
    if day.tz_localize('Asia/Kolkata')+pd.Timedelta(hours=15,minutes=30)>pd.Timestamp.now(tz='UTC'):
        raise ValueError('Official-history backfill requires a completed source session')
    start=max(low,day-pd.Timedelta(days=max(270,int(goal_sessions)*2)))
    calendar=load(root/'action_audit/calendar.json',None)
    expected=expected_sessions(start,day,calendar)
    dates=expected if expected is not None else list(pd.bdate_range(start,day))
    dates=sorted(dates,reverse=True)[:max(1,int(goal_sessions))]
    attempts_path=out/'primary_history_attempts.json';attempts=load(attempts_path,{})
    folder=root/'official';folder.mkdir(exist_ok=True)
    cached={str(d.date()) for d in dates if (folder/(str(d.date())+'_official_market.parquet')).exists()}
    pending=sorted([d for d in dates if str(d.date()) not in cached],key=lambda d:(attempts.get(str(d.date()),'0000'),-d.value))
    errors=[];downloaded=[];started=time.monotonic()
    for date in pending[:max(0,min(8,int(budget)))]:
        if time.monotonic()-started>=seconds:break
        key=str(date.date());attempts[key]=datetime.now(timezone.utc).isoformat()
        try:
            receipt=collect(key,folder);downloaded.append(receipt);cached.add(key)
        except Exception as exc:
            error=str(exc)[:180];errors.append({'date':key,'error':error})
            if any(s in error.lower() for s in ('429','too many requests','403','forbidden')):break
    attempts_path.write_text(json.dumps(attempts,indent=2))
    result={'source_asof':asof,'goal_session_files':len(dates),'available_session_files':len(cached),
      'remaining_session_files':len(dates)-len(cached),'new_session_files':len(downloaded),'batch_budget':min(8,max(0,int(budget))),
      'calendar_verified_for_entire_window':expected is not None,'receipts':downloaded,'errors':errors,
      'four_year_start':str(low.date()),'requested_start':str(min(dates).date()) if dates else None,
      'all_public_listings_per_date_not_private_stock_requests':True,
      'unadjusted_primary_prices_not_adjusted_forecast_history':True,'scoring_coverage_or_outcome_certification_added':False}
    (out/'primary_history_backfill.json').write_text(json.dumps(result,indent=2));return result

def completion_plan(root,report,observations):
    """Persist missing fields/project dependencies and elapsed-time milestones."""
    root=Path(root);out=root/'evidence';out.mkdir(exist_ok=True)
    fin=load(out/'financial_completeness.json',{})
    bundles=load(root/'catalyst_evidence/execution_bundles.json',[])
    reviews=load(root/'catalyst_evidence/quantified_execution_reviews.json',[])
    # Current market rows can have no bundle while reviewed projects belong
    # to other public issuers; preserve those dependencies separately.
    from v13_catalyst_execution import project_bundles
    now=pd.Timestamp(datetime.now(timezone.utc))
    reviewed_projects=[z for isin in sorted({r['isin'] for r in reviews}) for z in project_bundles(reviews,isin,now)]
    rows=int(report.get('current_verified_market_rows',0));steps=report.get('steps',{})
    requests=[]
    for r in fin.get('issuer_backlog',[]):
        missing=set(r.get('unknown_fields',[]))
        missing.update(k for k in r.get('expanded_unknown_fields',[]) if k in {
          'owner_PAT_yoy','owner_PAT_margin','verified_recurring_diluted_EPS_growth','reported_diluted_EPS_yoy_same_document'})
        if missing:requests.append({'isin':r['isin'],'symbol':r['symbol'],'missing_fields':sorted(missing),
          'requires_compatible_original_periods_and_basis':True,'unknown_reasons':r.get('unknown_reasons',{}),
          'owner_earnings_source_status':r.get('owner_earnings_source_status','UNKNOWN_OWNER_SOURCE_NOT_REVIEWED'),
          'cannot_infer_profitability_or_backdate_into_prior_forecasts':True})
    verified=observations[observations.get('entry_verified',pd.Series(False,index=observations.index)).eq(True)] if not observations.empty else observations
    entries=pd.to_datetime(verified['first_entry_date'],errors='coerce').dropna() if 'first_entry_date' in verified else pd.Series(dtype='datetime64[ns]')
    earliest=str(entries.min().date()) if not entries.empty else None
    milestones={str(h)+'m':str((entries.min()+pd.DateOffset(months=h)).date()) if not entries.empty else None for h in (6,12)}
    def pass_bound(step,key,budget):
        remaining=steps.get(step,{}).get(key)
        return {'remaining_catalogue_items':remaining,'maximum_items_per_run':budget,
          'optimistic_minimum_runs_for_one_catalogue_pass':math.ceil(remaining/budget) if isinstance(remaining,int) and budget>0 else None,
          'is_evidence_completion_or_calendar_deadline':False,'successful_identity_parse_and_disclosure_not_guaranteed':True}
    from v13_research_evaluation import paired_readiness
    result={'recorded_at_utc':now.isoformat(),'source_asof':report.get('asof'),
      'scope':'EVIDENCE_TASKS_NOT_FORECASTS_OR_PERCENT_ENGINEERING_DONE','frozen_records_modified':False,'promotion_authorised':False,
      'financial':{'complete_rows':fin.get('complete_rows',0),'current_market_rows':rows,'requests':requests,
        'owner_originals_have_reserved_collection_budget':True,**pass_bound('financials','remaining',steps.get('financials',{}).get('budget',800))},
      'catalyst':{'current_market_bundles':bundles,'reviewed_projects_including_other_public_issuers':reviewed_projects,
        'reviewed_sources_without_complete_chain':sum(not z['supported'] for z in reviewed_projects),
        'same_explicit_project_identity_required':True,'field_capture':steps.get('catalyst_field_capture',{}),
        **pass_bound('catalysts','backlog',steps.get('catalysts',{}).get('budget',120))},
      'valuation':{'source_supported_net2x_rows':report.get('source_bound_2x_scenario_rows',0),
        'required_original_inputs':['recurring_owner_earnings_bridge','forward_owners_PAT_and_period','forward_diluted_share_count_and_capital_actions','justified_forward_valuation_and_horizon','source_price_or_verified_entry_basis','net_transaction_costs'],
        'current_reported_PE_or_hypothetical_hurdles_do_not_complete_these_inputs':True},
      'entries_actions':{'registered_records':len(observations),'verified_entries':len(verified),
        'pending_or_unverified_records':len(observations)-len(verified),'notice_failures_rotate':True,
        'raw_history_is_not_a_no_action_interval':True,**pass_bound('action_notice_collection','remaining_public_notices',30)},
      'future_evaluation':{'paired_readiness':paired_readiness(observations),'earliest_verified_entry':earliest,
        'earliest_possible_full_horizon_labels':milestones,'these_dates_do_not_certify_source_complete_outcomes_or_promotion':True,
        'requires_six_mature_decision_months_500_pairs_20_positives_and_selected_samples':True,
        'monthly_refits_remain_experimental_and_frozen_heads_unchanged':True},
      'breadth':{'verified_market_rows':rows,'public_universe_rows':report.get('public_universe_rows'),
        'supplemental_primary_history':load(out/'primary_history_backfill.json',{}),
        'missing_listing_identity_or_adjustment_evidence_remains_excluded':True},
      'eta':{'active_software_batch':'COMPLETE_ONLY_AFTER_TESTS_AND_AUDITED_DEPLOYMENT',
        'all_available_financial_and_catalyst_inputs':'ISSUER_AND_SOURCE_DEPENDENT_NO_GUARANTEED_FINISH',
        'mature_validation':'ACTUAL_VERIFIED_ENTRY_PLUS_6_OR_12_MONTHS_AND_COMPLETE_ACTION_PROOF',
        'catalogue_pass_counts_are_not_prediction_accuracy_or_success_deadlines':True}}
    (out/'six_area_completion_plan.json').write_text(json.dumps(result,indent=2,allow_nan=False));return result
