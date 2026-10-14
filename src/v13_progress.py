"""Measured five-area evidence coverage; no subjective completion percentages."""
import json
from pathlib import Path
from datetime import datetime,timezone

def snapshot(root,report,observations):
    root=Path(root);rows=int(report.get('current_verified_market_rows',0));total=len(observations)
    mature=int(report.get('independently_verified_mature_outcomes',0))
    entries=int(observations.entry_verified.fillna(False).sum()) if total and 'entry_verified' in observations else 0
    values=[('Financial inputs',int(report.get('financial_complete_rows',0)),rows),
            ('Full execution proof',int(report.get('execution_chain_supported_rows',0)),rows),
            ('2x earnings/valuation',int(report.get('source_bound_2x_scenario_rows',0)),rows),
            ('Verified entry records',entries,total),('Mature unseen 12m outcomes',mature,total)]
    areas=[{'area':name,'verified':n,'denominator':den,'coverage_percent':round(100*n/den,2) if den else None} for name,n,den in values]
    financial=root/'evidence/financial_completeness.json'
    details=json.loads(financial.read_text()) if financial.exists() else {}
    result={'recorded_at_utc':datetime.now(timezone.utc).isoformat(),'source_asof':report.get('asof'),
      'metric':'EVIDENCE_COVERAGE_NOT_ENGINEERING_COMPLETION_OR_PREDICTION_ACCURACY','different_denominators_must_not_be_averaged':True,
      'areas':areas,'financial_quality_pass_rows':report.get('financial_quality_pass_rows'),
      'underlying_required_facts_complete_rows':details.get('underlying_required_facts_complete_rows'),
      'latest_reported_diluted_EPS_available_rows':details.get('latest_reported_diluted_EPS_available_rows'),
      'six_twelve_month_validation_requires_elapsed_market_time':True,'accuracy_approved':False,
      'engineering_eta_not_a_data_or_validation_deadline':True,'current_paired_cohort':report.get('paired_registration')}
    out=root/'evidence';out.mkdir(exist_ok=True);(out/'progress_status.json').write_text(json.dumps(result,indent=2));return result
