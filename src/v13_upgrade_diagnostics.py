"""Missing evidence and valuation horizon contracts for research reviews."""
import pandas as pd

def execution_diagnostics(bundle):
    v=bundle['observed_values'];states={}
    for key,ok in bundle['checks'].items():
        if ok:states[key]='SUPPORTED'
        elif key=='source_funding_sufficient':states[key]='FAILED' if v.get('funding_sufficient') is False else 'UNKNOWN'
        elif key=='required_operating_approvals':states[key]='FAILED' if v.get('approval_completed') is False else 'UNKNOWN'
        elif key=='positive_margin_with_profit_basis':states[key]='FAILED' if v.get('expected_margin_pct') is not None and v['expected_margin_pct']<=0 else 'UNKNOWN'
        elif key=='not_cancelled_or_certificate_pending':states[key]='FAILED'
        else:states[key]='FAILED' if v.get('order_status') in {'CANCELLED','EXECUTED'} or v.get('commissioned') is False else 'UNKNOWN'
    # Absence of an adverse notice is not independent proof of active status.
    if v.get('order_status') is None and v.get('execution_stage') is None:states['not_cancelled_or_certificate_pending']='UNKNOWN'
    return {**bundle,'research_check_states':states,
      'research_missing_checks':[k for k,v in states.items() if v=='UNKNOWN'],
      'research_failed_checks':[k for k,v in states.items() if v=='FAILED'],
      'owner_profit_bridge_status':'UNKNOWN_SOURCE_OWNER_PROFIT_AND_DILUTION_BRIDGE',
      'margin_is_owner_net_profit':v.get('margin_basis')=='PAT_AFTER_TAX',
      'order_value_is_recognised_revenue':False,'formal_execution_policy_changed':False}

def scenario_horizon(review,source_date):
    """A source-bound annual estimate must correspond to the stated horizon."""
    try:
        h=review['horizon_months'];date=pd.Timestamp(source_date).normalize()
        start=pd.Timestamp(review['forecast_period_start']);end=pd.Timestamp(review['forecast_period_end'])
        target=pd.Timestamp(review['valuation_target_date'])
        if any(pd.isna(x) or x.tzinfo is not None for x in (start,end,target)):return False
        if h not in (6,12) or not 330<=(end-start).days<=400:return False
        if not date<target<=date+pd.DateOffset(months=h):return False
        if end<date or end>target+pd.DateOffset(months=12):return False
        excerpt=review['input_source_excerpts']['forward_PAT']
        for x in (start,end):
            forms={x.strftime('%Y-%m-%d'),x.strftime('%d %B %Y'),x.strftime('%d %b %Y')}
            if not any(s.lower() in excerpt.lower() for s in forms):return False
        if review.get('forward_share_basis')!='FULLY_DILUTED_INCLUDING_REVIEWED_ACTIONS':return False
        return True
    except (KeyError,ValueError,TypeError):return False
