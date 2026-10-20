"""Financial coverage of every public issuer, independent of scored prices."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from v12_5_features import features
from v13_financial_evidence import balancing_quarters,financial_quality,financial_readiness,REQUIRED
from v13_information_clock import research_cutoffs,cutoff

NON_FACT_COLUMNS={'period','available'}

def audit(universe,wide,asof,root):
    date=pd.Timestamp(asof)
    clock=cutoff(research_cutoffs(pd.DataFrame({'date':[date]}),pd.Timestamp.now(tz='UTC')).iloc[0])
    # Counts refer to available originals, including retained versions. They
    # are not amounts of money, unique companies, or independent confirmations.
    raw=wide[wide.available.le(clock)&wide.period.le(date)].copy() if not wide.empty else wide.copy()
    isins=set(universe['isin']);raw=raw[raw.security_isin.isin(isins)] if not raw.empty else raw
    numeric=[c for c in raw.select_dtypes(include='number') if c not in NON_FACT_COLUMNS and c!='derived_balancing_quarter']
    values=int(raw[numeric].notna().sum().sum()) if numeric else 0
    expanded=balancing_quarters(raw)
    groups=dict(tuple(expanded.groupby('security_isin'))) if not expanded.empty else {}
    rows=[]
    for r in universe.to_dict('records'):
        isin=r['isin'];w=groups.get(isin,pd.DataFrame())
        f,_=features(w,date) if not w.empty else ({},[])
        quality=financial_quality(f);readiness=financial_readiness(w,date,f)
        rows.append({'symbol':r.get('nse_current_symbol',r.get('symbol')),'isin':isin,
          'source_asof':asof,'has_original_financials':not w.empty,
          'financial_observation_rows':len(w),'financial_complete':quality['complete'],
          'financial_quality_status':quality['status'],'financial_quality_pass':quality['quality_pass'],
          'unknown_fields':quality['unknown_fields'],'failed_checks':quality['failed_checks'],**readiness})
    summary={'source_asof':asof,'information_cutoff_utc':clock.isoformat(),
      'scope':'FULL_PUBLIC_UNIVERSE_INDEPENDENT_OF_PRICE_OR_SELECTION',
      'public_universe_issuers':len(rows),'issuers_with_financial_observations':sum(r['has_original_financials'] for r in rows),
      'original_period_basis_version_rows':len(raw),'nonmissing_numeric_fact_observations':values,
      'unique_source_document_hashes':int(raw.source_sha256.nunique()) if not raw.empty else 0,
      'unique_issuer_period_basis_kind_rows':int(len(raw.drop_duplicates(['security_isin','period_end','reporting_mode','period_kind']))) if not raw.empty else 0,
      'fresh_preferred_basis_quarter_issuers':sum(r['preferred_quarter_fresh'] for r in rows),
      'complete_seven_check_issuers':sum(r['financial_complete'] for r in rows),
      'quality_pass_issuers':sum(r['financial_quality_pass'] for r in rows),
      'known_failure_issuers':sum(r['financial_quality_status']=='KNOWN_FINANCIAL_FAILURE' for r in rows),
      'unknown_no_observed_failure_issuers':sum(r['financial_quality_status']=='UNKNOWN_FINANCIAL_INPUTS' for r in rows),
      'underlying_required_facts_complete_issuers':sum(r['underlying_required_facts_complete'] for r in rows),
      'latest_reported_diluted_EPS_available_issuers':sum(r['latest_reported_diluted_EPS_available'] for r in rows),
      'required_check_nonmissing':{k:sum(k not in r['unknown_fields'] for r in rows) for k in REQUIRED},
      'consolidated_basis_gap_issuers':sum(r['reporting_basis_status']=='MISSING_RECENT_CONSOLIDATED_QUARTER_NO_STANDALONE_SUBSTITUTION' for r in rows),
      'all_financial_data_complete':all(r['underlying_required_facts_complete'] for r in rows),
      'source_agreement_is_not_truth_certification':True,'rows':rows}
    out=Path(root)/'evidence';out.mkdir(parents=True,exist_ok=True)
    (out/'universe_financial_coverage.json').write_text(json.dumps(summary,indent=2))
    return {k:v for k,v in summary.items() if k!='rows'}
