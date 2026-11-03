"""Financial features with source-version coherence and dated comparatives.

The prior 450-day global filter could erase a needed prior-year quarter.
This module applies freshness to current observations, not to comparatives.
It never splices metrics from different original documents within a period.
The old frozen source and registered predictions are deliberately untouched.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

FIELDS = ('quarter_sales_yoy', 'quarter_pat_yoy', 'quarter_pat_margin',
          'annual_CFO_PAT', 'dated_debt_equity')


def coherent_period(known, kind, period=None):
    d = known[known.period_kind.eq(kind)]
    if period is not None:
        d = d[d.period.eq(period)]
    if d.empty:
        return d
    latest = d.period.max()
    d = d[d.period.eq(latest)]
    clock = d.available.max()
    d = d[d.available.eq(clock)]
    # Conflicting filings at the same clock are unresolved, not arbitrary ties.
    if d.source_sha256.nunique() != 1 or d.duplicated('metric').any():
        return d.iloc[:0]
    return d


def financial_row(facts, date, isin):
    date = pd.Timestamp(date)
    cut = (date.tz_localize('Asia/Kolkata') + pd.Timedelta(hours=15, minutes=30)).tz_convert('UTC')
    result = {c: np.nan for c in FIELDS}
    result.update(financial_status='UNKNOWN', financial_gate_status='UNKNOWN')
    known = facts[facts.available.le(cut) & facts.security_isin.eq(isin) & facts.period.le(date)]
    # Prefer a recent consolidated quarter; no fallback from incomplete facts
    # within the chosen basis to a different reporting basis.
    recent = known[known.period_kind.eq('quarter') & (date-known.period).dt.days.le(190)]
    if recent.empty:
        return result, []
    basis = 'consolidated' if recent.reporting_mode.eq('consolidated').any() else 'standalone'
    known = known[known.reporting_mode.eq(basis)]
    q = coherent_period(known, 'quarter')
    if q.empty or (date-q.period.iloc[0]).days > 190:
        return result, []
    used = [q]
    v = q.set_index('metric').value_INR
    result['reporting_mode'] = basis
    result['quarter_age_days'] = (date-q.period.iloc[0]).days
    if 'revenue' in v and 'pat' in v and v.revenue > 0:
        result['quarter_pat_margin'] = v.pat/v.revenue
    old = coherent_period(known, 'quarter', q.period.iloc[0]-pd.DateOffset(years=1))
    if len(old):
        used.append(old)
        prev = old.set_index('metric').value_INR
        for metric, feature in [('revenue', 'quarter_sales_yoy'), ('pat', 'quarter_pat_yoy')]:
            if metric in v and metric in prev and prev[metric] > 0:
                result[feature] = v[metric]/prev[metric]-1
    a = coherent_period(known, 'annual')
    if len(a) and (date-a.period.iloc[0]).days <= 450:
        used.append(a)
        av = a.set_index('metric').value_INR
        if 'pat' in av and 'cfo' in av and av.pat > 0:
            result['annual_CFO_PAT'] = av.cfo/av.pat
    b = coherent_period(known, 'instant')
    if len(b) and (date-b.period.iloc[0]).days <= 450:
        used.append(b)
        bv = b.set_index('metric').value_INR
        keys = ['equity','borrowings_current','borrowings_noncurrent']
        if all(k in bv for k in keys) and bv.equity > 0 and bv.borrowings_current >= 0 and bv.borrowings_noncurrent >= 0:
            result['dated_debt_equity'] = (bv.borrowings_current+bv.borrowings_noncurrent)/bv.equity
    values = np.array([result[c] for c in FIELDS], float)
    if np.isfinite(values).any():
        result['financial_status'] = 'PARTIAL_DATED_PRIMARY_FACTS'
    if np.isfinite(values).all():
        result['financial_status'] = 'FIVE_DATED_PREDICTORS_COMPLETE'
        passed = (result['quarter_sales_yoy'] >= .10 and result['quarter_pat_yoy'] >= .15
                  and result['quarter_pat_margin'] >= .08 and result['annual_CFO_PAT'] >= .6
                  and 0 <= result['dated_debt_equity'] <= 1)
        result['financial_gate_status'] = 'RESEARCH_PASS' if passed else 'RESEARCH_REJECT'
    # This research gate is NOT the original OPM/valuation/recurring-profit gate.
    sources = pd.concat(used).drop_duplicates(['source_sha256','metric','period_kind','period_end'])
    result['financial_available_at_utc'] = sources.available.max().isoformat()
    return result, sources.to_dict('records')


def build(panel_path, facts_path, old_matrix_path, output):
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(panel_path)
    facts = pd.read_parquet(facts_path)
    facts['available'] = pd.to_datetime(facts.available_at_utc, utc=True, format='mixed')
    facts['period'] = pd.to_datetime(facts.period_end)
    groups = {s:g for s,g in facts.groupby('symbol')}
    rows, links = [], []
    empty = facts.iloc[:0]
    for n, p in enumerate(panel[['date','symbol','isin']].to_dict('records'),1):
        fields, sources = financial_row(groups.get(p['symbol'],empty),p['date'],p['isin'])
        rows.append({**p,**fields})
        for s in sources:
            links.append({**p, **{k:s[k] for k in ['source_url','source_sha256','catalog_sha256','available_at_utc','period_end','period_start','period_kind','metric','reporting_mode']}})
        if n%10000 == 0: print('PIT rows',n,flush=True)
    frame = pd.DataFrame(rows)
    old = pd.read_parquet(old_matrix_path)
    old['date'] = pd.to_datetime(old.date);frame['date'] = pd.to_datetime(frame.date)
    frame = frame.merge(old[['date','symbol','isin','catalyst_status','order_disclosure_count90','capacity_disclosure_count90']], on=['date','symbol','isin'],validate='one_to_one')
    frame.to_parquet(out/'training_inputs.parquet',index=False)
    pd.DataFrame(links).to_parquet(out/'financial_source_links.parquet',index=False)
    summary = {'rows':len(frame),'financial_status':frame.financial_status.value_counts().to_dict(),
               'research_financial_gate':frame.financial_gate_status.value_counts().to_dict(),
               'issuers_with_partial_or_complete_predictors':frame.loc[frame.financial_status.ne('UNKNOWN'),'symbol'].nunique(),
               'nonnull_features':frame[list(FIELDS)].notna().sum().to_dict(),
               'full_original_financial_gate_validated':False,'verified_earnings_catalyst_rows':0,
               'historical_valuation_available':False,'production_approved':False}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--panel',required=True);p.add_argument('--facts',required=True);p.add_argument('--old-matrix',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    build(a.panel,a.facts,a.old_matrix,a.output)
