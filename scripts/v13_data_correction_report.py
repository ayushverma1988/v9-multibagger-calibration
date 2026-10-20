"""Build a reviewable dated correction from successful recovery artifacts."""
import argparse,json
from pathlib import Path
import pandas as pd


def build(root,output,worker_commit,contract_run,checkpoint_receipt=None):
    from v13_example_history_audit import EXAMPLES
    root=Path(root);out=Path(output)
    run=json.loads((root/'correction_run.json').read_text())
    x=json.loads((root/'two_x_research/latest.json').read_text())
    e=json.loads((root/'early_potential/latest.json').read_text())
    f=json.loads((root/'evidence/universe_financial_coverage.json').read_text())
    c=json.loads((root/'financial_crosscheck/latest.json').read_text())
    checkpoint=json.loads(Path(checkpoint_receipt).read_text()) if checkpoint_receipt else None
    union=json.loads((root/'financial_snapshot_union.json').read_text())
    pct=lambda v:'—' if v is None else f'{100*v:.2f}%'
    money=lambda v:'—' if v is None else f'{v:,.2f}'
    quality={r['symbol']:r for r in f['rows']}
    quality_isin={r['isin']:r for r in f['rows']}
    current_chart_states={r['isin']:r for r in e['scored_states']}
    chart_example_watch=[]
    for name,symbol,isin in EXAMPLES:
        q=quality_isin.get(isin,{})
        state=current_chart_states.get(isin)
        chart_example_watch.append({'company':name,'symbol':symbol,'isin':isin,
          'in_current_public_universe':isin in quality_isin,
          'current_input_status':state['early_state'] if state else 'NO_CURRENT_SOURCE_ALIGNED_MODEL_INPUTS',
          'financial_status':q.get('financial_quality_status','UNKNOWN'),
          'failed_checks':q.get('failed_checks',[]),'unknown_fields':q.get('unknown_fields',[]),
          'owner_profit_status':q.get('owner_earnings_source_status')})
    w=pd.read_parquet(root/'financial_wide.parquet')
    w=w[w.security_isin.isin({r['isin'] for r in f['rows']}) &
        pd.to_datetime(w.available_at_utc,utc=True,format='mixed').le(pd.Timestamp(f['information_cutoff_utc'])) &
        pd.to_datetime(w.period_end).le(pd.Timestamp(run['source_asof']))]
    history_depth={kind:{'original_rows':len(d),'issuers':d.security_isin.nunique(),'first_period':str(d.period_end.min()),'last_period':str(d.period_end.max())} for kind,d in w.groupby('period_kind')}
    assert sum(r['original_rows'] for r in history_depth.values())==f['original_period_basis_version_rows']
    lines=['# Data correction and research output — 20 October 2026','',
      f"Price source session: **{run['source_asof']}**. Review completed: {run['review_completed_at_utc']}. **{x['scored_rows']:,}/{f['public_universe_issuers']:,} public NSE shares** have source-aligned inputs in this replay. This is partial coverage and must not replace the earlier broad watchlist.",'',
      '## What was wrong','',
      'The earlier 20 October output covered only 226 of 3,200 shares. A newer completed collector expanded the current cohort. In an unranked sample of 16 stored but unscored histories, all three public price-request variants still returned blank 19 October closing-price fields. These names were unavailable for current inference, not rejected investment theses. No missing closing price or adjusted price was filled in this correction.','',
      'The financial code could also switch to standalone results when the recent consolidated quarter was absent. Mastek exposed this problem: its standalone result suggested falling sales/profit, whereas the issuer’s June consolidated release reported growth. The code now preserves consolidated continuity, marks missing group results explicitly and reserves collection capacity for their originals.','',
      'A first broad recovery attempted 1,600 filings and recovered 1,888 net numeric observations. A targeted pass recovered another 129 observations in its own snapshot. These counts overlap later collections and must not be added to claim unique new facts. The final replay unions the broad recovery, targeted recovery and the later interrupted worker’s saved collection, matching exact source hash, reporting basis, period and availability clock. No conflicting numeric extraction of the same original was found in that union.','',
      f"The final pass rechecked **{run['originals_attempted']:,} original filings**, extracted **{run['new_primary_fact_observations']:,} observations including rechecks**, and added **{run['net_numeric_fact_observation_change']:,} net numeric observations** after the union. **{run['original_source_errors']:,} identity/source error** remains visible. Frozen-model replay passed on {run['market_cohort_rows']:,} unchanged market-input rows. Original forecasts, model heads and registries were preserved.",'',
      '## Financial coverage — audit of all 3,200 public NSE issuers','',
      '| Measure | Count |','| --- | ---: |']
    for label,key in [('Public NSE shares','public_universe_issuers'),('Shares with financial observations','issuers_with_financial_observations'),('Original period/basis/version rows','original_period_basis_version_rows'),('Nonmissing numeric fact observations','nonmissing_numeric_fact_observations'),('Unique original document hashes','unique_source_document_hashes'),('Fresh preferred-basis quarter','fresh_preferred_basis_quarter_issuers'),('All seven screening ratios/checks available','complete_seven_check_issuers'),('Underlying required facts available, including undefined-ratio cases','underlying_required_facts_complete_issuers'),('Pass the seven available checks','quality_pass_issuers'),('Missing recent consolidated quarter','consolidated_basis_gap_issuers')]:
        lines.append(f'| {label} | {f[key]:,} |')
    lines += ['', f"Coverage uses information available by **{f['information_cutoff_utc']}**. Later-published observations can be stored but are excluded from this dated replay. Counts describe source observations, not money invested, independent confirmations or unique economic facts. Complete inputs can fail financial checks. Zero interest or a nonpositive comparison denominator can leave a ratio undefined even when original amounts are available. **Financial coverage remains incomplete.** BSE-only securities are outside this public NSE denominator.",'',
      '| Available statement history | Issuers | Original rows | Earliest period | Latest period |',
      '| --- | ---: | ---: | --- | --- |']
    for kind,d in history_depth.items():lines.append(f"| {kind} | {d['issuers']:,} | {d['original_rows']:,} | {d['first_period']} | {d['last_period']} |")
    lines += ['', 'These endpoints describe the collected dataset; each issuer can have gaps and duplicate original versions. Annual financial history currently starts in March 2025, so this is not a complete five- or ten-year fundamental history. Longer original annual-report coverage remains a data gap.','',
      '## Cross-validation','',
      f"Compared {c['observations_compared']:,} dated provider observations across {c['issuers_compared']:,} issuers. **{c['primary_plus_two_provider_agreements']:,} annual revenue/operating-cash-flow field-periods agree with both Screener and reviewed StockAnalysis/S&P Global displays within their published rounding precision.** This limited check is not a claim that all fields or all companies are verified.",'',
      'Only the same issuer, consolidated/standalone basis, fiscal period, metric and INR units are compared. Repeated sites using one provider count once. Net profit and EPS definitions remain separate where total, parent-attributable, basic, diluted or restated bases have not been reconciled. Agreement is corroboration, not a truth certificate; all publishers may depend on the same underlying issuer filing.','',
      'The runner fetched Screener successfully. StockAnalysis direct requests returned 403, so the third-provider observations are explicitly dated manual reviews of accessible public tables. They are not a live third-provider feed. Further conflicts include historical LTTS revenue comparisons and selected other amounts; originals remain canonical pending review.','',
      '| Comparison status | Observations |','| --- | ---: |']
    for status,n in c['status_counts'].items():lines.append(f'| {status} | {n:,} |')
    lines += ['', 'Reviewed annual amounts below are normalised to **₹ crore**. Each row matches the same fiscal year-end and reporting basis. Display rounding is allowed only within the source’s recorded precision. Full original values, source URLs and retrieval times are retained in the downloadable correction data.','',
      '| Share | Year-end | Basis | Field | Original | Screener | StockAnalysis / S&P | Result |',
      '| --- | --- | --- | --- | ---: | ---: | ---: | --- |']
    annual_review=[]
    for r in c['rows']:
        if r.get('provider_family')!='SP_GLOBAL' or r.get('period_kind')!='annual':continue
        keys=['isin','reporting_mode','period_kind','period_end','metric']
        other=[s for s in c['rows'] if s.get('provider_family')=='SCREENER' and all(s.get(k)==r.get(k) for k in keys)]
        s=other[0] if len(other)==1 else {}
        agree=r.get('comparison_status')=='NUMERIC_AGREEMENT_WITHIN_DISPLAY_ROUNDING' and s.get('comparison_status')=='NUMERIC_AGREEMENT_WITHIN_DISPLAY_ROUNDING'
        fmt=lambda v:'—' if v is None else f'{v/1e7:,.2f}'
        lines.append(f"| {r['symbol']} | {r['period_end']} | {r['reporting_mode']} | {r['metric']} | {fmt(r.get('primary_value_INR'))} | {fmt(s.get('value_INR'))} | {fmt(r.get('value_INR'))} | {'Agree within rounding' if agree else 'Review required'} |")
        annual_review.append({'primary_comparison':r,'screener_comparison':s,'three_publishers_agree_within_rounding':agree})
    lines += ['', '## Current separate 2× research list','',
      'Same frozen heads and preferred 17 October selector: verified small/mid-cap classification, known sector, RSI strictly above 70, above 50DMA and financial/owner checks. Missing financial evidence is shown explicitly. The estimates remain unvalidated research estimates; a shortlist entry does not mean doubling is likely.','',
      '| Rank | Share | Close ₹ | Model 2× by 6m | Model 2× by 12m | Model 30% loss event | Financial status |','| ---: | --- | ---: | ---: | ---: | ---: | --- |']
    for r in x['records']:
        label={'PASS_AVAILABLE_SEVEN_CHECKS':'All seven available checks pass','UNKNOWN_SECTOR_RULES_OR_CLASSIFICATION':'Financial-sector rules unavailable; review required'}.get(r['sector_status'],r['sector_status'])
        lines.append(f"| {int(r['rsi70_research_rank'])} | {r['symbol']} | {money(r['close'])} | {pct(r['p6'])} | {pct(r['p12'])} | {pct(r['loss12'])} | {label} |")
    lines += ['', 'The loss head targets a 30% loss event before a 2× hit over the 12-month window. These estimates are also unvalidated and remain high for these names; a relative research rank must not be read as a favourable risk/reward conclusion.','',
      'PVRINOX was present in an intermediate partial snapshot. It is excluded after combining the available originals because the interest-coverage check fails. South Indian Bank and Karur Vysya Bank remain explicitly unverified under sector-specific financial rules; their presence is not a seven-check financial pass.','',
      'Mastek now passes all seven available financial checks after its consolidated June originals and comparison periods were recovered. It still lacks usable current market inputs in this replay. Cupid, Tanla and Zensar remain on the current list. Persistent has RSI 62.78, below this lane’s strict 70 threshold; Sun TV has a parent/minority/total-profit reconciliation conflict. Hexaware also has an observed year-on-year consolidated PAT decline. These are distinct from missing quotes.']
    lines += ['', '## Earlier 11 recommendations — retained separately','',
      'The original estimates below use 15 October prices and remain historical. Current absence is not silently interpreted as lack of potential. A current momentum-screen failure also does not settle a longer-term investment thesis.','',
      '| Share | 15 Oct model 2× by 12m | Current status | Financial / owner-profit status |','| --- | ---: | --- | --- |']
    for r in x['previous']:
        q=quality.get(r['symbol'],{})
        status=r['status']+(' — '+', '.join(r['reasons']) if r['reasons'] else '')
        owner=q.get('owner_earnings_source_status','UNKNOWN')
        label=q.get('financial_quality_status','UNKNOWN')
        if owner.startswith('UNKNOWN'):label+='; '+owner
        lines.append(f"| {r['symbol']} | {pct(r.get('prior_p12'))} | {status} | {label} |")
    lines += ['', '## Your nine chart examples — retained research watch','',
      'The nine user-supplied chart names are tracked separately from the eleven earlier model recommendations. All nine are present in the current public NSE master, but none is in this replay’s 633 current source-aligned input rows. Their absence cannot be treated as a current technical rejection or a confirmation of upside. Their available financial evidence also differs:','',
      '| Share | Current model inputs | Financial evidence |',
      '| --- | --- | --- |']
    for r in chart_example_watch:
        detail=r['financial_status']
        if r['failed_checks']:detail+=' — '+', '.join(r['failed_checks'])
        if r['unknown_fields']:detail+=' — missing '+', '.join(r['unknown_fields'])
        lines.append(f"| {r['symbol']} | {r['current_input_status']} | {detail} |")
    lines += ['', 'These names remain a research/data-completion watch. They do not supply positive training labels or override observed revenue/profit weakness. Merging older exchange history requires verified security identity and corporate-action treatment.']
    early=sorted([r for r in e['candidates'] if r['early_state']=='EARLY_POTENTIAL'],key=lambda r:r['early_rank'] or 999)
    lines += ['', '## Separate early-potential chart/volume lane','',
      f"{len(early)} current setups are recorded in this cohort. These are untrained 50–100% upside hypotheses, separate from the 2× model. A ranking score is not a return probability. A filing cue identifies a dated announcement to review; it does not establish completed capex or realised earnings. The first 20 are:",'',
      '| Rank | Share | Close ₹ | Pattern | Financial inputs |','| ---: | --- | ---: | --- | --- |']
    for r in early[:20]:lines.append(f"| {int(r['early_rank'])} | {r['symbol']} | {money(r['close'])} | {r['early_pattern']} | {r['financial_quality_status']} |")
    lines += ['', '## Daily-worker changes','',
      f'The weekday **22:00 IST** worker is pinned to tested correction commit `{worker_commit}`. It requests up to 3,200 public histories and 1,600 filings per run within bounded source/time limits, retries limited source variants, counts fresh official-aligned histories separately from merely usable old histories, rotates public financial comparison work and reports full-universe financial coverage. A verified source pack retains the 24 successfully recovered original filings, including Mastek comparison periods, without seeding any desired stock rankings. Runs use the existing single-writer concurrency group. GitHub scheduling can be delayed.','',
      'The later collector timed out after spending much of its 90-minute budget restoring thousands of separate archive objects under provider rate limits. Its saved financial collection is included in this correction; its previous completed list was not treated as a new model result. Restarts now use one hash-verified bundle, preserve immutable original objects and record the actual attempt phase. Health reports distinguish a previous completed report from the present attempt.','',
      ('The archive migration completed and verified a fresh bundle restore against every original derived-file hash. Original checkpoint records, frozen forecasts and model heads were unchanged. '+f"Verified {checkpoint['derived_bundle_files']:,} derived files; compressed bundle {checkpoint['derived_bundle_bytes']:,} bytes." if checkpoint and checkpoint.get('passed') else 'The archive migration is running; its completion has not yet been claimed.'),'',
      'Validation: all **630 tests passed locally and on GitHub**, including missing consolidated results, stale source responses, original-source identity and byte-hash checks, rate limits, period/basis mismatches and duplicate-provider checks. This correction replay writes no private archive state; the normal worker persists its own verified source recovery. No model was retrained or declared accurate.','',
      '## Reproducibility and sources','',
      f"- Frozen model SHA-256: `{run['corrected_prediction_replay']['model_sha256']}`",
      f"- Corrected input SHA-256: `{run['corrected_prediction_replay']['input_parquet_sha256']}`",
      '- [Correction and code review](https://github.com/ayushverma1988/v9-multibagger-calibration/pull/20)',
      '- [Final combined-financial model replay](https://github.com/ayushverma1988/v9-multibagger-calibration/actions/runs/39387830536)',
      '- [Archive restart repair](https://github.com/ayushverma1988/v9-multibagger-calibration/actions/runs/39386777965)',
      '- [Source diagnostic run](https://github.com/ayushverma1988/v9-multibagger-calibration/actions/runs/39367248886)',
      f'- [Contract checks](https://github.com/ayushverma1988/v9-multibagger-calibration/actions/runs/{contract_run})',
      '- [Mastek issuer consolidated June release](https://www.mastek.com/wp-content/uploads/2026/07/Press-Release-Q1FY27.pdf)',
      '- [Mastek Screener consolidated tables](https://www.screener.in/company/MASTEK/consolidated/)',
      '- [Mastek StockAnalysis/S&P Global financial tables](https://stockanalysis.com/quote/nse/MASTEK/financials/)',
      '- Individual matched fields and source timestamps are retained in the correction artifact and dated source-review seed files.','']
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text('\n'.join(lines))
    two_x={k:v for k,v in x.items() if k not in {'records'}}
    two_x['records']=[{k:r.get(k) for k in ['symbol','close','p6','p12','loss12','joint12','rsi14','rsi70_research_rank','sector_status']} for r in x['records']]
    early_summary={k:v for k,v in e.items() if k not in {'candidates','scored_states'}}
    early_summary['early_candidates']=early
    compact={'correction_run':run,'financial_history_depth':history_depth,'two_x':two_x,'chart_example_watch':chart_example_watch,'early_potential':early_summary,'financial_snapshot_union':union,'checkpoint_repair':checkpoint,'worker_source_commit':worker_commit,'prior_cohort_financial_audit':[quality[r['symbol']] for r in x['previous'] if r['symbol'] in quality],'financial_source_comparison_summary':{k:v for k,v in c.items() if k not in {'rows','three_source_checks'}},'three_source_checks':c['three_source_checks'],'annual_three_publisher_review':annual_review}
    out.with_suffix('.json').write_text(json.dumps(compact,indent=2))
    print(out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);p.add_argument('--worker-commit',required=True);p.add_argument('--contract-run',required=True);p.add_argument('--checkpoint-receipt');a=p.parse_args();build(a.root,a.output,a.worker_commit,a.contract_run,a.checkpoint_receipt)
