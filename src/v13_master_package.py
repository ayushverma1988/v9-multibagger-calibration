"""Export and verify the auditable user-facing master and numeric receipts."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile
import pandas as pd

def build(root,output,report=None):
    root=Path(root);folder=root/'master_data';output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    summary=json.loads((folder/'summary.json').read_text())
    master=pd.read_parquet(folder/'master_inputs.parquet');cells=pd.read_parquet(folder/'master_input_sources.parquet')
    assert len(master)==summary['issuers']==3200
    assert len(cells)==summary['required_input_cells']==86400
    assert not cells.duplicated(['isin','column']).any()
    assert int(cells.value.notna().sum())==summary['original_observed_cells']
    assert int(cells.resolved_value.isna().sum())==summary['remaining_missing_or_unresolved_cells']
    assert int(cells.eligible_current_two_source_input.sum())==summary['current_two_source_input_cells']
    assert int(cells.recovered_missing_original.sum())==summary['recovered_missing_original_cells']
    assert cells.loc[cells.recovered_missing_original,'value'].isna().all()
    assert cells.loc[cells.eligible_current_two_source_input,'validated_value'].notna().all()
    recovery=cells[cells.value.isna()&cells.resolved_value.notna()].copy()
    recovery.to_csv(folder/'recovered_missing_values.csv',index=False)
    original=json.loads((root/'evidence/universe_financial_coverage.json').read_text())
    prior=json.loads((folder/'earlier_twenty_comparison.json').read_text())
    assert len(prior)==20 and sum(r['current_market_input_available'] for r in prior)==20
    if (root/'redevelopment/latest.json').exists():
        observed=hashlib.sha256((root/'redevelopment/latest.json').read_bytes()).hexdigest()
    else:observed=None
    providers={s:len(list((folder/'provider_inputs').glob('*_'+s+'.json'))) for s in ('yahoo','finology','screener')}
    api_count=len(list((folder/'api_quote_inputs').glob('*.json')))
    review={**{k:v for k,v in summary.items() if k!='column_coverage'},
        'recovered_cells_all_periods':len(recovery),'source_statement_files_by_provider':providers,
        'indian_stock_api_quote_receipts':api_count,
        'original_numeric_financial_observations':original['nonmissing_numeric_fact_observations'],
        'unchanged_latest_completed_forecast_file_sha256':observed,
        'packaged_at_utc':datetime.now(timezone.utc).isoformat()}
    (folder/'package_verification.json').write_text(json.dumps(review,indent=2))
    readme=f'''# Multibagger master data — 21 October 2026

Market snapshot: {summary['market_asof']}.
Source review clock: {summary['review_clock_utc']}.

Open master_data/master_inputs.csv for the full 3,200-company master.
Original fields are preserved. resolved_ fields include separately corroborated
substitutes; validated_ fields additionally require sufficient freshness and
two distinct sources total. A wrapper and its upstream count once.

Recovered missing values, all observed periods: {len(recovery)}.
Recovered missing values meeting freshness limits: {summary['recovered_missing_original_cells']}.
Remaining missing or unresolved values: {summary['remaining_missing_or_unresolved_cells']}.
Current cells with two matching sources: {summary['current_two_source_input_cells']}.

The source ledger maps all 86,400 cells. recovered_missing_values.csv isolates
the recovered gaps. Provider statement files retain observed numeric facts,
URLs, hashes and clocks. Quote API receipts are quote snapshots, not dated
quarterly earnings, and are excluded from financial-statement recovery counts.

The earlier twenty-name comparison retains the old estimates. No new prediction
or retraining was performed by this recovery export. Source agreement does not
guarantee accuracy or stock returns. Unreported/ambiguous values remain missing.
Full original copyrighted HTML/PDF pages are excluded from this download.
'''
    selected=[]
    for path in folder.iterdir():
        if path.is_file() and path.suffix in {'.json','.csv','.parquet','.md'}:selected.append((path,'master_data/'+path.name))
    for directory in ('provider_inputs','api_quote_inputs'):
        selected.extend((p,'master_data/'+directory+'/'+p.name) for p in (folder/directory).glob('*.json'))
    selected.append((root/'financial_wide.parquet','financial_wide.parquet'))
    selected.append((root/'universe.parquet','universe.parquet'))
    selected.append((root/'evidence/universe_financial_coverage.json','universe_financial_coverage.json'))
    if report:selected.append((Path(report),'Recovery_report.md'))
    manifest=[{'path':name,'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()} for path,name in selected]
    with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        archive.writestr('README.md',readme)
        archive.writestr('manifest.json',json.dumps(manifest,indent=2))
        for path,name in selected:archive.write(path,name)
    with zipfile.ZipFile(output) as archive:assert archive.testzip() is None
    return {'path':str(output.resolve()),'bytes':output.stat().st_size,'sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'verification':review}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--state',required=True);p.add_argument('--output',required=True);p.add_argument('--report');a=p.parse_args();print(json.dumps(build(a.state,a.output,a.report)))
