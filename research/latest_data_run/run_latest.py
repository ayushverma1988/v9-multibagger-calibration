"""Bounded latest-data V13 paper run; keep source failures and prior results dated."""
import argparse,hashlib,io,json,os,re,time
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
from urllib.parse import urlencode,quote
import joblib,numpy as np,pandas as pd,requests
from primary_history_extension import extend,ANCHOR,TARGET

def plain(x):
    if isinstance(x,dict):return {str(k):plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [plain(v) for v in x]
    if isinstance(x,(np.integer,np.bool_)):return x.item()
    if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
    if isinstance(x,(pd.Timestamp,datetime)):return x.isoformat()
    if x is pd.NA or x is pd.NaT:return None
    return x
def save(path,value):Path(path).write_text(json.dumps(plain(value),indent=2,allow_nan=False))
def protected(root):
    paths=list(root.glob('registries/*/*.json'))+list(root.glob('paired_cohorts/*.json'))
    paths += [root/n for n in ['active_heads.joblib','active_model.json','challenger/research_heads.joblib','challenger/freeze.json','prospective_observations.parquet']]
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}

def run(root,out):
    from v13_model import probabilities
    from v13_runner import current_frame,paired_registration_coverage_gate
    from v13_history_views import canonical_histories
    from v12_3_official_monitor import collect
    from v12_public_history import parse_chart
    import v13_available_financials as available
    from v13_priority_research import rank,WEIGHTS
    from v13_ranking_candidate import rank_candidate,WEIGHTS as TRIAL_WEIGHTS
    root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=True);src=out/'sources';src.mkdir(exist_ok=True)
    before=protected(root);started=datetime.now(timezone.utc)
    report=json.loads((root/'latest_run.json').read_text());asof=report['asof']
    if pd.Timestamp(asof)!=TARGET:raise ValueError('The verified session changed; do not silently use another day')
    heads=joblib.load(root/'active_heads.joblib');model_sha=hashlib.sha256((root/'active_heads.joblib').read_bytes()).hexdigest()
    source_manifest_sha=hashlib.sha256((root/'remote_manifest.json').read_bytes()).hexdigest()
    replay=report['exact_prediction_replay']['champion']
    def replay_input(receipt):
        folder=root/'prediction_replay'/receipt['input_parquet_sha256'];raw=(folder/'inputs.parquet').read_bytes()
        if hashlib.sha256(raw).hexdigest()!=receipt['input_parquet_sha256'] or receipt['model_sha256']!=model_sha:raise ValueError('Replay source differs')
        saved=json.loads((folder/('champion_'+receipt['receipt_sha256']+'.json')).read_text())
        d=pd.read_parquet(io.BytesIO(raw));values=probabilities(heads,d)
        if d['isin'].tolist()!=saved['isins']:raise ValueError('Replay identities differ')
        for c in ['p6','p12','joint12','loss12']:
            if not np.allclose(values[c],saved['probabilities'][c],atol=1e-12,rtol=0):raise ValueError('Replay probabilities differ')
            d[c]=values[c]
        return d
    collapsed=replay_input(replay)
    initial_gate=paired_registration_coverage_gate(root,len(collapsed))
    save(out/'latest_collector_diagnostic.json',{'source_asof':asof,'rows':len(collapsed),'coverage_gate':initial_gate,
        'priority_selection':report.get('priority_research'),'not_a_whole_market_no_opportunity_signal':True})
    broad=[]
    for p in (root/'priority_research').glob('*.json'):
        obj=json.loads(p.read_text());rec=obj.get('prediction_replay')
        if not rec or not obj.get('records') or obj.get('source_asof','')>=asof:continue
        q=root/'prediction_replay'/rec.get('input_parquet_sha256','')/'inputs.parquet'
        if not q.exists():continue
        import pyarrow.parquet as pq
        n=pq.ParquetFile(q).metadata.num_rows
        if n>=initial_gate['minimum_rows']:broad.append((obj['source_asof'],n,p.name,obj))
    previous=max(broad,key=lambda x:(x[0],x[1],x[2])) if broad else None
    if previous:
        old=replay_input(previous[3]['prediction_replay'])
        for r in previous[3]['records']:
            q=old[old['isin'].eq(r['isin'])]
            if len(q)!=1:raise ValueError('Previous snapshot identity missing')
            for c in ['p6','p12','joint12','loss12']:
                if not np.isclose(q.iloc[0][c],r[c],atol=1e-12,rtol=0):raise ValueError('Previous selected probability differs')
        save(out/'last_broad_priority_result.json',{**previous[3],'verified_market_rows':previous[1],
            'replayed_during_this_review':True,'fresh_for_latest_session':False})
    universe=pd.read_parquet(root/'universe.parquet');sym='nse_current_symbol' if 'nse_current_symbol' in universe else 'symbol'
    wanted=set(zip(universe['isin'],universe[sym]))
    probes=[]
    previous_probe=out/'provider_probe_receipts.json'
    if previous_probe.exists():
        probes=json.loads(previous_probe.read_text())
        for p in probes:
            raw=src/('probe_'+p['symbol']+'.raw')
            if p.get('sha256') and (not raw.exists() or hashlib.sha256(raw.read_bytes()).hexdigest()!=p['sha256']):raise ValueError('Captured provider probe differs')
    for symbol in ([] if probes else ['RELIANCE','SBIN']):
        item=universe[universe[sym].eq(symbol)]
        if len(item)!=1:continue
        p1=int((TARGET-pd.DateOffset(years=4)).tz_localize('Asia/Kolkata').timestamp())
        p2=int((TARGET+pd.Timedelta(days=1)).tz_localize('Asia/Kolkata').timestamp())
        url='https://query1.finance.yahoo.com/v8/finance/chart/'+quote(symbol+'.NS',safe='')+f'?interval=1d&period1={p1}&period2={p2}&events=div%2Csplits'
        status={'symbol':symbol,'url':url,'requested_at_utc':datetime.now(timezone.utc).isoformat()}
        try:
            response=requests.get(url,headers={'User-Agent':'Mozilla/5.0'},timeout=(10,30));raw=response.content
            (src/('probe_'+symbol+'.raw')).write_bytes(raw)
            status.update(http_status=response.status_code,retry_after=response.headers.get('Retry-After'),sha256=hashlib.sha256(raw).hexdigest())
            if response.status_code==200:
                bars,meta=parse_chart(raw,symbol,str(item.iloc[0]['isin']),asof,{'first_retrieved_utc':datetime.now(timezone.utc).isoformat()})
                status.update(latest_bar=meta['latest_bar'],fresh=meta['latest_bar']==asof)
            else:status['fresh']=False
        except Exception as exc:status.update(fresh=False,error_type=type(exc).__name__)
        probes.append(status)
        if status.get('http_status')==429:break
        time.sleep(1)
    save(out/'provider_probe_receipts.json',probes)
    # One correctly scoped official action-catalogue request, no retries after
    # access denial or rate limiting. Unknown catalogue coverage cannot mean 0.
    action_url='https://www.nseindia.com/api/corporates-corporateActions?'+urlencode({'index':'equities','from_date':'10-11-2026','to_date':'11-11-2026'})
    action_proof={'url':action_url,'requested_at_utc':datetime.now(timezone.utc).isoformat()};actions=None;action_symbols=set();calendar_ok=False
    try:
        captured=out/'action_catalogue_receipt.json'
        if captured.exists() and (src/'primary_action_catalogue.raw').exists():
            action_proof=json.loads(captured.read_text());raw=(src/'primary_action_catalogue.raw').read_bytes()
            if action_proof['url']!=action_url or hashlib.sha256(raw).hexdigest()!=action_proof['sha256']:raise ValueError('Captured action catalogue differs')
            if action_proof.get('http_status')!=200:raise ValueError('Captured action response is not successful')
            action_proof.update(reused_same_source_capture=True,reverified_at_utc=datetime.now(timezone.utc).isoformat())
            body=json.loads(raw)
        else:
            response=requests.get(action_url,headers={'User-Agent':'Mozilla/5.0','Accept':'application/json','Referer':'https://www.nseindia.com/companies-listing/corporate-filings-actions'},timeout=(10,35))
            raw=response.content;(src/'primary_action_catalogue.raw').write_bytes(raw)
            action_proof.update(http_status=response.status_code,retry_after=response.headers.get('Retry-After'),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
            response.raise_for_status();body=response.json()
        actions=body if isinstance(body,list) else body.get('data')
        if not isinstance(actions,list) or not actions:raise ValueError('No complete nonempty primary action catalogue')
        if isinstance(body,dict) and any(int(body[k])>len(actions) for k in ['total','totalRecords'] if isinstance(body.get(k),(int,float))):raise ValueError('Truncated action catalogue')
        for row in actions:
            symbol=str(row.get('symbol','')).strip();day=pd.to_datetime(row.get('exDate') or row.get('ex_date'),dayfirst=True,errors='coerce')
            if not symbol or pd.isna(day) or not ANCHOR<day<=TARGET:raise ValueError('Action response outside requested interval')
            action_symbols.add(symbol)
        action_proof.update(validated=True,rows=len(actions),affected_symbols=sorted(action_symbols),scope='MAIN_BOARD_EQUITIES_ONLY')
        calendar_url='https://nsearchives.nseindia.com/content/circulars/CMTR71775.pdf'
        calendar_file=src/'calendar_2026.pdf';old_calendar=out/'calendar_receipt.json'
        if calendar_file.exists() and old_calendar.exists():
            calendar_raw=calendar_file.read_bytes();prior_calendar=json.loads(old_calendar.read_text())
            if prior_calendar['url']!=calendar_url or hashlib.sha256(calendar_raw).hexdigest()!=prior_calendar['sha256']:raise ValueError('Captured calendar differs')
        else:
            r=requests.get(calendar_url,headers={'User-Agent':'Mozilla/5.0'},timeout=(10,30));r.raise_for_status()
            calendar_raw=r.content;calendar_file.write_bytes(calendar_raw)
        import pdfplumber
        with pdfplumber.open(io.BytesIO(calendar_raw)) as pdf:text=' '.join(p.extract_text() or '' for p in pdf.pages)
        calendar_ok=bool(re.search(r'November\s+10,?\s+2026\s+Tuesday\s+Diwali',text,re.I))
        if not calendar_ok:raise ValueError('Intervening holiday not verified')
        save(out/'calendar_receipt.json',{'url':calendar_url,'sha256':hashlib.sha256(calendar_raw).hexdigest(),
            '10_Nov_2026_is_trading_holiday':True,'9_to_11_Nov_is_one_next_session_interval':True})
    except Exception as exc:action_proof.update(validated=False,error_type=type(exc).__name__)
    save(out/'action_catalogue_receipt.json',action_proof)
    receipt={'started_at_utc':started,'source_asof':asof,'source_manifest_sha256':source_manifest_sha,'model_sha256':model_sha,
        'workflow_run':os.environ.get('GITHUB_RUN_ID'),'collector_market_rows':len(collapsed),'collector_coverage_gate':initial_gate,
        'provider_probes':probes,'primary_action_catalogue':action_proof,
        'last_broad_source_asof':previous[0] if previous else None,'last_broad_market_rows':previous[1] if previous else None,
        'archive_writes':False,'production_changed':False,'forecasts_rewritten':False,'new_formal_forecast_registered':False,
        'trial_weights_adopted_in_production':False,'latest_model_status':'BLOCKED_CURRENT_SOURCE_COVERAGE',
        'new_primary_rows':0,'recovered_market_rows':0}
    if action_proof.get('validated') and calendar_ok:
        for day in [ANCHOR,TARGET]:collect(str(day.date()),src/'official')
        anchor=pd.read_parquet(src/'official'/(str(ANCHOR.date())+'_official_market.parquet'))
        current=pd.read_parquet(src/'official'/(asof+'_official_market.parquet'))
        a={r['isin']:r for r in anchor.to_dict('records') if r['source_status']=='VALID_POSITIVE_VOLUME_SESSION'}
        c={r['isin']:r for r in current.to_dict('records') if r['source_status']=='VALID_POSITIVE_VOLUME_SESSION'}
        histories,identity_audit=canonical_histories(root/'history/securities',wanted)
        view=out/'research_state';view.mkdir();(view/'history/securities').mkdir(parents=True);(view/'evidence').mkdir()
        for p in root.iterdir():
            if p.name not in {'history','official','evidence'}:(view/p.name).symlink_to(p.resolve(),target_is_directory=p.is_dir())
        (view/'official').symlink_to((src/'official').resolve(),target_is_directory=True)
        reasons=Counter();lineage=[]
        for (isin,symbol),(path,bars) in histories.items():
            bars=bars.copy();bars['date']=pd.to_datetime(bars.date);bars=bars[bars.date.le(TARGET)]
            if bars.empty:reasons['EMPTY_HISTORY']+=1;continue
            if bars.date.max()==TARGET:
                target=view/'history/securities'/(isin+'.parquet');bars.to_parquet(target,index=False,compression='zstd')
                reasons['NATIVE_CURRENT_HISTORY']+=1;continue
            if isin not in a or isin not in c:reasons['MISSING_PRIMARY_IDENTITY']+=1;continue
            try:
                new,proof=extend(bars,a[isin],c[isin],action_symbols,True,True)
                new.to_parquet(view/'history/securities'/(isin+'.parquet'),index=False,compression='zstd')
                proof['original_history_sha256']=hashlib.sha256(path.read_bytes()).hexdigest();lineage.append(proof)
                reasons['PRIMARY_NO_ACTION_EXTENSION']+=1
            except ValueError as exc:reasons[str(exc)]+=1
        save(out/'history_extension_lineage.json',lineage)
        save(out/'history_extension_coverage.json',{'counts':dict(reasons),'histories_considered':len(histories),
            'current_new_primary_rows':len(lineage),'main_board_action_free_extension_only':True,
            'SME_extended_without_separate_action_feed':False,'current_provider_independence_claimed':False})
        print('Primary extension view',dict(reasons),flush=True)
        frame,_=current_frame(view,asof)
        gate=paired_registration_coverage_gate(root,len(frame))
        receipt.update(new_primary_rows=len(lineage),recovered_market_rows=len(frame),recovered_coverage_gate=gate,
            history_coverage=dict(reasons),source_method='NSE_PRIMARY_CURRENT_BAR_WITH_MATCHED_PRIOR_VENDOR_HISTORY_AND_ACTION_FREE_INTERVAL',
            latest_session_independent_vendor_verification=False)
        if gate['passed'] and not frame.empty:
            import v13_financial_evidence as financial
            import v13_statement_eps as statement
            import v13_catalyst_execution as catalyst
            from v13_dated_sector import attach
            wide=pd.read_parquet(root/'financial_wide.parquet')
            frame=financial.enrich_evidence(frame,wide,view,asof);frame=attach(frame,view)
            frame=statement.enrich(frame,view);frame=catalyst.join(frame,view);frame=financial.apply_scenarios(frame,view)
            for k,v in probabilities(heads,frame).items():frame[k]=v
            frame.to_parquet(out/'latest_model_inputs_and_predictions.parquet',index=False,compression='zstd')
            old_threshold=available.RSI_MIN;results={}
            try:
                for threshold in [70,50]:
                    available.RSI_MIN=threshold;d,checks,audit=available.screen(frame,wide)
                    baseline=rank(d,d.available_research_eligible);trial=rank_candidate(d,d.available_research_eligible)
                    pd.testing.assert_frame_equal(baseline[['isin','p6','p12','joint12','loss12']],trial[['isin','p6','p12','joint12','loss12']])
                    cols=['isin','symbol','series','source_close_INR','rsi14','p6','p12','joint12','loss12',
                        'available_financial_tier','available_research_group']
                    chosen=baseline[baseline.priority_rank.notna()].sort_values('priority_rank')
                    experimental=trial[trial.candidate_rank.notna()].sort_values('candidate_rank')
                    obj={'source_asof':asof,'RSI_strictly_greater_than':threshold,'market_rows':len(frame),'eligible':int(d.available_research_eligible.sum()),
                        'selected':len(chosen),'baseline_weights':WEIGHTS,'trial_weights':TRIAL_WEIGHTS,
                        'records':json.loads(chosen[cols+['priority_rank','priority_score']].to_json(orient='records',double_precision=15)),
                        'trial_records':json.loads(experimental[cols+['candidate_rank','candidate_score','candidate_price_discipline']].to_json(orient='records',double_precision=15)),
                        'source_method':receipt['source_method'],'latest_session_independent_vendor_verification':False,
                        'research_only':True,'formal_registration_unchanged':True,'probabilities_accuracy_validated':False,
                        'financial_tiers':audit['tiers']}
                    save(out/f'latest_RSI{threshold}.json',obj);results[str(threshold)]={k:v for k,v in obj.items() if k not in ['records','trial_records']}
            finally:available.RSI_MIN=old_threshold
            receipt.update(latest_model_status='COMPLETED_PRIMARY_DATA_RESEARCH_RUN',lists=results,
                financial_complete_rows=int(frame.financial_complete.sum()),financial_quality_pass_rows=int(frame.financial_quality_pass.sum()))
    if protected(root)!=before:raise ValueError('Frozen files changed')
    receipt.update(completed_at_utc=datetime.now(timezone.utc),protected_files_count=len(before),protected_files_unchanged=True)
    save(out/'run_receipt.json',receipt)
    print(json.dumps(plain(receipt),indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);a=p.parse_args();run(a.root,a.out)
