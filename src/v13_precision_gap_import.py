"""Import precisely named source observations, preserving original master values.

Reported bank interest/income, fixed assets and EPS are not substituted for
industrial metrics. The source equation, currency, context and actual period
are checked before a single source can fill an unresolved cell.
"""
import base64,gzip,hashlib,json,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from xml.etree import ElementTree as ET
import pandas as pd
import requests
from v12_dated_training_inputs import exact_facts
from v11_4_longterm_fundamentals import local,context_map
from v11_4_strict_annual_numeric_features import dimensional_contexts
from v11_4_verified_current_financials import numeric
from v12_5_expanded_facts import unit_types
from v13_master_data import KEYS


def load_packet(path):
    return json.loads(gzip.decompress(base64.b64decode(Path(path).read_text())))


def validate_xml_observation(raw,document,observation,master_isin):
    r=document['record'];o=observation
    _,identity=exact_facts(raw,r,'2026-10-21',allow_empty=True)
    if identity['security_isin']!=master_isin or o['isin']!=master_isin:raise ValueError('Equity identity differs')
    if o['reporting_mode']!=r['reporting_mode']:raise ValueError('Observation reporting basis differs')
    root=ET.fromstring(raw);contexts=context_map(root);dims=dimensional_contexts(root);units=unit_types(root)
    ref=o['source_locator']['context'];context=contexts.get(ref,{})
    if ref in identity['contexts_with_conflicting_economic_dates_rejected'] or dims.get(ref,0):raise ValueError('Ambiguous or dimensional economic context')
    if str(context.get('end'))!=o['period_end'] or str(context.get('start') or context.get('end'))!=o['source_locator']['period_start']:raise ValueError('Economic period differs')
    days=context.get('duration');kind='instant' if days is None else 'annual' if 330<=days<=400 else 'quarter' if 60<=days<=120 else 'ytd'
    if kind!=o['period_kind'] or kind=='ytd':raise ValueError('Economic duration differs')
    tags=o['source_locator']['tags'];values=[]
    for tag in tags:
        unit='INR/SHARES' if 'EarningsLossPerShare' in tag else 'INR'
        matches={numeric(e.text) for e in root if local(e.tag)==tag and e.get('contextRef')==ref and units.get(e.get('unitRef'))==unit and numeric(e.text) is not None}
        if len(matches)!=1:raise ValueError('Missing, conflicting or wrong-unit original tag')
        values.append(matches.pop())
    if len(tags)==1:
        allowed={'ProfitLossForThePeriod','ProfitLossOfMinorityInterest','ProfitLossAfterTaxesMinorityInterestAndShareOfProfitLossOfAssociates'}
        if tags[0] not in allowed:raise ValueError('Unreviewed direct taxonomy mapping')
        value=values[0]
    elif tags==['ProfitLossForPeriodBeforeMinorityInterest','ShareOfProfitLossOfAssociates'] or tags==['ProfitLossForThePeriod','ShareOfProfitLossOfAssociates']:
        if o['metric']!='pat':raise ValueError('PAT equation used for another metric')
        value=sum(values)
    elif tags==['DilutedEarningsLossPerShareFromContinuingAndDiscontinuedOperations','ProfitLossFromDiscontinuingOperationAfterTax']:
        if o['metric']!='eps_diluted_continuing' or values[1]!=0:raise ValueError('Discontinued profit is not explicitly zero')
        value=values[0]
    elif tags==['Assets','Deposits','Borrowings','OtherLiabilitiesAndProvisions','Capital','ReservesAndSurplus']:
        if o['metric']!='equity':raise ValueError('Equity equation used for another metric')
        value=values[0]-sum(values[1:4])
        if abs(value-sum(values[4:]))>1:raise ValueError('Bank balance sheet does not reconcile')
    elif tags==['CashAndBalancesWithReserveBankOfIndia','BalancesWithBanksAndMoneyAtCallAndShortNotice','CashAndCashEquivalentsCashFlowStatement']:
        if o['metric']!='cash' or abs(sum(values[:2])-values[2])>1:raise ValueError('Reported closing cash does not reconcile')
        value=sum(values[:2])
    else:raise ValueError('Unreviewed source equation')
    if abs(value-o['value_INR'])>1e-6:raise ValueError('Transcribed value differs from original source equation')
    return True


def apply(root,packet_path):
    root=Path(root);packet=load_packet(packet_path)
    embedded_parts=sorted(Path(packet_path).parent.glob('precision_gap_original_bytes_20261021.part*.b64'))
    embedded=json.loads(gzip.decompress(base64.b64decode(''.join(p.read_text() for p in embedded_parts)))) if embedded_parts else {}
    universe=pd.read_parquet(root/'universe.parquet');master=dict(zip(universe.nse_current_symbol,universe['isin']))
    by_hash={d['receipt']['sha256']:d for d in packet['documents']};valid={};failures=[]
    raw_folder=root/'master_data/raw';raw_folder.mkdir(parents=True,exist_ok=True)
    host_access_stopped=Event()
    def fetch(document):
        proof=document['receipt'];digest=proof['sha256'];r=document['record'];suffix='.pdf' if document['source_type']=='PDF_MANUALLY_REVIEWED' else '.xml';path=raw_folder/digest/('original'+suffix)
        try:
            if proof.get('status')!=200 or proof['requested_url']!=r['source_url']:raise ValueError('Original receipt mismatch')
            if digest in embedded:raw=gzip.decompress(base64.b64decode(embedded[digest]))
            elif path.exists():raw=path.read_bytes()
            else:
                # Local source publication caches use URL and availability hashes.
                cache=root/'filing_raw'/hashlib.sha256(r['available_at_utc'].encode()).hexdigest()/(hashlib.sha256(r['source_url'].encode()).hexdigest()+'.bin')
                if cache.exists():raw=cache.read_bytes()
                else:
                    if host_access_stopped.is_set():raise ValueError('Source access stopped after access denial or rate limit')
                    time.sleep(.08)
                    response=requests.get(r['source_url'],timeout=(10,40))
                    if response.status_code in {401,403,429}:host_access_stopped.set()
                    response.raise_for_status();raw=response.content
            if hashlib.sha256(raw).hexdigest()!=digest:raise ValueError('Original source version changed')
            path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw);path.with_suffix('.receipt.json').write_text(json.dumps(proof,indent=2))
            return digest,raw,None
        except Exception as exc:return digest,None,str(exc)[:180]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,(digest,raw,error) in enumerate(pool.map(fetch,packet['documents']),1):
            if error:failures.append({'source_sha256':digest,'source_url':by_hash[digest]['record']['source_url'],'error':error})
            else:valid[digest]=raw
            if i%100==0:print('checked original source versions',i,'usable',len(valid),'errors',len(failures),flush=True)
    accepted=[];rejected=[]
    for o in packet['observations']:
        digest=o['source_sha256'];document=by_hash[digest];symbol=o['symbol']
        if digest not in valid or symbol not in master:continue
        try:
            if document['source_type']=='NSE_XBRL':validate_xml_observation(valid[digest],document,o,master[symbol])
            else:
                # Reviewed page 8, exact consolidated June-quarter columns.
                if symbol!='MIDWESTLTD' or master[symbol]!='INE0XAD01024' or digest!='24a82051ea25fc6a44efa03cc05c33e489f8d339e85fd0f7c897613ba94b27d6' or o['source_locator']['physical_pdf_page']!=8:raise ValueError('Unreviewed issuer PDF identity or table')
                if not valid[digest].startswith(b'%PDF'):raise ValueError('Issuer source is not a PDF')
            accepted.append(o)
        except Exception as exc:rejected.append({'symbol':symbol,'metric':o['metric'],'period_end':o['period_end'],'source_sha256':digest,'error':str(exc)[:180]})
    folder=root/'master_data/provider_inputs';folder.mkdir(exist_ok=True)
    by_isin={}
    for o in accepted:by_isin.setdefault(o['isin'],[]).append(o)
    for isin,observations in by_isin.items():
        path=folder/(isin+'.json');old=json.loads(path.read_text()) if path.exists() else []
        unique={}
        for o in old+observations:
            key=tuple(o.get(k) for k in (*KEYS,'provider_family','source_sha256','first_observed_utc','value_INR','rounding_half_width_INR','definition_status'));unique[key]=o
        path.write_text(json.dumps(list(unique.values()),indent=2))
    result={'packet_sha256':hashlib.sha256(Path(packet_path).read_bytes()).hexdigest(),'embedded_source_documents':len(embedded),'source_documents_expected':len(packet['documents']),'source_documents_validated':len(valid),'source_fetch_failures':failures,'accepted_observations':len(accepted),'issuer_count':len(by_isin),'observation_validation_errors':rejected,'single_source_sufficient':True,'existing_original_values_overwritten':False,'frozen_forecasts_modified':False}
    (root/'master_data/precision_source_import.json').write_text(json.dumps(result,indent=2));return result
