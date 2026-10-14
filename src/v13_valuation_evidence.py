"""Whole-market official PE/dated peer snapshots and explicit 2x hurdles.

Snapshot clocks are first observations, not historical publication claims.
Reported PE is not proof of recurring owner EPS or a justified future multiple.
No probability, forecast, selection or trained head is modified here.
"""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,io
import numpy as np
import pandas as pd
from v11_4_direct_exchange_pe_valuation import fetch_original_company_pe,normalize_day_market
from v11_4_live_nse_market_catalyst import exchange_primary_for_day
from v12_dated_source_collection import Store

MEMBERS='https://nsearchives.nseindia.com/content/indices/ind_niftytotalmarket_list.csv'

def peer_hurdles(company,members,min_peers=5):
    needed={'symbol','isin','close','company_pe'}
    if not needed.issubset(company) or not {'symbol','isin','industry'}.issubset(members):raise ValueError('Missing dated company/peer identity')
    c=company.copy();m=members.copy()
    if c.duplicated(['symbol','isin']).any() or m.duplicated(['symbol','isin']).any():raise ValueError('Ambiguous valuation/sector identity')
    c=c.merge(m[['symbol','isin','industry']],on=['symbol','isin'],how='left',validate='1:1')
    c['company_pe']=pd.to_numeric(c.company_pe,errors='coerce')
    c.loc[~c.company_pe.between(.000001,19999.999),'company_pe']=np.nan
    records=[];target=2*1.005/.995
    for z in c.to_dict('records'):
        peers=c[c.industry.eq(z.get('industry'))&c['isin'].ne(z['isin'])&c.company_pe.notna()] if pd.notna(z.get('industry')) and str(z['industry']).strip() else c.iloc[:0]
        row={'symbol':z['symbol'],'isin':z['isin'],'source_close_INR':float(z['close']),
          'reported_company_PE':float(z['company_pe']) if np.isfinite(z['company_pe']) else None,
          'industry':z.get('industry') if pd.notna(z.get('industry')) else None,'other_positive_PE_peers':len(peers),
          'valuation_status':'OBSERVED_OFFICIAL_PE_AND_COVERED_PEER_SAMPLE_NOT_FORWARD_VALUATION',
          'owner_recurring_EPS_verified':False,'2x_scenario_supported':False,'hypothetical_sensitivities':[]}
        if len(peers)>=min_peers:
            row['covered_peer_PE_quantiles']={str(q):float(peers.company_pe.quantile(q)) for q in (.25,.5,.75)}
        if row['reported_company_PE'] is not None:
            assumptions={'no_rerating':row['reported_company_PE'],'20pct_derating':row['reported_company_PE']*.8,'20pct_rerating':row['reported_company_PE']*1.2}
            if len(peers)>=min_peers:assumptions['covered_peer_median']=float(peers.company_pe.median())
            for name,pe in assumptions.items():
                for share in (1.,1.1,1.25):row['hypothetical_sensitivities'].append({'case':name,'assumed_forward_PE':pe,
                  'assumed_forward_share_multiple':share,'required_earnings_growth_for_net_2x':target*share*row['reported_company_PE']/pe-1,
                  'assumptions_not_source_verified':True})
        records.append(row)
    return records

def collect(root,asof):
    root=Path(root);out=root/'evidence';out.mkdir(parents=True,exist_ok=True)
    try:
        market,market_proof=exchange_primary_for_day(pd.Timestamp(asof))
        raw,url,company,attempts=fetch_original_company_pe(asof)
        clock=datetime.now(timezone.utc).isoformat();digest=hashlib.sha256(raw).hexdigest()
        rawdir=root/'valuation'/'raw'/asof;rawdir.mkdir(parents=True,exist_ok=True);(rawdir/(digest+'.bin')).write_bytes(raw)
        m=normalize_day_market(market);joined=m.merge(company,on='symbol',how='left',validate='1:1')
        member_raw,proof=Store(rawdir).get(MEMBERS)
        members=pd.read_csv(io.BytesIO(member_raw),encoding='utf-8-sig');members.columns=members.columns.str.strip()
        fields={'Symbol':'symbol','ISIN Code':'isin','Industry':'industry'}
        if not set(fields).issubset(members) or len(members)<500:raise ValueError('Invalid original sector-membership file')
        members=members[list(fields)].rename(columns=fields)
        for col in ['symbol','isin']:members[col]=members[col].astype(str).str.strip().str.upper()
        rows=peer_hurdles(joined,members)
        snapshot={'source_asof':asof,'first_observed_utc':clock,'company_PE_url':url,'company_PE_sha256':digest,
          'membership_url':MEMBERS,'membership_sha256':proof['sha256'],'membership_first_observed_utc':proof['first_retrieved_utc'],
          'publication_time_verified':False,'current_membership_not_backfilled_into_history':True,'same_day_primary_market_crosscheck':market_proof,
          'rows':len(rows),'reported_positive_company_PE_rows':sum(z['reported_company_PE'] is not None for z in rows),
          'other_peers_at_least_five_rows':sum(z['other_positive_PE_peers']>=5 for z in rows),
          'source_bound_forward_2x_support_rows':0,'status':'DATED_RESEARCH_SNAPSHOT_NOT_FORECAST_INPUT_OR_PROBABILITY',
          'future_retrieval_cannot_be_backdated':True,'source_attempts':attempts}
        folder=root/'valuation_snapshots';folder.mkdir(exist_ok=True)
        key=hashlib.sha256((asof+digest+proof['sha256']).encode()).hexdigest();p=folder/(key+'.json')
        if p.exists():snapshot=json.loads(p.read_text())
        else:p.write_text(json.dumps(snapshot,indent=2,allow_nan=False))
        (out/'valuation_snapshot.json').write_text(json.dumps(snapshot,indent=2,allow_nan=False))
        (out/'valuation_research_hurdles.json').write_text(json.dumps(rows,indent=2,allow_nan=False))
        return snapshot
    except Exception as exc:
        failure={'source_asof':asof,'status':'UNKNOWN_VALUATION_SOURCE_UNAVAILABLE','error':str(exc)[:300],'probabilities_or_heads_changed':False}
        (out/'valuation_snapshot.json').write_text(json.dumps(failure,indent=2));return failure
