"""Whole-market NSE monitoring; separate from the immutable outcome protocol.

Downloads contain no private watchlist. Raw exchange prices are NOT adjusted
returns and cannot replace the registered total-return proxy silently.
"""
from __future__ import annotations
import argparse,hashlib,io,json,zipfile
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from v12_prospective_registry import validate_registry,digest


def official_url(date):
    return 'https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_'+pd.Timestamp(date).strftime('%Y%m%d')+'_F_0000.csv.zip'


def parse_bhavcopy(raw,date,receipt):
    day=pd.Timestamp(date).normalize();seen=pd.Timestamp(receipt['first_retrieved_utc'])
    if seen.tzinfo is None or day.tz_localize('Asia/Kolkata')+pd.Timedelta(hours=15,minutes=30)>seen:
        raise ValueError('Unfinished session or missing source clock')
    if receipt['requested_url']!=official_url(day) or hashlib.sha256(raw).hexdigest()!=receipt['sha256']:
        raise ValueError('Official source URL/hash mismatch')
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        files=[n for n in z.namelist() if n.lower().endswith('.csv')]
        if len(files)!=1:raise ValueError('Ambiguous official archive')
        d=pd.read_csv(io.BytesIO(z.read(files[0])),low_memory=False)
    required=['TradDt','Src','Sgmt','ISIN','TckrSymb','SctySrs','OpnPric','HghPric','LwPric','ClsPric','TtlTradgVol','TtlNbOfTxsExctd','NewBrdLotQty']
    if not set(required).issubset(d):raise ValueError('Official bhavcopy schema missing')
    if not d.Src.eq('NSE').all() or not d.Sgmt.eq('CM').all() or not pd.to_datetime(d.TradDt).eq(day).all():
        raise ValueError('Wrong official venue/segment/date')
    x=d[d.SctySrs.isin(['EQ','BE','BZ','SM','ST']) & d.ISIN.astype(str).str.match(r'^INE[A-Z0-9]{8}[0-9]$')].copy()
    x=x.rename(columns={'TradDt':'date','ISIN':'isin','TckrSymb':'symbol','SctySrs':'series','OpnPric':'open','HghPric':'high','LwPric':'low','ClsPric':'close','TtlTradgVol':'volume','TtlNbOfTxsExctd':'trades','NewBrdLotQty':'lot_size'})
    cols=['open','high','low','close','volume','trades','lot_size']
    for c in cols:x[c]=pd.to_numeric(x[c],errors='coerce')
    finite=np.isfinite(x[cols].to_numpy(float)).all(axis=1)
    good=(finite & x[['open','high','low','close','lot_size']].gt(0).all(axis=1) & x.volume.gt(0) & x.trades.gt(0)
          & x.low.le(x[['open','close']].min(axis=1)) & x.high.ge(x[['open','close']].max(axis=1)))
    x['source_status']=np.where(good,'VALID_POSITIVE_VOLUME_SESSION','UNKNOWN_PRICE_VOLUME')
    x.loc[x.duplicated('isin',keep=False),'source_status']='UNKNOWN_MULTIPLE_SERIES_OR_SESSIONS'
    x['source_sha256']=receipt['sha256'];x['first_retrieved_utc']=receipt['first_retrieved_utc']
    x['venue']='NSE';x['currency']='INR';x['corporate_actions_verified']=False
    return x[['date','isin','symbol','series',*cols,'source_status','source_sha256','first_retrieved_utc','venue','currency','corporate_actions_verified']]


def collect(date,output):
    out=Path(output);rawdir=out/'raw';rawdir.mkdir(parents=True,exist_ok=True)
    day=pd.Timestamp(date).normalize();now=pd.Timestamp.now(tz='UTC')
    if day.tz_localize('Asia/Kolkata')+pd.Timedelta(hours=15,minutes=30)>now:raise ValueError('Do not request future or unfinished session')
    url=official_url(day);key=hashlib.sha256(url.encode()).hexdigest();body=rawdir/(key+'.bin');meta=rawdir/(key+'.json')
    if body.exists() and meta.exists():raw=body.read_bytes();receipt=json.loads(meta.read_text())
    else:
        r=requests.get(url,headers={'User-Agent':'Mozilla/5.0','Accept':'text/csv,application/zip,*/*','Referer':'https://www.nseindia.com/'},timeout=(30,45))
        r.raise_for_status()
        if r.url!=url:raise ValueError('Unexpected official source redirect')
        raw=r.content;receipt={'requested_url':url,'status':r.status_code,'first_retrieved_utc':datetime.now(timezone.utc).isoformat(),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'body_file':body.name}
        body.write_bytes(raw);meta.write_text(json.dumps(receipt,indent=2))
    frame=parse_bhavcopy(raw,day,receipt)
    frame.to_parquet(out/(str(day.date())+'_official_market.parquet'),index=False)
    result={'date':str(day.date()),'official_equity_rows':len(frame),'source_status':frame.source_status.value_counts().to_dict(),'source_sha256':receipt['sha256'],'prices':'UNADJUSTED_PRIMARY_EXCHANGE','unseen_outcomes_adjudicated':0}
    (out/(str(day.date())+'_collection_receipt.json')).write_text(json.dumps(result,indent=2))
    return result


def audit_registration(registry,official_path,output):
    reg,receipt=validate_registry(registry);x=pd.read_parquet(official_path);rows=[]
    for r in reg['records']:
        found=x[x['isin'].eq(r['isin'])];status='UNKNOWN_OFFICIAL_SECURITY_MISSING'
        if len(found)==1:
            q=found.iloc[0];open_clock=pd.Timestamp(q.date).tz_localize('Asia/Kolkata')+pd.Timedelta(hours=9,minutes=15)
            if open_clock<=pd.Timestamp(r['decision_at_utc']):status='PENDING_POST_REGISTRATION_SESSION'
            elif q.symbol!=r['symbol']:status='UNKNOWN_SYMBOL_CHANGED_REQUIRES_IDENTITY_REVIEW'
            elif q.source_status!='VALID_POSITIVE_VOLUME_SESSION':status=str(q.source_status)
            else:status='OBSERVED_TRADE_REQUIRES_FIRST_ENTRY_AND_ACTION_RECONCILIATION'
        rows.append({'isin':r['isin'],'symbol':r['symbol'],'status':status})
    report={'registration_sha256':receipt['registration_sha256'],'official_file_sha256':hashlib.sha256(Path(official_path).read_bytes()).hexdigest(),
            'status_counts':pd.Series([r['status'] for r in rows]).value_counts().to_dict(),'registry_unchanged':True,'matured_unseen_outcomes':0,
            'entry_proof_complete':False,'corporate_actions_reconciled':False,'rows':rows}
    Path(output).write_text(json.dumps(report,indent=2));return {k:v for k,v in report.items() if k!='rows'}


def reconcile_observations(bars,official):
    """Extra preflight; cannot turn the old registry's flags into primary proof.

Raw prices must match the primary exchange. Split-adjusted provider OHLC
cannot pass until action factors are independently reconciled. This function
does not append or alter a registered evaluation or assert a first entry.
"""
    required={'date','isin','open','high','low','close','volume'}
    if not required.issubset(bars) or not required.issubset(official):raise ValueError('Missing reconciliation schema')
    b=bars.copy();o=official.copy();b['date']=pd.to_datetime(b.date);o['date']=pd.to_datetime(o.date)
    if b.duplicated(['date','isin']).any() or o.duplicated(['date','isin']).any():raise ValueError('Ambiguous primary observation identity')
    x=b.merge(o[['date','isin','open','high','low','close','volume','source_status']],on=['date','isin'],how='left',validate='one_to_one',suffixes=('_provider','_official'))
    numeric=x[['open_provider','high_provider','low_provider','close_provider','volume_provider']].to_numpy(float)
    good=np.isfinite(numeric).all(axis=1)&(numeric>0).all(axis=1)
    for c in ['open','high','low','close']:
        good &= (x[c+'_provider']-x[c+'_official']).abs().le(.011).to_numpy()
    good &= x.volume_provider.eq(x.volume_official).to_numpy()
    good &= x.source_status.eq('VALID_POSITIVE_VOLUME_SESSION').to_numpy()
    x['primary_observation_check']=np.where(good,'MATCHED_RAW_PRICE_AND_VOLUME','UNKNOWN_OR_CONFLICT')
    return x


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--date',required=True);p.add_argument('--output',required=True);p.add_argument('--registry');a=p.parse_args()
    print(json.dumps(collect(a.date,a.output),indent=2))
    if a.registry:print(json.dumps(audit_registration(a.registry,Path(a.output)/(a.date+'_official_market.parquet'),Path(a.output)/(a.date+'_registry_monitor.json')),indent=2))
