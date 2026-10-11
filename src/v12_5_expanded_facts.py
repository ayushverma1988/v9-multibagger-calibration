"""Additional original-XBRL facts with explicit currency/per-share units.

Reuses immutable dated public filings; never interprets reported diluted EPS
as recurring EPS or cash-flow profitability as verified future earnings.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
from xml.etree import ElementTree as ET
import pandas as pd
from v12_dated_training_inputs import exact_facts,FACT_TAGS
from v11_4_verified_current_financials import INSTANT
from v11_4_verified_current_financials import numeric
from v11_4_longterm_fundamentals import context_map,local
from v11_4_strict_annual_numeric_features import dimensional_contexts
from v12_dated_source_collection import Store,sha

TAGS={
 'assets':(['Assets','TotalAssets'],'INR','instant'),
 'current_liabilities':(['CurrentLiabilities'],'INR','instant'),
 'eps_diluted_continuing':(['DilutedEarningsLossPerShareFromContinuingOperations'],'INR/SHARES','duration'),
 'paidup_equity_capital':(['PaidUpValueOfEquityShareCapital'],'INR','either'),
 'face_value':(['FaceValueOfEquityShareCapital'],'INR/SHARES','either'),
 'weighted_diluted_shares':(['WeightedAverageNumberOfDilutedSharesOutstanding','WeightedAverageNumberOfEquitySharesOutstandingDiluted'],'SHARES','duration')}
TAGS={**{k:(v,'INR','instant' if k in INSTANT else 'duration') for k,v in FACT_TAGS.items()},**TAGS}

def unit_types(root):
    result={}
    for e in root.iter():
        if local(e.tag)!='unit':continue
        measures=[(x.text or '').strip().split(':')[-1].upper() for x in e.iter() if local(x.tag)=='measure']
        div=[x for x in e if local(x.tag)=='divide']
        if not div and len(measures)==1:result[e.get('id')]=measures[0]
        elif len(div)==1:
            parts={local(x.tag):[(y.text or '').strip().split(':')[-1].upper() for y in x.iter() if local(y.tag)=='measure'] for x in div[0]}
            if parts=={'unitNumerator':['INR'],'unitDenominator':['SHARES']}:result[e.get('id')]='INR/SHARES'
    return result

def parse_extra(raw,record,asof):
    _,identity=exact_facts(raw,record,asof)
    root=ET.fromstring(raw);ctx=context_map(root);dims=dimensional_contexts(root);units=unit_types(root)
    bad=set(identity['contexts_with_conflicting_economic_dates_rejected'])
    # Original accepted legacy contexts can be reconstructed only from explicit
    # start/end pairs, never guessed from a context name.
    explicit={}
    for e in root.iter():
        if local(e.tag) in {'DateOfStartOfReportingPeriod','DateOfEndOfReportingPeriod'} and e.get('contextRef') and e.text:
            explicit.setdefault(e.get('contextRef'),{})[local(e.tag)]=e.text.strip()
    for ref,v in explicit.items():
        if ref in ctx or ref in bad:continue
        if len(v)==2:
            a=pd.Timestamp(v['DateOfStartOfReportingPeriod']).date();b=pd.Timestamp(v['DateOfEndOfReportingPeriod']).date()
            ctx[ref]={'start':a,'end':b,'duration':(b-a).days}
    end=pd.Timestamp(record['period_end']);low=pd.Timestamp(asof)-pd.DateOffset(years=4)
    choices={}
    for metric,(tags,unit,kind_rule) in TAGS.items():
        for e in root.iter():
            ref=e.get('contextRef');c=ctx.get(ref,{})
            if local(e.tag) not in tags or ref in bad or dims.get(ref,0) or units.get(e.get('unitRef'))!=unit:continue
            v=numeric(e.text)
            if v is None or not c.get('end'):continue
            period=pd.Timestamp(c['end']);start=pd.Timestamp(c.get('start') or c['end']);days=c.get('duration')
            if not low<=start<=period<=end:continue
            kind='instant' if days is None else 'annual' if 330<=days<=400 else 'quarter' if 60<=days<=120 else 'ytd'
            if kind=='ytd' or (kind_rule=='instant' and kind!='instant') or (kind_rule=='duration' and kind=='instant'):continue
            if metric=='cfo' and kind!='annual':continue
            row={'metric':metric,'value':v,'unit':unit,'tag':local(e.tag),'period_start':str(start.date()),'period_end':str(period.date()),'period_kind':kind,'context':ref,**identity}
            choices.setdefault((metric,str(period.date()),kind),[]).append(row)
    rows=[]
    for (metric,_,_),options in choices.items():
        tags=TAGS[metric][0];priority=min(tags.index(x['tag']) for x in options)
        opts=[x for x in options if tags.index(x['tag'])==priority]
        if len({x['value'] for x in opts})==1:rows.append(opts[0])
    return rows

def collect(records_path,raw_dir,output,asof='2026-10-09'):
    out=Path(output);out.mkdir(parents=True,exist_ok=True);store=Store(raw_dir)
    records=pd.read_parquet(records_path);rows=[];errors=[]
    for i,r in enumerate(records.to_dict('records'),1):
        try:
            raw,proof=store.get(r['source_url'])
            for f in parse_extra(raw,r,asof):
                rows.append({**r,**f,'source_sha256':proof['sha256'],'first_retrieved_utc':proof['first_retrieved_utc']})
        except Exception as e:errors.append({'source_url':r['source_url'],'symbol':r['symbol'],'error':str(e)[:200]})
        if i%500==0:print('expanded originals',i,'facts',len(rows),'errors',len(errors),flush=True)
    d=pd.DataFrame(rows);d.to_parquet(out/'extra_facts.parquet',index=False)
    (out/'errors.json').write_text(json.dumps(errors,indent=2))
    summary={'documents_attempted':len(records),'extra_facts':len(d),'issuer_count':d.security_isin.nunique(),'metrics':d.metric.value_counts().to_dict(),'errors':len(errors),'recurring_EPS_verified':False}
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--records',required=True);p.add_argument('--raw',required=True);p.add_argument('--output',required=True);a=p.parse_args();collect(a.records,a.raw,a.output)
