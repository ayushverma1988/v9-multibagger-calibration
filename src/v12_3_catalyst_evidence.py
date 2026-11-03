"""Expanded original-document candidates with conservative evidence fields.

Disclosed money, dates, capacity and conditions are evidence for review.
They are not labelled verified future profits by an extraction expression.
"""
from __future__ import annotations
import argparse,json,re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd
from v12_dated_source_collection import Store,sha
from v11_4_primary_document_catalysts import extract_pdf,classify_document


def enrich(text,info):
    flat=re.sub(r'\s+',' ',text)
    snippets={}
    patterns={'execution':r'.{0,90}(?:execut(?:ion|ed)|deliver(?:y|ed)|complet(?:ion|ed)|commission(?:ing|ed)).{0,160}',
              'conditions':r'.{0,70}(?:subject to|conditional|cancel(?:lation|led)|termination|approval pending).{0,160}',
              'funding':r'.{0,70}(?:fund(?:ed|ing)|capital expenditure|capex|internal accruals|borrowings).{0,160}',
              'profit':r'.{0,70}(?:profit margin|EBITDA margin|incremental profit|incremental earnings).{0,160}'}
    for k,pattern in patterns.items(): snippets[k]=list(dict.fromkeys(re.findall(pattern,flat,re.I)))[:8]
    quantities=[]
    for e in info['quantified_evidence']:
        q=dict(e)
        if q['category']=='order':
            unit=q['unit_as_disclosed'].lower().rstrip('.')
            factor=1e7 if unit in {'crore','crores','cr'} else 1e5 if unit in {'lakh','lakhs','lac','lacs'} else 1e6 if unit=='million' else 1e9 if unit=='billion' else None
            q['order_value_INR']=q['quantity']*factor if factor else None
            # Turnover, order book and one-off contracts must be distinguished by review.
            q['standalone_new_order_verified']=False
        quantities.append(q)
    amounts=sorted(set(q['order_value_INR'] for q in quantities if q.get('order_value_INR') is not None))
    durations=[]
    for excerpt in snippets['execution']:
        for n,u in re.findall(r'\b(\d+(?:\.\d+)?)\s*(months?|years?)\b',excerpt,re.I):
            durations.append(float(n)*(12 if u.lower().startswith('year') else 1))
        if re.search(r'\bone year\b',excerpt,re.I):durations.append(12.)
    return {**info,'quantified_evidence':quantities,'evidence_snippets':snippets,
            'distinct_order_amounts_INR_for_review':amounts,'amounts_must_not_be_summed':True,
            'execution_month_mentions_for_review':sorted(set(durations)),
            'tax_inclusive_amount_mentioned':bool(re.search(r'includ(?:ing|es|ed)|inclusive',flat,re.I) and re.search(r'\bGST\b',flat)),
            'longer_than_12_month_execution_mentioned':any(d>12 for d in durations),
            'execution_schedule_verified':False,'funding_sufficiency_verified':False,
            'incremental_profit_verified':False,'complete_causal_chain_verified':False}


def collect(events_path,output,per_year=64):
    out=Path(output);out.mkdir(parents=True,exist_ok=True);store=Store(out/'raw')
    x=pd.read_parquet(events_path)
    x=x[(x.order_disclosure|x.capacity_disclosure)&x.attachment_url.str.match(r'https://nsearchives\.nseindia\.com/.*\.pdf$',case=False)].copy()
    x=x.sort_values('available_at_utc').drop_duplicates('attachment_url',keep='last')
    x['year']=x.available_at_utc.str[:4];x['hash']=x.attachment_url.map(lambda s:sha(s.encode()))
    x=x.sort_values('hash').groupby('year').head(per_year)
    x.to_parquet(out/'public_dated_request_universe.parquet',index=False)
    def read(r):
        try:
            raw,p=store.get(r['attachment_url']);text=extract_pdf(raw)
            info=enrich(text,classify_document(text,r['symbol'],r['issuer_name']))
            (out/(p['sha256']+'.txt')).write_text(text)
            return {**{k:r[k] for k in ['symbol','isin','available_at_utc','issuer_name','attachment_url','catalog_sha256','event_id']},**info,
                    'source_sha256':p['sha256'],'first_retrieved_utc':p['first_retrieved_utc']},None
        except Exception as e:return None,{'symbol':r['symbol'],'source_url':r['attachment_url'],'error':str(e)[:250]}
    records=[];errors=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for n,(r,e) in enumerate(pool.map(read,x.to_dict('records')),1):
            if r:records.append(r)
            if e:errors.append(e)
            if n%20==0:print('catalyst documents',n,'read',len(records),'errors',len(errors),flush=True)
    (out/'document_evidence.json').write_text(json.dumps(records,indent=2))
    (out/'errors.json').write_text(json.dumps(errors,indent=2))
    summary={'requested':len(x),'documents_read':len(records),'issuer_matched':sum(r['issuer_identity_read'] for r in records),
             'quantified_candidates':sum(len(r['quantified_evidence']) for r in records),'documents_with_execution_snippets':sum(bool(r['evidence_snippets']['execution']) for r in records),
             'documents_with_profit_snippets':sum(bool(r['evidence_snippets']['profit']) for r in records),'complete_verified_earnings_chains':0,'errors':len(errors),'production_approved':False}
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))


def add_materiality(documents,facts_path,output):
    """Disclosed gross contract / last known annual revenue, never profit."""
    rows=json.loads(Path(documents).read_text());f=pd.read_parquet(facts_path)
    f=f[f.metric.eq('revenue') & f.period_kind.eq('annual')].copy()
    f['clock']=pd.to_datetime(f.available_at_utc,utc=True,format='mixed');f['period']=pd.to_datetime(f.period_end)
    result=[]
    for r in rows:
        rec={k:r[k] for k in ['symbol','isin','available_at_utc','attachment_url','source_sha256']}
        text_path=Path(documents).parent/(r['source_sha256']+'.txt')
        info=enrich(text_path.read_text(),r)
        rec.update({k:info[k] for k in ['distinct_order_amounts_INR_for_review','execution_month_mentions_for_review','tax_inclusive_amount_mentioned','longer_than_12_month_execution_mentioned']})
        rec['gross_order_to_annual_revenue_ratios']=[];rec['materiality_status']='UNKNOWN';rec['earnings_impact_verified']=False
        cut=pd.Timestamp(r['available_at_utc']);day=cut.tz_convert('Asia/Kolkata').tz_localize(None).normalize()
        known=f[f.symbol.eq(r['symbol']) & f.security_isin.eq(r['isin']) & f.clock.le(cut) & (day-f.period).dt.days.between(0,450)]
        if len(known) and info['issuer_identity_read']:
            basis='consolidated' if known.reporting_mode.eq('consolidated').any() else 'standalone'
            known=known[known.reporting_mode.eq(basis)];known=known[known.period.eq(known.period.max())];known=known[known.clock.eq(known.clock.max())]
            if len(known)==1 and known.value_INR.iloc[0]>0:
                q=known.iloc[0];rec['annual_revenue_INR']=float(q.value_INR);rec['financial_source_sha256']=q.source_sha256;rec['financial_available_at_utc']=q.available_at_utc
                rec['gross_order_to_annual_revenue_ratios']=[v/float(q.value_INR) for v in info['distinct_order_amounts_INR_for_review']]
                if rec['gross_order_to_annual_revenue_ratios']:rec['materiality_status']='GROSS_CONTRACT_SCALE_ONLY_NOT_12_MONTH_REVENUE_OR_PROFIT'
        result.append(rec)
    Path(output).write_text(json.dumps(result,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--events',required=True);p.add_argument('--output',required=True);p.add_argument('--per-year',type=int,default=64);a=p.parse_args();collect(a.events,a.output,a.per_year)
