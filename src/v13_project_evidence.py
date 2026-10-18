"""Partial original project facts, separate publication/review clocks and scenarios."""
import argparse,concurrent.futures,hashlib,json,re
from datetime import datetime,timezone
from pathlib import Path
from urllib.parse import urlparse
import pandas as pd,requests
from v11_4_primary_document_catalysts import extract_pdf

# Outcome-blind source review cases discovered in the chronological 2023-25
# metadata sample. Administrative false positives are retained as counterexamples.
CASES=[
('RAILTEL','2023-01-03T10:54:22Z','ORDER_AWARDED','186.19 Crores','INR 186.19 crore including GST over five years; not one-year revenue or profit.'),
('BOROLTD','2023-01-02T12:52:50Z','COMMERCIAL_PRODUCTION_STARTED','additional capacity of 42 TPD','Second Jaipur furnace commercial production; utilisation, margins and attributable profit unknown.'),
('SAFARI','2023-01-02T12:50:10Z','COMMERCIAL_PRODUCTION_STARTED','2,25,000 pieces per month','Subsidiary capacity increased from 125000 to 225000 pieces/month; capacity is not sales.'),
('KERNEX','2024-01-01T11:42:20Z','ORDER_AWARDED','109.465 crores','600-day JV order; company work allocation unknown. Do not assign entire JV contract to owner revenue.'),
('UFLEX','2024-01-02T14:02:25Z','COMMERCIAL_PRODUCTION_STARTED','18000 MT per year','Egypt step-down subsidiary PCR plant commissioned; owner profit bridge unknown.'),
('GMDCLTD','2024-01-08T11:05:49Z','OPERATING_APPROVAL_RECEIVED','3 MTPA to 5 MTPA','Environmental clearance for Surkha(N) expansion; not proof of completed production ramp-up.'),
('POWERMECH','2025-01-01T05:44:01Z','ORDER_AWARDED','₹294 Crore','INR 294 crore excluding GST and conditional incentives; schedule depends on notice to proceed.'),
('GOODLUCK','2025-01-01T07:41:13Z','COMMERCIAL_PRODUCTION_STARTED','successfully commenced with effect from 01st January, 2025','Hydraulic tube commercial production started; capacity utilisation and incremental earnings unknown.'),
('BALKRISHNA','2023-04-07T09:22:44Z','NON_PROJECT_ADMINISTRATIVE',None,'Dematerialisation certificate cancellation, not a failed business project.'),
('BALKRISHNA','2024-01-02T10:55:39Z','NON_PROJECT_ADMINISTRATIVE',None,'Dematerialisation certificate cancellation, not a failed business project.'),
('BALKRISHNA','2025-04-08T11:39:53Z','NON_PROJECT_ADMINISTRATIVE',None,'Dematerialisation certificate cancellation, not a failed business project.'),
('EIHOTEL','2023-01-10T08:00:23Z','NON_PROJECT_ADMINISTRATIVE',None,'Promoter reclassification approval, not an operating approval.'),
('DCBBANK','2024-01-02T13:34:39Z','NON_PROJECT_ADMINISTRATIVE',None,'Shareholding permission, not project approval or revenue.'),
('AUBANK','2025-01-04T03:00:14Z','NON_PROJECT_ADMINISTRATIVE',None,'Shareholding permission, not project approval or revenue.')]

def run(events,output):
    out=Path(output);out.mkdir(parents=True,exist_ok=True);e=pd.read_parquet(events)
    def fetch(case):
        symbol,clock,stage,excerpt,note=case
        z=dict(symbol=symbol,published_at_utc=clock,stage=stage,review_note=note,full_profit_chain_verified=False)
        try:
            k=e[e.symbol.eq(symbol)&e.available_at_utc.eq(pd.Timestamp(clock))]
            if len(k)!=1:raise ValueError('Ambiguous original metadata identity')
            r=k.iloc[0];u=urlparse(r.document_url)
            if u.hostname not in {'nsearchives.nseindia.com','archives.nseindia.com'}:raise ValueError('Unexpected original host')
            response=requests.get(r.document_url,timeout=(8,30));response.raise_for_status()
            if urlparse(response.url).hostname!=u.hostname:raise ValueError('Unexpected redirect')
            raw=response.content
            if not raw.startswith(b'%PDF') or len(raw)>15000000:raise ValueError('Not a bounded original PDF')
            text=extract_pdf(raw);digest=hashlib.sha256(raw).hexdigest()
            if symbol not in text and r['isin'] not in text:raise ValueError('Printed identity absent')
            if excerpt and re.sub(r'\s+',' ',excerpt) not in re.sub(r'\s+',' ',text):raise ValueError('Reviewed excerpt absent')
            (out/(digest+'.pdf')).write_bytes(raw);(out/(digest+'.txt')).write_text(text)
            z.update(isin=r['isin'],event_id=r.event_id,document_url=r.document_url,metadata_sha256=r.metadata_sha256,
              source_sha256=digest,text_sha256=hashlib.sha256(text.encode()).hexdigest(),excerpt=excerpt,
              reviewed_at_utc=datetime.now(timezone.utc).isoformat(),status='PARTIAL_ORIGINAL_FACT_REVIEWED' if stage!='NON_PROJECT_ADMINISTRATIVE' else 'ADMINISTRATIVE_COUNTEREXAMPLE_REVIEWED',
              evidence_scope='RETROSPECTIVE_REVIEW_NOT_AS_OPERATED_HISTORICAL_FEATURE',
              incremental_owner_profit=None,forward_diluted_EPS=None)
        except Exception as ex:z.update(status='REVIEW_BLOCKED',error=str(ex)[:250])
        return z
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:rows=list(ex.map(fetch,CASES))
    summary={'selection':'source-review cases selected without market outcomes','records':rows,
      'partial_project_facts':sum(r['status']=='PARTIAL_ORIGINAL_FACT_REVIEWED' for r in rows),
      'administrative_counterexamples':sum(r['status']=='ADMINISTRATIVE_COUNTEREXAMPLE_REVIEWED' for r in rows),
      'full_profit_chains':0,'historical_evaluation_enriched_with_reviewed_facts':False}
    (out/'summary.json').write_text(json.dumps(summary,indent=2));return summary

def scenario_bridge(base_owner_pat,incremental_revenue,owner_net_margin,forward_diluted_shares,price,forward_pe,entry_cost=.005,exit_cost=.005):
    """Assumed attributable revenue/margin, not order value or EBITDA margin."""
    import math
    vals=[base_owner_pat,incremental_revenue,owner_net_margin,forward_diluted_shares,price,forward_pe,entry_cost,exit_cost]
    if any(v is None or isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in vals):return dict(status='UNKNOWN_INPUTS',net_multiple=None)
    if min(base_owner_pat,incremental_revenue)<0 or not 0<=owner_net_margin<=1 or min(forward_diluted_shares,price,forward_pe)<=0 or not 0<=entry_cost<1 or not 0<=exit_cost<1:raise ValueError('Invalid scenario basis')
    pat=base_owner_pat+incremental_revenue*owner_net_margin;eps=pat/forward_diluted_shares
    return dict(status='ASSUMPTION_SCENARIO_NOT_FORECAST',forward_owner_PAT_INR=pat,forward_diluted_EPS=eps,net_multiple=eps*forward_pe*(1-exit_cost)/(price*(1+entry_cost)),source_supported=False,probability=None)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--events',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(a.events,a.output)
