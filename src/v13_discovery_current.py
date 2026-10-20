"""Two-stage watchlist, dated classifications and unchanged frozen probabilities."""
import argparse,hashlib,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np,pandas as pd
from v13_sector_quality import annotate,dated_members
from v13_historical_discovery import signals
from v13_price_policy import valid_price,POLICY as PRICE_POLICY

def classify_current(frame,classification_dir,prior_members,now):
    root=Path(classification_dir);receipts={r['name']:r for r in json.loads((root/'receipt.json').read_text()) if 'error' not in r}
    for name in ['amfi.xlsx','bank.csv','insurance.csv']:
        receipt=receipts[name];local_name=receipt.get('normalized_name',name)
        expected=receipt.get('normalized_sha256',receipt['sha256'])
        if hashlib.sha256((root/local_name).read_bytes()).hexdigest()!=expected:raise ValueError('Classification original or normalized snapshot changed')
    d=frame.copy();d['original_information_cutoff_utc']=d.information_cutoff_utc;d['information_cutoff_utc']=now
    m=prior_members.copy();m['size_category']=None;detailed=[]
    for filename,industry in [('bank.csv','Banks'),('insurance.csv','Insurance')]:
        x=pd.read_csv(root/filename);x.columns=x.columns.str.strip()
        for r in x.to_dict('records'):detailed.append(dict(isin=r['ISIN Code'],symbol=r['Symbol'],industry=industry,size_category=None,available_at_utc=receipts[filename]['first_observed_utc'],source_sha256=receipts[filename]['sha256']))
    m=pd.concat([m,pd.DataFrame(detailed)],ignore_index=True);d=dated_members(d,m)
    if receipts['amfi.xlsx'].get('normalized_name'):
        amfi=pd.read_csv(root/receipts['amfi.xlsx']['normalized_name'])
    else:
        amfi=pd.read_excel(root/'amfi.xlsx',header=1);amfi.columns=amfi.columns.str.strip()
        category=next(c for c in amfi if c.startswith('Categorization'))
        amfi=amfi.rename(columns={'ISIN':'isin','NSE Symbol':'symbol',category:'size_category'})
    for c in ['isin','symbol','size_category']:amfi[c]=amfi[c].astype(str).str.strip()
    if amfi.duplicated(['isin','symbol']).any():raise ValueError('Ambiguous AMFI identity')
    lookup=amfi.set_index(['isin','symbol'])
    if pd.Timestamp(receipts['amfi.xlsx']['first_observed_utc'])>pd.Timestamp(now):raise ValueError('Future size snapshot')
    for i,r in d.iterrows():
        key=(r['isin'],r.symbol)
        if key in lookup.index:d.loc[i,'size_category']=lookup.loc[key,'size_category']
    d['size_source_sha256']=receipts['amfi.xlsx']['sha256'];d['size_first_observed_utc']=receipts['amfi.xlsx']['first_observed_utc']
    return annotate(d)
def watchlists(frame,documents,now):
    d=frame.copy();cut=pd.Timestamp(now);rows=[]
    for r in documents:
        published=pd.to_datetime(r.get('available_at_utc'),utc=True,errors='coerce');retrieved=pd.to_datetime(r.get('first_retrieved_utc'),utc=True,errors='coerce')
        if pd.isna(published) or pd.isna(retrieved) or max(published,retrieved)>cut or published<cut-pd.Timedelta(days=90):continue
        if r.get('issuer_identity_read') is not True or not r.get('source_sha256'):continue
        rows.append(dict(isin=r['isin'],source_sha256=r['source_sha256'],**signals(r.get('headline',''),r.get('text',''))))
    counts=pd.DataFrame(rows).drop_duplicates(['isin','source_sha256']).groupby('isin').sum(numeric_only=True) if rows else pd.DataFrame()
    for signal in ['order','capacity','commissioning_language','approval_language','adverse_project']:d['current_'+signal+'_documents']=d['isin'].map(counts[signal]) if signal in counts else np.nan
    positive=d[['current_order_documents','current_capacity_documents','current_commissioning_language_documents','current_approval_language_documents']].gt(0).any(axis=1)
    no_bad=~d.sector_status.eq('KNOWN_FINANCIAL_FAILURE')
    no_bad &= ~(d.sector_sector_kind.eq('UNKNOWN')&d.get('research_financial_tier',pd.Series('',index=d.index)).eq('OBSERVED_FINANCIAL_FAILURE'))
    for col in ['owner_PAT_yoy','owner_PAT_margin']:
        values=pd.to_numeric(d.get(col,pd.Series(np.nan,index=d.index)),errors='coerce')
        no_bad &= ~(values.lt(0) if col.endswith('yoy') else values.le(0))
    no_bad &= ~d.get('current_owner_profit_status',pd.Series('',index=d.index)).eq('UNKNOWN_OWNER_PLUS_MINORITY_DOES_NOT_RECONCILE_TOTAL_PAT')
    market=valid_price(d.close)&d.feature_eligible.fillna(False).astype(bool)
    market &= d.p6.between(0,1)&d.p12.between(0,1)&d.loss12.between(0,.95)&d.p6.le(d.p12+1e-12)&d.joint12.between(0,1)&d.joint12.le(d.p12+1e-12)&(d.joint12+d.loss12).le(1+1e-12)
    size=d.size_status.eq('SMALL_MID_VERIFIED')
    d['early_watch_eligible']=market&size&no_bad&positive&~d.current_adverse_project_documents.gt(0)
    d['rsi70_research_eligible']=market&size&no_bad&d.rsi14.gt(70)&d.distance_ma50.gt(0)&d.log_ret63.le(np.log(2.5))&d.sector_sector_kind.ne('UNKNOWN')
    for prefix in ['early_watch','rsi70_research']:
        valid=d[prefix+'_eligible']
        score=(.70*d.p6.where(valid).rank(pct=True)+.25*d.redevelopment_price_discipline+.05*d.research_within_200dma_band).where(valid)
        d[prefix+'_score']=score;pool=d[valid].sort_values([prefix+'_score','isin'],ascending=[False,True]).head(20)
        d[prefix+'_rank']=d['isin'].map({v:i+1 for i,v in enumerate(pool['isin'])})
    d['price_policy']=PRICE_POLICY
    d['review_asof_utc']=now;d['probabilities_status']='UNCHANGED_FROZEN_ESTIMATES_FROM_ORIGINAL_INPUT_NOT_RECALIBRATED'
    return d

def run(input_path,classification_dir,evidence,output):
    e=Path(evidence);out=Path(output);out.mkdir(parents=True,exist_ok=True);f=pd.read_parquet(input_path)
    s=json.loads((e/'evidence/valuation_snapshot.json').read_text());m=pd.DataFrame(json.loads((e/'evidence/valuation_research_hurdles.json').read_text()))[['isin','symbol','industry']]
    m['available_at_utc']=s['membership_first_observed_utc'];m['source_sha256']=s['membership_sha256'];now=datetime.now(timezone.utc).isoformat()
    d=watchlists(classify_current(f,classification_dir,m,now),json.loads((e/'catalyst_evidence/document_evidence.json').read_text()),now)
    d.to_parquet(out/'current_two_stage.parquet',index=False)
    cols=['symbol','isin','close','p6','p12','loss12','rsi14','size_category','industry','sector_status','dd_breakout_state','early_watch_rank','rsi70_research_rank']
    summary={'review_asof_utc':now,'price_source_date':str(pd.Timestamp(d.date.max()).date()),'ranking_weights_are_preferences_not_probabilities':{'p6_percentile':.70,'price_discipline':.25,'optional_DMA_band':.05},
      'size_counts':d.size_status.value_counts().to_dict(),'sector_counts':d.sector_sector_kind.value_counts().to_dict(),
      'frozen_probabilities_exactly_preserved':bool(d[['p6','p12','joint12','loss12']].equals(f[['p6','p12','joint12','loss12']]))}
    for prefix in ['early_watch','rsi70_research']:summary[prefix]=json.loads(d[d[prefix+'_rank'].notna()].sort_values(prefix+'_rank')[cols].to_json(orient='records'))
    (out/'current_summary.json').write_text(json.dumps(summary,indent=2));return summary
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ['input','classification-dir','evidence','output']:p.add_argument('--'+k,required=True)
    a=p.parse_args();run(a.input,a.classification_dir,a.evidence,a.output)
