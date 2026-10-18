"""Source-bound retrospective announcement metadata; never execution proof."""
import hashlib,json,re
from urllib.parse import urlparse
import numpy as np,pandas as pd
from v13_source_quality import business_order_status,capacity_disclosure,LEGAL
SIGNALS=('order','capacity','commissioning_language','approval_language','adverse_project','dilution','promoter_purchase_language')
FEATURES=[f'hd_{s}_90d' for s in SIGNALS]+['hd_positive_age_days']
def signals(headline,details):
    text=str(headline)+' '+str(details)
    unconfirmed=bool(re.search(r'clarification|verification|response.*awaited|rumou?r|denies|denial',text,re.I))
    adverse=bool(re.search(r'\b(cancelled|canceled|terminated|delayed|postponed|suspended)\b|extended.{0,20}time.?lines',text,re.I))
    project=bool(re.search(r'plant|project|capacity|commercial|contract|(?:work|purchase) order|commissioning',text,re.I))
    admin=bool(re.search(r'demateriali[sz]|reclassification|voting rights|aggregate holding|share capital|regulation 74',text,re.I))
    operating=bool(re.search(r'product|drug|FDA|environmental|plant|project|factory|facility|marketing authori[sz]|operating licen[cs]e',text,re.I))
    intent=bool(re.search(r'\b(will|planned|proposed|expected|target|intend)\b|subject to',text,re.I))
    legal=bool(re.search(LEGAL,text,re.I))
    return dict(order=int(not unconfirmed and business_order_status(details,headline)=='BUSINESS_ORDER_DISCLOSURE_NOT_REVENUE'),
      capacity=int(not unconfirmed and not admin and capacity_disclosure(text)),
      commissioning_language=int(not unconfirmed and not adverse and not intent and not legal and bool(re.search(r'commissioned|commenced.{0,35}(?:production|operations)|commencement of commercial',text,re.I))),
      approval_language=int(not unconfirmed and not adverse and not intent and not legal and not admin and operating and bool(re.search(r'(?:received|obtained|granted).{0,50}(?:approval|licen[cs]e)|(?:approval|licen[cs]e).{0,50}(?:received|granted)',text,re.I))),
      adverse_project=int(adverse and project and not admin),
      dilution=int(bool(re.search(r'allotment.{0,70}(?:equity|shares|warrants)|preferential issue|qualified institution',text,re.I))),
      promoter_purchase_language=int(not unconfirmed and bool(re.search(r'promoter',text,re.I)) and bool(re.search(r'purchase|acquisition',text,re.I))))
def normalize(events):
    rows=[];rejected={}
    def reject(reason):rejected[reason]=rejected.get(reason,0)+1
    for r in events.to_dict('records'):
        try:
            raw=json.loads(r['source_json'])
            if r.get('source')!='NSE' or raw.get('symbol')!=r['symbol'] or raw.get('sm_isin')!=r['isin']:reject('SOURCE_IDENTITY_MISMATCH');continue
            published=pd.Timestamp(r['published_ts'])
            if published.tzinfo is None:reject('NAIVE_PUBLICATION_CLOCK');continue
            original=pd.to_datetime(raw.get('an_dt'),format='%d-%b-%Y %H:%M:%S',errors='coerce')
            if pd.isna(original):reject('UNKNOWN_ORIGINAL_BROADCAST');continue
            if original.tz_localize('Asia/Kolkata').tz_convert('UTC')!=published:reject('BROADCAST_CLOCK_MISMATCH');continue
            received=pd.to_datetime(r.get('exchange_received_ts'),utc=True,errors='coerce')
            clock=max(published,received) if pd.notna(received) else published
            url=r.get('document_url');u=urlparse(str(url))
            if u.scheme!='https' or u.hostname not in {'nsearchives.nseindia.com','archives.nseindia.com'}:reject('NO_PRIMARY_ATTACHMENT');continue
            if raw.get('attchmntFile')!=url:reject('ATTACHMENT_MISMATCH');continue
            if str(raw.get('desc',''))!=str(r.get('headline','')) or str(raw.get('attchmntText',''))!=str(r.get('details','')):reject('TEXT_MISMATCH');continue
            rows.append(dict(isin=r['isin'],symbol=r['symbol'],event_id=r['event_id'],available_at_utc=clock,document_url=url,metadata_sha256=hashlib.sha256(r['source_json'].encode()).hexdigest(),headline=r['headline'],details=r['details'],**signals(r['headline'],r['details'])))
        except (ValueError,TypeError,KeyError):reject('MALFORMED_ORIGINAL_METADATA')
    d=pd.DataFrame(rows)
    if d.empty:return d,dict(accepted=0,rejected=rejected)
    before=len(d);d=d.sort_values('available_at_utc').drop_duplicates(['isin','document_url','metadata_sha256'])
    return d,dict(accepted=len(d),duplicates_removed=before-len(d),rejected=rejected,evidence_level='ORIGINAL_METADATA_NOT_REVIEWED_EXECUTION',retrospective_reconstruction=True)
def attach(frame,events,coverage_end):
    out=frame.copy()
    if out.duplicated(['date','isin']).any():raise ValueError('Duplicate decision identity')
    end=pd.Timestamp(coverage_end)
    if end.tzinfo is None:raise ValueError('Archive end must include timezone')
    for c in FEATURES:out[c]=np.nan
    out['hd_coverage_status']='OUTSIDE_ARCHIVE_OR_UNKNOWN'
    groups={k:g.sort_values('available_at_utc') for k,g in events.groupby('isin')} if len(events) else {}
    for i,r in out.iterrows():
        cut=r.get('information_cutoff_utc')
        if cut is None or pd.isna(cut):cut=pd.Timestamp(r.date).normalize().tz_localize('Asia/Kolkata')+pd.Timedelta(hours=15,minutes=30)
        cut=pd.Timestamp(cut)
        if cut.tzinfo is None:raise ValueError('Naive decision clock')
        if cut>end:continue
        e=groups.get(r['isin'])
        if e is None:continue
        prior=e[e.available_at_utc.le(cut)]
        if prior.empty:continue
        recent=prior[prior.available_at_utc.ge(cut-pd.Timedelta(days=90))]
        for s in SIGNALS:out.loc[i,f'hd_{s}_90d']=int(recent[s].sum())
        positive=prior[prior[['order','capacity','commissioning_language','approval_language']].any(axis=1)]
        if len(positive):out.loc[i,'hd_positive_age_days']=min(366,(cut-positive.available_at_utc.max()).total_seconds()/86400)
        out.loc[i,'hd_coverage_status']='ARCHIVED_METADATA_OBSERVED_NOT_EXHAUSTIVENESS_PROOF'
    return out
