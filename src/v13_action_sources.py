"""Mine original action notices from whole-public NSE feeds for review.

Metadata, empty responses and downloaded notices do not certify complete
action intervals. No private forecast list enters a request queue.
"""
from pathlib import Path
import json,re
import pandas as pd
from v12_dated_source_collection import Store,sha
from v13_parallel_collection import bounded_map
from v11_4_primary_document_catalysts import extract_pdf

def collect(root,asof,budget=30):
    root=Path(root);out=root/'action_audit';out.mkdir(exist_ok=True)
    paths=list((root/'sources').glob('*/events.parquet'))
    docs_path=out/'original_notice_candidates.json';old=json.loads(docs_path.read_text()) if docs_path.exists() else []
    if not paths:return {'notices':len(old),'new_notices':0,'complete_intervals_certified':0}
    e=pd.concat([pd.read_parquet(p) for p in paths],ignore_index=True)
    date=pd.Timestamp(asof);low=date-pd.DateOffset(years=4)
    clock=pd.to_datetime(e.available_at_utc,utc=True,format='mixed')
    # All public issuers, selected and unselected, share this queue.
    action=e.text.str.contains(r'\bbonus\b|\bsplit\b|sub.?division|\bdividend\b|\brights\b|merger|demerger|change.*(?:symbol|ISIN)',case=False,na=False)
    e=e[action&clock.ge(low.tz_localize('Asia/Kolkata'))&clock.lt((date+pd.Timedelta(days=1)).tz_localize('Asia/Kolkata'))]
    e=e[e.attachment_url.str.match(r'https://nsearchives\.nseindia\.com/.*\.pdf$',case=False)].drop_duplicates(['attachment_url','available_at_utc'])
    known={(z['attachment_url'],z['available_at_utc']) for z in old}
    e=e[[(r.attachment_url,r.available_at_utc) not in known for r in e.itertuples()]]
    queue=e.sort_values(['available_at_utc','symbol'],ascending=[False,True]).head(budget);errors=[];added=0
    store=Store(out/'raw'/'notices'/asof)
    def fetch(r):
        try:
            raw,proof=store.get(r['attachment_url']);text=extract_pdf(raw)
            excerpts=[re.sub(r'\s+',' ',z).strip()[:300] for z in text.split('\n') if re.search(r'ex[ -]?date|record date|bonus|split|dividend|rights|merger|ISIN',z,re.I)]
            (out/(proof['sha256']+'.txt')).write_text(text)
            return {'notice':{**r,'source_sha256':proof['sha256'],'first_retrieved_utc':proof['first_retrieved_utc'],'text_sha256':sha(text.encode()),
              'review_excerpts':excerpts[:20],'status':'ORIGINAL_DOWNLOADED_FIELDS_AND_INTERVAL_STILL_REQUIRE_VERIFICATION',
              'ex_date_not_inferred_from_record_date':True,'all_primary_notices_reviewed':False}}
        except Exception as exc:return {'error':str(exc)[:200]}
    for r,result in bounded_map(queue.to_dict('records'),fetch,workers=4,seconds=150):
        if result.get('notice'):old.append(result['notice']);added+=1
        else:errors.append({'source_url':r['attachment_url'],'error':result['error']})
    docs_path.write_text(json.dumps(old,indent=2))
    summary={'source_asof':asof,'notices':len(old),'new_notices':added,'remaining_public_notices':len(e)-added,
      'errors':errors,'complete_intervals_certified':0,'absence_is_not_no_action_proof':True,'private_forecasts_used_in_requests':False}
    (out/'notice_collection.json').write_text(json.dumps(summary,indent=2));return summary
