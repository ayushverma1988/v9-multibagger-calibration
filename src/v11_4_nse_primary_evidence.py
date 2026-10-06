from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import pandas as pd
import requests
from pypdf import PdfReader

import v11_4_catalyst_graph as cg

UA="Mozilla/5.0 (compatible; V11.4CatalystResearch/1.0)"

def pdf_text(url:str)->str:
    if not url:
        return ""
    try:
        r=requests.get(url,headers={"User-Agent":UA,"Referer":"https://www.nseindia.com/"},timeout=35)
        if r.status_code!=200:
            return ""
        ct=r.headers.get("content-type","").lower()
        if "pdf" not in ct and not url.lower().endswith(".pdf"):
            return ""
        reader=PdfReader(io.BytesIO(r.content))
        parts=[]
        for p in reader.pages[:25]:
            try:
                parts.append(p.extract_text() or "")
            except Exception:
                pass
        return " ".join(parts)[:150000]
    except Exception:
        return ""

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--events",required=True)
    ap.add_argument("--queries-config",required=True)
    ap.add_argument("--output",required=True)
    ap.add_argument("--fetch-attachments",action="store_true")
    args=ap.parse_args()

    cfg=json.load(open(args.queries_config))
    ev=pd.read_parquet(args.events).copy()
    rows=[]
    fetched=0
    for r in ev.itertuples(index=False):
        symbol=str(getattr(r,"symbol","") or "").strip().upper()
        if not symbol:
            continue
        headline=str(getattr(r,"headline","") or "")
        details=str(getattr(r,"details","") or "")
        url=str(getattr(r,"document_url","") or "")
        base=(headline+" "+details).strip()
        c0=cg.catalyst_types(base)
        # Fetch official attachment only for announcements already carrying
        # catalyst/order/capacity/product/promoter clues in headline/details.
        body=""
        if args.fetch_attachments and (c0 or cg.stage(base)[1] >= 0.40):
            body=pdf_text(url)
            fetched += 1 if body else 0
        text=(base+" "+body).strip()
        cts=cg.catalyst_types(text)
        th=cg.themes(text,cfg)
        st,sw=cg.stage(text)
        neg=bool(cg.NEGATIVE_PATTERNS.search(text))
        ts=pd.to_datetime(getattr(r,"published_ts",None),utc=True,errors="coerce")
        recid=str(getattr(r,"source_record_id","") or "")
        raw="|".join(["NSE",recid,symbol,str(ts),headline])
        eid=hashlib.sha256(raw.encode()).hexdigest()
        conf=cg.event_confidence(1.0,sw,cts,neg,1)
        rows.append({
            "evidence_id":eid,
            "published_ts":ts,
            "symbol":symbol,
            "company_name":None,
            "isin":getattr(r,"isin",None),
            "domain":"nseindia.com",
            "url":url,
            "title":headline,
            "query_family":"nse_primary",
            "source_tier":1,
            "source_trust":1.0,
            "catalyst_types":json.dumps(cts),
            "themes":json.dumps(th),
            "stage":st,
            "stage_weight":sw,
            "money_crore_max":cg.extract_money_crore(text),
            "capacity_pct_max":cg.extract_capacity_pct(text),
            "negative_flag":neg,
            "corroboration_count":1,
            "evidence_confidence":conf,
            "attachment_text_used":bool(body),
        })

    df=pd.DataFrame(rows)
    if len(df):
        df=df.sort_values("published_ts",ascending=False).drop_duplicates("evidence_id")
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
    df.to_parquet(out,index=False)

    meaningful=df[
        (df["stage_weight"]>=0.40)
        | df["catalyst_types"].ne("[]")
    ] if len(df) else df
    summary={
        "input_rows":int(len(ev)),
        "output_rows":int(len(df)),
        "meaningful_rows":int(len(meaningful)),
        "companies":int(meaningful["symbol"].nunique()) if len(meaningful) else 0,
        "attachments_extracted":int(fetched),
        "stages":meaningful["stage"].value_counts().to_dict() if len(meaningful) else {},
    }
    json.dump(summary,open(out.parent/"nse_primary_summary.json","w"),indent=2,default=str)
    print(json.dumps(summary,indent=2,default=str))

if __name__=="__main__":
    main()
