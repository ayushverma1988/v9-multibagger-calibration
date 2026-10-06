from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd

import v11_4_catalyst_graph as cg
from v11_4_nse_primary_evidence import pdf_text

DOC_RE=re.compile(
    r"\b(press release|investor presentation|corporate presentation|business update|"
    r"earnings call|conference call|transcript|analyst meet|investor meet|"
    r"outcome of board meeting|regulation 30|material event|credit rating)\b",
    re.I,
)

def parse_ts(v):
    return pd.to_datetime(v,utc=True,errors="coerce")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--events",required=True)
    ap.add_argument("--secondary",required=True)
    ap.add_argument("--initial-primary",required=True)
    ap.add_argument("--queries-config",required=True)
    ap.add_argument("--output",required=True)
    ap.add_argument("--lookback-days",type=int,default=240)
    ap.add_argument("--max-seed-symbols",type=int,default=140)
    ap.add_argument("--max-docs-per-symbol",type=int,default=12)
    args=ap.parse_args()

    cfg=json.load(open(args.queries_config))
    sec=pd.read_parquet(args.secondary)
    pri=pd.read_parquet(args.initial_primary)
    events=pd.read_parquet(args.events).copy()
    events["published_ts"]=pd.to_datetime(events["published_ts"],utc=True,errors="coerce")
    cutoff=pd.Timestamp.now(tz="UTC")-pd.Timedelta(days=int(args.lookback_days))
    events=events[events["published_ts"].notna() & (events["published_ts"]>=cutoff)].copy()

    # Secondary evidence only seeds a primary-source search; it never qualifies
    # a company by itself.
    if "linked_evidence_score" in sec:
        rank=sec.groupby("symbol")["linked_evidence_score"].max().sort_values(ascending=False)
    else:
        rank=sec.groupby("symbol").size().sort_values(ascending=False)
    secondary_seeds=list(rank.head(int(args.max_seed_symbols)).index.astype(str).str.upper())

    # Rank primary candidates too; cap the COMBINED deep-document universe.
    if len(pri):
        p=pri.copy()
        p["_pstage"]=pd.to_numeric(p.get("stage_weight"),errors="coerce").fillna(0)
        p["_pconf"]=pd.to_numeric(p.get("evidence_confidence"),errors="coerce").fillna(0)
        prank=p.groupby("symbol").agg(stage=("_pstage","max"),conf=("_pconf","max"))
        prank["score"]=0.65*prank["stage"].clip(0,1)+0.35*prank["conf"].clip(0,1)
        primary_seeds=list(prank.sort_values("score",ascending=False).head(int(args.max_seed_symbols)).index.astype(str).str.upper())
    else:
        primary_seeds=[]

    interleaved=[]
    for i in range(max(len(secondary_seeds),len(primary_seeds))):
        if i<len(secondary_seeds):interleaved.append(secondary_seeds[i])
        if i<len(primary_seeds):interleaved.append(primary_seeds[i])
    seeds=list(dict.fromkeys(interleaved))[:int(args.max_seed_symbols)]

    rows=[]; fetched=0; errors=[]
    for sym in seeds:
        g=events[events["symbol"].astype(str).str.upper().eq(sym)].sort_values("published_ts",ascending=False)
        if g.empty:continue
        chosen=[]
        for r in g.itertuples(index=False):
            headline=str(getattr(r,"headline","") or "")
            details=str(getattr(r,"details","") or "")
            url=str(getattr(r,"document_url","") or "")
            if not url:continue
            base=(headline+" "+details).strip()
            if DOC_RE.search(base) or cg.catalyst_types(base) or cg.stage(base)[1]>=0.40:
                chosen.append(r)
            if len(chosen)>=int(args.max_docs_per_symbol):break

        for r in chosen:
            headline=str(getattr(r,"headline","") or "")
            details=str(getattr(r,"details","") or "")
            url=str(getattr(r,"document_url","") or "")
            body=pdf_text(url)
            if not body:continue
            fetched+=1
            text=(headline+" "+details+" "+body).strip()
            cts=cg.catalyst_types(text)
            st,sw=cg.stage(text)
            if not cts and sw<0.40:
                continue
            capctx=cg.catalyst_context(text)
            th=cg.themes(capctx,cfg)
            neg=bool(cg.NEGATIVE_PATTERNS.search(text))
            capm=cg.extract_capacity_metrics(capctx)
            ts=parse_ts(getattr(r,"published_ts",None))
            recid=str(getattr(r,"source_record_id","") or "")
            eid=hashlib.sha256("|".join(["NSE_DEEP",recid,sym,str(ts),headline]).encode()).hexdigest()
            rows.append({
                "evidence_id":eid,
                "published_ts":ts,
                "symbol":sym,
                "company_name":None,
                "isin":getattr(r,"isin",None),
                "domain":"nseindia.com",
                "url":url,
                "title":headline,
                "query_family":"nse_deep_primary",
                "source_tier":1,
                "source_trust":1.0,
                "catalyst_types":json.dumps(cts),
                "themes":json.dumps(th),
                "stage":st,
                "stage_weight":sw,
                "money_crore_max":cg.extract_catalyst_money_crore(capctx),
                "capacity_pct_max":capm["capacity_pct"],
                "capacity_pct_direct":capm["capacity_pct_direct"],
                "capacity_pct_inferred":capm["capacity_pct_inferred"],
                "capacity_inference_method":capm["capacity_inference_method"],
                "capacity_quantities":json.dumps(capm["capacity_quantities"]),
                "negative_flag":neg,
                "corroboration_count":1,
                "evidence_confidence":cg.event_confidence(1.0,sw,cts,neg,1),
                "attachment_text_used":True,
            })

    df=pd.DataFrame(rows)
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    if len(df):
        df=df.drop_duplicates("evidence_id").sort_values("published_ts",ascending=False)
    df.to_parquet(out,index=False)
    summary={
        "seed_symbols":len(seeds),
        "documents_fetched":fetched,
        "material_deep_rows":int(len(df)),
        "material_companies":int(df["symbol"].nunique()) if len(df) else 0,
        "stages":df["stage"].value_counts().to_dict() if len(df) else {},
        "query_families":df["query_family"].value_counts().to_dict() if len(df) else {},
        "errors":errors[:20],
    }
    json.dump(summary,open(out.parent/"deep_primary_summary.json","w"),indent=2,default=str)
    print(json.dumps(summary,indent=2,default=str))

if __name__=="__main__":
    main()
