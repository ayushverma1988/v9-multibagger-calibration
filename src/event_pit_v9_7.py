from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED = [
    "event_id","published_ts","source","source_record_id",
    "symbol","isin","security_key","headline","event_type",
    "event_direction","event_strength","document_url","raw_text_hash",
]


def normalize_events(df: pd.DataFrame) -> pd.DataFrame:
    x=df.copy()
    missing=[c for c in REQUIRED if c not in x.columns]
    if missing:
        raise ValueError(f"Missing event columns: {missing}")

    x["published_ts"]=pd.to_datetime(x["published_ts"],errors="coerce",utc=True)
    x["symbol"]=x["symbol"].astype(str).str.upper().str.strip()
    x["isin"]=x["isin"].astype(str).str.upper().str.strip()
    x.loc[x["isin"].isin(["","NAN","NONE","<NA>"]),"isin"]=np.nan
    x["security_key"]=x["security_key"].astype(str).str.strip()
    x["event_type"]=x["event_type"].astype(str).str.strip()
    x["event_direction"]=pd.to_numeric(x["event_direction"],errors="coerce")
    x["event_strength"]=pd.to_numeric(x["event_strength"],errors="coerce")

    x=x.dropna(subset=["published_ts","source","source_record_id","headline","event_type"])
    x=x.sort_values(["published_ts","source","source_record_id"])

    # Revisions are append-only; exact source-record duplicates keep the earliest
    # publication row and later revisions require a distinct source_record_id.
    x=x.drop_duplicates(["source","source_record_id"],keep="first")
    return x.reset_index(drop=True)


def security_key(symbol, isin):
    if pd.notna(isin) and str(isin).strip():
        return "ISIN:"+str(isin).upper().strip()
    return "SYM:"+str(symbol).upper().strip()


def validate_events(df: pd.DataFrame, taxonomy: dict) -> dict:
    x=normalize_events(df)
    allowed=set(taxonomy["event_types"])
    bad_types=sorted(set(x["event_type"])-allowed)
    if bad_types:
        raise ValueError(f"Unknown event types: {bad_types[:20]}")

    expected=[
        security_key(s,i)
        for s,i in zip(x["symbol"],x["isin"])
    ]
    bad_key=int((x["security_key"].astype(str)!=pd.Series(expected,index=x.index)).sum())
    if bad_key:
        raise ValueError(f"{bad_key} event rows have non-PIT security keys")

    if ((x["event_direction"]<-1)|(x["event_direction"]>1)).any():
        raise ValueError("event_direction must be in [-1,1]")
    if ((x["event_strength"]<0)|(x["event_strength"]>1)).any():
        raise ValueError("event_strength must be in [0,1]")

    return {
        "rows":int(len(x)),
        "sources":int(x["source"].nunique()),
        "event_types":int(x["event_type"].nunique()),
        "start":str(x["published_ts"].min()) if len(x) else None,
        "end":str(x["published_ts"].max()) if len(x) else None,
        "duplicates_removed":int(len(df)-len(x)),
        "valid":True,
    }


def asof_event_features(
    snapshots: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    ev=normalize_events(events)
    snaps=snapshots[["date","symbol"] + (["isin"] if "isin" in snapshots.columns else [])].copy()
    snaps["date"]=pd.to_datetime(snaps["date"])
    if "isin" not in snaps.columns:
        snaps["isin"]=np.nan
    snaps["security_key"]=[
        security_key(s,i) for s,i in zip(snaps["symbol"],snaps["isin"])
    ]

    rows=[]
    for r in snaps.itertuples(index=False):
        # Snapshot date is interpreted as UTC midnight. Events at the same
        # timestamp/day are excluded; only strictly prior publications count.
        ts=pd.Timestamp(r.date).tz_localize("UTC")
        q=ev[
            (ev["security_key"]==r.security_key)
            & (ev["published_ts"]<ts)
        ].copy()

        def window(days):
            start=ts-pd.Timedelta(days=days)
            return q[q["published_ts"]>=start].copy()

        w30=window(30)
        w90=window(90)
        pos90=w90[w90["event_direction"]>0]
        neg90=w90[w90["event_direction"]<0]

        novelty=(
            float(w90["event_type"].nunique())/max(1,len(w90))
            if len(w90) else 0.0
        )

        def days_since(z):
            if z.empty:
                return np.nan
            return float((ts-z["published_ts"].max()).total_seconds()/86400.0)

        rows.append({
            "date":pd.Timestamp(r.date),
            "symbol":r.symbol,
            "event_count_30d":int(len(w30)),
            "event_count_90d":int(len(w90)),
            "positive_event_strength_90d":float(
                (pos90["event_strength"]*pos90["event_direction"]).sum()
            ) if len(pos90) else 0.0,
            "negative_event_strength_90d":float(
                (-neg90["event_strength"]*neg90["event_direction"]).sum()
            ) if len(neg90) else 0.0,
            "event_novelty_90d":novelty,
            "days_since_positive_event":days_since(pos90),
            "days_since_negative_event":days_since(neg90),
        })
    return pd.DataFrame(rows)


def synthetic_self_test(taxonomy: dict) -> dict:
    ev=pd.DataFrame([
        {
            "event_id":"e1","published_ts":"2025-01-10T10:00:00Z",
            "source":"TEST","source_record_id":"1","symbol":"ABC",
            "isin":"INE000TEST01","security_key":"ISIN:INE000TEST01",
            "headline":"Order win","event_type":"order_win",
            "event_direction":1.0,"event_strength":0.8,
            "document_url":"https://example.invalid/1","raw_text_hash":"h1",
        },
        {
            "event_id":"e2","published_ts":"2025-02-01T00:00:00Z",
            "source":"TEST","source_record_id":"2","symbol":"ABC",
            "isin":"INE000TEST01","security_key":"ISIN:INE000TEST01",
            "headline":"Debt reduction","event_type":"debt_reduction",
            "event_direction":1.0,"event_strength":0.5,
            "document_url":"https://example.invalid/2","raw_text_hash":"h2",
        },
    ])
    validate_events(ev,taxonomy)
    snaps=pd.DataFrame([
        {"date":"2025-02-01","symbol":"ABC","isin":"INE000TEST01"},
        {"date":"2025-02-02","symbol":"ABC","isin":"INE000TEST01"},
    ])
    z=asof_event_features(snaps,ev)
    assert int(z.iloc[0]["event_count_30d"])==1
    assert int(z.iloc[1]["event_count_30d"])==2
    return {
        "same_day_event_excluded":True,
        "next_day_event_included":True,
        "passed":True,
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--taxonomy",default="config/event_taxonomy_v9_7.json")
    ap.add_argument("--events")
    ap.add_argument("--snapshots")
    ap.add_argument("--output")
    ap.add_argument("--self-test",action="store_true")
    args=ap.parse_args()

    taxonomy=json.load(open(args.taxonomy))
    if args.self_test:
        print(json.dumps(synthetic_self_test(taxonomy),indent=2))
        return

    if not args.events:
        raise SystemExit("--events required unless --self-test")

    p=Path(args.events)
    ev=pd.read_parquet(p) if p.suffix.lower() in {".parquet",".pq"} else pd.read_csv(p)
    report=validate_events(ev,taxonomy)

    if args.snapshots and args.output:
        sp=Path(args.snapshots)
        snaps=pd.read_parquet(sp) if sp.suffix.lower() in {".parquet",".pq"} else pd.read_csv(sp)
        out=asof_event_features(snaps,ev)
        op=Path(args.output)
        op.parent.mkdir(parents=True,exist_ok=True)
        if op.suffix.lower() in {".parquet",".pq"}:
            out.to_parquet(op,index=False)
        else:
            out.to_csv(op,index=False)
        report["feature_rows"]=int(len(out))

    print(json.dumps(report,indent=2,default=str))


if __name__=="__main__":
    main()
