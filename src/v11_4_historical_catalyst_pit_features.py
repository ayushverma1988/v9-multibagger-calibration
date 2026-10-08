"""Historical NSE catalyst *metadata* PIT features for all 18 frozen folds.

Only event types already present in canonical original NSE filing metadata.
Use later of source publication and exchange-received times if both exist.
Do not invent monetary order size or classify "other" headlines as verified
capacity commissioning. This module is research only; no model weights.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd

CATALYST_TYPES=("capacity_expansion","order_win","regulatory","promoter_activity",
                "corporate_action","buyback","dilution","earnings")
LOOKBACKS=(90,180,365)

def fold_close(date):
    return (pd.Timestamp(date).normalize().tz_localize("Asia/Kolkata")
            +pd.Timedelta(hours=15,minutes=30)).tz_convert("UTC")

def source_availability(published,received):
    x=pd.to_datetime(published,utc=True,errors="coerce",format="mixed")
    y=pd.to_datetime(received,utc=True,errors="coerce",format="mixed")
    # Use the later timestamp conservatively, never an early disputed record.
    return x.where(y.isna()|(x>=y),y)

def fold_event_counts(events,universe,asof,lookback=180):
    cutoff=fold_close(asof)
    e=events.copy()
    if "asof_utc" not in e:
        e["asof_utc"]=source_availability(e["published_ts"],e["exchange_received_ts"])
    start=cutoff-pd.Timedelta(days=lookback)
    e=e[e["asof_utc"].notna()&(e["asof_utc"]<=cutoff)&
        (e["asof_utc"]>=start)&e["symbol"].isin(universe)]
    return e.groupby(["symbol","event_type"]).size()

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--events",required=True)
    p.add_argument("--snapshot",required=True)
    p.add_argument("--folds",required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    col=["event_id","symbol","published_ts","exchange_received_ts",
         "event_type","event_direction","document_url"]
    e=pd.read_parquet(a.events,columns=col)
    raw_n=len(e)
    e=e.drop_duplicates("event_id").copy()
    e["symbol"]=e["symbol"].astype(str).str.upper().str.strip()
    e["event_type"]=e["event_type"].astype(str).str.lower().str.strip()
    e=e[e["event_type"].isin(CATALYST_TYPES)&e["symbol"].ne("")]
    e["published_ts"]=pd.to_datetime(e["published_ts"],utc=True,errors="coerce",format="mixed")
    e["exchange_received_ts"]=pd.to_datetime(e["exchange_received_ts"],utc=True,errors="coerce",format="mixed")
    e["asof_utc"]=source_availability(e["published_ts"],e["exchange_received_ts"])
    e=e[e["asof_utc"].notna()&e["published_ts"].notna()]
    e["event_direction"]=pd.to_numeric(e["event_direction"],errors="coerce").fillna(0)
    snap=pd.read_parquet(a.snapshot,columns=["date","symbol"])
    snap["date"]=pd.to_datetime(snap["date"],errors="coerce").dt.normalize()
    snap["symbol"]=snap["symbol"].astype(str).str.upper().str.strip()
    fdates=pd.to_datetime(pd.read_csv(a.folds)["date"],errors="coerce").dropna().dt.normalize().sort_values().unique()
    if len(fdates)!=18:raise SystemExit("Frozen 18-fold dates must remain unchanged")
    rows=[];checks=[];evidence=[]
    for d in fdates:
        day=pd.Timestamp(d).normalize()
        cutoff=fold_close(day)
        universe=sorted(set(snap.loc[snap["date"].eq(day),"symbol"]))
        base=pd.DataFrame({"date":str(day.date()),"symbol":universe})
        f=e[e["symbol"].isin(universe)&(e["asof_utc"]<=cutoff)]
        proof=f[(f["asof_utc"]>=cutoff-pd.Timedelta(days=365))&
                (f["document_url"].astype(str).str.startswith("https://nsearchives.nseindia.com/"))]
        evidence.extend([
            {"fold":str(day.date()),"symbol":r.symbol,"event_type":r.event_type,
             "published_ts":str(r.published_ts),"source_available_utc":str(r.asof_utc),
             "original_NSE_document_url":r.document_url,"event_id":r.event_id}
            for r in proof.sort_values("asof_utc",ascending=False).head(120).itertuples(index=False)
        ])
        for window in LOOKBACKS:
            recent=f[f["asof_utc"]>=cutoff-pd.Timedelta(days=window)]
            for typ in CATALYST_TYPES:
                sub=recent[recent["event_type"].eq(typ)]
                cnt=sub.groupby("symbol").size()
                pos=sub[sub["event_direction"]>0].groupby("symbol").size()
                colname=f"nse_{typ}_{window}d"
                base[colname]=base["symbol"].map(cnt).fillna(0).astype("int32")
                base[colname+"_positive"]=base["symbol"].map(pos).fillna(0).astype("int32")
        base["historical_asof_utc"]=cutoff.isoformat()
        base["nse_catalyst_total_180d"]=base[[f"nse_{t}_180d" for t in CATALYST_TYPES]].sum(axis=1)
        rows.append(base)
        checks.append({
            "fold":str(day.date()),"universe_symbols":len(universe),
            "companies_with_180d_known_catalyst_types":int((base["nse_catalyst_total_180d"]>0).sum()),
            "documented_catalyst_metadata_180d":int(base["nse_catalyst_total_180d"].sum()),
            "feature_rows":len(base),"events_in_original_archive_asof":len(f)
        })
        print("PIT_CATALYST",checks[-1],flush=True)
    frame=pd.concat(rows,ignore_index=True)
    if frame.duplicated(["date","symbol"]).any():raise SystemExit("Duplicate symbol in frozen universe")
    if len(frame)<14000:raise SystemExit("Insufficient historical stocks for 18 folds")
    frame.to_parquet(out/"canonical_NSE_18fold_catalyst_metadata_PIT_RESEARCH.parquet",
                     index=False,compression="zstd")
    pd.DataFrame(evidence).to_csv(out/"sample_source_provenance_by_fold.csv",index=False)
    pd.DataFrame(checks).to_csv(out/"18fold_catalyst_coverage.csv",index=False)
    summary={
        "scope":"18FOLD_SOURCE_METADATA_CATALYST_REHEARSAL_ONLY",
        "source":"Original canonical NSE exchange disclosure records 2016-2026",
        "source_rows_before_type_filter":raw_n,
        "eligible_event_type_rows":len(e),
        "source_events_restricted_to_frozen_folds":True,
        "point_in_time_eligibility":"max(published_ts,exchange_received_ts) <= local 15:30 exchange close",
        "original_fold_count":len(checks),"matrix_rows":len(frame),
        "coverages":checks,
        "official_document_sample_links":len(evidence),
        "no_unverified_commissioning_or_order_value_claims":True,
        "GDELT_or_Google_historical_availability_verified":False,
        "six_month_market_backtest_completed":False,
        "v10_model_changed":False,
    }
    (out/"historical_catalyst_18fold_summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k!="coverages"},indent=2),flush=True)
if __name__=="__main__":main()
