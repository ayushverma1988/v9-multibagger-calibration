from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd

import calibrate_v9_2 as base


def norm_isin(s):
    x=s.astype(str).str.upper().str.strip()
    valid=s.notna() & ~x.isin(["","NAN","NONE","<NA>"])
    return x.where(valid)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--oos",required=True)
    ap.add_argument("--events",required=True)
    ap.add_argument("--legacy-dir",required=True)
    ap.add_argument("--output",required=True)
    args=ap.parse_args()

    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)

    oos=pd.read_parquet(args.oos)
    oos["date"]=pd.to_datetime(oos["date"])
    oos["isin_norm"]=norm_isin(oos["isin"]) if "isin" in oos.columns else np.nan
    sel=oos[oos.get("selected_v941",False)==True].copy()

    missing_by_year=(oos.assign(year=oos["date"].dt.year)
        .groupby("year")["isin_norm"]
        .agg(rows="size",known=lambda x:x.notna().sum())
        .reset_index())
    missing_by_year["missing_fraction"]=1-missing_by_year["known"]/missing_by_year["rows"]
    missing_by_year.to_csv(out/"missing_isin_by_year.csv",index=False)

    selected_missing={
        "selected_rows":int(len(sel)),
        "selected_missing_isin_rows":int(sel["isin_norm"].isna().sum()) if len(sel) else 0,
        "selected_missing_isin_fraction":float(sel["isin_norm"].isna().mean()) if len(sel) else 0.0,
        "selected_missing_isin_symbols":sorted(sel.loc[sel["isin_norm"].isna(),"symbol"].astype(str).unique().tolist()) if len(sel) else [],
    }

    market=base.load_market(2003,2026,args.legacy_dir)
    market["date"]=pd.to_datetime(market["date"])
    market=market.sort_values(["symbol","date"]).copy()
    market["adj_ret"]=market.groupby("symbol")["adj_close"].pct_change()
    market["raw_ret"]=market.groupby("symbol")["close"].pct_change()
    market["abs_adj_ret"]=market["adj_ret"].abs()

    jumps=market[np.isfinite(market["adj_ret"]) & (market["abs_adj_ret"]>=.50)].copy()
    jumps["year"]=jumps["date"].dt.year
    jumps.to_csv(out/"suspicious_adj_jumps_50pct.csv",index=False)

    # Selected-universe jump exposure.
    selected_symbols=set(sel["symbol"].astype(str)) if len(sel) else set()
    selected_jumps=jumps[jumps["symbol"].astype(str).isin(selected_symbols)].copy()
    selected_jumps.to_csv(out/"selected_symbols_suspicious_jumps.csv",index=False)

    # Cross-check 2016+ split/bonus announcements from the PIT event archive.
    events=pd.read_parquet(args.events)
    events["published_ts"]=pd.to_datetime(events["published_ts"],utc=True,errors="coerce")
    txt=(events.get("headline","").fillna("").astype(str)+" "+events.get("details","").fillna("").astype(str)).str.lower()
    ca=events[txt.str.contains(r"\bsplit\b|\bbonus\b",regex=True,na=False)].copy()
    ca["event_date"]=ca["published_ts"].dt.tz_convert("Asia/Kolkata").dt.tz_localize(None).dt.normalize()
    ca["symbol"]=ca["symbol"].astype(str).str.upper().str.strip()

    checks=[]
    for r in jumps[jumps["date"]>=pd.Timestamp("2016-01-01")].itertuples(index=False):
        q=ca[
            (ca["symbol"]==str(r.symbol).upper().strip())
            & (ca["event_date"]>=pd.Timestamp(r.date)-pd.Timedelta(days=7))
            & (ca["event_date"]<=pd.Timestamp(r.date)+pd.Timedelta(days=7))
        ]
        checks.append({
            "date":pd.Timestamp(r.date),
            "symbol":r.symbol,
            "adj_ret":float(r.adj_ret),
            "raw_ret":float(r.raw_ret) if np.isfinite(r.raw_ret) else np.nan,
            "nearby_split_bonus_announcement":bool(len(q)>0),
            "nearby_event_count":int(len(q)),
        })
    cj=pd.DataFrame(checks)
    cj.to_csv(out/"jump_corporate_action_crosscheck.csv",index=False)

    summary={
        "model":"V10.2 residual integrity audit",
        "oos_rows":int(len(oos)),
        "oos_missing_isin_fraction":float(oos["isin_norm"].isna().mean()),
        **selected_missing,
        "daily_market_rows":int(len(market)),
        "suspicious_adj_jumps_50pct":int(len(jumps)),
        "suspicious_adj_jumps_100pct":int((jumps["abs_adj_ret"]>=1.0).sum()),
        "selected_symbol_suspicious_jumps_50pct":int(len(selected_jumps)),
        "post2016_jumps_crosschecked":int(len(cj)),
        "post2016_jumps_without_nearby_split_bonus_announcement":int((~cj["nearby_split_bonus_announcement"]).sum()) if len(cj) else 0,
        "note":"diagnostic only; no model changes are made by this audit",
    }
    json.dump(summary,open(out/"summary.json","w"),indent=2,default=str)
    print(json.dumps(summary,indent=2,default=str))

if __name__=="__main__":
    main()
