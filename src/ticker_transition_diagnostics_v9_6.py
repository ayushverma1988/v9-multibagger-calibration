from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import calibrate_v9_2 as base


def transition_rows(d: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    q=d.dropna(subset=["isin"]).copy()
    q["isin"]=q["isin"].astype(str).str.upper().str.strip()
    q=q[q["isin"].str.startswith("INE")]
    for isin,g in q.groupby("isin",sort=False):
        g=g.sort_values(["date","symbol"]).copy()
        syms=list(g["symbol"].dropna().astype(str).unique())
        if len(syms)<=1:
            continue
        # Find chronological symbol segments.
        seq=g[["date","symbol","series","close","adj_close","turnover"]].copy()
        seq["prev_symbol"]=seq["symbol"].shift()
        starts=seq[(seq["prev_symbol"].notna()) & (seq["symbol"]!=seq["prev_symbol"])].copy()
        for _,r in starts.iterrows():
            old=str(r["prev_symbol"])
            new=str(r["symbol"])
            td=pd.Timestamp(r["date"])
            old_hist=seq[(seq["symbol"]==old)&(seq["date"]<td)].sort_values("date")
            new_hist=seq[(seq["symbol"]==new)&(seq["date"]>=td)].sort_values("date")
            if old_hist.empty or new_hist.empty:
                continue
            a=old_hist.iloc[-1]
            b=new_hist.iloc[0]
            adj_ratio=(float(b["adj_close"])/float(a["adj_close"])) if float(a["adj_close"])!=0 else np.nan
            close_ratio=(float(b["close"])/float(a["close"])) if float(a["close"])!=0 else np.nan
            rows.append({
                "isin":isin,
                "old_symbol":old,
                "new_symbol":new,
                "old_date":pd.Timestamp(a["date"]),
                "new_date":pd.Timestamp(b["date"]),
                "calendar_gap_days":int((pd.Timestamp(b["date"])-pd.Timestamp(a["date"])).days),
                "old_series":a["series"],
                "new_series":b["series"],
                "old_close":float(a["close"]),
                "new_close":float(b["close"]),
                "old_adj_close":float(a["adj_close"]),
                "new_adj_close":float(b["adj_close"]),
                "raw_ratio":close_ratio,
                "adj_ratio":adj_ratio,
                "old_turnover":float(a["turnover"]) if np.isfinite(a["turnover"]) else np.nan,
                "new_turnover":float(b["turnover"]) if np.isfinite(b["turnover"]) else np.nan,
            })
    return pd.DataFrame(rows)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--start-year",type=int,default=2010)
    ap.add_argument("--end-year",type=int,default=2026)
    ap.add_argument("--output",required=True)
    args=ap.parse_args()

    outdir=Path(args.output)
    outdir.mkdir(parents=True,exist_ok=True)

    d=base.load_market(args.start_year,args.end_year,None)
    # Same-date same-security duplicates can distort rolling features if stitched.
    dup=(
        d.dropna(subset=["security_key"])
         .groupby(["date","security_key"])
         .agg(
             rows=("symbol","size"),
             symbols=("symbol","nunique"),
             series=("series","nunique"),
         )
         .reset_index()
    )
    dup=dup[dup["rows"]>1].copy()

    tr=transition_rows(d)
    if len(tr):
        tr["adj_jump_abs_log"]=np.abs(np.log(tr["adj_ratio"].clip(lower=1e-12)))
        tr["plausible_continuity"]=tr["adj_ratio"].between(0.67,1.50)
        tr["near_continuity"]=tr["adj_ratio"].between(0.80,1.25)
        tr["series_changed"]=tr["old_series"].astype(str)!=tr["new_series"].astype(str)
        tr=tr.sort_values(["plausible_continuity","adj_jump_abs_log"],ascending=[True,False])

    tr.to_csv(outdir/"ticker_transition_diagnostics.csv",index=False)
    dup.to_csv(outdir/"duplicate_date_security_key.csv",index=False)

    summary={
        "market_rows":int(len(d)),
        "date_security_duplicates":int(len(dup)),
        "duplicate_rows_total":int(dup["rows"].sum()) if len(dup) else 0,
        "ticker_transition_rows":int(len(tr)),
        "transition_isins":int(tr["isin"].nunique()) if len(tr) else 0,
        "adj_ratio_median":float(tr["adj_ratio"].median()) if len(tr) else None,
        "adj_ratio_p05":float(tr["adj_ratio"].quantile(.05)) if len(tr) else None,
        "adj_ratio_p95":float(tr["adj_ratio"].quantile(.95)) if len(tr) else None,
        "near_continuity_fraction":float(tr["near_continuity"].mean()) if len(tr) else None,
        "plausible_continuity_fraction":float(tr["plausible_continuity"].mean()) if len(tr) else None,
        "series_change_fraction":float(tr["series_changed"].mean()) if len(tr) else None,
        "diagnostic_goal":"determine whether full-ISIN stitching should be restricted by same-date deduplication, series transitions, or price-scale continuity",
    }
    json.dump(summary,open(outdir/"summary.json","w"),indent=2,default=str)
    print(json.dumps(summary,indent=2,default=str))


if __name__=="__main__":
    main()
