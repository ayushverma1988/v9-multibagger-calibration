from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--oos",required=True)
    ap.add_argument("--output",required=True)
    args=ap.parse_args()

    outdir=Path(args.output)
    outdir.mkdir(parents=True,exist_ok=True)

    d=pd.read_parquet(args.oos)
    d["date"]=pd.to_datetime(d["date"])
    d["symbol"]=d["symbol"].astype(str).str.upper().str.strip()
    if "isin" not in d.columns:
        d["isin"]=np.nan
    d["isin_norm"]=d["isin"].astype(str).str.upper().str.strip()
    d.loc[d["isin"].isna() | d["isin_norm"].isin(["","NAN","NONE"]),"isin_norm"]=np.nan

    dup_symbol=(
        d.groupby(["date","symbol"],dropna=False)
         .size().reset_index(name="rows")
    )
    dup_symbol=dup_symbol[dup_symbol["rows"]>1].copy()

    known=d.dropna(subset=["isin_norm"]).copy()
    dup_isin=(
        known.groupby(["date","isin_norm"])
             .agg(rows=("symbol","size"),symbols=("symbol","nunique"))
             .reset_index()
    )
    dup_isin=dup_isin[(dup_isin["rows"]>1)|(dup_isin["symbols"]>1)].copy()

    isin_symbols=(
        known.groupby("isin_norm")
             .agg(
                 symbol_count=("symbol","nunique"),
                 first_date=("date","min"),
                 last_date=("date","max"),
                 rows=("symbol","size"),
             )
             .reset_index()
    )
    multi_symbol_isin=isin_symbols[isin_symbols["symbol_count"]>1].copy()

    if len(multi_symbol_isin):
        details=(
            known[known["isin_norm"].isin(multi_symbol_isin["isin_norm"])]
            .groupby(["isin_norm","symbol"])
            .agg(first_date=("date","min"),last_date=("date","max"),rows=("date","size"))
            .reset_index()
            .sort_values(["isin_norm","first_date","symbol"])
        )
    else:
        details=pd.DataFrame(columns=["isin_norm","symbol","first_date","last_date","rows"])

    symbol_isins=(
        known.groupby("symbol")
             .agg(
                 isin_count=("isin_norm","nunique"),
                 first_date=("date","min"),
                 last_date=("date","max"),
                 rows=("isin_norm","size"),
             )
             .reset_index()
    )
    multi_isin_symbol=symbol_isins[symbol_isins["isin_count"]>1].copy()

    selected_mask=pd.Series(False,index=d.index)
    for col in ["selected_v931","selected_v95"]:
        if col in d.columns:
            selected_mask=selected_mask | d[col].fillna(False).astype(bool)

    churn_isins=set(multi_symbol_isin["isin_norm"].dropna().astype(str))
    churn_rows=d["isin_norm"].isin(churn_isins)
    selected_churn=int((churn_rows & selected_mask).sum())

    # Same-ISIN ticker transitions are likely continuity candidates only when
    # symbols do not coexist on the same snapshot date.
    overlap_isins=set(dup_isin.loc[dup_isin["symbols"]>1,"isin_norm"].astype(str))
    sequential_isins=sorted(churn_isins-overlap_isins)

    missing_isin=float(d["isin_norm"].isna().mean()) if len(d) else 0.0

    dup_symbol.to_csv(outdir/"duplicate_date_symbol.csv",index=False)
    dup_isin.to_csv(outdir/"duplicate_date_isin.csv",index=False)
    multi_symbol_isin.to_csv(outdir/"isin_with_multiple_symbols.csv",index=False)
    details.to_csv(outdir/"isin_symbol_timeline.csv",index=False)
    multi_isin_symbol.to_csv(outdir/"symbols_with_multiple_isins.csv",index=False)

    summary={
        "rows":int(len(d)),
        "dates":int(d["date"].nunique()),
        "symbols":int(d["symbol"].nunique()),
        "known_isins":int(d["isin_norm"].nunique()),
        "missing_isin_fraction":missing_isin,
        "duplicate_date_symbol_groups":int(len(dup_symbol)),
        "duplicate_date_isin_groups":int(len(dup_isin)),
        "isins_with_multiple_symbols":int(len(multi_symbol_isin)),
        "sequential_ticker_change_isins":int(len(sequential_isins)),
        "overlapping_multi_symbol_isins":int(len(overlap_isins)),
        "symbols_with_multiple_isins":int(len(multi_isin_symbol)),
        "selected_rows_on_multi_symbol_isin":selected_churn,
        "continuity_policy_status":"diagnostic_only",
        "next_rule":(
            "do not rewrite history automatically; review sequential same-ISIN "
            "ticker changes separately from overlapping symbols and symbol reuse"
        ),
    }
    json.dump(summary,open(outdir/"summary.json","w"),indent=2,default=str)
    print(json.dumps(summary,indent=2,default=str))


if __name__=="__main__":
    main()
