from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import calibrate_v9 as base


def compute_symbol(g: pd.DataFrame, asof: pd.Timestamp) -> dict | None:
    g=g.sort_values("date").copy()
    g=g[g["date"]<=asof].copy()
    if len(g)<60:
        return None
    p=pd.to_numeric(g["adj_close"],errors="coerce")
    v=pd.to_numeric(g["volume"],errors="coerce")
    r=p.pct_change()

    dma50_prev=p.shift(1).rolling(50,min_periods=40).mean()
    dma200_prev=p.shift(1).rolling(200,min_periods=160).mean()
    hi252_prev=p.shift(1).rolling(252,min_periods=126).max()
    lo252_prev=p.shift(1).rolling(252,min_periods=126).min()
    hi60_prev=p.shift(1).rolling(60,min_periods=40).max()

    last=g.index[-1]
    price=float(p.loc[last]) if pd.notna(p.loc[last]) else np.nan
    d50=float(dma50_prev.loc[last]) if pd.notna(dma50_prev.loc[last]) else np.nan
    d200=float(dma200_prev.loc[last]) if pd.notna(dma200_prev.loc[last]) else np.nan
    h252=float(hi252_prev.loc[last]) if pd.notna(hi252_prev.loc[last]) else np.nan
    l252=float(lo252_prev.loc[last]) if pd.notna(lo252_prev.loc[last]) else np.nan
    h60=float(hi60_prev.loc[last]) if pd.notna(hi60_prev.loc[last]) else np.nan

    def chg(s,n):
        a=s.iloc[-1] if len(s) else np.nan
        b=s.iloc[-1-n] if len(s)>n else np.nan
        if pd.isna(a) or pd.isna(b) or b==0:return np.nan
        return float(a/b-1)

    d50s=chg(dma50_prev.dropna(),20)
    d200s=chg(dma200_prev.dropna(),20)

    upv=v.where(r>0).tail(20).mean()
    dnv=v.where(r<0).tail(20).mean()
    accum=float(upv/dnv) if pd.notna(upv) and pd.notna(dnv) and dnv>0 else np.nan
    vol20=float(v.tail(20).mean()) if len(v)>=20 else np.nan
    vol_break=float(v.iloc[-1]/vol20) if pd.notna(vol20) and vol20>0 and pd.notna(v.iloc[-1]) else np.nan

    h20=p.tail(20).max(); l20=p.tail(20).min()
    compression=float(h20/l20-1) if pd.notna(h20) and pd.notna(l20) and l20>0 else np.nan

    ret20=chg(p,20); ret60=chg(p,60); ret120=chg(p,120)
    return {
        "symbol":str(g.iloc[-1]["symbol"]).upper(),
        "asof_date":str(pd.Timestamp(g.iloc[-1]["date"]).date()),
        "price":price,
        "dma50_prev":d50,
        "dma200_prev":d200,
        "price_gt_dma50_prev":bool(pd.notna(price) and pd.notna(d50) and price>d50),
        "price_lt_dma200_prev":bool(pd.notna(price) and pd.notna(d200) and price<d200),
        "price_gt_dma200_prev":bool(pd.notna(price) and pd.notna(d200) and price>d200),
        "price_above_both_dma":bool(pd.notna(price) and pd.notna(d50) and pd.notna(d200) and price>d50 and price>d200),
        "dma50_slope_20":d50s,
        "dma200_slope_20":d200s,
        "dma50_slope_positive":bool(pd.notna(d50s) and d50s>0),
        "dma200_flat_or_positive":bool(pd.notna(d200s) and d200s>-0.01),
        "up_from_52w_low":float(price/l252-1) if pd.notna(price) and pd.notna(l252) and l252>0 else np.nan,
        "down_from_52w_high":float(1-price/h252) if pd.notna(price) and pd.notna(h252) and h252>0 else np.nan,
        "near_or_breaking_60d_high":bool(pd.notna(price) and pd.notna(h60) and price>=0.98*h60),
        "ret_20_chart":ret20,
        "ret_60_chart":ret60,
        "ret_120_chart":ret120,
        "up_down_volume_ratio_20":accum,
        "volume_breakout_ratio":vol_break,
        "base_compression_20":compression,
        "base_compression_flag":bool(pd.notna(compression) and compression<0.18),
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--symbols-file",required=True)
    ap.add_argument("--symbol-column",default="symbol")
    ap.add_argument("--asof",default="")
    ap.add_argument("--output",required=True)
    args=ap.parse_args()

    asof=pd.Timestamp(args.asof).normalize() if args.asof else pd.Timestamp.today().normalize()
    src=pd.read_csv(args.symbols_file)
    syms=set(src[args.symbol_column].dropna().astype(str).str.upper().str.strip())

    start_year=max(2010,int(asof.year)-2)
    daily=base.load_market(start_year,int(asof.year))
    daily["date"]=pd.to_datetime(daily["date"]).dt.normalize()
    daily=daily[daily["symbol"].astype(str).str.upper().isin(syms)].copy()

    rows=[]
    for _,g in daily.groupby(daily["symbol"].astype(str).str.upper()):
        z=compute_symbol(g,asof)
        if z:rows.append(z)

    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    df=pd.DataFrame(rows)
    if len(df):
        for c in ["ret_20_chart","ret_60_chart","ret_120_chart"]:
            df[c+"_rank"]=df[c].rank(pct=True,method="average")
        df["chart_trigger_score"]=(
            0.20*df["price_gt_dma50_prev"].astype(float)
            +0.12*df["dma50_slope_positive"].astype(float)
            +0.10*df["dma200_flat_or_positive"].astype(float)
            +0.12*df["near_or_breaking_60d_high"].astype(float)
            +0.12*(pd.to_numeric(df["up_down_volume_ratio_20"],errors="coerce").fillna(1).clip(0,2)/2)
            +0.10*(pd.to_numeric(df["volume_breakout_ratio"],errors="coerce").fillna(1).clip(0,3)/3)
            +0.08*df["base_compression_flag"].astype(float)
            +0.16*pd.to_numeric(df["ret_60_chart_rank"],errors="coerce").fillna(0.5)
        ).clip(0,1)
    df.to_csv(out,index=False)
    summary={
        "asof":str(asof.date()),
        "symbols_requested":len(syms),
        "symbols_extracted":int(len(df)),
        "dma50_coverage":float(df["dma50_prev"].notna().mean()) if len(df) else 0,
        "dma200_coverage":float(df["dma200_prev"].notna().mean()) if len(df) else 0,
        "recovery_setup_count":int((df["price_gt_dma50_prev"]&df["price_lt_dma200_prev"]).sum()) if len(df) else 0,
        "trend_confirmation_count":int(df["price_above_both_dma"].sum()) if len(df) else 0,
    }
    json.dump(summary,open(out.parent/"chart_trigger_summary.json","w"),indent=2)
    print(json.dumps(summary,indent=2))

if __name__=="__main__":
    main()
