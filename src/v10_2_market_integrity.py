from __future__ import annotations

import numpy as np
import pandas as pd

import calibrate_v9_2 as base

LAST_ACTION_RESOLUTION_AUDIT=[]


def apply_split_bonus_adjustment_resolved(df: pd.DataFrame, start_year: int, end_year: int) -> pd.Series:
    """Split/bonus-only adjusted close with action identity resolved through
    point-in-time NSE symbol history. Dividends and complex restructurings are
    deliberately not scaled, matching the V9.4.1 price-return definition.
    """
    global LAST_ACTION_RESOLUTION_AUDIT
    a=base.load_actions(start_year,end_year)
    if a.empty:
        LAST_ACTION_RESOLUTION_AUDIT=[]
        return df["close"].astype(float).copy()

    a=a[a["type"].isin(["split","bonus"])].copy()
    a["ex_date"]=pd.to_datetime(a["ex_date"])

    try:
        hp=base.hf_hub_download(base.REPO,"symbol_history/nse.parquet",repo_type="dataset")
        h=pd.read_parquet(hp)
        h["valid_from"]=pd.to_datetime(h["valid_from"])
        h["valid_to"]=pd.to_datetime(h["valid_to"])
        h["isin_norm"]=h["isin"].astype(str).str.upper().str.strip()
    except Exception as exc:
        print("WARNING: symbol-history resolution unavailable:",repr(exc),flush=True)
        h=pd.DataFrame()

    resolved=[]; audit=[]
    for r in a.itertuples(index=False):
        orig=str(getattr(r,"symbol","") or "").upper().strip()
        isin=str(getattr(r,"isin","") or "").upper().strip()
        ex=pd.Timestamp(getattr(r,"ex_date"))
        rs=orig; method="action_symbol"
        if len(h) and isin not in {"","NAN","NONE","<NA>"}:
            q=h[(h["isin_norm"]==isin)&(h["valid_from"]<=ex)&(h["valid_to"]>=ex)]
            if len(q):
                rs=str(q.sort_values("valid_from").iloc[-1]["symbol"]).upper().strip()
                method="isin_symbol_history"
        resolved.append(rs)
        if rs!=orig:
            audit.append({
                "ex_date":str(ex.date()),"type":getattr(r,"type",None),
                "isin":isin,"action_symbol":orig,"resolved_symbol":rs,
                "resolution_method":method,"raw_subject":getattr(r,"raw_subject",None),
            })
    a["resolved_symbol"]=resolved
    LAST_ACTION_RESOLUTION_AUDIT=audit

    def fac(r):
        try:
            if r["type"]=="split" and pd.notna(r.get("face_value_from")) and pd.notna(r.get("face_value_to")) and float(r["face_value_from"])>0:
                return float(r["face_value_to"])/float(r["face_value_from"])
            if r["type"]=="bonus" and pd.notna(r.get("ratio_num")) and pd.notna(r.get("ratio_den")):
                n,d=float(r["ratio_num"]),float(r["ratio_den"])
                return d/(n+d) if n+d>0 else 1.0
        except Exception:
            pass
        return 1.0

    a["factor"]=a.apply(fac,axis=1)
    a=a[(a["factor"]>0)&(a["factor"]<1.01)]
    amap={k:g[["ex_date","factor"]].sort_values("ex_date") for k,g in a.groupby("resolved_symbol")}

    out=np.empty(len(df),dtype=float)
    for sym,idx0 in df.groupby("symbol",sort=False).groups.items():
        inds=np.asarray(list(idx0),dtype=int)
        dates=df.loc[inds,"date"].to_numpy(dtype="datetime64[ns]")
        close=df.loc[inds,"close"].to_numpy(float)
        factors=np.ones(len(inds),dtype=float)
        g=amap.get(str(sym).upper().strip())
        if g is not None:
            for d,ff in zip(g["ex_date"].to_numpy(dtype="datetime64[ns]"),g["factor"].to_numpy(float)):
                pos=np.searchsorted(dates,d,side="left")
                if pos>0:
                    factors[:pos]*=float(ff)
        out[inds]=close*factors
    return pd.Series(out,index=df.index)


def install_on_base() -> None:
    base.apply_split_bonus_adjustment=apply_split_bonus_adjustment_resolved
