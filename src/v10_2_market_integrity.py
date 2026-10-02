from __future__ import annotations

from functools import lru_cache
import numpy as np
import pandas as pd

import calibrate_v9_2 as base

LAST_ACTION_RESOLUTION_AUDIT=[]
LAST_SYMBOL_STITCH_AUDIT=[]
LAST_VOLUME_NORMALIZATION_AUDIT=[]


def _norm(v):
    if pd.isna(v):
        return None
    s=str(v).strip().upper()
    return None if s in {"","NAN","NONE","<NA>","NULL"} else s


@lru_cache(maxsize=1)
def load_symbol_history() -> pd.DataFrame:
    hp=base.hf_hub_download(base.REPO,"symbol_history/nse.parquet",repo_type="dataset")
    h=pd.read_parquet(hp).copy()
    h["valid_from"]=pd.to_datetime(h["valid_from"],errors="coerce")
    h["valid_to"]=pd.to_datetime(h["valid_to"],errors="coerce")
    h["isin_norm"]=h["isin"].map(_norm)
    h["symbol_norm"]=h["symbol"].map(_norm)
    return h


def resolve_action_table(start_year:int,end_year:int,types=None) -> pd.DataFrame:
    """Return structured NSE actions with symbol resolved point-in-time by ISIN.

    This protects action joins from later symbol/name changes. Resolution is
    metadata-only and never uses returns or outcome labels.
    """
    global LAST_ACTION_RESOLUTION_AUDIT
    a=base.load_actions(start_year,end_year).copy()
    if a.empty:
        LAST_ACTION_RESOLUTION_AUDIT=[]
        return a
    a["type"]=a["type"].fillna("").astype(str).str.lower()
    a["ex_date"]=pd.to_datetime(a["ex_date"],errors="coerce")
    if types is not None:
        a=a[a["type"].isin(set(types))].copy()

    try:
        h=load_symbol_history()
    except Exception as exc:
        print("WARNING: symbol-history resolution unavailable:",repr(exc),flush=True)
        h=pd.DataFrame()

    resolved=[]; resolved_isins=[]; audit=[]
    for r in a.itertuples(index=False):
        orig=_norm(getattr(r,"symbol",None))
        isin=_norm(getattr(r,"isin",None))
        ex=pd.Timestamp(getattr(r,"ex_date"))
        rs=orig; risin=isin; method="action_symbol"

        if len(h) and isin:
            # First choice: the action ISIN is genuinely active on the event date.
            q=h[
                (h["isin_norm"]==isin)
                & (h["valid_from"]<=ex)
                & ((h["valid_to"].isna()) | (h["valid_to"]>=ex))
            ]
            if len(q):
                z=q.sort_values("valid_from").iloc[-1]
                rs=_norm(z["symbol"]) or orig
                risin=_norm(z["isin"]) or isin
                method="isin_symbol_history"
            else:
                # Some action files are backfilled with a later ISIN. Bridge
                # through the nearest symbol used by that ISIN, then find the
                # predecessor ISIN/symbol that was actually active at ex-date.
                q2=h[h["isin_norm"]==isin].copy()
                if len(q2):
                    q2=q2.assign(
                        dist=np.where(
                            q2["valid_from"]>=ex,
                            (q2["valid_from"]-ex).dt.days,
                            (ex-q2["valid_to"].fillna(q2["valid_from"])).dt.days.abs()+100000
                        )
                    )
                    bridge=q2.sort_values(["dist","valid_from"]).iloc[0]
                    bridge_symbol=_norm(bridge["symbol"])
                    if bridge_symbol:
                        q3=h[
                            (h["symbol_norm"]==bridge_symbol)
                            & (h["valid_from"]<=ex)
                            & ((h["valid_to"].isna()) | (h["valid_to"]>=ex))
                        ]
                        if len(q3):
                            z=q3.sort_values("valid_from").iloc[-1]
                            rs=_norm(z["symbol"]) or bridge_symbol
                            risin=_norm(z["isin"]) or isin
                            method="future_isin_symbol_bridge_to_event_identity"
                        else:
                            rs=bridge_symbol
                            method="future_isin_symbol_bridge_no_event_interval"

        resolved.append(rs)
        resolved_isins.append(risin)
        if rs!=orig or risin!=isin or method!="action_symbol":
            audit.append({
                "ex_date":str(ex.date()) if pd.notna(ex) else None,
                "type":getattr(r,"type",None),
                "action_isin":isin,
                "resolved_isin":risin,
                "action_symbol":orig,
                "resolved_symbol":rs,
                "resolution_method":method,
                "raw_subject":getattr(r,"raw_subject",None),
            })
    a["resolved_symbol"]=resolved
    a["resolved_isin"]=resolved_isins
    LAST_ACTION_RESOLUTION_AUDIT=audit
    return a


def _action_factor(r) -> float:
    try:
        if r["type"]=="split" and pd.notna(r.get("face_value_from")) and pd.notna(r.get("face_value_to")) and float(r["face_value_from"])>0:
            return float(r["face_value_to"])/float(r["face_value_from"])
        if r["type"]=="bonus" and pd.notna(r.get("ratio_num")) and pd.notna(r.get("ratio_den")):
            n,d=float(r["ratio_num"]),float(r["ratio_den"])
            return d/(n+d) if n+d>0 else 1.0
    except Exception:
        pass
    return 1.0


def split_bonus_action_table(start_year:int,end_year:int) -> pd.DataFrame:
    a=resolve_action_table(start_year,end_year,types={"split","bonus"})
    if a.empty:
        return a
    a=a.copy()
    a["factor"]=a.apply(_action_factor,axis=1)
    return a[(a["factor"]>0)&(a["factor"]<1.01)].copy()


def apply_split_bonus_adjustment_resolved(df: pd.DataFrame, start_year: int, end_year: int) -> pd.Series:
    """Split/bonus-only price normalization with symbol-history identity repair.

    Dividends are deliberately excluded. Merger/demerger economics are not
    forced into a synthetic multiplicative factor.
    """
    a=split_bonus_action_table(start_year,end_year)
    if a.empty:
        return df["close"].astype(float).copy()

    amap={k:g[["ex_date","factor"]].sort_values("ex_date") for k,g in a.groupby("resolved_symbol")}
    out=np.empty(len(df),dtype=float)
    for sym,idx0 in df.groupby("symbol",sort=False).groups.items():
        inds=np.asarray(list(idx0),dtype=int)
        dates=df.loc[inds,"date"].to_numpy(dtype="datetime64[ns]")
        close=df.loc[inds,"close"].to_numpy(float)
        factors=np.ones(len(inds),dtype=float)
        g=amap.get(_norm(sym))
        if g is not None:
            for d,ff in zip(g["ex_date"].to_numpy(dtype="datetime64[ns]"),g["factor"].to_numpy(float)):
                pos=np.searchsorted(dates,d,side="left")
                if pos>0:
                    factors[:pos]*=float(ff)
        out[inds]=close*factors
    return pd.Series(out,index=df.index)


def normalize_split_bonus_volume(df:pd.DataFrame,start_year:int,end_year:int) -> pd.DataFrame:
    """Normalize share volume across splits/bonuses.

    A 10:1 split has price factor 0.1; pre-split share volume is multiplied by
    1/0.1 so volume-acceleration features do not see a mechanical 10x surge.
    Monetary turnover is left unchanged.
    """
    global LAST_VOLUME_NORMALIZATION_AUDIT
    x=df.copy()
    a=split_bonus_action_table(start_year,end_year)
    if a.empty or "volume" not in x:
        LAST_VOLUME_NORMALIZATION_AUDIT=[]
        return x
    amap={k:g[["ex_date","factor","type"]].sort_values("ex_date") for k,g in a.groupby("resolved_symbol")}
    audit=[]
    out=x["volume"].astype(float).to_numpy(copy=True)
    for sym,idx0 in x.groupby("symbol",sort=False).groups.items():
        inds=np.asarray(list(idx0),dtype=int)
        dates=x.loc[inds,"date"].to_numpy(dtype="datetime64[ns]")
        vol=x.loc[inds,"volume"].to_numpy(float)
        factors=np.ones(len(inds),dtype=float)
        g=amap.get(_norm(sym))
        if g is not None:
            for _,r in g.iterrows():
                d=np.datetime64(pd.Timestamp(r["ex_date"]))
                ff=float(r["factor"])
                pos=np.searchsorted(dates,d,side="left")
                if pos>0 and ff>0:
                    factors[:pos]/=ff
                    audit.append({
                        "symbol":_norm(sym),"ex_date":str(pd.Timestamp(r["ex_date"]).date()),
                        "type":r["type"],"price_factor":ff,"pre_event_volume_factor":1.0/ff,
                    })
        out[inds]=vol*factors
    x["volume"]=out
    LAST_VOLUME_NORMALIZATION_AUDIT=audit
    return x


def stitch_symbol_changes_same_isin(df:pd.DataFrame) -> pd.DataFrame:
    """Stitch pure symbol/name changes when ISIN is unchanged.

    Same-ISIN symbol changes are identifier changes, not economic events.
    Historical rows are mapped to the latest observed symbol for that ISIN so
    feature/label history remains continuous. ISIN changes are *not* stitched
    here; split-related ISIN changes remain continuous naturally when symbol is
    unchanged, while mergers/demergers are handled as structural boundaries.
    """
    global LAST_SYMBOL_STITCH_AUDIT
    x=df.copy()
    x["_isin_norm"]=x.get("isin",pd.Series(index=x.index,dtype=object)).map(_norm)
    x["_symbol_original"]=x["symbol"].astype(str)
    valid=x[x["_isin_norm"].notna()].copy()
    if valid.empty:
        LAST_SYMBOL_STITCH_AUDIT=[]
        return x.drop(columns=["_isin_norm","_symbol_original"])

    latest=(
        valid.sort_values(["_isin_norm","date"])
        .groupby("_isin_norm",as_index=False)
        .tail(1)[["_isin_norm","symbol"]]
        .rename(columns={"symbol":"_canonical_symbol"})
    )
    mp=dict(zip(latest["_isin_norm"],latest["_canonical_symbol"]))
    audit=[]
    for isin,g in valid.groupby("_isin_norm"):
        syms=sorted({_norm(v) for v in g["symbol"] if _norm(v)})
        if len(syms)>1:
            canon=_norm(mp.get(isin))
            audit.append({
                "isin":isin,"symbols":syms,"canonical_symbol":canon,
                "first_date":str(pd.Timestamp(g["date"].min()).date()),
                "last_date":str(pd.Timestamp(g["date"].max()).date()),
            })
    canon=x["_isin_norm"].map(mp)
    x["symbol"]=canon.where(canon.notna(),x["symbol"])

    # Overlap around a rename can produce duplicate canonical rows. Keep the
    # more liquid row deterministically; this uses only same-day market data.
    if x.duplicated(["date","symbol","series"]).any():
        x=x.sort_values(["date","symbol","series","turnover"],ascending=[True,True,True,False])
        x=x.drop_duplicates(["date","symbol","series"],keep="first")
    LAST_SYMBOL_STITCH_AUDIT=audit
    return x.drop(columns=["_isin_norm","_symbol_original"]).sort_values(["symbol","date"]).reset_index(drop=True)


def structured_identity_events(start_year:int,end_year:int) -> pd.DataFrame:
    """Explicit structural events that must reset/censor continuity.

    Split/bonus are corrected multiplicatively and therefore excluded here.
    Merger/demerger cannot generally be represented by one price factor.
    Name/symbol changes are non-blocking when identity continuity is established
    through ISIN/symbol history.
    """
    a=resolve_action_table(start_year,end_year)
    if a.empty:
        return a
    structural={"merger","demerger"}
    name_like={"name_change","name change","symbol_change","symbol change"}
    z=a[a["type"].isin(structural|name_like)].copy()
    if len(z):
        z["identity_policy"]=np.where(
            z["type"].isin(structural),"structural_reset","identity_continuity"
        )
    return z


def install_on_base() -> None:
    base.apply_split_bonus_adjustment=apply_split_bonus_adjustment_resolved
