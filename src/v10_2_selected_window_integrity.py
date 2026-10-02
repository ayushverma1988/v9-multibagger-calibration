from __future__ import annotations

import argparse, json, re
from pathlib import Path

import numpy as np
import pandas as pd

import calibrate_v9_2 as base
import v10_2_market_integrity as market_integrity


CA_RE = re.compile(
    r"\bsplit\b|\bbonus\b|\bdemerger\b|de-merger|\bmerger\b|"
    r"scheme of arrangement|sub[- ]?division|subdivision|face value|"
    r"consolidation of shares|capital reduction|corporate action",
    re.I,
)


def norm_str(v):
    if pd.isna(v):
        return None
    x=str(v).strip().upper()
    return None if x in {"","NAN","NONE","<NA>","NULL"} else x


def prepare_events(path):
    ev=pd.read_parquet(path)
    ev["published_ts"]=pd.to_datetime(ev["published_ts"],utc=True,errors="coerce")
    ev["event_date"]=ev["published_ts"].dt.tz_convert("Asia/Kolkata").dt.tz_localize(None).dt.normalize()
    ev["symbol_norm"]=ev.get("symbol",pd.Series(index=ev.index,dtype=object)).map(norm_str)
    ev["isin_norm"]=ev.get("isin",pd.Series(index=ev.index,dtype=object)).map(norm_str)
    txt=(
        ev.get("headline",pd.Series("",index=ev.index)).fillna("").astype(str)+" "+
        ev.get("details",pd.Series("",index=ev.index)).fillna("").astype(str)
    )
    et=ev.get("event_type",pd.Series("",index=ev.index)).fillna("").astype(str).str.lower()
    ev["is_corporate_action"]=(et=="corporate_action") | txt.str.contains(CA_RE,na=False)
    return ev[ev["is_corporate_action"]].copy()


def match_corporate_action(ca, jump_date, symbol, isin):
    jd=pd.Timestamp(jump_date).normalize()
    lo,hi=jd-pd.Timedelta(days=365),jd+pd.Timedelta(days=14)
    q=ca[(ca["event_date"]>=lo)&(ca["event_date"]<=hi)].copy()

    isin=norm_str(isin); sym=norm_str(symbol)
    if isin:
        qi=q[q["isin_norm"]==isin]
        if len(qi):
            q=qi
        else:
            q=q[q["symbol_norm"]==sym]
    else:
        q=q[q["symbol_norm"]==sym]

    if q.empty:
        return {
            "ca_match":False,"ca_event_count":0,"nearest_ca_event_date":None,
            "nearest_ca_event_type":None,"nearest_ca_headline":None,
            "days_from_nearest_ca_event":np.nan,
        }

    q=q.assign(abs_days=(q["event_date"]-jd).abs().dt.days)
    r=q.sort_values(["abs_days","event_date"]).iloc[0]
    return {
        "ca_match":True,
        "ca_event_count":int(len(q)),
        "nearest_ca_event_date":str(pd.Timestamp(r["event_date"]).date()),
        "nearest_ca_event_type":str(r.get("event_type","")),
        "nearest_ca_headline":str(r.get("headline",""))[:300],
        "days_from_nearest_ca_event":int((pd.Timestamp(r["event_date"])-jd).days),
    }


def classify_jump(adj_ret, raw_ret, ca_match):
    ar=float(adj_ret) if np.isfinite(adj_ret) else np.nan
    rr=float(raw_ret) if np.isfinite(raw_ret) else np.nan
    if np.isfinite(ar) and np.isfinite(rr):
        if abs(ar-rr)>=0.25 and abs(rr)<0.30:
            return "adjustment_factor_discontinuity"
        if abs(rr)>=0.50 and ca_match:
            return "corporate_action_or_restructuring"
        if abs(rr)>=0.50:
            return "raw_market_move_or_unresolved"
    return "other"


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--oos",required=True)
    ap.add_argument("--events",required=True)
    ap.add_argument("--legacy-dir",required=True)
    ap.add_argument("--output",required=True)
    ap.add_argument("--resolve-split-bonus-symbol-history",action="store_true")
    args=ap.parse_args()

    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)

    oos=pd.read_parquet(args.oos)
    oos["date"]=pd.to_datetime(oos["date"])
    if "selected_v941" not in oos.columns:
        raise ValueError("selected_v941 missing from frozen OOS artifact")
    sel=oos[oos["selected_v941"]==True].copy()
    sel["symbol_norm"]=sel["symbol"].map(norm_str)
    sel["isin_norm"]=sel.get("isin",pd.Series(index=sel.index,dtype=object)).map(norm_str)

    if args.resolve_split_bonus_symbol_history:
        market_integrity.install_on_base()
    market=base.load_market(2003,2026,args.legacy_dir)
    market["date"]=pd.to_datetime(market["date"])
    market=market.sort_values(["symbol","date"]).copy()
    market["symbol_norm"]=market["symbol"].map(norm_str)
    market["isin_norm"]=market.get("isin",pd.Series(index=market.index,dtype=object)).map(norm_str)
    market["adj_ret"]=market.groupby("symbol_norm")["adj_close"].pct_change()
    market["raw_ret"]=market.groupby("symbol_norm")["close"].pct_change()
    market["abs_adj_ret"]=market["adj_ret"].abs()
    market["is_jump50"]=np.isfinite(market["adj_ret"]) & (market["abs_adj_ret"]>=.50)

    ca=prepare_events(args.events)

    jump_records=[]
    selected_symbols=set(sel["symbol_norm"].dropna())
    sj=market[market["is_jump50"] & market["symbol_norm"].isin(selected_symbols)].copy()
    for r in sj.itertuples(index=False):
        m=match_corporate_action(ca,r.date,r.symbol,getattr(r,"isin",None))
        rec={
            "jump_date":pd.Timestamp(r.date),
            "symbol":r.symbol,
            "isin":getattr(r,"isin",None),
            "adj_ret":float(r.adj_ret),
            "raw_ret":float(r.raw_ret) if np.isfinite(r.raw_ret) else np.nan,
            "abs_adj_ret":float(r.abs_adj_ret),
            **m,
        }
        rec["jump_class"]=classify_jump(rec["adj_ret"],rec["raw_ret"],rec["ca_match"])
        jump_records.append(rec)
    jump_df=pd.DataFrame(jump_records)
    if len(jump_df):
        jump_df.to_csv(out/"selected_symbol_jump_classification.csv",index=False)

    # Trading-day-relative exposure for every selected OOS row.
    exposure=[]
    groups={s:g.reset_index(drop=True) for s,g in market.groupby("symbol_norm",sort=False)}
    for srow in sel.itertuples(index=False):
        sym=norm_str(srow.symbol)
        g=groups.get(sym)
        if g is None or g.empty:
            continue
        dates=g["date"].to_numpy(dtype="datetime64[ns]")
        sd=np.datetime64(pd.Timestamp(srow.date))
        pos=int(np.searchsorted(dates,sd,side="right")-1)
        if pos<0:
            continue
        jidx=np.flatnonzero(g["is_jump50"].to_numpy())
        for jp in jidx:
            off=int(jp-pos)
            windows=[]
            if -252<=off<=0: windows.append("feature_252")
            if 1<=off<=126: windows.append("label_y6_126")
            if 1<=off<=252: windows.append("label_y12_252")
            if 1<=off<=504: windows.append("label_y24_504")
            if not windows:
                continue
            jr=g.iloc[jp]
            m=match_corporate_action(ca,jr["date"],jr["symbol"],jr.get("isin",None))
            cls=classify_jump(jr["adj_ret"],jr["raw_ret"],m["ca_match"])
            for w in windows:
                exposure.append({
                    "selected_date":pd.Timestamp(srow.date),
                    "symbol":srow.symbol,
                    "selected_isin":getattr(srow,"isin",None),
                    "jump_date":pd.Timestamp(jr["date"]),
                    "trading_day_offset":off,
                    "window":w,
                    "adj_ret":float(jr["adj_ret"]),
                    "raw_ret":float(jr["raw_ret"]) if np.isfinite(jr["raw_ret"]) else np.nan,
                    "jump_class":cls,
                    **m,
                })

    ex=pd.DataFrame(exposure)
    if len(ex):
        ex=ex.drop_duplicates(["selected_date","symbol","jump_date","window"])
        ex.to_csv(out/"selected_window_jump_exposures.csv",index=False)

    def rows_exposed(window, classes=None):
        if ex.empty:return 0
        q=ex[ex["window"]==window]
        if classes is not None:q=q[q["jump_class"].isin(classes)]
        return int(q[["selected_date","symbol"]].drop_duplicates().shape[0])

    blocking_classes={"adjustment_factor_discontinuity","corporate_action_or_restructuring"}
    summary={
        "model":"V10.2 selected-window residual integrity audit",
        "selected_oos_rows":int(len(sel)),
        "selected_symbols":int(sel["symbol_norm"].nunique()),
        "selected_symbol_jumps_50pct":int(len(jump_df)),
        "jump_class_counts":jump_df["jump_class"].value_counts(dropna=False).to_dict() if len(jump_df) else {},
        "corporate_action_matches":int(jump_df["ca_match"].sum()) if len(jump_df) else 0,
        "selected_rows_exposed_feature_252_any_jump":rows_exposed("feature_252"),
        "selected_rows_exposed_y6_any_jump":rows_exposed("label_y6_126"),
        "selected_rows_exposed_y12_any_jump":rows_exposed("label_y12_252"),
        "selected_rows_exposed_y24_any_jump":rows_exposed("label_y24_504"),
        "selected_rows_exposed_feature_252_blocking":rows_exposed("feature_252",blocking_classes),
        "selected_rows_exposed_y6_blocking":rows_exposed("label_y6_126",blocking_classes),
        "selected_rows_exposed_y12_blocking":rows_exposed("label_y12_252",blocking_classes),
        "selected_rows_exposed_y24_blocking":rows_exposed("label_y24_504",blocking_classes),
        "primary_integrity_pass":bool(
            rows_exposed("feature_252",blocking_classes)==0 and
            rows_exposed("label_y6_126",blocking_classes)==0
        ),
        "primary_scope":"V9.4.1 includes ret_252, so the feature-integrity check covers trading-day offsets -252 through 0 (253 price rows); the primary y6 outcome is 126 trading sessions. y12/y24 are diagnostic.",
        "note":"No model parameters or outcome-based selection rules are changed by this audit.",
    }
    json.dump(summary,open(out/"summary.json","w"),indent=2,default=str)
    print(json.dumps(summary,indent=2,default=str))


if __name__=="__main__":
    main()
