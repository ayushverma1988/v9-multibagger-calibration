from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

import calibrate_v9_2 as base
import calibrate_v9_4 as v94
import calibrate_v9_4_1 as v941


REPO=base.REPO


def norm(v):
    if pd.isna(v):
        return None
    s=str(v).strip().upper()
    return None if s in {"","NAN","NONE","<NA>","NULL"} else s


def load_bse_history(start_year=2024,end_year=2026):
    frames=[]
    for y in range(start_year,end_year+1):
        p=hf_hub_download(REPO,f"bse/year={y}/bse_{y}.parquet",repo_type="dataset")
        d=pd.read_parquet(p,columns=["date","symbol","series","isin","close","volume","turnover"])
        frames.append(d)
    x=pd.concat(frames,ignore_index=True)
    x["date"]=pd.to_datetime(x["date"])
    x=x[x["isin"].notna() & x["isin"].astype(str).str.startswith("INE")].copy()
    x["symbol"]=x["symbol"].astype(str).str.upper().str.strip()
    x=x.sort_values(["symbol","date","turnover"],ascending=[True,True,False])
    x=x.drop_duplicates(["date","symbol","series"],keep="first").reset_index(drop=True)
    return x


def load_bse_actions(start_year=2024,end_year=2026):
    z=[]
    for y in range(start_year,end_year+1):
        try:
            p=hf_hub_download(REPO,f"actions/bse_{y}.parquet",repo_type="dataset")
            z.append(pd.read_parquet(p))
        except Exception:
            pass
    if not z:
        return pd.DataFrame()
    a=pd.concat(z,ignore_index=True)
    a["ex_date"]=pd.to_datetime(a["ex_date"],errors="coerce")
    a["type"]=a["type"].fillna("").astype(str).str.lower()
    a["symbol_norm"]=a["symbol"].map(norm)
    a["isin_norm"]=a["isin"].map(norm)
    return a


def load_bse_symbol_history():
    p=hf_hub_download(REPO,"symbol_history/bse.parquet",repo_type="dataset")
    h=pd.read_parquet(p).copy()
    h["valid_from"]=pd.to_datetime(h["valid_from"],errors="coerce")
    h["valid_to"]=pd.to_datetime(h["valid_to"],errors="coerce")
    h["symbol_norm"]=h["symbol"].map(norm)
    h["isin_norm"]=h["isin"].map(norm)
    return h


def resolve_actions(a,h):
    if a.empty:
        return a
    rows=[]
    for r in a.itertuples(index=False):
        ex=pd.Timestamp(r.ex_date)
        orig_sym=norm(getattr(r,"symbol",None))
        orig_isin=norm(getattr(r,"isin",None))
        rs=orig_sym; ri=orig_isin; method="action_identity"
        q=pd.DataFrame()
        if orig_isin:
            q=h[
                (h["isin_norm"]==orig_isin)
                & (h["valid_from"]<=ex)
                & ((h["valid_to"].isna()) | (h["valid_to"]>=ex))
            ]
        if q.empty and orig_sym:
            q=h[
                (h["symbol_norm"]==orig_sym)
                & (h["valid_from"]<=ex)
                & ((h["valid_to"].isna()) | (h["valid_to"]>=ex))
            ]
        if len(q):
            z=q.sort_values("valid_from").iloc[-1]
            rs=norm(z["symbol"]) or rs
            ri=norm(z["isin"]) or ri
            method="point_in_time_symbol_history"
        rec=r._asdict()
        rec.update(resolved_symbol=rs,resolved_isin=ri,identity_resolution=method)
        rows.append(rec)
    return pd.DataFrame(rows)


def action_factor(r):
    try:
        if r["type"]=="split" and pd.notna(r.get("face_value_from")) and pd.notna(r.get("face_value_to")) and float(r["face_value_from"])>0:
            return float(r["face_value_to"])/float(r["face_value_from"])
        if r["type"]=="bonus" and pd.notna(r.get("ratio_num")) and pd.notna(r.get("ratio_den")):
            n,d=float(r["ratio_num"]),float(r["ratio_den"])
            return d/(n+d) if n+d>0 else 1.0
    except Exception:
        pass
    return 1.0


def apply_bse_split_bonus(x,a):
    y=x.copy().sort_values(["symbol","date"]).reset_index(drop=True)
    sb=a[a["type"].isin(["split","bonus"])].copy()
    if len(sb):
        sb["factor"]=sb.apply(action_factor,axis=1)
        sb=sb[(sb["factor"]>0)&(sb["factor"]<1.01)]
    amap={k:g.sort_values("ex_date") for k,g in sb.groupby("resolved_symbol")} if len(sb) else {}

    adj=np.empty(len(y),float)
    vol=y["volume"].astype(float).to_numpy(copy=True)
    for sym,idx0 in y.groupby("symbol",sort=False).groups.items():
        inds=np.asarray(list(idx0),dtype=int)
        dates=y.loc[inds,"date"].to_numpy(dtype="datetime64[ns]")
        close=y.loc[inds,"close"].to_numpy(float)
        vv=y.loc[inds,"volume"].to_numpy(float)
        pf=np.ones(len(inds),float)
        vf=np.ones(len(inds),float)
        g=amap.get(norm(sym))
        if g is not None:
            for _,r in g.iterrows():
                ex=np.datetime64(pd.Timestamp(r["ex_date"]))
                f=float(r["factor"])
                pos=np.searchsorted(dates,ex,side="left")
                if pos>0:
                    pf[:pos]*=f
                    vf[:pos]/=f
        adj[inds]=close*pf
        vol[inds]=vv*vf
    y["adj_close"]=adj
    y["volume"]=vol
    return y,sb


def stitch_same_isin(x):
    y=x.copy()
    y["isin_norm"]=y["isin"].map(norm)
    latest=(
        y[y["isin_norm"].notna()]
        .sort_values(["isin_norm","date"])
        .groupby("isin_norm",as_index=False)
        .tail(1)[["isin_norm","symbol"]]
        .rename(columns={"symbol":"canonical_symbol"})
    )
    mp=dict(zip(latest["isin_norm"],latest["canonical_symbol"]))
    y["symbol_original"]=y["symbol"]
    y["symbol"]=y["isin_norm"].map(mp).fillna(y["symbol"])
    y=y.sort_values(["date","symbol","series","turnover"],ascending=[True,True,True,False])
    y=y.drop_duplicates(["date","symbol","series"],keep="first")
    audit=(
        y.groupby("isin_norm")
        .agg(symbols_before=("symbol_original",lambda s:"|".join(sorted(set(map(str,s))))),
             canonical_symbol=("symbol","last"),
             first_date=("date","min"),last_date=("date","max"))
        .reset_index()
    )
    audit=audit[audit["symbols_before"].str.contains("\\|",regex=True,na=False)].copy()
    return y.sort_values(["symbol","date"]).reset_index(drop=True),audit


def mark_bse_structural_integrity(x,a):
    y=x.copy().sort_values(["symbol","date"]).reset_index(drop=True)
    y["_blocking_event"]=False
    y["_blocking_type"]=""
    structural=a[a["type"].isin(["merger","demerger"])].copy()
    event_rows=[]
    for r in structural.itertuples(index=False):
        ex=pd.Timestamp(r.ex_date)
        ri=norm(getattr(r,"resolved_isin",None))
        rs=norm(getattr(r,"resolved_symbol",None))
        q=y[y["isin"].map(norm)==ri] if ri else pd.DataFrame()
        if q.empty and rs:
            q=y[y["symbol"].map(norm)==rs]
        if q.empty:
            event_rows.append({"type":r.type,"ex_date":ex,"resolved_isin":ri,"resolved_symbol":rs,"mapped":False})
            continue
        q=q.sort_values("date")
        dates=q["date"].to_numpy(dtype="datetime64[ns]")
        pos=int(np.searchsorted(dates,np.datetime64(ex),side="left"))
        idx=None
        if pos<len(q):
            idx=q.index[pos]
        else:
            li=q.index[-1]
            if 0 <= (ex-pd.Timestamp(q.loc[li,"date"])).days <= 30:
                idx=li
        if idx is not None:
            y.at[idx,"_blocking_event"]=True
            y.at[idx,"_blocking_type"]=str(r.type)
            event_rows.append({"type":r.type,"ex_date":ex,"resolved_isin":ri,"resolved_symbol":rs,"mapped":True,"mapped_date":y.at[idx,"date"]})
        else:
            event_rows.append({"type":r.type,"ex_date":ex,"resolved_isin":ri,"resolved_symbol":rs,"mapped":False})

    clean=pd.Series(True,index=y.index,dtype=bool)
    for _,inds0 in y.groupby("symbol",sort=False).groups.items():
        inds=list(inds0)
        b=y.loc[inds,"_blocking_event"].astype(int)
        dirty=b.rolling(253,min_periods=1).max().astype(bool)
        clean.loc[inds]=~dirty.to_numpy()
    y["integrity_feature_clean"]=clean

    # Any residual split-like discontinuity after structured split/bonus
    # normalization invalidates the current 252-session feature window.
    y["_adj_ret"]=y.groupby("symbol")["adj_close"].pct_change()
    y["_raw_ret"]=y.groupby("symbol")["close"].pct_change()
    residual=(
        np.isfinite(y["_adj_ret"]) & np.isfinite(y["_raw_ret"])
        & ((y["_adj_ret"]-y["_raw_ret"]).abs()>=0.25)
        & (y["_raw_ret"].abs()<0.30)
    )
    y["_unresolved_adjustment"]=residual
    for _,inds0 in y.groupby("symbol",sort=False).groups.items():
        inds=list(inds0)
        b=y.loc[inds,"_unresolved_adjustment"].astype(int)
        dirty=b.rolling(253,min_periods=1).max().astype(bool)
        clean.loc[inds]=clean.loc[inds].to_numpy() & (~dirty.to_numpy())
    y["integrity_feature_clean"]=clean
    return y,pd.DataFrame(event_rows)


def find_file(root,name):
    hits=list(Path(root).rglob(name))
    if not hits:
        raise FileNotFoundError(f"{name} not found under {root}")
    return hits[0]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--accepted-dir",required=True)
    ap.add_argument("--output",required=True)
    ap.add_argument("--config",default="config_v9_4_1.json")
    args=ap.parse_args()
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    cfg=json.load(open(args.config))
    base.cfg_h24=int(cfg["label_days"]["y24"])

    data=pd.read_parquet(find_file(args.accepted_dir,"snapshot_dataset.parquet"))
    oos=pd.read_parquet(find_file(args.accepted_dir,"oos_predictions.parquet"))
    nse=pd.read_csv(find_file(args.accepted_dir,"current_selection.csv"))
    model_summary=json.load(open(find_file(args.accepted_dir,"summary.json")))

    print("Loading and normalizing BSE history...",flush=True)
    bse=load_bse_history(2024,2026)
    actions=resolve_actions(load_bse_actions(2024,2026),load_bse_symbol_history())
    bse,sb=apply_bse_split_bonus(bse,actions)
    bse,rename_audit=stitch_same_isin(bse)
    bse,struct_audit=mark_bse_structural_integrity(bse,actions)
    bse=base.add_features(bse)

    # Exclude current rows whose feature history crosses a merger/demerger or
    # unresolved adjustment discontinuity, without altering model parameters.
    last=pd.Timestamp(bse["date"].max())
    dirty_current=bse[(bse["date"]==last)&(~bse["integrity_feature_clean"])][
        ["date","symbol","isin","close","_blocking_type"]
    ].copy()
    bse.loc[(bse["date"]==last)&(~bse["integrity_feature_clean"]),"avg_turnover_63"]=0.0

    best100,_=base.evaluate_calibrators(oos,"y6","p_raw")
    bestdd,_=base.evaluate_calibrators(oos,"dd30_6m","p_dd30_raw")
    _,threshold_info=v94.calibrate_thresholds(oos.copy())

    print("Scoring BSE with frozen NSE-trained models...",flush=True)
    # Bootstrap confidence bands are diagnostic only and are not consumed by
    # the V9.4.1 selector. Disable them for the BSE transfer run so current
    # scoring does not repeat 500 block-resamples without changing selection.
    score_cfg=dict(cfg)
    score_cfg["bootstrap_blocks"]=0
    bcur=base.fit_current(data,bse,oos,best100,bestdd,score_cfg)
    tcur=v94.current_threshold_predictions(data,bse,oos,threshold_info,score_cfg)

    spec=dict(model_summary["production_risk_config"])
    shares_dict=model_summary["production_ladder_shares_within_upside_weight"]
    shares=(float(shares_dict["p100"]),float(shares_dict["p50"]),float(shares_dict["p25"]))
    gates=model_summary["production_signal_quality"]
    bcur=v941.build_current_v941(bcur,tcur,oos,threshold_info,spec,shares,gates,cfg)
    bcur["exchange"]="BSE"
    bcur["cross_exchange_extrapolation"]=True
    bcur["validation_domain"]="NSE-trained; BSE current transfer"

    # One current BSE representation per ISIN: highest 63d traded value wins.
    bcur=bcur.sort_values(["isin","avg_turnover_63"],ascending=[True,False])
    bcur=bcur.drop_duplicates("isin",keep="first").reset_index(drop=True)
    bcur.to_csv(out/"bse_current_predictions.csv",index=False)

    # Accepted NSE current predictions remain unchanged.
    nse["exchange"]="NSE"
    nse["cross_exchange_extrapolation"]=False
    nse["validation_domain"]="NSE validated"
    nse["isin"]=nse["isin"].astype(str)

    # Prefer the validated NSE representation for dual-listed eligible ISINs.
    combined=pd.concat([nse,bcur],ignore_index=True,sort=False)
    combined["_exchange_pref"]=combined["exchange"].map({"NSE":0,"BSE":1}).fillna(2)
    combined=combined.sort_values(["isin","_exchange_pref","avg_turnover_63"],ascending=[True,True,False])
    before=len(combined)
    dual=combined[combined.duplicated("isin",keep=False)]["isin"].nunique()
    combined=combined.drop_duplicates("isin",keep="first").drop(columns="_exchange_pref").reset_index(drop=True)

    # Re-run only the already-frozen selector over the deduplicated combined
    # candidate set. No weights/gates are changed.
    sel=v94.select_ladder_topk(combined,spec,shares,int(cfg.get("selection_k",10)))
    combined["selected_v103"]=False
    combined["selection_score_v103"]=np.nan
    combined["selection_rank_v103"]=np.nan
    for rank,idx in enumerate(sel.index.tolist(),start=1):
        combined.loc[idx,"selected_v103"]=True
        combined.loc[idx,"selection_score_v103"]=float(sel.loc[idx,"selection_score_v94"])
        combined.loc[idx,"selection_rank_v103"]=rank

    combined=combined.sort_values(
        ["selected_v103","selection_rank_v103","p100_cal"],
        ascending=[False,True,False],na_position="last"
    ).reset_index(drop=True)
    combined.to_csv(out/"combined_nse_bse_current_universe.csv",index=False)
    top=combined[combined["selected_v103"]].head(10).copy()
    top.to_csv(out/"combined_nse_bse_top10.csv",index=False)

    # Audits
    sb.to_csv(out/"bse_split_bonus_actions_resolved.csv",index=False)
    rename_audit.to_csv(out/"bse_symbol_name_change_audit.csv",index=False)
    struct_audit.to_csv(out/"bse_merger_demerger_audit.csv",index=False)
    dirty_current.to_csv(out/"bse_current_integrity_exclusions.csv",index=False)

    bse_only=int((combined["exchange"]=="BSE").sum())
    nse_only=int((combined["exchange"]=="NSE").sum())
    top_bse=int((top["exchange"]=="BSE").sum())
    pdiag={}
    for ex,g in combined.groupby("exchange"):
        pdiag[ex]={
            "n":int(len(g)),
            "p100_mean":float(g["p100_cal"].mean()),
            "p100_median":float(g["p100_cal"].median()),
            "p100_p95":float(g["p100_cal"].quantile(.95)),
            "dd30_mean":float(g["p_dd30_cal"].mean()),
        }

    summary={
        "stage":"V10.3 NSE+BSE current-universe extension",
        "model_weights_changed":False,
        "selector_rules_changed":False,
        "nse_validation_source":"accepted V10.2 run 37059786518",
        "bse_history_available":"2024-01-01 through 2026-10-01",
        "bse_backtest_status":"insufficient history for an 18-fold independent BSE validation; BSE-only names are explicitly tagged transfer/extrapolation",
        "current_date":str(last.date()),
        "nse_eligible_rows":int(len(nse)),
        "bse_eligible_rows_before_isin_dedupe":int(len(bcur)),
        "combined_rows_before_cross_exchange_dedupe":int(before),
        "dual_listed_isins_preferred_to_nse":int(dual),
        "combined_unique_isins":int(len(combined)),
        "combined_nse_rows":nse_only,
        "combined_bse_only_rows":bse_only,
        "bse_current_integrity_exclusions":int(len(dirty_current)),
        "top10_bse_only_count":top_bse,
        "probability_distribution_diagnostic":pdiag,
        "production_risk_config":spec,
        "production_ladder_shares":{"p100":shares[0],"p50":shares[1],"p25":shares[2]},
        "dedupe_rule":"ISIN; prefer eligible NSE listing, otherwise highest-liquidity BSE representation",
        "bse_corporate_action_rules":{
            "split_bonus":"point-in-time BSE symbol history resolution; price and share-volume normalization",
            "name_symbol_change":"same ISIN stitched to continuous history",
            "merger_demerger":"structural feature reset; current row excluded if event lies inside prior 252 sessions",
        },
        "promotion_note":"This extends current coverage only. It does not claim BSE-specific historical validation equivalent to NSE.",
    }
    json.dump(summary,open(out/"summary.json","w"),indent=2,default=str)
    print(json.dumps(summary,indent=2,default=str))
    cols=[c for c in ["selection_rank_v103","exchange","symbol","isin","close","p100_cal","p_dd30_cal","selection_score_v103","cross_exchange_extrapolation"] if c in top]
    print(top[cols].to_string(index=False),flush=True)


if __name__=="__main__":
    main()
