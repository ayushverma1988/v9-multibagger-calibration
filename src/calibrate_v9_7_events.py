from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


EVENT_TYPES = [
    "order_win","capacity_expansion","debt_reduction","credit_rating",
    "promoter_activity","pledge_change","dilution","buyback",
    "management_change","auditor_change","regulatory","litigation",
    "customer_supplier","earnings","corporate_action",
]

META_FEATURES = [
    "baseline_rank",
    "event_count_30d","event_count_90d",
    "positive_event_strength_90d","negative_event_strength_90d",
    "event_type_count_90d","days_since_event","days_since_positive_event",
    "days_since_negative_event",
] + [f"evt_{x}_90d" for x in EVENT_TYPES]


def security_key(symbol, isin):
    if pd.notna(isin) and str(isin).strip() and str(isin).upper() not in {"NAN","NONE","<NA>"}:
        return "ISIN:" + str(isin).upper().strip()
    return "SYM:" + str(symbol).upper().strip()


def add_event_features(snapshots: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    s=snapshots.copy()
    s["date"]=pd.to_datetime(s["date"])
    if "isin" not in s.columns:
        s["isin"]=np.nan
    s["security_key"]=[security_key(a,b) for a,b in zip(s["symbol"],s["isin"])]

    e=events.copy()
    e["published_ts"]=pd.to_datetime(e["published_ts"],utc=True,errors="coerce")
    e=e.dropna(subset=["published_ts","security_key"]).copy()
    e["event_date"]=e["published_ts"].dt.tz_convert("Asia/Kolkata").dt.tz_localize(None).dt.normalize()
    e["event_direction"]=pd.to_numeric(e["event_direction"],errors="coerce").fillna(0.0)
    e["event_strength"]=pd.to_numeric(e["event_strength"],errors="coerce").fillna(0.0)

    out=[]
    base_cols=["date","symbol","isin","security_key"]
    for td,g in s.groupby("date",sort=True):
        td=pd.Timestamp(td).normalize()
        # Same-day announcements are excluded. Only events strictly before the
        # snapshot date are eligible.
        w90=e[(e["event_date"]<td)&(e["event_date"]>=td-pd.Timedelta(days=90))].copy()
        w30=w90[w90["event_date"]>=td-pd.Timedelta(days=30)].copy()

        keys=g[base_cols].copy()
        agg90=w90.groupby("security_key").agg(
            event_count_90d=("event_id","size"),
            positive_event_strength_90d=("event_strength",lambda z:0.0),
            negative_event_strength_90d=("event_strength",lambda z:0.0),
            event_type_count_90d=("event_type","nunique"),
            last_event_date=("event_date","max"),
        ).reset_index()

        if len(w90):
            pos=(w90["event_direction"]>0)
            neg=(w90["event_direction"]<0)
            ps=(w90.loc[pos].assign(v=lambda x:x["event_strength"]*x["event_direction"])
                .groupby("security_key")["v"].sum())
            ns=(w90.loc[neg].assign(v=lambda x:-x["event_strength"]*x["event_direction"])
                .groupby("security_key")["v"].sum())
            agg90["positive_event_strength_90d"]=agg90["security_key"].map(ps).fillna(0.0)
            agg90["negative_event_strength_90d"]=agg90["security_key"].map(ns).fillna(0.0)

        agg30=w30.groupby("security_key").size().rename("event_count_30d").reset_index()
        keys=keys.merge(agg90,on="security_key",how="left").merge(agg30,on="security_key",how="left")

        # Event-type counts over 90 days.
        if len(w90):
            ct=(w90.groupby(["security_key","event_type"]).size()
                .unstack(fill_value=0))
            for et in EVENT_TYPES:
                keys[f"evt_{et}_90d"]=keys["security_key"].map(ct[et] if et in ct.columns else pd.Series(dtype=float)).fillna(0.0)
        else:
            for et in EVENT_TYPES:
                keys[f"evt_{et}_90d"]=0.0

        keys["event_count_90d"]=keys["event_count_90d"].fillna(0.0)
        keys["event_count_30d"]=keys["event_count_30d"].fillna(0.0)
        keys["positive_event_strength_90d"]=keys["positive_event_strength_90d"].fillna(0.0)
        keys["negative_event_strength_90d"]=keys["negative_event_strength_90d"].fillna(0.0)
        keys["event_type_count_90d"]=keys["event_type_count_90d"].fillna(0.0)
        keys["days_since_event"]=(td-pd.to_datetime(keys["last_event_date"])).dt.days.astype(float)

        for label,mask in [
            ("positive",w90["event_direction"]>0),
            ("negative",w90["event_direction"]<0),
        ]:
            if len(w90):
                last=w90.loc[mask].groupby("security_key")["event_date"].max()
                keys[f"days_since_{label}_event"]=[
                    float((td-last.get(k)).days) if pd.notna(last.get(k)) else np.nan
                    for k in keys["security_key"]
                ]
            else:
                keys[f"days_since_{label}_event"]=np.nan
        out.append(keys.drop(columns=["last_event_date"],errors="ignore"))
    return pd.concat(out,ignore_index=True) if out else pd.DataFrame()


def spec_from_row(r):
    return {
        "pool_n":int(r.pool_n),
        "risk_drop":float(r.risk_drop),
        "w_safety":float(r.w_safety),
        "w_consensus":float(r.w_consensus),
        "baseline":bool(r.fell_back_to_v92),
    }


def survivor_pool(g,spec,k):
    q=g.dropna(subset=["p_cal","p_dd30_cal","model_dispersion"]).copy()
    if len(q)<k:
        return pd.DataFrame()
    if spec["baseline"]:
        q["selection_score"]=q["p_cal"]
        return q.sort_values(["selection_score","p_cal","model_dispersion"],ascending=[False,False,True])

    pool=q.sort_values(["p_cal","model_dispersion"],ascending=[False,True]).head(min(spec["pool_n"],len(q))).copy()
    if len(pool)<k:
        return pd.DataFrame()
    if spec["risk_drop"]>0:
        cutoff=float(pool["p_dd30_cal"].quantile(1.0-spec["risk_drop"]))
        pool=pool[pool["p_dd30_cal"]<=cutoff].copy()
    if len(pool)<k:
        return pd.DataFrame()

    pool["comp_alpha"]=pool["p_cal"].rank(pct=True,method="average")
    pool["comp_safety"]=pool["p_dd30_cal"].rank(pct=True,method="average",ascending=False)
    pool["comp_consensus"]=pool["model_dispersion"].rank(pct=True,method="average",ascending=False)
    wa=max(0.0,1.0-spec["w_safety"]-spec["w_consensus"])
    pool["selection_score"]=(
        wa*pool["comp_alpha"]
        +spec["w_safety"]*pool["comp_safety"]
        +spec["w_consensus"]*pool["comp_consensus"]
    )
    return pool.sort_values(["selection_score","p_cal","model_dispersion"],ascending=[False,False,True])


def fit_meta(train):
    m=Pipeline([
        ("impute",SimpleImputer(strategy="median",add_indicator=True)),
        ("scale",RobustScaler()),
        ("clf",LogisticRegression(
            C=0.10,class_weight="balanced",max_iter=3000,random_state=20261001
        )),
    ])
    m.fit(train[META_FEATURES],train["y6"].astype(int))
    return m


def agg(t):
    if t.empty:
        return None
    lift=np.clip(t["lift_2x"].replace([np.inf,-np.inf],np.nan).dropna(),0,10)
    return {
        "folds":int(t["date"].nunique()),
        "mean_precision_2x":float(t["precision_2x"].mean()),
        "median_precision_2x":float(t["precision_2x"].median()),
        "mean_capped_lift_2x":float(lift.mean()) if len(lift) else np.nan,
        "hit_fold_rate":float(t["hit"].mean()),
        "mean_dd30_rate":float(t["dd30_rate"].mean()),
    }


def retain(a,b,ratio=0.95):
    for f in ["mean_precision_2x","median_precision_2x","mean_capped_lift_2x","hit_fold_rate"]:
        av=a.get(f,np.nan); bv=b.get(f,np.nan)
        if not(np.isfinite(av) and np.isfinite(bv)):
            return False
        if bv>0 and av+1e-12<ratio*bv:
            return False
    return True


def utility(a,b):
    rs=[]
    for f in ["mean_precision_2x","mean_capped_lift_2x","hit_fold_rate"]:
        rs.append(a[f]/b[f] if b[f]>0 else 0.0)
    dd=(b["mean_dd30_rate"]-a["mean_dd30_rate"])/b["mean_dd30_rate"] if b["mean_dd30_rate"]>0 else 0.0
    return 0.45*rs[0]+0.35*rs[1]+0.20*rs[2]+0.10*dd


def evaluate_weight(pools,w,k=10):
    rows=[]
    for td,g in pools.groupby("date"):
        q=g.dropna(subset=["y6","selection_score"]).copy()
        if len(q)<k or q["p_event_meta"].notna().sum()<k:
            continue
        q["baseline_rank"]=q["selection_score"].rank(pct=True,method="average")
        q["event_rank"]=q["p_event_meta"].rank(pct=True,method="average").fillna(0.5)
        q["score_evt"]=(1-w)*q["baseline_rank"]+w*q["event_rank"]
        s=q.sort_values(["score_evt","baseline_rank","p_cal"],ascending=[False,False,False]).head(k)
        br=float(q["y6"].mean()); pr=float(s["y6"].mean())
        rows.append({
            "date":pd.Timestamp(td),"precision_2x":pr,
            "lift_2x":pr/br if br>0 else np.nan,
            "hit":float(pr>0),"dd30_rate":float(s["dd30_6m"].mean()),
        })
    return pd.DataFrame(rows)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--events",required=True)
    ap.add_argument("--snapshot",required=True)
    ap.add_argument("--oos",required=True)
    ap.add_argument("--chosen931",required=True)
    ap.add_argument("--config",required=True)
    ap.add_argument("--output",required=True)
    args=ap.parse_args()

    cfg=json.load(open(args.config))
    k=int(cfg.get("selection_k",10))
    min_train_rows=500
    min_train_pos=20
    min_prior_folds=8
    min_policy_folds=4
    min_gain=0.02
    weights=[0.0,0.02,0.05,0.10]

    events=pd.read_parquet(args.events)
    snap=pd.read_parquet(args.snapshot)
    snap["date"]=pd.to_datetime(snap["date"])
    oos=pd.read_parquet(args.oos)
    oos["date"]=pd.to_datetime(oos["date"])
    chosen=pd.read_csv(args.chosen931)
    chosen["date"]=pd.to_datetime(chosen["date"])
    cmap={pd.Timestamp(r.date):r for r in chosen.itertuples(index=False)}

    feat=add_event_features(snap[["date","symbol","isin"]].drop_duplicates(),events)
    merged=oos.merge(feat.drop(columns=["isin","security_key"],errors="ignore"),on=["date","symbol"],how="left")

    # y6 maturity date is sourced from the snapshot artifact.
    mature=snap[["date","symbol","y6_mature_date"]].drop_duplicates()
    mature["y6_mature_date"]=pd.to_datetime(mature["y6_mature_date"],errors="coerce")
    merged=merged.merge(mature,on=["date","symbol"],how="left")

    pools=[]
    for td,r in cmap.items():
        p=survivor_pool(merged[merged["date"]==td],spec_from_row(r),k)
        if p.empty:
            continue
        p["baseline_rank"]=p["selection_score"].rank(pct=True,method="average")
        pools.append(p)
    pools=pd.concat(pools,ignore_index=True)
    pools["p_event_meta"]=np.nan

    # Strictly forward meta predictions.
    for td in sorted(pools["date"].unique()):
        prior=pools[(pools["date"]<td)&pools["y6"].notna()].copy()
        prior=prior[pd.to_datetime(prior["y6_mature_date"],errors="coerce")<pd.Timestamp(td)]
        idx=pools.index[pools["date"]==td]
        if (
            prior["date"].nunique()<min_prior_folds
            or len(prior)<min_train_rows
            or int(prior["y6"].sum())<min_train_pos
            or prior["y6"].nunique()<2
        ):
            continue
        model=fit_meta(prior)
        pools.loc[idx,"p_event_meta"]=model.predict_proba(pools.loc[idx,META_FEATURES])[:,1]

    decisions=[]
    selected=[]
    for td in sorted(pools["date"].unique()):
        hist=pools[pools["date"]<td].copy()
        base=agg(evaluate_weight(hist,0.0,k))
        best_w=0.0; best_u=1.0
        if base and base["folds"]>=min_policy_folds:
            for w in weights[1:]:
                a=agg(evaluate_weight(hist,w,k))
                if not a or a["folds"]<min_policy_folds or not retain(a,base,0.95):
                    continue
                u=utility(a,base)
                if u>=1.0+min_gain and u>best_u:
                    best_w=w; best_u=u

        cur=pools[pools["date"]==td].copy()
        if cur["p_event_meta"].notna().sum()<k:
            best_w=0.0
        cur["event_rank"]=cur["p_event_meta"].rank(pct=True,method="average").fillna(0.5)
        cur["score_evt"]=(1-best_w)*cur["baseline_rank"]+best_w*cur["event_rank"]
        sel=cur.sort_values(["score_evt","baseline_rank","p_cal"],ascending=[False,False,False]).head(k)
        for rank,(_,r) in enumerate(sel.iterrows(),1):
            selected.append({
                "date":td,"symbol":r["symbol"],"rank":rank,"weight":best_w,
                "y6":r["y6"],"dd30_6m":r["dd30_6m"]
            })
        decisions.append({
            "date":td,"weight":best_w,
            "prior_policy_folds":base["folds"] if base else 0,
            "train_utility":best_u,
        })

    selected=pd.DataFrame(selected)
    decisions=pd.DataFrame(decisions)

    # Compare selected event-overlay vs exact baseline on same folds.
    evt_rows=[]; base_rows=[]
    for td,g in selected.groupby("date"):
        q=pools[pools["date"]==td].dropna(subset=["y6"]).copy()
        if len(g)!=k or q.empty:
            continue
        br=float(q["y6"].mean())
        pr=float(g["y6"].mean())
        evt_rows.append({
            "date":td,"precision_2x":pr,
            "lift_2x":pr/br if br>0 else np.nan,
            "hit":float(pr>0),"dd30_rate":float(g["dd30_6m"].mean()),
            "weight":float(g["weight"].iloc[0]),
        })
        bs=q.sort_values(["selection_score","p_cal","model_dispersion"],ascending=[False,False,True]).head(k)
        bpr=float(bs["y6"].mean())
        base_rows.append({
            "date":td,"precision_2x":bpr,
            "lift_2x":bpr/br if br>0 else np.nan,
            "hit":float(bpr>0),"dd30_rate":float(bs["dd30_6m"].mean()),
        })

    evt=pd.DataFrame(evt_rows); base=pd.DataFrame(base_rows)
    a=agg(evt); b=agg(base)

    outdir=Path(args.output); outdir.mkdir(parents=True,exist_ok=True)
    feat.to_parquet(outdir/"event_features_by_snapshot.parquet",index=False)
    pools.to_parquet(outdir/"event_meta_predictions.parquet",index=False)
    decisions.to_csv(outdir/"event_weight_decisions.csv",index=False)
    evt.to_csv(outdir/"event_overlay_metrics.csv",index=False)
    base.to_csv(outdir/"baseline_metrics.csv",index=False)

    summary={
        "model":"V9.7 PIT event meta-overlay",
        "events_rows":int(len(events)),
        "snapshot_feature_rows":int(len(feat)),
        "method":"fixed V9.4.1 risk-survivor pool + prior-only regularized event meta-model + max 10% event weight",
        "leakage_policy":"same-day announcements excluded; training labels must mature strictly before decision date; weight selected only from prior meta-predicted folds",
        "V9.7":a,
        "V9.4.1_same_folds":b,
        "retention_passed":bool(a and b and retain(a,b,0.95)),
        "active_event_folds":int((decisions["weight"]>0).sum()) if len(decisions) else 0,
        "production_gate":bool(
            a and b and retain(a,b,0.95)
            and utility(a,b)>=1.0+min_gain
            and int((decisions["weight"]>0).sum())>=4
        ),
        "utility_ratio":float(utility(a,b)) if a and b else None,
    }
    json.dump(summary,open(outdir/"summary.json","w"),indent=2,default=str)
    print(json.dumps(summary,indent=2,default=str))


if __name__=="__main__":
    main()
