from __future__ import annotations

import argparse, glob, json
from pathlib import Path

import numpy as np
import pandas as pd

import calibrate_v9_4 as v94


def find_one(root,name):
    hits=glob.glob(str(Path(root)/"**"/name),recursive=True)
    if not hits:
        raise FileNotFoundError(f"{name} not found under {root}")
    return hits[0]


def fold_spec(g):
    r=g.iloc[0]
    return {
        "name":str(r.get("chosen_config","")),
        "pool_n":int(float(r.get("chosen_pool_n",100))),
        "risk_drop":float(r.get("chosen_risk_drop",0.0)),
        "w_safety":float(r.get("chosen_w_safety",0.0)),
        "w_consensus":float(r.get("chosen_w_consensus",0.0)),
        "baseline":str(r.get("chosen_config",""))=="V9.2_baseline",
    }


def fold_shares(g):
    p100=pd.to_numeric(g.get("v941_share_p100"),errors="coerce").dropna()
    p50=pd.to_numeric(g.get("v941_share_p50"),errors="coerce").dropna()
    p25=pd.to_numeric(g.get("v941_share_p25"),errors="coerce").dropna()
    if len(p100)==0:
        return (1.0,0.0,0.0)
    return (float(p100.iloc[0]),float(p50.iloc[0]),float(p25.iloc[0]))


def ranked_candidates(g,spec,shares,n=20):
    q=g.dropna(subset=["p100_cal","p50_cal","p25_cal","p_dd30_cal","model_dispersion"]).copy()
    if spec.get("baseline",False):
        q=q.sort_values(["p100_cal","model_dispersion"],ascending=[False,True]).copy()
        q["selection_score_v94"]=q["p100_cal"]
        return q.head(n)
    k=min(n,max(10,int(spec["pool_n"])))
    sel=v94.select_ladder_topk(q,spec,shares,k)
    return sel


def overlay(topn,risk_col="p_dd30_v104",final_k=10,veto_n=2):
    x=topn.copy()
    if len(x)<final_k:
        return pd.DataFrame()
    x=x.reset_index().rename(columns={"index":"orig_index"})
    x["baseline_rank"]=np.arange(1,len(x)+1)
    original=x.head(final_k).copy()

    # Frozen rule: among the top-20 V10.2 candidates, mark exactly the two
    # highest V10.4 downside-risk names as veto candidates. They alter the
    # portfolio only if they are inside the original top-10.
    risk=x.dropna(subset=[risk_col]).sort_values(risk_col,ascending=False)
    veto=set(risk.head(veto_n)["orig_index"].tolist())

    kept=original[~original["orig_index"].isin(veto)].copy()
    need=final_k-len(kept)
    if need>0:
        refill=x[(x["baseline_rank"]>final_k)&(~x["orig_index"].isin(veto))].copy()
        refill=refill.sort_values("baseline_rank").head(need)
        kept=pd.concat([kept,refill],ignore_index=True)
    if len(kept)!=final_k:
        return pd.DataFrame()
    kept=kept.sort_values("baseline_rank").reset_index(drop=True)
    kept["overlay_rank"]=np.arange(1,final_k+1)
    kept["veto_count_applied"]=len(original[original["orig_index"].isin(veto)])
    return kept


def summarize(folds,prefix):
    q=folds.copy()
    return {
        "folds":int(len(q)),
        "mean_precision_100":float(q[f"{prefix}_precision"].mean()),
        "median_precision_100":float(q[f"{prefix}_precision"].median()),
        "hit_fold_rate_100":float(q[f"{prefix}_hit"].mean()),
        "mean_capped_lift_100":float(q[f"{prefix}_lift"].clip(upper=10).mean()),
        "mean_dd30_rate":float(q[f"{prefix}_dd"].mean()),
        "median_dd30_rate":float(q[f"{prefix}_dd"].median()),
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--baseline-dir",required=True)
    ap.add_argument("--v104-dir",required=True)
    ap.add_argument("--output",default="outputs_v10_5b_risk_overlay")
    args=ap.parse_args()
    outdir=Path(args.output); outdir.mkdir(parents=True,exist_ok=True)

    base=pd.read_parquet(find_one(args.baseline_dir,"oos_predictions.parquet"))
    v104=pd.read_parquet(find_one(args.v104_dir,"oos_predictions.parquet"))
    base["date"]=pd.to_datetime(base["date"]); v104["date"]=pd.to_datetime(v104["date"])
    risk=v104[["date","symbol","p_dd30_cal","p_dd30_raw"]].copy()
    # Coverage-only repair frozen after missingness audit:
    # use calibrated V10.4 DD30 when available; otherwise use the same
    # point-in-time V10.4 raw DD30 prediction. No outcome data or selector
    # parameter is used to choose the fallback.
    risk["p_dd30_v104"]=pd.to_numeric(risk["p_dd30_cal"],errors="coerce")
    raw=pd.to_numeric(risk["p_dd30_raw"],errors="coerce")
    risk["p_dd30_v104"]=risk["p_dd30_v104"].fillna(raw)
    risk["p_dd30_v104_source"]=np.where(
        pd.to_numeric(risk["p_dd30_cal"],errors="coerce").notna(),
        "calibrated",
        np.where(raw.notna(),"raw_fallback","missing"),
    )
    z=base.merge(risk[["date","symbol","p_dd30_v104","p_dd30_v104_source"]],on=["date","symbol"],how="left")
    risk_coverage=float(z["p_dd30_v104"].notna().mean())

    rows=[]; selected_rows=[]; reproduction=[]
    dates=sorted(pd.to_datetime(z.loc[z["selected_v941"].fillna(False),"date"].unique()))
    for td in dates:
        g=z[z["date"]==td].copy()
        control=g[g["selected_v941"].fillna(False)].copy()
        if len(control)!=10:
            continue
        spec=fold_spec(g); shares=fold_shares(g)
        top20=ranked_candidates(g,spec,shares,20)
        if len(top20)<10:
            continue
        recon=set(top20.head(10)["symbol"].astype(str))
        actual=set(control["symbol"].astype(str))
        reproduction.append(len(recon&actual)/10.0)

        cand=overlay(top20,"p_dd30_v104",10,2)
        if len(cand)!=10:
            continue
        universe=g.dropna(subset=["y6"]).copy()
        br=float(universe["y6"].mean()) if len(universe) else np.nan

        cp=float(control["y6"].mean()); cd=float(control["dd30_6m"].mean())
        pp=float(cand["y6"].mean()); pdn=float(cand["dd30_6m"].mean())
        rows.append({
            "date":td,
            "control_precision":cp,
            "control_lift":cp/br if br>0 else np.nan,
            "control_hit":float(cp>0),
            "control_dd":cd,
            "candidate_precision":pp,
            "candidate_lift":pp/br if br>0 else np.nan,
            "candidate_hit":float(pp>0),
            "candidate_dd":pdn,
            "veto_count_applied":int(cand["veto_count_applied"].iloc[0]),
            "control_reproduction_jaccard":len(recon&actual)/len(recon|actual) if len(recon|actual) else np.nan,
        })
        q=cand[["date","symbol","baseline_rank","overlay_rank","p100_cal","p_dd30_cal","p_dd30_v104","p_dd30_v104_source","y6","dd30_6m"]].copy()
        selected_rows.append(q)

    folds=pd.DataFrame(rows)
    if folds.empty:
        raise RuntimeError("No evaluable overlay folds")
    control=summarize(folds,"control")
    cand=summarize(folds,"candidate")

    def ratio(a,b):
        return float(a/b) if np.isfinite(a) and np.isfinite(b) and b>0 else None
    ratios={
        "mean_precision_100":ratio(cand["mean_precision_100"],control["mean_precision_100"]),
        "median_precision_100":ratio(cand["median_precision_100"],control["median_precision_100"]),
        "hit_fold_rate_100":ratio(cand["hit_fold_rate_100"],control["hit_fold_rate_100"]),
        "mean_capped_lift_100":ratio(cand["mean_capped_lift_100"],control["mean_capped_lift_100"]),
    }
    dd_delta=float(cand["mean_dd30_rate"]-control["mean_dd30_rate"])

    recent=folds[pd.to_datetime(folds["date"])>=pd.Timestamp("2023-01-01")].copy()
    recent_control=summarize(recent,"control") if len(recent) else {}
    recent_cand=summarize(recent,"candidate") if len(recent) else {}
    recent_ratios={
        "precision":ratio(recent_cand.get("mean_precision_100",np.nan),recent_control.get("mean_precision_100",np.nan)),
        "lift":ratio(recent_cand.get("mean_capped_lift_100",np.nan),recent_control.get("mean_capped_lift_100",np.nan)),
        "hit":ratio(recent_cand.get("hit_fold_rate_100",np.nan),recent_control.get("hit_fold_rate_100",np.nan)),
        "dd_delta":float(recent_cand.get("mean_dd30_rate",np.nan)-recent_control.get("mean_dd30_rate",np.nan)) if len(recent) else None,
        "folds":int(len(recent)),
    }

    gates={
        "risk_join_coverage_ge_0_99":risk_coverage>=0.99,
        "evaluable_folds_ge_18":len(folds)>=18,
        "control_reproduction_mean_ge_0_99":float(np.mean(reproduction))>=0.99 if reproduction else False,
        "mean_precision_retention_ge_0_95":(ratios["mean_precision_100"] or 0)>=0.95,
        "capped_lift_retention_ge_0_95":(ratios["mean_capped_lift_100"] or 0)>=0.95,
        "hit_fold_retention_ge_0_95":(ratios["hit_fold_rate_100"] or 0)>=0.95,
        "median_precision_retention_ge_0_95":(ratios["median_precision_100"] or 0)>=0.95,
        "dd30_improves_at_least_3pp":dd_delta<=-0.03,
        "recent_folds_ge_4":recent_ratios["folds"]>=4,
        "recent_precision_retention_ge_0_90":(recent_ratios["precision"] or 0)>=0.90,
        "recent_lift_retention_ge_0_90":(recent_ratios["lift"] or 0)>=0.90,
        "recent_hit_retention_ge_0_90":(recent_ratios["hit"] or 0)>=0.90,
        "recent_dd_not_worse_2pp":recent_ratios["dd_delta"] is not None and recent_ratios["dd_delta"]<=0.02,
    }

    # Current snapshot.
    bcur=pd.read_csv(find_one(args.baseline_dir,"current_selection.csv"))
    rcur=pd.read_csv(find_one(args.v104_dir,"current_selection.csv"))
    rr=rcur[["date","symbol","p_dd30_cal","p_dd30_raw"]].copy()
    rr["p_dd30_v104"]=pd.to_numeric(rr["p_dd30_cal"],errors="coerce").fillna(
        pd.to_numeric(rr["p_dd30_raw"],errors="coerce")
    )
    cur=bcur.merge(rr[["date","symbol","p_dd30_v104"]],on=["date","symbol"],how="left")
    prod=json.load(open("published_v10_2_production/model_summary.json"))
    spec=dict(prod["production_risk_config"])
    shares=(1.0,0.0,0.0)
    top20=ranked_candidates(cur,spec,shares,20)
    current_overlay=overlay(top20,"p_dd30_v104",10,2)
    if len(current_overlay):
        current_overlay.to_csv(outdir/"current_top10_v105b.csv",index=False)

    summary={
        "stage":"V10.5B alpha-preserve risk overlay",
        "control":"V10.2 integrity-corrected V9.4.1",
        "risk_source":"V10.4 point-in-time augmented DD30 probability only; calibrated when available, same-model raw fallback only for calibration-availability gaps; V10.4 alpha predictions ignored",
        "coverage_repair":"p_dd30_cal else p_dd30_raw, frozen from missingness audit before this rerun",
        "frozen_overlay_rule":"reconstruct V10.2 top20; flag two highest V10.4 DD30-risk names; if flagged names are in top10, veto them and refill by next V10.2 rank",
        "model_tuning":False,
        "risk_join_coverage":risk_coverage,
        "control_reproduction_mean_top10_overlap":float(np.mean(reproduction)) if reproduction else None,
        "control":control,
        "candidate":cand,
        "alpha_retention_ratios":ratios,
        "dd_delta":dd_delta,
        "recent_2023plus":{"control":recent_control,"candidate":recent_cand,"ratios":recent_ratios},
        "gates":gates,
        "acceptance_gate":bool(all(gates.values())),
        "action_if_pass":"eligible for production snapshot verification; V10.2 alpha model remains unchanged",
        "action_if_fail":"retain V10.2 production; do not tune veto count or threshold post hoc",
    }
    json.dump(summary,open(outdir/"summary.json","w"),indent=2,default=str)
    folds.to_csv(outdir/"fold_metrics.csv",index=False)
    if selected_rows:
        pd.concat(selected_rows,ignore_index=True).to_csv(outdir/"selected_rows.csv",index=False)
    print(json.dumps(summary,indent=2,default=str))

if __name__=="__main__":
    main()
