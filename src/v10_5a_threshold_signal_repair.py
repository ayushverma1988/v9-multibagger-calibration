from __future__ import annotations

import argparse, glob, json, math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

import calibrate_v9_4 as v94
import calibrate_v9_4_1 as v941

EPS=1e-6

SIGNALS={
    "p25":{"target":"hit25_6m","raw":"hit25_6m_raw","disp":"hit25_6m_dispersion","old":"hit25_6m_cal_raw"},
    "p50":{"target":"hit50_6m","raw":"hit50_6m_raw","disp":"hit50_6m_dispersion","old":"hit50_6m_cal_raw"},
}

def find_one(root,name):
    hits=glob.glob(str(Path(root)/"**"/name),recursive=True)
    if not hits:
        raise FileNotFoundError(f"{name} not found under {root}")
    return hits[0]

def xmat(df,spec):
    p100=np.clip(pd.to_numeric(df["p100_cal"],errors="coerce").to_numpy(float),EPS,1-EPS)
    raw=np.clip(pd.to_numeric(df[spec["raw"]],errors="coerce").to_numpy(float),EPS,1-EPS)
    disp=pd.to_numeric(df[spec["disp"]],errors="coerce").to_numpy(float)
    return np.c_[logit(p100),logit(raw),disp]

def model():
    return Pipeline([
        ("impute",SimpleImputer(strategy="median")),
        ("scale",RobustScaler()),
        ("clf",LogisticRegression(C=0.20,penalty="l2",solver="lbfgs",max_iter=2000,random_state=20260928)),
    ])

def forward_stack(df,spec):
    out=df.copy()
    col=f'{spec["target"]}_v105_repaired'
    out[col]=np.nan
    dates=sorted(pd.to_datetime(out["date"].dropna().unique()))
    for td in dates:
        prior=out[pd.to_datetime(out["date"])<td].copy()
        tr=prior.dropna(subset=[spec["target"],"p100_cal",spec["raw"]]).copy()
        idx=out.index[(pd.to_datetime(out["date"])==td)&out["p100_cal"].notna()&out[spec["raw"]].notna()]
        if len(idx)==0:
            continue
        if tr["date"].nunique()<8 or len(tr)<3000 or tr[spec["target"]].sum()<100 or tr[spec["target"]].nunique()<2:
            continue
        m=model().fit(xmat(tr,spec),tr[spec["target"]].astype(int))
        out.loc[idx,col]=m.predict_proba(xmat(out.loc[idx],spec))[:,1]
    return out,col

def fit_current(oos,current,spec):
    tr=oos.dropna(subset=[spec["target"],"p100_cal",spec["raw"]]).copy()
    if tr["date"].nunique()<8 or len(tr)<3000 or tr[spec["target"]].sum()<100:
        return np.full(len(current),np.nan)
    m=model().fit(xmat(tr,spec),tr[spec["target"]].astype(int))
    return m.predict_proba(xmat(current,spec))[:,1]

def cal_slope(y,p):
    y=np.asarray(y).astype(int)
    p=np.clip(np.asarray(p,float),EPS,1-EPS)
    if len(np.unique(y))<2:
        return np.nan,np.nan
    lr=LogisticRegression(C=1e6,solver="lbfgs",max_iter=2000)
    lr.fit(logit(p).reshape(-1,1),y)
    return float(lr.coef_[0,0]),float(lr.intercept_[0])

def metrics(df,target,pcol):
    q=df.dropna(subset=[target,pcol]).copy()
    if q.empty or q[target].nunique()<2:
        return {"n":int(len(q)),"folds":int(q["date"].nunique()) if len(q) else 0}
    y=q[target].astype(int).to_numpy(); p=np.clip(q[pcol].to_numpy(float),EPS,1-EPS)
    br=float(y.mean()); pr=float(average_precision_score(y,p))
    slope,intercept=cal_slope(y,p)
    return {
        "n":int(len(q)),"folds":int(q["date"].nunique()),"positives":int(y.sum()),
        "base_rate":br,"pr_auc":pr,"pr_auc_lift":float(pr/br) if br>0 else np.nan,
        "brier":float(brier_score_loss(y,p)),"brier_climatology":float(br*(1-br)),
        "logloss":float(log_loss(y,p)),"calibration_slope":slope,"calibration_intercept":intercept,
    }

def gate_signal(df,spec,pcol):
    target=spec["target"]
    q=df.dropna(subset=[target,pcol]).copy()
    cand=metrics(q,target,pcol)
    old=metrics(q,target,spec["old"])
    recent=q[pd.to_datetime(q["date"])>=pd.Timestamp("2023-01-01")].copy()
    rec=metrics(recent,target,pcol)
    gates={
        "rows_ge_3000":cand.get("n",0)>=3000,
        "folds_ge_8":cand.get("folds",0)>=8,
        "positives_ge_100":cand.get("positives",0)>=100,
        "pr_auc_lift_ge_1_05":cand.get("pr_auc_lift",-np.inf)>=1.05,
        "calibration_slope_ge_0_25":cand.get("calibration_slope",-np.inf)>=0.25,
        "calibration_slope_le_2_50":cand.get("calibration_slope",np.inf)<=2.50,
        "positive_brier_skill":cand.get("brier",np.inf)<cand.get("brier_climatology",-np.inf),
        "pr_auc_not_worse_than_old":cand.get("pr_auc",-np.inf)>=old.get("pr_auc",np.inf),
        "recent_pr_auc_lift_ge_1_00":rec.get("pr_auc_lift",-np.inf)>=1.00,
        "recent_calibration_slope_positive":rec.get("calibration_slope",-np.inf)>0.0,
    }
    return {"candidate":cand,"old_same_rows":old,"recent_2023plus":rec,"gates":gates,"passed":bool(all(gates.values()))}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--baseline-dir",required=True)
    ap.add_argument("--config",default="config_v9_4_1.json")
    ap.add_argument("--output",default="outputs_v10_5a_threshold_repair")
    args=ap.parse_args()
    outdir=Path(args.output); outdir.mkdir(parents=True,exist_ok=True)
    cfg=json.load(open(args.config))
    oos=pd.read_parquet(find_one(args.baseline_dir,"oos_predictions.parquet"))
    oos["date"]=pd.to_datetime(oos["date"])
    chosen=pd.read_csv(find_one(args.baseline_dir,"chosen_v931_config_by_fold.csv"))
    chosen["date"]=pd.to_datetime(chosen["date"])
    current=pd.read_csv(find_one(args.baseline_dir,"current_selection.csv"))
    current["date"]=pd.to_datetime(current["date"])

    reports={}
    repaired_cols={}
    for name,spec in SIGNALS.items():
        oos,col=forward_stack(oos,spec)
        repaired_cols[name]=col
        reports[name]=gate_signal(oos,spec,col)

    # Keep pre-coherence repaired signals separate for quality gating.
    oos["hit25_6m_cal_raw_v105"]=oos[repaired_cols["p25"]]
    oos["hit50_6m_cal_raw_v105"]=oos[repaired_cols["p50"]]

    # Only a passing repaired signal is allowed to replace its old diagnostic.
    for name,spec in SIGNALS.items():
        if reports[name]["passed"]:
            oos[spec["old"]]=oos[repaired_cols[name]]

    p100=oos["p100_cal"].to_numpy(float)
    p50r=oos["hit50_6m_cal_raw"].to_numpy(float)
    p25r=oos["hit25_6m_cal_raw"].to_numpy(float)
    oos["p50_cal"]=np.clip(np.where(np.isfinite(p50r)&np.isfinite(p100),np.maximum(p50r,p100),np.nan),EPS,1-EPS)
    oos["p25_cal"]=np.clip(np.where(np.isfinite(p25r)&oos["p50_cal"].notna(),np.maximum(p25r,oos["p50_cal"].to_numpy(float)),np.nan),EPS,1-EPS)

    # Re-run the existing forward selector exactly; signal gates still use prior folds only.
    oos_new,chosen941=v941.forward_v941(oos,chosen,cfg)
    comp=v941.comparison_by_fold(oos_new,cfg)
    cand=v94.summarize_comparison(comp,"V9.4.1")

    control=json.load(open("published_v10_2_production/model_summary.json"))["forward_history"]["V9.4.1"]
    ratios={}
    for k in ["mean_precision_100","mean_capped_lift_100","hit_fold_rate_100","median_precision_100"]:
        a=float(cand.get(k,np.nan)); b=float(control.get(k,np.nan))
        ratios[k]=a/b if np.isfinite(a) and np.isfinite(b) and b>0 else None
    dd_delta=float(cand.get("mean_dd30_rate",np.nan))-float(control["mean_dd30_rate"])

    spec=dict(json.load(open("published_v10_2_production/model_summary.json"))["production_risk_config"])
    shares,search,baseline,prod_gates=v941.optimize_ladder_gated(oos_new,spec,cfg)

    # Current repaired estimates use all historical OOS rows, but only for signals that passed.
    for name,s in SIGNALS.items():
        pred=fit_current(oos,current,s)
        current[f'{s["target"]}_v105_repaired']=pred
        if reports[name]["passed"]:
            current[s["old"]]=pred
    current["p50_cal"]=np.maximum(current["hit50_6m_cal_raw"].to_numpy(float),current["p100_cal"].to_numpy(float))
    current["p25_cal"]=np.maximum(current["hit25_6m_cal_raw"].to_numpy(float),current["p50_cal"].to_numpy(float))
    top=v94.select_ladder_topk(current,spec,shares,int(cfg.get("selection_k",10)))
    if len(top):
        top=top.copy()
        top["selection_rank_v105a"]=np.arange(1,len(top)+1)
        top.to_csv(outdir/"current_top10_v105a.csv",index=False)

    alpha_ok=all((v is not None and v>=0.95) for v in [ratios["mean_precision_100"],ratios["mean_capped_lift_100"],ratios["hit_fold_rate_100"]])
    selector_gate=bool(alpha_ok and dd_delta<=0.03)
    summary={
        "stage":"V10.5A P25/P50 temporal signal repair",
        "method":"fixed low-dimensional temporal logistic stack: logit(P100), logit(direct threshold raw), direct-model dispersion",
        "coefficient_policy":"fit only on earlier OOS folds; C=0.20 fixed before results; no hyperparameter search",
        "p100_changed":False,
        "signals":reports,
        "signals_passed":[k for k,v in reports.items() if v["passed"]],
        "candidate_forward_history":cand,
        "control_forward_history":control,
        "alpha_retention_ratios":ratios,
        "dd_delta":dd_delta,
        "production_ladder_shares_if_used":{"p100":shares[0],"p50":shares[1],"p25":shares[2]},
        "production_signal_gates_after_repair":prod_gates,
        "selector_alpha_gate":selector_gate,
        "promotion_allowed":bool(selector_gate and (shares!=(1.0,0.0,0.0))),
        "rule":"A repaired signal can pass independently. Production weight remains zero unless the unchanged ladder optimizer and alpha-retention gates also pass.",
    }
    json.dump(summary,open(outdir/"summary.json","w"),indent=2,default=str)
    oos_new.to_parquet(outdir/"oos_predictions_v105a.parquet",index=False)
    comp.to_csv(outdir/"selection_metrics_by_fold_v105a.csv",index=False)
    chosen941.to_csv(outdir/"chosen_v105a_by_fold.csv",index=False)
    if not search.empty: search.to_csv(outdir/"ladder_search_v105a.csv",index=False)
    print(json.dumps(summary,indent=2,default=str))

if __name__=="__main__":
    main()
