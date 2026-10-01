from __future__ import annotations

import argparse, hashlib, json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

from v10_2_raw_rank_stability import (
    Policy as RawPolicy,
    spec_from_row,
    select_policy,
    perturb_for_policy,
    outcome,
    agg,
    alpha_pass,
    seed_for,
)


@dataclass(frozen=True)
class SwitchPolicy:
    name:str
    span_trigger:float
    raw_secondary_scale:float


POLICIES=[
    SwitchPolicy("compress_010_raw100",.010,1.00),
    SwitchPolicy("compress_015_raw100",.015,1.00),
    SwitchPolicy("compress_020_raw100",.020,1.00),
    SwitchPolicy("compress_030_raw100",.030,1.00),
    SwitchPolicy("compress_010_raw50",.010,.50),
    SwitchPolicy("compress_015_raw50",.015,.50),
    SwitchPolicy("compress_020_raw50",.020,.50),
    SwitchPolicy("compress_030_raw50",.030,.50),
    SwitchPolicy("compress_010_raw20",.010,.20),
    SwitchPolicy("compress_015_raw20",.015,.20),
    SwitchPolicy("compress_020_raw20",.020,.20),
    SwitchPolicy("compress_030_raw20",.030,.20),
]

INC=RawPolicy("incumbent","p_cal","p_dd30_cal")
def raw_policy(scale):
    return RawPolicy(f"raw_s{scale}","p_raw","p_dd30_raw",secondary_scale=float(scale))


def compression_span(g,spec):
    q=g.dropna(subset=["p_cal","model_dispersion"]).copy()
    if len(q)<20:return np.inf
    n=min(max(20,int(spec["pool_n"])),len(q))
    band=q.sort_values(["p_cal","model_dispersion"],ascending=[False,True]).head(n)
    p=pd.to_numeric(band["p_cal"],errors="coerce").dropna()
    if len(p)<10:return np.inf
    med=float(p.median())
    if not np.isfinite(med) or abs(med)<1e-12:return np.inf
    # Robust Top-band relative span. 2018's beta-calibrated scores collapse
    # into a much narrower range than the 1% perturbation stress.
    return float((p.quantile(.90)-p.quantile(.10))/abs(med))


def select_switch(g,spec,sp,k):
    span=compression_span(g,spec)
    p=raw_policy(sp.raw_secondary_scale) if span<=sp.span_trigger else INC
    s=select_policy(g,spec,p,k)
    if len(s):
        s=s.copy()
        s["_compression_span"]=span
        s["_used_raw"]=float(p.alpha_col=="p_raw")
    return s


def perturb_all(g,sigma,rng):
    z=g.copy()
    for c in ["p_cal","p_raw","p_dd30_cal","p_dd30_raw","model_dispersion"]:
        if c not in z:continue
        a=pd.to_numeric(z[c],errors="coerce").to_numpy(float)
        a=a*np.exp(rng.normal(0,sigma,len(a)))
        if c!="model_dispersion":
            a=np.clip(a,1e-8,1-1e-8)
        else:
            a=np.maximum(a,1e-12)
        z[c]=a
    return z


def stability(g,spec,sp,k,sims=500,sigma=.01):
    base=select_switch(g,spec,sp,k)
    if len(base)!=k:return None
    bs=set(base["symbol"].astype(str))
    top=str(base.iloc[0]["symbol"])
    js=[];tops=[];repl=[];rawuse=[]
    rng=np.random.default_rng(seed_for(g["date"].iloc[0],sp.name,"external"))
    for _ in range(sims):
        z=perturb_all(g,sigma,rng)
        s=select_switch(z,spec,sp,k)
        if len(s)!=k:continue
        ss=set(s["symbol"].astype(str))
        js.append(len(bs&ss)/len(bs|ss))
        tops.append(float(str(s.iloc[0]["symbol"])==top))
        repl.append(len(bs-ss)/k)
        rawuse.append(float(s["_used_raw"].iloc[0]))
    if not js:return None
    return {
        "mean_jaccard":float(np.mean(js)),
        "p05_jaccard":float(np.quantile(js,.05)),
        "top1_stability":float(np.mean(tops)),
        "names_replaced_fraction":float(np.mean(repl)),
        "raw_switch_fraction_under_perturbation":float(np.mean(rawuse)),
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--oos",required=True)
    ap.add_argument("--chosen931",required=True)
    ap.add_argument("--config",required=True)
    ap.add_argument("--output",required=True)
    ap.add_argument("--sims",type=int,default=500)
    args=ap.parse_args()

    cfg=json.load(open(args.config));k=int(cfg.get("selection_k",10))
    oos=pd.read_parquet(args.oos);oos["date"]=pd.to_datetime(oos["date"])
    chosen=pd.read_csv(args.chosen931);chosen["date"]=pd.to_datetime(chosen["date"])
    cmap={pd.Timestamp(r.date):r for r in chosen.itertuples(index=False)}

    # Baseline outcome
    base_rows=[]
    for td,r in cmap.items():
        g=oos[oos["date"]==td].copy()
        if g.empty:continue
        s=select_policy(g,spec_from_row(r),INC,k)
        m=outcome(g,s,k)
        if m:base_rows.append({"date":td,**m})
    baseline=agg(pd.DataFrame(base_rows))

    mets=[];sts=[];switches=[]
    for sp in POLICIES:
        for td,r in cmap.items():
            g=oos[oos["date"]==td].copy()
            if g.empty:continue
            spec=spec_from_row(r)
            span=compression_span(g,spec)
            s=select_switch(g,spec,sp,k)
            m=outcome(g,s,k)
            if m:mets.append({"policy":sp.name,"date":td,"span":span,"used_raw":float(span<=sp.span_trigger),**m})
            st=stability(g,spec,sp,k,args.sims,.01)
            if st:sts.append({"policy":sp.name,"date":td,"span":span,**st})
            switches.append({"policy":sp.name,"date":td,"span":span,"used_raw":float(span<=sp.span_trigger)})

    mt=pd.DataFrame(mets);st=pd.DataFrame(sts);sw=pd.DataFrame(switches)
    rows=[]
    for sp in POLICIES:
        a=agg(mt[mt["policy"]==sp.name])
        s=st[st["policy"]==sp.name]
        u=sw[sw["policy"]==sp.name]
        if not a or s.empty:continue
        mj=float(s["mean_jaccard"].mean());wp=float(s["p05_jaccard"].min());t1=float(s["top1_stability"].mean())
        ap=alpha_pass(a,baseline,.95)
        stab=(mj>=.80 and wp>=.60 and t1>=.70)
        dd=a["mean_dd30_rate"]<=baseline["mean_dd30_rate"]+.03
        rows.append({
            **asdict(sp),**a,
            "mean_jaccard":mj,
            "worst_p05_jaccard":wp,
            "mean_top1_stability":t1,
            "mean_names_replaced_fraction":float(s["names_replaced_fraction"].mean()),
            "worst_fold":str(s.loc[s["p05_jaccard"].idxmin(),"date"]),
            "raw_switch_folds":int(u["used_raw"].sum()),
            "alpha_retention_passed":ap,
            "stability_passed":stab,
            "dd_not_worse_3pp":dd,
            "all_gates_passed":bool(ap and stab and dd),
        })

    table=pd.DataFrame(rows).sort_values(
        ["all_gates_passed","stability_passed","alpha_retention_passed","mean_jaccard","mean_precision_2x"],
        ascending=[False,False,False,False,False]
    )
    passing=table[table["all_gates_passed"]]
    chosen_policy=None if passing.empty else str(passing.iloc[0]["name"])

    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    mt.to_csv(out/"compression_metrics_by_fold.csv",index=False)
    st.to_csv(out/"compression_stability_by_fold.csv",index=False)
    sw.to_csv(out/"compression_switches.csv",index=False)
    table.to_csv(out/"compression_policy_comparison.csv",index=False)
    summary={
        "model":"V10.2 calibration-compression switch",
        "principle":"use calibrated selector when score separation exceeds perturbation scale; switch to raw ensemble ordering only when calibration compresses the top candidate band",
        "baseline":baseline,
        "chosen_policy":chosen_policy,
        "promotion_gate":bool(chosen_policy is not None),
    }
    json.dump(summary,open(out/"summary.json","w"),indent=2,default=str)
    print(table.to_string(index=False))
    print(json.dumps(summary,indent=2,default=str))

if __name__=="__main__":
    main()
