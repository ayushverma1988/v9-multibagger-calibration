from __future__ import annotations

import argparse, json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

from v10_2_raw_rank_stability import (
    Policy, spec_from_row, select_policy, perturb_for_policy,
    outcome, agg, alpha_pass, seed_for
)

CANDIDATES=[
    Policy("incumbent","p_cal","p_dd30_cal"),
    Policy("raw_alpha_raw_risk","p_raw","p_dd30_raw"),
    Policy("raw_secondary20","p_raw","p_dd30_raw",secondary_scale=.20),
    Policy("raw_pure_alpha","p_raw","p_dd30_raw",secondary_scale=0.0),
]

def jaccard(a,b):
    a=set(map(str,a)); b=set(map(str,b))
    return len(a&b)/len(a|b) if (a or b) else np.nan

def stability(g,spec,policy,k,sims,sigma,salt):
    base=select_policy(g,spec,policy,k)
    if len(base)!=k:return None
    bs=set(base["symbol"].astype(str))
    js=[]; top=[]
    rng=np.random.default_rng(seed_for(g["date"].iloc[0],policy.name,salt))
    for _ in range(sims):
        z=perturb_for_policy(g,policy,sigma,rng)
        s=select_policy(z,spec,policy,k)
        if len(s)!=k:continue
        ss=set(s["symbol"].astype(str))
        js.append(len(bs&ss)/len(bs|ss))
        top.append(float(str(s.iloc[0]["symbol"])==str(base.iloc[0]["symbol"])))
    if not js:return None
    return {
        "mean_jaccard":float(np.mean(js)),
        "p05_jaccard":float(np.quantile(js,.05)),
        "top1_stability":float(np.mean(top)),
    }

def choose_policy(g,spec,k,internal_sims=150,sigma=.01):
    incumbent=CANDIDATES[0]
    inc_sel=select_policy(g,spec,incumbent,k)
    inc_st=stability(g,spec,incumbent,k,internal_sims,sigma,"adaptive_internal_inc")
    if len(inc_sel)!=k or inc_st is None:
        return incumbent,inc_st,[]

    # Stable folds remain exactly incumbent.
    if inc_st["p05_jaccard"]>=.60 and inc_st["mean_jaccard"]>=.80:
        return incumbent,inc_st,[{"policy":"incumbent",**inc_st,"overlap_incumbent":1.0}]

    rows=[]
    for p in CANDIDATES:
        s=select_policy(g,spec,p,k)
        st=stability(g,spec,p,k,internal_sims,sigma,"adaptive_internal")
        if len(s)!=k or st is None:continue
        ov=jaccard(inc_sel["symbol"],s["symbol"])
        rows.append({"policy":p.name,**st,"overlap_incumbent":ov})

    # Preserve at least 8 of 10 incumbent names where possible
    # (Jaccard >= 8/12 = .6667). Policy choice uses stability only, never labels.
    eligible=[r for r in rows if r["overlap_incumbent"]>=2/3]
    if not eligible:
        return incumbent,inc_st,rows

    eligible=sorted(
        eligible,
        key=lambda r:(r["p05_jaccard"]>=.60, r["p05_jaccard"], r["mean_jaccard"], r["top1_stability"]),
        reverse=True
    )
    chosen=next(p for p in CANDIDATES if p.name==eligible[0]["policy"])
    return chosen,inc_st,rows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--oos",required=True)
    ap.add_argument("--chosen931",required=True)
    ap.add_argument("--config",required=True)
    ap.add_argument("--output",required=True)
    ap.add_argument("--internal-sims",type=int,default=150)
    ap.add_argument("--external-sims",type=int,default=500)
    args=ap.parse_args()

    cfg=json.load(open(args.config))
    k=int(cfg.get("selection_k",10))
    oos=pd.read_parquet(args.oos); oos["date"]=pd.to_datetime(oos["date"])
    chosen=pd.read_csv(args.chosen931); chosen["date"]=pd.to_datetime(chosen["date"])
    cmap={pd.Timestamp(r.date):r for r in chosen.itertuples(index=False)}

    metric_rows=[]; stab_rows=[]; decisions=[]
    for td,r in cmap.items():
        g=oos[oos["date"]==td].copy()
        if g.empty:continue
        spec=spec_from_row(r)
        p,inc_st,cands=choose_policy(g,spec,k,args.internal_sims,.01)
        sel=select_policy(g,spec,p,k)
        m=outcome(g,sel,k)
        st=stability(g,spec,p,k,args.external_sims,.01,"adaptive_external")
        if m is not None:metric_rows.append({"date":td,"policy":p.name,**m})
        if st is not None:stab_rows.append({"date":td,"policy":p.name,**st})
        decisions.append({
            "date":td,
            "chosen_policy":p.name,
            "inc_internal_mean_jaccard":inc_st["mean_jaccard"] if inc_st else None,
            "inc_internal_p05_jaccard":inc_st["p05_jaccard"] if inc_st else None,
            "candidate_diagnostics":json.dumps(cands,default=str),
        })

    mt=pd.DataFrame(metric_rows); st=pd.DataFrame(stab_rows); dec=pd.DataFrame(decisions)
    a=agg(mt)
    # incumbent baseline realized outcomes from same folds
    brows=[]
    for td,r in cmap.items():
        g=oos[oos["date"]==td].copy()
        s=select_policy(g,spec_from_row(r),CANDIDATES[0],k)
        m=outcome(g,s,k)
        if m is not None:brows.append({"date":td,**m})
    b=agg(pd.DataFrame(brows))

    stability_summary={
        "mean_jaccard":float(st["mean_jaccard"].mean()),
        "worst_p05_jaccard":float(st["p05_jaccard"].min()),
        "mean_top1_stability":float(st["top1_stability"].mean()),
        "worst_fold":str(st.loc[st["p05_jaccard"].idxmin(),"date"]) if len(st) else None,
    }
    gates={
        "alpha_retention":bool(alpha_pass(a,b,.95)),
        "mean_jaccard":bool(stability_summary["mean_jaccard"]>=.80),
        "worst_p05_jaccard":bool(stability_summary["worst_p05_jaccard"]>=.60),
        "top1":bool(stability_summary["mean_top1_stability"]>=.70),
        "dd_not_worse_3pp":bool(a["mean_dd30_rate"]<=b["mean_dd30_rate"]+.03),
    }
    summary={
        "model":"V10.2 fragile-fold-only adaptive stability selector",
        "principle":"stable folds remain incumbent; unstable folds choose among predeclared raw-score selectors using label-free internal perturbation stability only",
        "adaptive":a,
        "incumbent":b,
        "stability":stability_summary,
        "policy_counts":dec["chosen_policy"].value_counts().to_dict(),
        "gates":gates,
        "promotion_gate":bool(all(gates.values())),
    }

    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    mt.to_csv(out/"adaptive_metrics_by_fold.csv",index=False)
    st.to_csv(out/"adaptive_stability_by_fold.csv",index=False)
    dec.to_csv(out/"adaptive_policy_decisions.csv",index=False)
    json.dump(summary,open(out/"summary.json","w"),indent=2,default=str)
    print(json.dumps(summary,indent=2,default=str))

if __name__=="__main__":
    main()
