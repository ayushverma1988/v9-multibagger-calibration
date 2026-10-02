from __future__ import annotations

import argparse, json
from pathlib import Path

import pandas as pd

from v10_2_bagged_topk import BagPolicy, evaluate
from v10_2_raw_rank_stability import (
    Policy as RawPolicy, spec_from_row, select_policy, outcome, agg, alpha_pass
)

POLICY = BagPolicy("bag50_w50", 50, 0.50)
INC = RawPolicy("incumbent", "p_cal", "p_dd30_cal")


def baseline_by_fold(oos, cmap, k):
    rows=[]
    for td,r in cmap.items():
        g=oos[oos["date"]==td].copy()
        if g.empty:
            continue
        s=select_policy(g,spec_from_row(r),INC,k)
        m=outcome(g,s,k)
        if m:
            rows.append({"date":td,**m})
    return pd.DataFrame(rows)


def summarize(mt, st, baseline):
    a=agg(mt)
    mj=float(st["mean_jaccard"].mean())
    wp=float(st["p05_jaccard"].min())
    t1=float(st["top1_stability"].mean())
    out={
        **a,
        "mean_jaccard":mj,
        "worst_p05_jaccard":wp,
        "mean_top1_stability":t1,
        "mean_names_replaced_fraction":float(st["names_replaced_fraction"].mean()),
        "worst_fold":str(st.loc[st["p05_jaccard"].idxmin(),"date"]),
        "alpha_retention_passed":bool(alpha_pass(a,baseline,.95)),
        "mean_jaccard_passed":bool(mj>=.80),
        "worst_p05_passed":bool(wp>=.60),
        "top1_passed":bool(t1>=.70),
        "dd_not_worse_3pp":bool(a["mean_dd30_rate"]<=baseline["mean_dd30_rate"]+.03),
    }
    out["stability_passed"]=bool(
        out["mean_jaccard_passed"] and out["worst_p05_passed"] and out["top1_passed"]
    )
    out["all_gates_passed"]=bool(
        out["stability_passed"] and out["alpha_retention_passed"] and out["dd_not_worse_3pp"]
    )
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--oos",required=True)
    ap.add_argument("--chosen931",required=True)
    ap.add_argument("--config",required=True)
    ap.add_argument("--output",required=True)
    ap.add_argument("--screen-sims",type=int,default=100)
    ap.add_argument("--confirm-sims",type=int,default=500)
    args=ap.parse_args()

    cfg=json.load(open(args.config))
    k=int(cfg.get("selection_k",10))
    oos=pd.read_parquet(args.oos); oos["date"]=pd.to_datetime(oos["date"])
    chosen=pd.read_csv(args.chosen931); chosen["date"]=pd.to_datetime(chosen["date"])
    cmap={pd.Timestamp(r.date):r for r in chosen.itertuples(index=False)}

    bfold=baseline_by_fold(oos,cmap,k)
    baseline=agg(bfold)

    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    bfold.to_csv(out/"baseline_metrics_by_fold.csv",index=False)

    # Stage 1 is stability-only rejection. Outcome metrics are recorded but do
    # not determine whether a different policy is tried: the policy is frozen.
    mt100,st100=evaluate(oos,cmap,cfg,POLICY,args.screen_sims)
    s100=summarize(mt100,st100,baseline)
    mt100.to_csv(out/"screen_metrics_by_fold.csv",index=False)
    st100.to_csv(out/"screen_stability_by_fold.csv",index=False)

    confirmation=None
    if s100["stability_passed"]:
        mt500,st500=evaluate(oos,cmap,cfg,POLICY,args.confirm_sims)
        confirmation=summarize(mt500,st500,baseline)
        mt500.to_csv(out/"chosen_metrics_by_fold.csv",index=False)
        st500.to_csv(out/"chosen_stability_by_fold.csv",index=False)

    summary={
        "model":"V10.2 targeted Bag50 confirmation",
        "policy":{"name":POLICY.name,"internal_sims":POLICY.internal_sims,"freq_weight":POLICY.freq_weight},
        "selection_rationale":"least-aggressive predeclared Bag50 policy after label-free blocker pass; no outcome-based policy search",
        "baseline":baseline,
        "screen_100":s100,
        "confirmation_500":confirmation,
        "promotion_gate":bool(confirmation and confirmation["all_gates_passed"]),
    }
    json.dump(summary,open(out/"summary.json","w"),indent=2,default=str)
    print(json.dumps(summary,indent=2,default=str))


if __name__=="__main__":
    main()
