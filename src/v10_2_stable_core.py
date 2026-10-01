from __future__ import annotations

import argparse, hashlib, json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

from v10_2_rank_aggregation import (
    Policy as RAPolicy,
    spec_from_row,
    select_policy as ra_select,
    perturb as ra_perturb,
    outcome,
    agg,
    alpha_pass,
    seed_for,
)


@dataclass(frozen=True)
class CorePolicy:
    name:str
    base_name:str
    core_n:int
    candidate_limit:int=16
    internal_sims:int=150


BASES={
    "raw":RAPolicy("raw","raw",risk_col="p_dd30_raw"),
    "borda4":RAPolicy("borda4","borda",risk_col="p_dd30_raw"),
    "hybrid50":RAPolicy("hybrid50","borda",raw_weight=.50,risk_col="p_dd30_raw"),
    "hybrid75":RAPolicy("hybrid75","borda",raw_weight=.75,risk_col="p_dd30_raw"),
}

POLICIES=[
    CorePolicy("raw_core8","raw",8),
    CorePolicy("borda_core8","borda4",8),
    CorePolicy("hybrid50_core7","hybrid50",7),
    CorePolicy("hybrid50_core8","hybrid50",8),
    CorePolicy("hybrid50_core9","hybrid50",9),
    CorePolicy("hybrid75_core7","hybrid75",7),
    CorePolicy("hybrid75_core8","hybrid75",8),
    CorePolicy("hybrid75_core9","hybrid75",9),
]


def stable_core_select(g,spec,cp,k=10,salt="base"):
    basep=BASES[cp.base_name]

    # Need a candidate slate wider than k. Reuse the same robust alpha/risk
    # mechanics, but temporarily request a larger final list by increasing k.
    slate_n=max(k,int(cp.candidate_limit))
    slate=ra_select(g,spec,basep,slate_n)
    if len(slate)<k:
        # Some historical configs have small survivor sets. Fall back to all
        # available robust candidates but never silently return <k.
        slate=ra_select(g,spec,basep,k)
        if len(slate)<k:
            return pd.DataFrame()

    slate_idx=set(slate.index)
    freq=pd.Series(0.0,index=slate.index)
    avg_rank=pd.Series(0.0,index=slate.index)
    rank_seen=pd.Series(0.0,index=slate.index)

    rng=np.random.default_rng(seed_for(g["date"].iloc[0],cp.name,salt))
    for sim in range(int(cp.internal_sims)):
        z=ra_perturb(g,basep,.01,rng)
        s=ra_select(z,spec,basep,k)
        if len(s)!=k:
            continue
        order={idx:r for r,idx in enumerate(s.index,1)}
        for idx in slate.index:
            if idx in order:
                freq.loc[idx]+=1.0
                avg_rank.loc[idx]+=order[idx]
                rank_seen.loc[idx]+=1.0

    denom=max(1,int(cp.internal_sims))
    freq=freq/denom
    mean_rank=avg_rank/rank_seen.replace(0,np.nan)

    cand=slate.copy()
    cand["_freq"]=freq.reindex(cand.index).fillna(0.0)
    cand["_mean_rank"]=mean_rank.reindex(cand.index).fillna(k+5)

    # Stable core: high survival frequency first, then better expected rank.
    core=cand.sort_values(
        ["_freq","_mean_rank","selection_score_ra","p_raw","symbol"],
        ascending=[False,True,False,False,True],
    ).head(int(cp.core_n)).copy()

    # Preserve alpha in the flexible slots by filling from the base robust rank.
    remaining=cand.loc[~cand.index.isin(core.index)].sort_values(
        ["selection_score_ra","p_raw","model_dispersion","symbol"],
        ascending=[False,False,True,True],
    )
    fill=remaining.head(k-len(core)).copy()
    out=pd.concat([core,fill],axis=0)

    if len(out)<k:
        rest=g.loc[~g.index.isin(out.index)].copy()
        rest=rest.dropna(subset=["p_raw"]).sort_values(
            ["p_raw","model_dispersion","symbol"],
            ascending=[False,True,True],
        ).head(k-len(out))
        out=pd.concat([out,rest],axis=0)

    if len(out)!=k:
        return pd.DataFrame()

    out["_core_flag"]=out.index.isin(core.index)
    out["_stable_order_score"]=(
        1000*out["_core_flag"].astype(int)
        +100*out.get("_freq",0).fillna(0)
        +out["selection_score_ra"].rank(pct=True,method="average")
    )
    return out.sort_values(
        ["_stable_order_score","selection_score_ra","p_raw","symbol"],
        ascending=[False,False,False,True],
    ).head(k).copy()


def external_stability(g,spec,cp,k,sims=500):
    base=stable_core_select(g,spec,cp,k,salt="unperturbed")
    if len(base)!=k:
        return None
    bs=set(base["symbol"].astype(str))
    top=str(base.iloc[0]["symbol"])
    js=[]; tops=[]; repl=[]
    basep=BASES[cp.base_name]
    rng=np.random.default_rng(seed_for(g["date"].iloc[0],cp.name,"external"))
    for i in range(sims):
        z=ra_perturb(g,basep,.01,rng)
        s=stable_core_select(z,spec,cp,k,salt=f"external_inner_{i}")
        if len(s)!=k:
            continue
        ss=set(s["symbol"].astype(str))
        js.append(len(bs&ss)/len(bs|ss))
        tops.append(float(str(s.iloc[0]["symbol"])==top))
        repl.append(len(bs-ss)/k)
    if not js:
        return None
    return {
        "mean_jaccard":float(np.mean(js)),
        "p05_jaccard":float(np.quantile(js,.05)),
        "top1_stability":float(np.mean(tops)),
        "names_replaced_fraction":float(np.mean(repl)),
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--oos",required=True)
    ap.add_argument("--chosen931",required=True)
    ap.add_argument("--config",required=True)
    ap.add_argument("--output",required=True)
    ap.add_argument("--sims",type=int,default=500)
    args=ap.parse_args()

    cfg=json.load(open(args.config)); k=int(cfg.get("selection_k",10))
    oos=pd.read_parquet(args.oos); oos["date"]=pd.to_datetime(oos["date"])
    chosen=pd.read_csv(args.chosen931); chosen["date"]=pd.to_datetime(chosen["date"])
    cmap={pd.Timestamp(r.date):r for r in chosen.itertuples(index=False)}

    # exact incumbent for alpha comparison
    incumbent=RAPolicy("incumbent","incumbent",risk_col="p_dd30_cal")
    base_rows=[]
    for td,r in cmap.items():
        g=oos[oos["date"]==td].copy()
        if g.empty:continue
        s=ra_select(g,spec_from_row(r),incumbent,k)
        m=outcome(g,s,k)
        if m:base_rows.append({"date":td,**m})
    baseline=agg(pd.DataFrame(base_rows))

    mets=[]; sts=[]
    for cp in POLICIES:
        for td,r in cmap.items():
            g=oos[oos["date"]==td].copy()
            if g.empty:continue
            spec=spec_from_row(r)
            s=stable_core_select(g,spec,cp,k)
            m=outcome(g,s,k)
            if m:mets.append({"policy":cp.name,"date":td,**m})
            st=external_stability(g,spec,cp,k,args.sims)
            if st:sts.append({"policy":cp.name,"date":td,**st})

    mt=pd.DataFrame(mets); st=pd.DataFrame(sts)
    rows=[]
    for cp in POLICIES:
        a=agg(mt[mt["policy"]==cp.name])
        s=st[st["policy"]==cp.name]
        if not a or s.empty:continue
        mj=float(s["mean_jaccard"].mean())
        wp=float(s["p05_jaccard"].min())
        t1=float(s["top1_stability"].mean())
        ap=alpha_pass(a,baseline,.95)
        sp=(mj>=.80 and wp>=.60 and t1>=.70)
        dd=a["mean_dd30_rate"]<=baseline["mean_dd30_rate"]+.03
        rows.append({
            **asdict(cp),**a,
            "mean_jaccard":mj,
            "worst_p05_jaccard":wp,
            "mean_top1_stability":t1,
            "mean_names_replaced_fraction":float(s["names_replaced_fraction"].mean()),
            "worst_fold":str(s.loc[s["p05_jaccard"].idxmin(),"date"]),
            "alpha_retention_passed":ap,
            "stability_passed":sp,
            "dd_not_worse_3pp":dd,
            "all_gates_passed":bool(ap and sp and dd),
        })

    table=pd.DataFrame(rows).sort_values(
        ["all_gates_passed","stability_passed","alpha_retention_passed","mean_jaccard","mean_precision_2x"],
        ascending=[False,False,False,False,False],
    )
    passing=table[table["all_gates_passed"]]
    chosen_policy=None if passing.empty else str(passing.iloc[0]["name"])

    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    mt.to_csv(out/"stable_core_metrics_by_fold.csv",index=False)
    st.to_csv(out/"stable_core_stability_by_fold.csv",index=False)
    table.to_csv(out/"stable_core_policy_comparison.csv",index=False)
    summary={
        "model":"V10.2 stable-core top-k selector",
        "principle":"robust perturbation survival core + alpha-driven flexible boundary slots",
        "baseline":baseline,
        "chosen_policy":chosen_policy,
        "promotion_gate":bool(chosen_policy is not None),
        "required":{
            "alpha_retention":.95,
            "mean_jaccard":.80,
            "worst_p05_jaccard":.60,
            "top1":.70,
        },
    }
    json.dump(summary,open(out/"summary.json","w"),indent=2,default=str)
    print(table.to_string(index=False))
    print(json.dumps(summary,indent=2,default=str))

if __name__=="__main__":
    main()
