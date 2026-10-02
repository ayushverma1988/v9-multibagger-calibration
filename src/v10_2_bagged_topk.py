from __future__ import annotations

import argparse, hashlib, json
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import pandas as pd

from v10_2_compression_switch import SwitchPolicy, select_switch, perturb_all
from v10_2_raw_rank_stability import spec_from_row, outcome, agg, alpha_pass, select_policy, Policy as RawPolicy

BASE=SwitchPolicy("compress_060_raw20", .060, .20)
INC=RawPolicy("incumbent","p_cal","p_dd30_cal")

@dataclass(frozen=True)
class BagPolicy:
    name:str
    internal_sims:int
    freq_weight:float

POLICIES=[
    BagPolicy("bag50_w50",50,.50),
    BagPolicy("bag50_w75",50,.75),
    BagPolicy("bag50_w100",50,1.00),
    BagPolicy("bag100_w50",100,.50),
    BagPolicy("bag100_w75",100,.75),
    BagPolicy("bag100_w100",100,1.00),
]

def seed_for(date,name,salt):
    h=hashlib.sha256(f"{pd.Timestamp(date).date()}|{name}|{salt}".encode()).hexdigest()
    return int(h[:8],16)

def bag_select(g,spec,bp,k=10):
    base=select_switch(g,spec,BASE,k)
    if len(base)!=k:return pd.DataFrame()

    # Common random numbers make the bagging operator deterministic for a
    # given fold/policy and remove Monte-Carlo noise from stability testing.
    rng=np.random.default_rng(seed_for(g["date"].iloc[0],bp.name,"internal"))
    counts={}
    mean_rank_sum={}
    mean_rank_n={}
    union=set(base.index)

    for _ in range(int(bp.internal_sims)):
        z=perturb_all(g,.01,rng)
        s=select_switch(z,spec,BASE,k)
        if len(s)!=k:continue
        for rank,idx in enumerate(s.index,1):
            union.add(idx)
            counts[idx]=counts.get(idx,0)+1
            mean_rank_sum[idx]=mean_rank_sum.get(idx,0)+rank
            mean_rank_n[idx]=mean_rank_n.get(idx,0)+1

    q=g.loc[list(union)].copy()
    if len(q)<k:return base

    denom=max(1,int(bp.internal_sims))
    q["_freq"]=[counts.get(i,0)/denom for i in q.index]
    q["_mean_rank"]=[
        mean_rank_sum.get(i,0)/mean_rank_n.get(i,1) if mean_rank_n.get(i,0)>0 else k+10
        for i in q.index
    ]

    # Alpha anchor from the raw ensemble score. Frequency is the robust
    # component; p_raw only breaks ties / preserves predictive ordering.
    q["_alpha_rank"]=q["p_raw"].rank(pct=True,method="average")
    q["_bag_score"]=bp.freq_weight*q["_freq"]+(1-bp.freq_weight)*q["_alpha_rank"]

    return q.sort_values(
        ["_bag_score","_freq","_mean_rank","p_raw","p_cal","symbol"],
        ascending=[False,False,True,False,False,True]
    ).head(k).copy()

def external_stability(g,spec,bp,k,sims):
    base=bag_select(g,spec,bp,k)
    if len(base)!=k:return None
    bs=set(base["symbol"].astype(str)); top=str(base.iloc[0]["symbol"])
    rng=np.random.default_rng(seed_for(g["date"].iloc[0],bp.name,"external"))
    js=[];tops=[];rep=[]
    for _ in range(int(sims)):
        z=perturb_all(g,.01,rng)
        s=bag_select(z,spec,bp,k)
        if len(s)!=k:continue
        ss=set(s["symbol"].astype(str))
        js.append(len(bs&ss)/len(bs|ss))
        tops.append(float(str(s.iloc[0]["symbol"])==top))
        rep.append(len(bs-ss)/k)
    if not js:return None
    return {
        "mean_jaccard":float(np.mean(js)),
        "p05_jaccard":float(np.quantile(js,.05)),
        "top1_stability":float(np.mean(tops)),
        "names_replaced_fraction":float(np.mean(rep)),
    }

def incumbent_metrics(oos,cmap,cfg):
    k=int(cfg.get("selection_k",10));rows=[]
    for td,r in cmap.items():
        g=oos[oos["date"]==td].copy()
        s=select_policy(g,spec_from_row(r),INC,k)
        m=outcome(g,s,k)
        if m:rows.append({"date":td,**m})
    return agg(pd.DataFrame(rows))

def evaluate(oos,cmap,cfg,bp,sims):
    k=int(cfg.get("selection_k",10)); mets=[]; sts=[]
    for td,r in cmap.items():
        g=oos[oos["date"]==td].copy()
        if g.empty:continue
        spec=spec_from_row(r)
        s=bag_select(g,spec,bp,k)
        m=outcome(g,s,k)
        if m:mets.append({"date":td,**m})
        st=external_stability(g,spec,bp,k,sims)
        if st:sts.append({"date":td,**st})
    return pd.DataFrame(mets),pd.DataFrame(sts)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--oos",required=True)
    ap.add_argument("--chosen931",required=True)
    ap.add_argument("--config",required=True)
    ap.add_argument("--output",required=True)
    ap.add_argument("--grid-sims",type=int,default=100)
    ap.add_argument("--confirm-sims",type=int,default=500)
    args=ap.parse_args()

    cfg=json.load(open(args.config))
    oos=pd.read_parquet(args.oos);oos["date"]=pd.to_datetime(oos["date"])
    chosen=pd.read_csv(args.chosen931);chosen["date"]=pd.to_datetime(chosen["date"])
    cmap={pd.Timestamp(r.date):r for r in chosen.itertuples(index=False)}
    baseline=incumbent_metrics(oos,cmap,cfg)

    rows=[]
    for bp in POLICIES:
        mt,st=evaluate(oos,cmap,cfg,bp,args.grid_sims)
        a=agg(mt)
        if not a or st.empty:continue
        mj=float(st["mean_jaccard"].mean());wp=float(st["p05_jaccard"].min());t1=float(st["top1_stability"].mean())
        rows.append({
            **asdict(bp),**a,
            "mean_jaccard":mj,"worst_p05_jaccard":wp,"mean_top1_stability":t1,
            "mean_names_replaced_fraction":float(st["names_replaced_fraction"].mean()),
            "worst_fold":str(st.loc[st["p05_jaccard"].idxmin(),"date"]),
            "alpha_retention_passed":bool(alpha_pass(a,baseline,.95)),
            "stability_passed":bool(mj>=.80 and wp>=.60 and t1>=.70),
            "dd_not_worse_3pp":bool(a["mean_dd30_rate"]<=baseline["mean_dd30_rate"]+.03),
        })

    tab=pd.DataFrame(rows)
    tab["all_gates_passed"]=tab["alpha_retention_passed"]&tab["stability_passed"]&tab["dd_not_worse_3pp"]
    tab=tab.sort_values(
        ["stability_passed","worst_p05_jaccard","mean_jaccard","internal_sims","freq_weight"],
        ascending=[False,False,False,True,True]
    )

    stable=tab[tab["stability_passed"]]
    chosen_name=str(stable.iloc[0]["name"]) if len(stable) else (str(tab.iloc[0]["name"]) if len(tab) else None)
    bp=next((x for x in POLICIES if x.name==chosen_name),None)

    confirmation=None
    if bp:
        mt,st=evaluate(oos,cmap,cfg,bp,args.confirm_sims)
        a=agg(mt);mj=float(st["mean_jaccard"].mean());wp=float(st["p05_jaccard"].min());t1=float(st["top1_stability"].mean())
        confirmation={
            **a,"mean_jaccard":mj,"worst_p05_jaccard":wp,"mean_top1_stability":t1,
            "mean_names_replaced_fraction":float(st["names_replaced_fraction"].mean()),
            "worst_fold":str(st.loc[st["p05_jaccard"].idxmin(),"date"]),
            "alpha_retention_passed":bool(alpha_pass(a,baseline,.95)),
            "mean_jaccard_passed":bool(mj>=.80),
            "worst_p05_passed":bool(wp>=.60),
            "top1_passed":bool(t1>=.70),
            "dd_not_worse_3pp":bool(a["mean_dd30_rate"]<=baseline["mean_dd30_rate"]+.03),
        }
        confirmation["all_gates_passed"]=bool(
            confirmation["alpha_retention_passed"] and confirmation["mean_jaccard_passed"] and
            confirmation["worst_p05_passed"] and confirmation["top1_passed"] and
            confirmation["dd_not_worse_3pp"]
        )

    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    tab.to_csv(out/"bagged_grid.csv",index=False)
    if bp:
        mt,st=evaluate(oos,cmap,cfg,bp,args.confirm_sims)
        mt.to_csv(out/"chosen_metrics_by_fold.csv",index=False)
        st.to_csv(out/"chosen_stability_by_fold.csv",index=False)
    summary={
        "model":"V10.2 bagged Top-K stability selection",
        "principle":"deterministic inclusion-frequency bagging under fixed 1% score perturbations; no labels used for selection",
        "base_selector":"compress_060_raw20",
        "baseline":baseline,
        "chosen_policy":asdict(bp) if bp else None,
        "confirmation_500":confirmation,
        "promotion_gate":bool(confirmation and confirmation["all_gates_passed"]),
    }
    json.dump(summary,open(out/"summary.json","w"),indent=2,default=str)
    print(tab.to_string(index=False))
    print(json.dumps(summary,indent=2,default=str))

if __name__=="__main__":
    main()
