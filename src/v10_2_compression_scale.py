from __future__ import annotations

import argparse, json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

from v10_2_compression_switch import compression_span, perturb_all
from v10_2_raw_rank_stability import (
    Policy as RawPolicy, spec_from_row, select_policy,
    outcome, agg, alpha_pass, seed_for
)

INC=RawPolicy("incumbent","p_cal","p_dd30_cal")


@dataclass(frozen=True)
class Policy:
    name:str
    span_trigger:float
    raw_secondary_scale:float


POLICIES=[
    Policy("compress_040_raw20",.04,.20),
    Policy("compress_040_raw50",.04,.50),
    Policy("compress_050_raw20",.05,.20),
    Policy("compress_050_raw50",.05,.50),
    Policy("compress_060_raw20",.06,.20),
    Policy("compress_060_raw50",.06,.50),
]


def raw_policy(scale):
    return RawPolicy(f"raw_s{scale}","p_raw","p_dd30_raw",secondary_scale=scale)


def select_switch(g,spec,p,k):
    span=compression_span(g,spec)
    use_raw=span<=p.span_trigger
    rp=raw_policy(p.raw_secondary_scale) if use_raw else INC
    s=select_policy(g,spec,rp,k)
    if len(s):
        s=s.copy()
        s["_span"]=span
        s["_used_raw"]=float(use_raw)
    return s


def stability(g,spec,p,k,sims=500):
    base=select_switch(g,spec,p,k)
    if len(base)!=k:return None
    bs=set(base["symbol"].astype(str));top=str(base.iloc[0]["symbol"])
    rng=np.random.default_rng(seed_for(g["date"].iloc[0],p.name,"external"))
    js=[];tops=[];rep=[];switch=[]
    for _ in range(sims):
        z=perturb_all(g,.01,rng)
        s=select_switch(z,spec,p,k)
        if len(s)!=k:continue
        ss=set(s["symbol"].astype(str))
        js.append(len(bs&ss)/len(bs|ss))
        tops.append(float(str(s.iloc[0]["symbol"])==top))
        rep.append(len(bs-ss)/k)
        switch.append(float(s["_used_raw"].iloc[0]))
    if not js:return None
    return {
        "mean_jaccard":float(np.mean(js)),
        "p05_jaccard":float(np.quantile(js,.05)),
        "top1_stability":float(np.mean(tops)),
        "names_replaced_fraction":float(np.mean(rep)),
        "raw_fraction_under_perturbation":float(np.mean(switch)),
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
    d=pd.read_parquet(args.oos);d["date"]=pd.to_datetime(d["date"])
    chosen=pd.read_csv(args.chosen931);chosen["date"]=pd.to_datetime(chosen["date"])
    cmap={pd.Timestamp(r.date):r for r in chosen.itertuples(index=False)}

    base_rows=[]
    for td,r in cmap.items():
        g=d[d["date"]==td].copy()
        s=select_policy(g,spec_from_row(r),INC,k)
        m=outcome(g,s,k)
        if m:base_rows.append({"date":td,**m})
    baseline=agg(pd.DataFrame(base_rows))

    mets=[];sts=[];sw=[]
    for p in POLICIES:
        for td,r in cmap.items():
            g=d[d["date"]==td].copy()
            if g.empty:continue
            spec=spec_from_row(r)
            span=compression_span(g,spec)
            s=select_switch(g,spec,p,k)
            m=outcome(g,s,k)
            if m:mets.append({"policy":p.name,"date":td,"span":span,"used_raw":float(span<=p.span_trigger),**m})
            st=stability(g,spec,p,k,args.sims)
            if st:sts.append({"policy":p.name,"date":td,"span":span,**st})
            sw.append({"policy":p.name,"date":td,"span":span,"used_raw":float(span<=p.span_trigger)})

    mt=pd.DataFrame(mets);st=pd.DataFrame(sts);sw=pd.DataFrame(sw)
    rows=[]
    for p in POLICIES:
        a=agg(mt[mt["policy"]==p.name])
        s=st[st["policy"]==p.name]
        u=sw[sw["policy"]==p.name]
        if not a or s.empty:continue
        mj=float(s["mean_jaccard"].mean())
        wp=float(s["p05_jaccard"].min())
        t1=float(s["top1_stability"].mean())
        apass=alpha_pass(a,baseline,.95)
        spass=bool(mj>=.80 and wp>=.60 and t1>=.70)
        dd=bool(a["mean_dd30_rate"]<=baseline["mean_dd30_rate"]+.03)
        rows.append({
            **asdict(p),**a,
            "mean_jaccard":mj,
            "worst_p05_jaccard":wp,
            "mean_top1_stability":t1,
            "mean_names_replaced_fraction":float(s["names_replaced_fraction"].mean()),
            "worst_fold":str(s.loc[s["p05_jaccard"].idxmin(),"date"]),
            "raw_switch_folds":int(u["used_raw"].sum()),
            "alpha_retention_passed":bool(apass),
            "stability_passed":spass,
            "dd_not_worse_3pp":dd,
            "all_gates_passed":bool(apass and spass and dd),
        })

    table=pd.DataFrame(rows).sort_values(
        ["stability_passed","span_trigger","raw_secondary_scale"],
        ascending=[False,True,True]
    )

    # Select by stability only: least intervention (smallest trigger, then
    # smallest secondary contribution) that clears all stability thresholds.
    stable=table[table["stability_passed"]]
    chosen_name=str(stable.iloc[0]["name"]) if len(stable) else None
    chosen_row=table[table["name"]==chosen_name].iloc[0].to_dict() if chosen_name else None
    promotion=bool(chosen_row and chosen_row["alpha_retention_passed"] and chosen_row["dd_not_worse_3pp"])

    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    mt.to_csv(out/"metrics_by_fold.csv",index=False)
    st.to_csv(out/"stability_by_fold.csv",index=False)
    sw.to_csv(out/"switches_by_fold.csv",index=False)
    table.to_csv(out/"policy_comparison.csv",index=False)
    summary={
        "model":"V10.2 perturbation-scale compression trigger",
        "principle":"compression threshold expressed as 4-6x the fixed 1% perturbation stress; policy chosen by stability only before outcome-retention check",
        "baseline":baseline,
        "chosen_policy":chosen_row,
        "promotion_gate":promotion,
        "required":{"mean_jaccard":.80,"worst_p05_jaccard":.60,"top1":.70,"alpha_retention":.95},
    }
    json.dump(summary,open(out/"summary.json","w"),indent=2,default=str)
    print(table.to_string(index=False))
    print(json.dumps(summary,indent=2,default=str))

if __name__=="__main__":
    main()
