from __future__ import annotations

import argparse, hashlib, json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

from v10_2_compression_switch import (
    SwitchPolicy, spec_from_row, select_switch, perturb_all
)
from v10_2_raw_rank_stability import outcome, agg, alpha_pass, select_policy, Policy as RawPolicy

BASE_SWITCH = SwitchPolicy("compress_030_raw50", .030, .50)
INCUMBENT = RawPolicy("incumbent","p_cal","p_dd30_cal")


@dataclass(frozen=True)
class CorePolicy:
    name:str
    core_n:int
    internal_sims:int
    fragile_mean_gate:float=.80
    fragile_p05_gate:float=.60


POLICIES=[
    CorePolicy("crn_core8_i25",8,25),
    CorePolicy("crn_core9_i25",9,25),
    CorePolicy("crn_core8_i40",8,40),
    CorePolicy("crn_core9_i40",9,40),
]


def seed_for(date,name,salt):
    raw=f"{pd.Timestamp(date).date()}|{name}|{salt}".encode()
    return int(hashlib.sha256(raw).hexdigest()[:8],16)


def base_select(g,spec,k):
    return select_switch(g,spec,BASE_SWITCH,k)


def simulated_sets(g,spec,k,cp):
    # Common random numbers: same internal perturbation stream is used for
    # unperturbed and externally perturbed copies of the same fold. This
    # removes Monte-Carlo estimator noise from the stability comparison.
    rng=np.random.default_rng(seed_for(g["date"].iloc[0],cp.name,"internal_crn"))
    sets=[]; counts={}; rank_sum={}; rank_n={}
    for _ in range(int(cp.internal_sims)):
        z=perturb_all(g,.01,rng)
        s=base_select(z,spec,k)
        if len(s)!=k:
            continue
        syms=s["symbol"].astype(str).tolist()
        sets.append(set(syms))
        for rank,sym in enumerate(syms,1):
            counts[sym]=counts.get(sym,0)+1
            rank_sum[sym]=rank_sum.get(sym,0)+rank
            rank_n[sym]=rank_n.get(sym,0)+1
    return sets,counts,rank_sum,rank_n


def stability_vs_base(base_symbols,sets):
    b=set(map(str,base_symbols))
    js=[len(b&s)/len(b|s) for s in sets if (b or s)]
    if not js:
        return np.nan,np.nan
    return float(np.mean(js)),float(np.quantile(js,.05))


def stable_core_select(g,spec,cp,k=10):
    base=base_select(g,spec,k)
    if len(base)!=k:
        return pd.DataFrame()

    base_syms=base["symbol"].astype(str).tolist()
    sets,counts,rank_sum,rank_n=simulated_sets(g,spec,k,cp)
    meanj,p05=stability_vs_base(base_syms,sets)

    if np.isfinite(meanj) and np.isfinite(p05) and meanj>=cp.fragile_mean_gate and p05>=cp.fragile_p05_gate:
        out=base.copy()
        out["_stable_core_used"]=0.0
        return out

    symbols=set(base_syms)
    for s in sets:
        symbols |= s
    q=g[g["symbol"].astype(str).isin(symbols)].copy()
    if len(q)<k:
        return base

    base_order={sym:i for i,sym in enumerate(base_syms,1)}
    denom=max(1,len(sets))
    rec=[]
    for r in q.itertuples():
        sym=str(r.symbol)
        freq=counts.get(sym,0)/denom
        mr=rank_sum.get(sym,0)/rank_n.get(sym,1) if rank_n.get(sym,0)>0 else k+10
        rec.append((r.Index,sym,freq,mr,base_order.get(sym,k+20),
                    float(getattr(r,"p_raw",np.nan)),
                    float(getattr(r,"p_cal",np.nan))))
    tab=pd.DataFrame(rec,columns=["idx","symbol","freq","mean_rank","base_rank","p_raw","p_cal"]).set_index("idx")

    core=tab.sort_values(
        ["freq","mean_rank","base_rank","p_raw","symbol"],
        ascending=[False,True,True,False,True]
    ).head(int(cp.core_n))

    selected=list(core.index)
    for idx,_ in base.iterrows():
        if idx not in selected:
            selected.append(idx)
        if len(selected)>=k:
            break
    if len(selected)<k:
        rest=tab.loc[~tab.index.isin(selected)].sort_values(
            ["freq","mean_rank","p_raw","symbol"],
            ascending=[False,True,False,True]
        )
        selected.extend(rest.index[:k-len(selected)].tolist())

    out=g.loc[selected[:k]].copy()
    if len(out)!=k:
        return base
    out["_stable_core_used"]=1.0
    return out


def external_stability(g,spec,cp,k,sims):
    base=stable_core_select(g,spec,cp,k)
    if len(base)!=k:
        return None
    bs=set(base["symbol"].astype(str))
    top=str(base.iloc[0]["symbol"])
    js=[];tops=[];repl=[]
    rng=np.random.default_rng(seed_for(g["date"].iloc[0],cp.name,"external"))
    for _ in range(int(sims)):
        z=perturb_all(g,.01,rng)
        s=stable_core_select(z,spec,cp,k)
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


def incumbent_metrics(oos,cmap,cfg):
    k=int(cfg.get("selection_k",10)); rows=[]
    for td,r in cmap.items():
        g=oos[oos["date"]==td].copy()
        if g.empty: continue
        s=select_policy(g,spec_from_row(r),INCUMBENT,k)
        m=outcome(g,s,k)
        if m: rows.append({"date":td,**m})
    return agg(pd.DataFrame(rows))


def evaluate(oos,cmap,cfg,cp,sims):
    k=int(cfg.get("selection_k",10)); mets=[]; sts=[]; uses=[]
    for td,r in cmap.items():
        g=oos[oos["date"]==td].copy()
        if g.empty: continue
        spec=spec_from_row(r)
        s=stable_core_select(g,spec,cp,k)
        m=outcome(g,s,k)
        if m: mets.append({"date":td,**m})
        uses.append({"date":td,"stable_core_used":float(s["_stable_core_used"].iloc[0]) if len(s) and "_stable_core_used" in s else 0})
        st=external_stability(g,spec,cp,k,sims)
        if st: sts.append({"date":td,**st})
    return pd.DataFrame(mets),pd.DataFrame(sts),pd.DataFrame(uses)


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
    oos=pd.read_parquet(args.oos); oos["date"]=pd.to_datetime(oos["date"])
    chosen=pd.read_csv(args.chosen931); chosen["date"]=pd.to_datetime(chosen["date"])
    cmap={pd.Timestamp(r.date):r for r in chosen.itertuples(index=False)}
    baseline=incumbent_metrics(oos,cmap,cfg)

    rows=[]
    for cp in POLICIES:
        mt,st,use=evaluate(oos,cmap,cfg,cp,args.grid_sims)
        a=agg(mt)
        if not a or st.empty: continue
        mj=float(st["mean_jaccard"].mean()); wp=float(st["p05_jaccard"].min()); t1=float(st["top1_stability"].mean())
        apass=alpha_pass(a,baseline,.95)
        dd=a["mean_dd30_rate"]<=baseline["mean_dd30_rate"]+.03
        rows.append({
            **asdict(cp),**a,
            "mean_jaccard":mj,"worst_p05_jaccard":wp,"mean_top1_stability":t1,
            "mean_names_replaced_fraction":float(st["names_replaced_fraction"].mean()),
            "worst_fold":str(st.loc[st["p05_jaccard"].idxmin(),"date"]),
            "stable_core_folds":int(use["stable_core_used"].sum()),
            "alpha_retention_passed":bool(apass),
            "stability_passed":bool(mj>=.80 and wp>=.60 and t1>=.70),
            "dd_not_worse_3pp":bool(dd),
        })

    table=pd.DataFrame(rows)
    table["all_gates_passed"]=table["alpha_retention_passed"] & table["stability_passed"] & table["dd_not_worse_3pp"]
    table=table.sort_values(
        ["all_gates_passed","stability_passed","worst_p05_jaccard","mean_jaccard","mean_precision_2x"],
        ascending=[False,False,False,False,False]
    )

    # Policy choice is stability-first and label-independent: choose the least
    # complex policy among those clearing the stability gate. Alpha is checked
    # only after this choice.
    stable=table[table["stability_passed"]]
    if len(stable):
        chosen_name=str(stable.sort_values(["internal_sims","core_n","mean_jaccard"],ascending=[True,True,False]).iloc[0]["name"])
    else:
        chosen_name=str(table.iloc[0]["name"]) if len(table) else None
    chosen_cp=next((p for p in POLICIES if p.name==chosen_name),None)

    confirmation=None
    if chosen_cp:
        mt,st,use=evaluate(oos,cmap,cfg,chosen_cp,args.confirm_sims)
        a=agg(mt)
        mj=float(st["mean_jaccard"].mean()); wp=float(st["p05_jaccard"].min()); t1=float(st["top1_stability"].mean())
        confirmation={
            **a,
            "mean_jaccard":mj,
            "worst_p05_jaccard":wp,
            "mean_top1_stability":t1,
            "mean_names_replaced_fraction":float(st["names_replaced_fraction"].mean()),
            "worst_fold":str(st.loc[st["p05_jaccard"].idxmin(),"date"]),
            "stable_core_folds":int(use["stable_core_used"].sum()),
            "alpha_retention_passed":bool(alpha_pass(a,baseline,.95)),
            "mean_jaccard_passed":bool(mj>=.80),
            "worst_p05_passed":bool(wp>=.60),
            "top1_passed":bool(t1>=.70),
            "dd_not_worse_3pp":bool(a["mean_dd30_rate"]<=baseline["mean_dd30_rate"]+.03),
        }
        confirmation["all_gates_passed"]=bool(
            confirmation["alpha_retention_passed"] and confirmation["mean_jaccard_passed"]
            and confirmation["worst_p05_passed"] and confirmation["top1_passed"]
            and confirmation["dd_not_worse_3pp"]
        )

    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    table.to_csv(out/"crn_grid.csv",index=False)
    if chosen_cp:
        mt,st,use=evaluate(oos,cmap,cfg,chosen_cp,args.confirm_sims)
        mt.to_csv(out/"chosen_metrics_by_fold.csv",index=False)
        st.to_csv(out/"chosen_stability_by_fold.csv",index=False)
        use.to_csv(out/"chosen_core_usage.csv",index=False)
    summary={
        "model":"V10.2 Stable-Core CRN",
        "principle":"deterministic stability-selection core with common random numbers across sensitivity comparisons",
        "base_selector":"compress_030_raw50",
        "baseline":baseline,
        "chosen_policy":asdict(chosen_cp) if chosen_cp else None,
        "confirmation_500":confirmation,
        "promotion_gate":bool(confirmation and confirmation["all_gates_passed"]),
    }
    json.dump(summary,open(out/"summary.json","w"),indent=2,default=str)
    print(table.to_string(index=False))
    print(json.dumps(summary,indent=2,default=str))

if __name__=="__main__":
    main()
