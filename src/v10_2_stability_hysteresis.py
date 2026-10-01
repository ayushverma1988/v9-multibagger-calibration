from __future__ import annotations

import argparse, json, hashlib
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Policy:
    name: str
    max_replacements: int
    keep_rank_limit: int
    fragile_only: bool
    fragility_p05_trigger: float = 0.60
    internal_sims: int = 60


POLICIES=[
    Policy("incumbent",10,10,False),
    Policy("sticky8_top15",2,15,False),
    Policy("sticky8_top20",2,20,False),
    Policy("sticky8_top30",2,30,False),
    Policy("sticky7_top20",3,20,False),
    Policy("sticky6_top20",4,20,False),
    Policy("fragile_sticky8_top20",2,20,True),
    Policy("fragile_sticky7_top20",3,20,True),
    Policy("fragile_sticky6_top20",4,20,True),
]


def seed(td,name,salt):
    h=hashlib.sha256(f"{pd.Timestamp(td).date()}|{name}|{salt}".encode()).hexdigest()
    return int(h[:8],16)


def spec_from_row(r):
    return dict(
        pool_n=int(r.pool_n),risk_drop=float(r.risk_drop),
        w_safety=float(r.w_safety),w_consensus=float(r.w_consensus),
        baseline=bool(r.fell_back_to_v92),
    )


def base_ranked(g,spec,k=10):
    q=g.dropna(subset=["p_cal","p_dd30_cal","model_dispersion"]).copy()
    if len(q)<k:return pd.DataFrame()
    if spec["baseline"]:
        q["selection_score"]=q["p_cal"]
        return q.sort_values(["selection_score","p_cal","model_dispersion","symbol"],
                             ascending=[False,False,True,True]).copy()
    pool=q.sort_values(["p_cal","model_dispersion"],ascending=[False,True]).head(min(spec["pool_n"],len(q))).copy()
    if len(pool)<k:return pd.DataFrame()
    if spec["risk_drop"]>0:
        cutoff=float(pool["p_dd30_cal"].quantile(np.clip(1-spec["risk_drop"],0,1)))
        pool=pool[pool["p_dd30_cal"]<=cutoff].copy()
    if len(pool)<k:return pd.DataFrame()
    pool["comp_alpha"]=pool["p_cal"].rank(pct=True,method="average")
    pool["comp_safety"]=pool["p_dd30_cal"].rank(pct=True,method="average",ascending=False)
    pool["comp_consensus"]=pool["model_dispersion"].rank(pct=True,method="average",ascending=False)
    wa=max(0.0,1-spec["w_safety"]-spec["w_consensus"])
    pool["selection_score"]=wa*pool["comp_alpha"]+spec["w_safety"]*pool["comp_safety"]+spec["w_consensus"]*pool["comp_consensus"]
    return pool.sort_values(["selection_score","p_cal","model_dispersion","symbol"],
                            ascending=[False,False,True,True]).copy()


def perturb(g,rng,sigma=.01):
    z=g.copy()
    for c in ["p_cal","p_dd30_cal","model_dispersion"]:
        a=pd.to_numeric(z[c],errors="coerce").to_numpy(float)
        a=a*np.exp(rng.normal(0,sigma,len(a)))
        if c!="model_dispersion": a=np.clip(a,1e-8,1-1e-8)
        else: a=np.maximum(a,1e-12)
        z[c]=a
    return z


def internal_fragility(g,spec,k,policy):
    base=base_ranked(g,spec,k).head(k)
    if len(base)!=k:return 0.0
    b=set(base["symbol"].astype(str))
    rng=np.random.default_rng(seed(g["date"].iloc[0],policy.name,"frag"))
    js=[]
    for _ in range(policy.internal_sims):
        s=base_ranked(perturb(g,rng),spec,k).head(k)
        if len(s)!=k:continue
        ss=set(s["symbol"].astype(str))
        js.append(len(b&ss)/len(b|ss))
    return float(np.quantile(js,.05)) if js else 0.0


def select_policy(g,spec,policy,prev_selected,k=10,fragility_override=None):
    ranked=base_ranked(g,spec,k)
    if len(ranked)<k:return pd.DataFrame(),False,np.nan
    if policy.name=="incumbent" or not prev_selected:
        return ranked.head(k).copy(),False,np.nan

    frag=internal_fragility(g,spec,k,policy) if fragility_override is None else fragility_override
    if policy.fragile_only and frag>=policy.fragility_p05_trigger:
        return ranked.head(k).copy(),False,frag

    ranked=ranked.copy()
    ranked["_rank"]=np.arange(1,len(ranked)+1)
    eligible_prev=ranked[
        ranked["symbol"].astype(str).isin(prev_selected)
        & (ranked["_rank"]<=policy.keep_rank_limit)
    ].copy()

    min_keep=max(0,k-policy.max_replacements)
    keep=eligible_prev.sort_values("_rank").head(min_keep).copy()

    # If fewer than the target number remain competitive, retain all competitive
    # prior names and fill the remainder from today's ranking.
    keep_syms=set(keep["symbol"].astype(str))
    need=k-len(keep)
    fill=ranked[~ranked["symbol"].astype(str).isin(keep_syms)].head(need).copy()
    sel=pd.concat([keep,fill],axis=0)
    sel=sel.sort_values(["_rank","symbol"],ascending=[True,True]).head(k).copy()
    return sel,True,frag


def sequential_unperturbed(oos,cmap,policy,k):
    selections={}
    diagnostics=[]
    prev=set()
    for td in sorted(cmap):
        g=oos[oos["date"]==td].copy()
        if g.empty:continue
        sel,active,frag=select_policy(g,spec_from_row(cmap[td]),policy,prev,k)
        if len(sel)!=k:continue
        syms=set(sel["symbol"].astype(str))
        selections[td]=syms
        diagnostics.append(dict(date=td,active=active,fragility_p05=frag,
                                retained_from_prev=len(prev&syms) if prev else 0))
        prev=syms
    return selections,pd.DataFrame(diagnostics)


def outcome(oos,cmap,selections,k):
    rows=[]
    for td,syms in selections.items():
        g=oos[oos["date"]==td].copy()
        q=g.dropna(subset=["y6","dd30_6m","p_cal"])
        s=q[q["symbol"].astype(str).isin(syms)].copy()
        if len(s)!=k or q.empty:continue
        br=float(q["y6"].mean());pr=float(s["y6"].mean())
        rows.append(dict(date=td,precision_2x=pr,lift_2x=pr/br if br>0 else np.nan,
                         hit=float(pr>0),dd30_rate=float(s["dd30_6m"].mean())))
    return pd.DataFrame(rows)


def aggregate(t):
    lift=np.clip(t["lift_2x"].replace([np.inf,-np.inf],np.nan).dropna(),0,10)
    return dict(folds=int(len(t)),mean_precision_2x=float(t["precision_2x"].mean()),
                median_precision_2x=float(t["precision_2x"].median()),
                mean_capped_lift_2x=float(lift.mean()) if len(lift) else np.nan,
                hit_fold_rate=float(t["hit"].mean()),
                mean_dd30_rate=float(t["dd30_rate"].mean()))


def alpha_pass(a,b,r=.95):
    for f in ["mean_precision_2x","median_precision_2x","mean_capped_lift_2x","hit_fold_rate"]:
        if not(np.isfinite(a[f]) and np.isfinite(b[f])):return False
        if b[f]>0 and a[f]+1e-12<r*b[f]:return False
    return True


def external_stability(oos,cmap,policy,base_selections,k,sims,sigma=.01):
    rows=[]
    # For each fold the previous portfolio is the policy's unperturbed previous
    # fold selection. That is exactly the information available in live use.
    dates=sorted([d for d in cmap if d in base_selections])
    for i,td in enumerate(dates):
        g=oos[oos["date"]==td].copy()
        prev=base_selections.get(dates[i-1],set()) if i>0 else set()
        base=base_selections[td]
        frag=internal_fragility(g,spec_from_row(cmap[td]),k,policy) if policy.name!="incumbent" else np.nan
        rng=np.random.default_rng(seed(td,policy.name,"external"))
        js=[];tops=[];repls=[]
        for _ in range(sims):
            zp=perturb(g,rng,sigma)
            s,_,_=select_policy(zp,spec_from_row(cmap[td]),policy,prev,k,fragility_override=frag)
            if len(s)!=k:continue
            ss=set(s["symbol"].astype(str))
            js.append(len(base&ss)/len(base|ss))
            # top1 is interpreted as the highest current base-ranked name in final selection
            btop=base_ranked(g,spec_from_row(cmap[td]),k).iloc[0]["symbol"]
            ptop=base_ranked(zp,spec_from_row(cmap[td]),k).iloc[0]["symbol"]
            tops.append(float(str(btop)==str(ptop)))
            repls.append(len(base-ss)/k)
        if js:
            rows.append(dict(date=td,mean_jaccard=float(np.mean(js)),
                             p05_jaccard=float(np.quantile(js,.05)),
                             top1_stability=float(np.mean(tops)),
                             names_replaced_fraction=float(np.mean(repls))))
    return pd.DataFrame(rows)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--oos",required=True)
    ap.add_argument("--chosen931",required=True)
    ap.add_argument("--config",required=True)
    ap.add_argument("--output",required=True)
    ap.add_argument("--sims",type=int,default=200)
    args=ap.parse_args()
    cfg=json.load(open(args.config));k=int(cfg.get("selection_k",10))
    oos=pd.read_parquet(args.oos);oos["date"]=pd.to_datetime(oos["date"])
    chosen=pd.read_csv(args.chosen931);chosen["date"]=pd.to_datetime(chosen["date"])
    cmap={pd.Timestamp(r.date):r for r in chosen.itertuples(index=False)}

    results=[];stab_all=[];diag_all=[]
    baseline=None
    policy_data={}
    for p in POLICIES:
        sels,diag=sequential_unperturbed(oos,cmap,p,k)
        ot=outcome(oos,cmap,sels,k)
        agg=aggregate(ot)
        if p.name=="incumbent":baseline=agg
        policy_data[p.name]=(sels,diag,ot,agg)

    for p in POLICIES:
        sels,diag,ot,agg=policy_data[p.name]
        st=external_stability(oos,cmap,p,sels,k,args.sims)
        passed_alpha=alpha_pass(agg,baseline,.95)
        meanj=float(st["mean_jaccard"].mean()) if len(st) else np.nan
        worst=float(st["p05_jaccard"].min()) if len(st) else np.nan
        top1=float(st["top1_stability"].mean()) if len(st) else np.nan
        passed_stab=bool(meanj>=.80 and worst>=.60 and top1>=.70)
        results.append({**asdict(p),**agg,
                        "mean_jaccard":meanj,"worst_p05_jaccard":worst,
                        "mean_top1_stability":top1,
                        "mean_names_replaced_fraction":float(st["names_replaced_fraction"].mean()) if len(st) else np.nan,
                        "alpha_retention_passed":passed_alpha,
                        "stability_passed":passed_stab,
                        "all_gates_passed":bool(passed_alpha and passed_stab)})
        if len(st):
            x=st.copy();x["policy"]=p.name;stab_all.append(x)
        if len(diag):
            x=diag.copy();x["policy"]=p.name;diag_all.append(x)

    table=pd.DataFrame(results).sort_values(
        ["all_gates_passed","stability_passed","mean_jaccard","mean_precision_2x"],
        ascending=[False,False,False,False])
    passing=table[table["all_gates_passed"]]
    chosen_name=str(passing.iloc[0]["name"]) if len(passing) else None

    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    table.to_csv(out/"policy_comparison.csv",index=False)
    if stab_all:pd.concat(stab_all,ignore_index=True).to_csv(out/"stability_by_fold.csv",index=False)
    if diag_all:pd.concat(diag_all,ignore_index=True).to_csv(out/"policy_diagnostics.csv",index=False)
    summary={
        "model":"V10.2 stability fix stage 2 - turnover-aware hysteresis",
        "scope":"selector only; uses only current scores plus previous known portfolio",
        "baseline":baseline,
        "chosen_policy":chosen_name,
        "promotion_gate":bool(chosen_name is not None),
        "required":{"alpha_retention":.95,"mean_jaccard":.80,"worst_p05_jaccard":.60,"top1":.70},
        "best_rows":table.head(5).to_dict(orient="records"),
    }
    json.dump(summary,open(out/"summary.json","w"),indent=2,default=str)
    print(table.to_string(index=False))
    print(json.dumps(summary,indent=2,default=str))

if __name__=="__main__":
    main()
