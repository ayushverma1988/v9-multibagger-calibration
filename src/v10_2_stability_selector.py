from __future__ import annotations

import argparse, json, hashlib
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit


@dataclass(frozen=True)
class Policy:
    name: str
    mode: str = "rank"
    pool_floor: int = 0
    consensus_weight: float = 0.0
    internal_sigma: float = 0.01
    internal_sims: int = 40


POLICIES = [
    Policy("incumbent"),
    Policy("smooth", mode="smooth"),
    Policy("smooth_floor30", mode="smooth", pool_floor=30),
    Policy("smooth_floor50", mode="smooth", pool_floor=50),
    Policy("cons_rank25", consensus_weight=0.25),
    Policy("cons_smooth25_f30", mode="smooth", pool_floor=30, consensus_weight=0.25),
    Policy("cons_smooth35_f30", mode="smooth", pool_floor=30, consensus_weight=0.35),
    Policy("cons_smooth35_f50", mode="smooth", pool_floor=50, consensus_weight=0.35),
]


def spec_from_row(r):
    return {
        "pool_n": int(r.pool_n),
        "risk_drop": float(r.risk_drop),
        "w_safety": float(r.w_safety),
        "w_consensus": float(r.w_consensus),
        "baseline": bool(r.fell_back_to_v92),
    }


def rank_high(s):
    return s.rank(pct=True, method="average")


def rank_low(s):
    return s.rank(pct=True, method="average", ascending=False)


def robust_component(values, higher_better=True):
    x=np.asarray(values,float)
    med=np.nanmedian(x)
    q25=np.nanquantile(x,.25)
    q75=np.nanquantile(x,.75)
    scale=max(q75-q25,1e-9)
    z=(x-med)/scale
    if not higher_better:
        z=-z
    return expit(np.clip(z,-8,8))


def candidate_scores(g, spec, policy, k=10):
    q=g.dropna(subset=["p_cal","p_dd30_cal","model_dispersion"]).copy()
    if len(q)<k:
        return pd.DataFrame()

    if spec["baseline"]:
        surv=q.copy()
        cutoff=np.nan
    else:
        pool_n=min(max(int(spec["pool_n"]),int(policy.pool_floor)),len(q))
        pool=q.sort_values(["p_cal","model_dispersion"],ascending=[False,True]).head(pool_n).copy()
        if len(pool)<k:
            return pd.DataFrame()
        rd=float(spec["risk_drop"])
        if rd>0:
            cutoff=float(pool["p_dd30_cal"].quantile(np.clip(1-rd,0,1)))
            surv=pool[pool["p_dd30_cal"]<=cutoff].copy()
        else:
            cutoff=float(pool["p_dd30_cal"].max())
            surv=pool.copy()
        if len(surv)<k:
            return pd.DataFrame()

    ws=float(spec.get("w_safety",0.0))
    wc=float(spec.get("w_consensus",0.0))
    wa=max(0.0,1.0-ws-wc)

    if spec["baseline"]:
        surv["base_score"]=surv["p_cal"]
    elif policy.mode=="smooth":
        pa=np.clip(surv["p_cal"].to_numpy(float),1e-6,1-1e-6)
        pdn=np.clip(surv["p_dd30_cal"].to_numpy(float),1e-6,1-1e-6)
        disp=np.maximum(surv["model_dispersion"].to_numpy(float),1e-12)
        alpha=robust_component(logit(pa),True)
        safety=robust_component(logit(pdn),False)
        consensus=robust_component(np.log(disp),False)
        surv["base_score"]=wa*alpha+ws*safety+wc*consensus
    else:
        surv["base_score"]=(
            wa*rank_high(surv["p_cal"])
            +ws*rank_low(surv["p_dd30_cal"])
            +wc*rank_low(surv["model_dispersion"])
        )

    surv["base_rank"]=rank_high(surv["base_score"])
    surv["risk_cutoff"]=cutoff
    return surv


def deterministic_seed(date, policy_name, salt):
    raw=f"{pd.Timestamp(date).date()}|{policy_name}|{salt}".encode()
    return int(hashlib.sha256(raw).hexdigest()[:8],16)


def perturb(g, sigma, rng):
    z=g.copy()
    for col in ["p_cal","p_dd30_cal","model_dispersion"]:
        a=pd.to_numeric(z[col],errors="coerce").to_numpy(float)
        a=a*np.exp(rng.normal(0,sigma,len(a)))
        if col!="model_dispersion":
            a=np.clip(a,1e-8,1-1e-8)
        else:
            a=np.maximum(a,1e-12)
        z[col]=a
    return z


def select_once(g,spec,policy,k=10,with_consensus=True,salt="base"):
    base=candidate_scores(g,spec,policy,k)
    if len(base)<k:
        return pd.DataFrame()

    if policy.consensus_weight<=0 or not with_consensus:
        return base.sort_values(
            ["base_score","p_cal","model_dispersion","symbol"],
            ascending=[False,False,True,True]
        ).head(k).copy()

    freq=pd.Series(0.0,index=g.index)
    rng=np.random.default_rng(deterministic_seed(g["date"].iloc[0],policy.name,salt))
    for _ in range(int(policy.internal_sims)):
        zp=perturb(g,policy.internal_sigma,rng)
        s=select_once(zp,spec,Policy(
            name=policy.name+"_inner",
            mode=policy.mode,
            pool_floor=policy.pool_floor,
            consensus_weight=0.0,
            internal_sigma=policy.internal_sigma,
            internal_sims=0,
        ),k,with_consensus=False)
        if len(s)==k:
            freq.loc[s.index]+=1.0
    freq=freq/max(1,int(policy.internal_sims))

    universe=base.copy()
    universe["selection_frequency"]=freq.reindex(universe.index).fillna(0.0)
    universe["robust_score"]=(
        (1.0-policy.consensus_weight)*universe["base_rank"]
        +policy.consensus_weight*universe["selection_frequency"]
    )
    return universe.sort_values(
        ["robust_score","selection_frequency","base_score","p_cal","symbol"],
        ascending=[False,False,False,False,True]
    ).head(k).copy()


def fold_outcome(g,sel,k=10):
    q=g.dropna(subset=["y6","dd30_6m","p_cal"]).copy()
    if len(sel)!=k or q.empty:
        return None
    y=sel["y6"].astype(float)
    dd=sel["dd30_6m"].astype(float)
    br=float(q["y6"].mean())
    pr=float(y.mean())
    return {
        "precision_2x":pr,
        "lift_2x":pr/br if br>0 else np.nan,
        "hit":float(pr>0),
        "dd30_rate":float(dd.mean()),
    }


def stability_fold(g,spec,policy,k,external_sims,external_sigma):
    base=select_once(g,spec,policy,k,salt="unperturbed")
    if len(base)!=k:
        return None
    b=set(base["symbol"].astype(str))
    js=[]; top1=[]; repl=[]
    rng=np.random.default_rng(deterministic_seed(g["date"].iloc[0],policy.name,"external"))
    for i in range(external_sims):
        zp=perturb(g,external_sigma,rng)
        s=select_once(zp,spec,policy,k,salt=f"ext{i}")
        if len(s)!=k:
            continue
        ss=set(s["symbol"].astype(str))
        js.append(len(b&ss)/len(b|ss))
        top1.append(float(str(s.iloc[0]["symbol"])==str(base.iloc[0]["symbol"])))
        repl.append(len(b-ss)/k)
    if not js:
        return None
    return {
        "mean_jaccard":float(np.mean(js)),
        "p05_jaccard":float(np.quantile(js,.05)),
        "top1_stability":float(np.mean(top1)),
        "names_replaced_fraction":float(np.mean(repl)),
    }


def aggregate_outcomes(t):
    lift=np.clip(t["lift_2x"].replace([np.inf,-np.inf],np.nan).dropna(),0,10)
    return {
        "folds":int(len(t)),
        "mean_precision_2x":float(t["precision_2x"].mean()),
        "median_precision_2x":float(t["precision_2x"].median()),
        "mean_capped_lift_2x":float(lift.mean()) if len(lift) else np.nan,
        "hit_fold_rate":float(t["hit"].mean()),
        "mean_dd30_rate":float(t["dd30_rate"].mean()),
    }


def alpha_pass(a,b,ratio=.95):
    fields=["mean_precision_2x","median_precision_2x","mean_capped_lift_2x","hit_fold_rate"]
    return all(
        np.isfinite(a.get(f,np.nan)) and np.isfinite(b.get(f,np.nan))
        and (b[f]<=0 or a[f]+1e-12>=ratio*b[f])
        for f in fields
    )


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--oos",required=True)
    ap.add_argument("--chosen931",required=True)
    ap.add_argument("--config",required=True)
    ap.add_argument("--output",required=True)
    ap.add_argument("--external-sims",type=int,default=100)
    ap.add_argument("--confirm-sims",type=int,default=500)
    args=ap.parse_args()

    cfg=json.load(open(args.config))
    k=int(cfg.get("selection_k",10))
    oos=pd.read_parquet(args.oos)
    oos["date"]=pd.to_datetime(oos["date"])
    chosen=pd.read_csv(args.chosen931)
    chosen["date"]=pd.to_datetime(chosen["date"])
    cmap={pd.Timestamp(r.date):r for r in chosen.itertuples(index=False)}

    all_out=[]; all_stab=[]
    for policy in POLICIES:
        for td,r in cmap.items():
            g=oos[oos["date"]==td].copy()
            if g.empty:
                continue
            spec=spec_from_row(r)
            sel=select_once(g,spec,policy,k,salt="metric")
            m=fold_outcome(g,sel,k)
            if m is not None:
                all_out.append({"policy":policy.name,"date":td,**m})
            st=stability_fold(g,spec,policy,k,args.external_sims,.01)
            if st is not None:
                all_stab.append({"policy":policy.name,"date":td,**st})

    outdf=pd.DataFrame(all_out)
    stdf=pd.DataFrame(all_stab)
    baseline=aggregate_outcomes(outdf[outdf["policy"]=="incumbent"])

    rows=[]
    for policy in POLICIES:
        ot=outdf[outdf["policy"]==policy.name]
        st=stdf[stdf["policy"]==policy.name]
        if ot.empty or st.empty:
            continue
        a=aggregate_outcomes(ot)
        meanj=float(st["mean_jaccard"].mean())
        worst=float(st["p05_jaccard"].min())
        top1=float(st["top1_stability"].mean())
        passed_alpha=alpha_pass(a,baseline,.95)
        passed_stability=(meanj>=.80 and worst>=.60 and top1>=.70)
        rows.append({
            **asdict(policy),**a,
            "mean_jaccard":meanj,
            "worst_p05_jaccard":worst,
            "mean_top1_stability":top1,
            "mean_names_replaced_fraction":float(st["names_replaced_fraction"].mean()),
            "alpha_retention_passed":passed_alpha,
            "stability_passed":passed_stability,
            "all_gates_passed":bool(passed_alpha and passed_stability),
        })
    table=pd.DataFrame(rows).sort_values(
        ["all_gates_passed","stability_passed","mean_jaccard","mean_precision_2x"],
        ascending=[False,False,False,False]
    )

    passing=table[table["all_gates_passed"]]
    chosen_policy=None
    confirm_rows=[]
    if len(passing):
        # Prefer smallest behavioral change: lowest consensus weight, then
        # smallest pool floor; use mean Jaccard only as the next tie-breaker.
        p=passing.sort_values(
            ["consensus_weight","pool_floor","mean_jaccard"],
            ascending=[True,True,False]
        ).iloc[0]
        chosen_policy=next(x for x in POLICIES if x.name==p["name"])

        for td,r in cmap.items():
            g=oos[oos["date"]==td].copy()
            if g.empty:continue
            st=stability_fold(g,spec_from_row(r),chosen_policy,k,args.confirm_sims,.01)
            if st:
                confirm_rows.append({"date":td,**st})

    confirm=pd.DataFrame(confirm_rows)
    confirmation=None
    if len(confirm):
        confirmation={
            "mean_jaccard":float(confirm["mean_jaccard"].mean()),
            "worst_p05_jaccard":float(confirm["p05_jaccard"].min()),
            "mean_top1_stability":float(confirm["top1_stability"].mean()),
            "mean_names_replaced_fraction":float(confirm["names_replaced_fraction"].mean()),
            "mean_jaccard_gate":bool(confirm["mean_jaccard"].mean()>=.80),
            "worst_p05_gate":bool(confirm["p05_jaccard"].min()>=.60),
            "top1_gate":bool(confirm["top1_stability"].mean()>=.70),
        }

    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    outdf.to_csv(out/"policy_outcomes_by_fold.csv",index=False)
    stdf.to_csv(out/"policy_stability_by_fold_grid.csv",index=False)
    table.to_csv(out/"policy_comparison.csv",index=False)
    confirm.to_csv(out/"chosen_policy_stability_500.csv",index=False)

    summary={
        "model":"V10.2 selector stability correction",
        "scope":"selector only; alpha model, labels and calibration unchanged",
        "baseline":baseline,
        "grid_policies":len(POLICIES),
        "chosen_policy":asdict(chosen_policy) if chosen_policy else None,
        "chosen_policy_confirmation_500":confirmation,
        "promotion_gate":bool(
            chosen_policy is not None and confirmation is not None
            and confirmation["mean_jaccard_gate"]
            and confirmation["worst_p05_gate"]
            and confirmation["top1_gate"]
        ),
        "required":{
            "alpha_retention":0.95,
            "mean_jaccard":0.80,
            "worst_p05_jaccard":0.60,
            "mean_top1_stability":0.70,
        }
    }
    json.dump(summary,open(out/"summary.json","w"),indent=2,default=str)
    print(table.to_string(index=False))
    print(json.dumps(summary,indent=2,default=str))


if __name__=="__main__":
    main()
