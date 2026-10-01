from __future__ import annotations

import argparse, hashlib, json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_COLS=["p_struct","p_elastic","p_gbm","p_survival"]


@dataclass(frozen=True)
class Policy:
    name:str
    alpha_mode:str
    raw_weight:float=0.0
    disagreement_penalty:float=0.0
    risk_col:str="p_dd30_raw"
    secondary_scale:float=1.0
    pool_floor:int=0


POLICIES=[
    Policy("incumbent","incumbent",risk_col="p_dd30_cal"),
    Policy("raw","raw",risk_col="p_dd30_raw"),
    Policy("borda4","borda",risk_col="p_dd30_raw"),
    Policy("median4","median",risk_col="p_dd30_raw"),
    Policy("trimmed4","trimmed",risk_col="p_dd30_raw"),
    Policy("lcb10","borda",disagreement_penalty=.10,risk_col="p_dd30_raw"),
    Policy("lcb20","borda",disagreement_penalty=.20,risk_col="p_dd30_raw"),
    Policy("hybrid25","borda",raw_weight=.25,risk_col="p_dd30_raw"),
    Policy("hybrid50","borda",raw_weight=.50,risk_col="p_dd30_raw"),
    Policy("hybrid75","borda",raw_weight=.75,risk_col="p_dd30_raw"),
    Policy("hybrid50_s50","borda",raw_weight=.50,risk_col="p_dd30_raw",secondary_scale=.50),
    Policy("hybrid75_s50","borda",raw_weight=.75,risk_col="p_dd30_raw",secondary_scale=.50),
]


def seed_for(date,name,salt):
    raw=f"{pd.Timestamp(date).date()}|{name}|{salt}".encode()
    return int(hashlib.sha256(raw).hexdigest()[:8],16)


def spec_from_row(r):
    return {
        "pool_n":int(r.pool_n),
        "risk_drop":float(r.risk_drop),
        "w_safety":float(r.w_safety),
        "w_consensus":float(r.w_consensus),
        "baseline":bool(r.fell_back_to_v92),
    }


def rank_high(s):
    return s.rank(pct=True,method="average")


def rank_low(s):
    return s.rank(pct=True,method="average",ascending=False)


def robust_alpha(q:pd.DataFrame,policy:Policy):
    if policy.alpha_mode=="incumbent":
        return rank_high(q["p_cal"].astype(float)), pd.Series(0.0,index=q.index)
    if policy.alpha_mode=="raw":
        return rank_high(q["p_raw"].astype(float)), pd.Series(0.0,index=q.index)

    ranks=pd.DataFrame(index=q.index)
    for c in MODEL_COLS:
        ranks[c]=rank_high(q[c].astype(float))

    if policy.alpha_mode=="median":
        core=ranks.median(axis=1)
    elif policy.alpha_mode=="trimmed":
        arr=np.sort(ranks.to_numpy(float),axis=1)
        core=pd.Series(arr[:,1:3].mean(axis=1),index=q.index)
    else:
        core=ranks.mean(axis=1)

    disagreement=ranks.std(axis=1,ddof=0)
    if policy.disagreement_penalty>0:
        core=core-policy.disagreement_penalty*disagreement

    if policy.raw_weight>0:
        raw_rank=rank_high(q["p_raw"].astype(float))
        core=(1-policy.raw_weight)*core+policy.raw_weight*raw_rank

    return core,disagreement


def select_policy(g,spec,policy,k=10):
    req=["p_cal","p_raw","p_dd30_cal","p_dd30_raw","model_dispersion"]+MODEL_COLS
    q=g.dropna(subset=req).copy()
    if len(q)<k:return pd.DataFrame()

    q["_alpha"],q["_model_rank_disp"]=robust_alpha(q,policy)

    if spec["baseline"]:
        q["selection_score_ra"]=q["_alpha"]
        return q.sort_values(
            ["selection_score_ra","p_raw","model_dispersion","symbol"],
            ascending=[False,False,True,True],
        ).head(k).copy()

    pool_n=min(max(int(spec["pool_n"]),int(policy.pool_floor)),len(q))
    pool=q.sort_values(
        ["_alpha","p_raw","model_dispersion","symbol"],
        ascending=[False,False,True,True],
    ).head(pool_n).copy()
    if len(pool)<k:return pd.DataFrame()

    rd=float(spec["risk_drop"])
    if rd>0:
        cutoff=float(pool[policy.risk_col].quantile(np.clip(1-rd,0,1)))
        surv=pool[pool[policy.risk_col]<=cutoff].copy()
    else:
        surv=pool.copy()
    if len(surv)<k:return pd.DataFrame()

    ws=float(spec["w_safety"])*policy.secondary_scale
    wc=float(spec["w_consensus"])*policy.secondary_scale
    wa=max(0.0,1.0-ws-wc)

    # For consensus, combine existing prediction dispersion with cross-model
    # rank disagreement. Lower disagreement is better.
    disagreement=(
        rank_low(surv["model_dispersion"])
        +rank_low(surv["_model_rank_disp"])
    )/2.0

    surv["selection_score_ra"]=(
        wa*rank_high(surv["_alpha"])
        +ws*rank_low(surv[policy.risk_col])
        +wc*disagreement
    )
    return surv.sort_values(
        ["selection_score_ra","_alpha","p_raw","model_dispersion","symbol"],
        ascending=[False,False,False,True,True],
    ).head(k).copy()


def perturb(g,policy,sigma,rng):
    z=g.copy()
    cols=set(MODEL_COLS+[
        "p_raw","p_cal",policy.risk_col,
        "p_dd30_raw","p_dd30_cal","model_dispersion"
    ])
    for c in cols:
        if c not in z:continue
        a=pd.to_numeric(z[c],errors="coerce").to_numpy(float)
        a=a*np.exp(rng.normal(0,sigma,len(a)))
        if c.startswith("p_") and c!="model_dispersion":
            a=np.clip(a,1e-8,1-1e-8)
        else:
            a=np.maximum(a,1e-12)
        z[c]=a
    return z


def stability(g,spec,policy,k,sims,sigma=.01):
    base=select_policy(g,spec,policy,k)
    if len(base)!=k:return None
    bs=set(base["symbol"].astype(str))
    js=[];tops=[];repl=[]
    rng=np.random.default_rng(seed_for(g["date"].iloc[0],policy.name,"external"))
    for _ in range(sims):
        z=perturb(g,policy,sigma,rng)
        s=select_policy(z,spec,policy,k)
        if len(s)!=k:continue
        ss=set(s["symbol"].astype(str))
        js.append(len(bs&ss)/len(bs|ss))
        tops.append(float(str(s.iloc[0]["symbol"])==str(base.iloc[0]["symbol"])))
        repl.append(len(bs-ss)/k)
    if not js:return None
    return {
        "mean_jaccard":float(np.mean(js)),
        "p05_jaccard":float(np.quantile(js,.05)),
        "top1_stability":float(np.mean(tops)),
        "names_replaced_fraction":float(np.mean(repl)),
    }


def outcome(g,s,k):
    q=g.dropna(subset=["y6","dd30_6m"]).copy()
    if len(s)!=k or q.empty:return None
    br=float(q["y6"].mean()); pr=float(s["y6"].mean())
    return {
        "precision_2x":pr,
        "lift_2x":pr/br if br>0 else np.nan,
        "hit":float(pr>0),
        "dd30_rate":float(s["dd30_6m"].mean()),
    }


def agg(t):
    lift=np.clip(t["lift_2x"].replace([np.inf,-np.inf],np.nan).dropna(),0,10)
    return {
        "folds":int(len(t)),
        "mean_precision_2x":float(t["precision_2x"].mean()),
        "median_precision_2x":float(t["precision_2x"].median()),
        "mean_capped_lift_2x":float(lift.mean()) if len(lift) else np.nan,
        "hit_fold_rate":float(t["hit"].mean()),
        "mean_dd30_rate":float(t["dd30_rate"].mean()),
    }


def alpha_pass(a,b,r=.95):
    for f in ["mean_precision_2x","median_precision_2x","mean_capped_lift_2x","hit_fold_rate"]:
        av=a.get(f,np.nan); bv=b.get(f,np.nan)
        if not(np.isfinite(av) and np.isfinite(bv)):return False
        if bv>0 and av+1e-12<r*bv:return False
    return True


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

    metrics=[]; stabs=[]
    for p in POLICIES:
        for td,r in cmap.items():
            g=oos[oos["date"]==td].copy()
            if g.empty:continue
            spec=spec_from_row(r)
            sel=select_policy(g,spec,p,k)
            m=outcome(g,sel,k)
            if m:metrics.append({"policy":p.name,"date":td,**m})
            st=stability(g,spec,p,k,args.sims,.01)
            if st:stabs.append({"policy":p.name,"date":td,**st})

    mt=pd.DataFrame(metrics); st=pd.DataFrame(stabs)
    baseline=agg(mt[mt["policy"]=="incumbent"])

    rows=[]
    for p in POLICIES:
        a=agg(mt[mt["policy"]==p.name])
        s=st[st["policy"]==p.name]
        if not a or s.empty:continue
        sj=float(s["mean_jaccard"].mean())
        wp=float(s["p05_jaccard"].min())
        t1=float(s["top1_stability"].mean())
        apass=alpha_pass(a,baseline,.95)
        stabpass=(sj>=.80 and wp>=.60 and t1>=.70)
        ddpass=a["mean_dd30_rate"]<=baseline["mean_dd30_rate"]+.03
        rows.append({
            **asdict(p),**a,
            "mean_jaccard":sj,
            "worst_p05_jaccard":wp,
            "mean_top1_stability":t1,
            "mean_names_replaced_fraction":float(s["names_replaced_fraction"].mean()),
            "worst_fold":str(s.loc[s["p05_jaccard"].idxmin(),"date"]),
            "alpha_retention_passed":apass,
            "stability_passed":stabpass,
            "dd_not_worse_3pp":ddpass,
            "all_gates_passed":bool(apass and stabpass and ddpass),
        })

    table=pd.DataFrame(rows).sort_values(
        ["all_gates_passed","stability_passed","alpha_retention_passed","mean_jaccard","mean_precision_2x"],
        ascending=[False,False,False,False,False]
    )

    passing=table[table["all_gates_passed"]]
    chosen_policy=None if passing.empty else str(passing.iloc[0]["name"])
    summary={
        "model":"V10.2 robust multi-model rank aggregation selector",
        "research_basis":"consensus rank aggregation + disagreement penalty; calibrated probability retained for reporting",
        "baseline":baseline,
        "chosen_policy":chosen_policy,
        "promotion_gate":bool(chosen_policy is not None),
        "required":{
            "mean_precision_retention":.95,
            "mean_jaccard":.80,
            "worst_p05_jaccard":.60,
            "top1_stability":.70,
            "dd_not_worse_by":.03,
        },
    }

    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    mt.to_csv(out/"rank_aggregation_metrics_by_fold.csv",index=False)
    st.to_csv(out/"rank_aggregation_stability_by_fold.csv",index=False)
    table.to_csv(out/"rank_aggregation_policy_comparison.csv",index=False)
    json.dump(summary,open(out/"summary.json","w"),indent=2,default=str)
    print(table.to_string(index=False))
    print(json.dumps(summary,indent=2,default=str))

if __name__=="__main__":
    main()
