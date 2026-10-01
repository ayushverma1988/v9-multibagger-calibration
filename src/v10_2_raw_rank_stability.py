from __future__ import annotations

import argparse, json, hashlib
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Policy:
    name: str
    alpha_col: str
    risk_col: str
    pool_floor: int = 0
    alpha_cal_blend: float = 0.0
    secondary_scale: float = 1.0


POLICIES=[
    Policy("incumbent","p_cal","p_dd30_cal"),
    Policy("raw_alpha","p_raw","p_dd30_cal"),
    Policy("raw_alpha_raw_risk","p_raw","p_dd30_raw"),
    Policy("raw_secondary90","p_raw","p_dd30_raw",secondary_scale=.90),
    Policy("raw_secondary80","p_raw","p_dd30_raw",secondary_scale=.80),
    Policy("raw_secondary70","p_raw","p_dd30_raw",secondary_scale=.70),
    Policy("raw_secondary60","p_raw","p_dd30_raw",secondary_scale=.60),
    Policy("raw_secondary50","p_raw","p_dd30_raw",secondary_scale=.50),
    Policy("raw_secondary40","p_raw","p_dd30_raw",secondary_scale=.40),
    Policy("raw_secondary30","p_raw","p_dd30_raw",secondary_scale=.30),
    Policy("raw_secondary25","p_raw","p_dd30_raw",secondary_scale=.25),
    Policy("raw_secondary20","p_raw","p_dd30_raw",secondary_scale=.20),
    Policy("raw_secondary10","p_raw","p_dd30_raw",secondary_scale=.10),
    Policy("raw_pure_alpha","p_raw","p_dd30_raw",secondary_scale=0.0),
    Policy("raw_secondary90_f30","p_raw","p_dd30_raw",pool_floor=30,secondary_scale=.90),
    Policy("raw_secondary80_f30","p_raw","p_dd30_raw",pool_floor=30,secondary_scale=.80),
    Policy("raw_secondary70_f30","p_raw","p_dd30_raw",pool_floor=30,secondary_scale=.70),
    Policy("raw_secondary60_f30","p_raw","p_dd30_raw",pool_floor=30,secondary_scale=.60),
    Policy("raw_secondary50_f30","p_raw","p_dd30_raw",pool_floor=30,secondary_scale=.50),
    Policy("raw_secondary40_f30","p_raw","p_dd30_raw",pool_floor=30,secondary_scale=.40),
    Policy("raw_secondary30_f30","p_raw","p_dd30_raw",pool_floor=30,secondary_scale=.30),
    Policy("raw_secondary20_f30","p_raw","p_dd30_raw",pool_floor=30,secondary_scale=.20),
    Policy("raw_secondary10_f30","p_raw","p_dd30_raw",pool_floor=30,secondary_scale=.10),
    Policy("raw_pure_alpha_f30","p_raw","p_dd30_raw",pool_floor=30,secondary_scale=0.0),
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


def select_policy(g,spec,policy,k=10):
    req=[policy.alpha_col,policy.risk_col,"model_dispersion","p_cal","p_dd30_cal"]
    q=g.dropna(subset=req).copy()
    if len(q)<k:return pd.DataFrame()

    alpha=q[policy.alpha_col].astype(float)
    if policy.alpha_cal_blend>0:
        q["_alpha_sel"]=(
            (1-policy.alpha_cal_blend)*rank_high(alpha)
            +policy.alpha_cal_blend*rank_high(q["p_cal"].astype(float))
        )
        alpha_col="_alpha_sel"
    else:
        alpha_col=policy.alpha_col

    if spec["baseline"]:
        out=q.sort_values(
            [alpha_col,"model_dispersion","symbol"],
            ascending=[False,True,True]
        ).head(k).copy()
        out["selection_score_stable"]=out[alpha_col]
        return out

    pool_n=min(max(int(spec["pool_n"]),int(policy.pool_floor)),len(q))
    pool=q.sort_values(
        [alpha_col,"model_dispersion","symbol"],
        ascending=[False,True,True]
    ).head(pool_n).copy()
    if len(pool)<k:return pd.DataFrame()

    rd=float(spec["risk_drop"])
    if rd>0:
        cutoff=float(pool[policy.risk_col].quantile(np.clip(1-rd,0,1)))
        surv=pool[pool[policy.risk_col]<=cutoff].copy()
    else:
        surv=pool.copy()
    if len(surv)<k:return pd.DataFrame()

    ws=float(spec["w_safety"])*float(policy.secondary_scale)
    wc=float(spec["w_consensus"])*float(policy.secondary_scale)
    wa=max(0.0,1.0-ws-wc)
    surv["selection_score_stable"]=(
        wa*rank_high(surv[alpha_col])
        +ws*rank_low(surv[policy.risk_col])
        +wc*rank_low(surv["model_dispersion"])
    )
    return surv.sort_values(
        ["selection_score_stable",alpha_col,"model_dispersion","symbol"],
        ascending=[False,False,True,True]
    ).head(k).copy()


def perturb_for_policy(g,policy,sigma,rng):
    z=g.copy()
    cols={policy.alpha_col,policy.risk_col,"model_dispersion"}
    if policy.alpha_cal_blend>0:
        cols.add("p_cal")
    for c in cols:
        if c not in z:continue
        a=pd.to_numeric(z[c],errors="coerce").to_numpy(float)
        a=a*np.exp(rng.normal(0,sigma,len(a)))
        if c.startswith("p_") and "dispersion" not in c:
            a=np.clip(a,1e-8,1-1e-8)
        else:
            a=np.maximum(a,1e-12)
        z[c]=a
    return z


def stability(g,spec,policy,k,sims,sigma=.01):
    base=select_policy(g,spec,policy,k)
    if len(base)!=k:return None
    bs=set(base["symbol"].astype(str))
    js=[];top=[];rep=[]
    rng=np.random.default_rng(seed_for(g["date"].iloc[0],policy.name,"external"))
    for _ in range(sims):
        z=perturb_for_policy(g,policy,sigma,rng)
        s=select_policy(z,spec,policy,k)
        if len(s)!=k:continue
        ss=set(s["symbol"].astype(str))
        js.append(len(bs&ss)/len(bs|ss))
        top.append(float(str(s.iloc[0]["symbol"])==str(base.iloc[0]["symbol"])))
        rep.append(len(bs-ss)/k)
    return {
        "mean_jaccard":float(np.mean(js)),
        "p05_jaccard":float(np.quantile(js,.05)),
        "top1_stability":float(np.mean(top)),
        "names_replaced_fraction":float(np.mean(rep)),
    } if js else None


def outcome(g,s,k):
    q=g.dropna(subset=["y6","dd30_6m"]).copy()
    if len(s)!=k or q.empty:return None
    br=float(q["y6"].mean());pr=float(s["y6"].mean())
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
        if not(np.isfinite(a.get(f,np.nan)) and np.isfinite(b.get(f,np.nan))):return False
        if b[f]>0 and a[f]+1e-12<r*b[f]:return False
    return True


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

    outs=[];stabs=[]
    for p in POLICIES:
        for td,r in cmap.items():
            g=d[d["date"]==td].copy()
            if g.empty:continue
            spec=spec_from_row(r)
            s=select_policy(g,spec,p,k)
            o=outcome(g,s,k)
            st=stability(g,spec,p,k,args.sims,.01)
            if o:outs.append({"policy":p.name,"date":td,**o})
            if st:stabs.append({"policy":p.name,"date":td,**st})

    od=pd.DataFrame(outs);sd=pd.DataFrame(stabs)
    base=agg(od[od["policy"]=="incumbent"])
    rows=[]
    for p in POLICIES:
        o=od[od["policy"]==p.name];s=sd[sd["policy"]==p.name]
        if o.empty or s.empty:continue
        a=agg(o)
        apass=alpha_pass(a,base,.95)
        spass=bool(s["mean_jaccard"].mean()>=.80 and s["p05_jaccard"].min()>=.60 and s["top1_stability"].mean()>=.70)
        rows.append({
            **asdict(p),**a,
            "mean_jaccard":float(s["mean_jaccard"].mean()),
            "worst_p05_jaccard":float(s["p05_jaccard"].min()),
            "mean_top1_stability":float(s["top1_stability"].mean()),
            "mean_names_replaced_fraction":float(s["names_replaced_fraction"].mean()),
            "worst_fold":str(s.loc[s["p05_jaccard"].idxmin(),"date"]),
            "alpha_retention_passed":apass,
            "stability_passed":spass,
            "dd_not_worse_3pp":bool(a["mean_dd30_rate"]<=base["mean_dd30_rate"]+.03),
            "all_gates_passed":bool(apass and spass and a["mean_dd30_rate"]<=base["mean_dd30_rate"]+.03),
        })

    tab=pd.DataFrame(rows).sort_values(
        ["all_gates_passed","stability_passed","mean_jaccard","mean_precision_2x"],
        ascending=[False,False,False,False]
    )
    passing=tab[tab["all_gates_passed"]]
    chosen_policy=None
    if len(passing):
        chosen_policy=passing.sort_values(
            ["alpha_cal_blend","pool_floor","mean_jaccard"],
            ascending=[True,True,False]
        ).iloc[0].to_dict()

    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    od.to_csv(out/"raw_rank_outcomes_by_fold.csv",index=False)
    sd.to_csv(out/"raw_rank_stability_by_fold.csv",index=False)
    tab.to_csv(out/"raw_rank_policy_comparison.csv",index=False)
    summary={
        "model":"V10.2 raw-score ranking stability correction",
        "principle":"calibrated probabilities retained for reporting; ranking may use pre-calibration raw alpha/downside scores",
        "baseline":base,
        "chosen_policy":chosen_policy,
        "promotion_gate":bool(chosen_policy is not None),
        "required":{"alpha_retention":.95,"mean_jaccard":.80,"worst_p05_jaccard":.60,"top1":.70},
    }
    json.dump(summary,open(out/"summary.json","w"),indent=2,default=str)
    print(tab.to_string(index=False))
    print(json.dumps(summary,indent=2,default=str))


if __name__=="__main__":
    main()
