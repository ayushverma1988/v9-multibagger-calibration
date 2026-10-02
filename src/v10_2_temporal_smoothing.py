from __future__ import annotations

import argparse, hashlib, json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

from v10_2_compression_switch import SwitchPolicy, compression_span, perturb_all
from v10_2_raw_rank_stability import (
    Policy as RawPolicy, spec_from_row, rank_high, rank_low,
    outcome, agg, alpha_pass, select_policy
)

BASE_SWITCH=SwitchPolicy("compress_030_raw50",.030,.50)
INC=RawPolicy("incumbent","p_cal","p_dd30_cal")


@dataclass(frozen=True)
class TemporalPolicy:
    name:str
    prior_weight:float
    fragile_only:bool=True
    internal_sims:int=50


POLICIES=[
    TemporalPolicy("temp05",.05),
    TemporalPolicy("temp10",.10),
    TemporalPolicy("temp15",.15),
    TemporalPolicy("temp20",.20),
]


def seed_for(date,name,salt):
    raw=f"{pd.Timestamp(date).date()}|{name}|{salt}".encode()
    return int(hashlib.sha256(raw).hexdigest()[:8],16)


def raw_policy(scale=.50):
    return RawPolicy("raw50","p_raw","p_dd30_raw",secondary_scale=scale)


def active_policy(g,spec):
    return raw_policy(.50) if compression_span(g,spec)<=BASE_SWITCH.span_trigger else INC


def full_rank(g,spec):
    p=active_policy(g,spec)
    req=[p.alpha_col,p.risk_col,"model_dispersion","p_cal","p_dd30_cal"]
    q=g.dropna(subset=req).copy()
    if len(q)<10:return pd.DataFrame()

    alpha=q[p.alpha_col].astype(float)
    alpha_col=p.alpha_col
    if spec["baseline"]:
        q["selection_score_temp"]=rank_high(alpha)
        return q.sort_values(["selection_score_temp",alpha_col,"model_dispersion","symbol"],
                             ascending=[False,False,True,True]).copy()

    pool_n=min(max(int(spec["pool_n"]),int(p.pool_floor)),len(q))
    pool=q.sort_values([alpha_col,"model_dispersion","symbol"],
                       ascending=[False,True,True]).head(pool_n).copy()
    if len(pool)<10:return pd.DataFrame()
    if float(spec["risk_drop"])>0:
        cutoff=float(pool[p.risk_col].quantile(np.clip(1-float(spec["risk_drop"]),0,1)))
        pool=pool[pool[p.risk_col]<=cutoff].copy()
    if len(pool)<10:return pd.DataFrame()

    ws=float(spec["w_safety"])*float(p.secondary_scale)
    wc=float(spec["w_consensus"])*float(p.secondary_scale)
    wa=max(0.0,1.0-ws-wc)
    pool["selection_score_temp"]=(
        wa*rank_high(pool[alpha_col])
        +ws*rank_low(pool[p.risk_col])
        +wc*rank_low(pool["model_dispersion"])
    )
    return pool.sort_values(["selection_score_temp",alpha_col,"model_dispersion","symbol"],
                            ascending=[False,False,True,True]).copy()


def base_select(g,spec,k=10):
    return full_rank(g,spec).head(k).copy()


def internal_p05(g,spec,tp,k=10):
    b=base_select(g,spec,k)
    if len(b)!=k:return 0.0
    bs=set(b["symbol"].astype(str))
    rng=np.random.default_rng(seed_for(g["date"].iloc[0],tp.name,"fragility"))
    js=[]
    for _ in range(tp.internal_sims):
        z=perturb_all(g,.01,rng)
        s=base_select(z,spec,k)
        if len(s)!=k:continue
        ss=set(s["symbol"].astype(str))
        js.append(len(bs&ss)/len(bs|ss))
    return float(np.quantile(js,.05)) if js else 0.0


def prior_rank_map(prev_g):
    if prev_g is None or prev_g.empty:
        return {}
    q=prev_g.dropna(subset=["p_raw"]).copy()
    if q.empty:return {}
    q["_prior_rank"]=rank_high(q["p_raw"].astype(float))
    return dict(zip(q["symbol"].astype(str),q["_prior_rank"].astype(float)))


def select_temporal(g,spec,tp,prev_map,k=10,frag_override=None):
    ranked=full_rank(g,spec)
    if len(ranked)<k:return pd.DataFrame(),False,np.nan
    frag=internal_p05(g,spec,tp,k) if frag_override is None else frag_override
    if tp.fragile_only and frag>=.60:
        out=ranked.head(k).copy();out["_temporal_used"]=0.0
        return out,False,frag

    ranked=ranked.copy()
    cur=rank_high(ranked["selection_score_temp"])
    prior=pd.Series(ranked["symbol"].astype(str).map(prev_map),index=ranked.index,dtype=float)
    prior=prior.where(prior.notna(),cur)
    ranked["_temporal_score"]=(1-tp.prior_weight)*cur+tp.prior_weight*prior

    out=ranked.sort_values(
        ["_temporal_score","selection_score_temp","p_raw","p_cal","model_dispersion","symbol"],
        ascending=[False,False,False,False,True,True]
    ).head(k).copy()
    out["_temporal_used"]=1.0
    return out,True,frag


def external_stability(g,spec,tp,prev_map,k,sims):
    base,_,frag=select_temporal(g,spec,tp,prev_map,k)
    if len(base)!=k:return None
    bs=set(base["symbol"].astype(str));top=str(base.iloc[0]["symbol"])
    rng=np.random.default_rng(seed_for(g["date"].iloc[0],tp.name,"external"))
    js=[];tops=[];rep=[]
    for _ in range(sims):
        z=perturb_all(g,.01,rng)
        s,_,_=select_temporal(z,spec,tp,prev_map,k,frag_override=frag)
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


def evaluate(oos,cmap,cfg,tp,sims):
    k=int(cfg.get("selection_k",10))
    dates=sorted(cmap)
    mets=[];sts=[];uses=[]
    for i,td in enumerate(dates):
        g=oos[oos["date"]==td].copy()
        if g.empty:continue
        prev_g=oos[oos["date"]==dates[i-1]].copy() if i>0 else None
        pmap=prior_rank_map(prev_g)
        spec=spec_from_row(cmap[td])
        s,active,frag=select_temporal(g,spec,tp,pmap,k)
        m=outcome(g,s,k)
        if m:mets.append({"date":td,**m})
        uses.append({"date":td,"temporal_used":float(active),"fragility_p05":frag})
        st=external_stability(g,spec,tp,pmap,k,sims)
        if st:sts.append({"date":td,**st})
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
    oos=pd.read_parquet(args.oos);oos["date"]=pd.to_datetime(oos["date"])
    chosen=pd.read_csv(args.chosen931);chosen["date"]=pd.to_datetime(chosen["date"])
    cmap={pd.Timestamp(r.date):r for r in chosen.itertuples(index=False)}
    baseline=incumbent_metrics(oos,cmap,cfg)

    rows=[]
    for tp in POLICIES:
        mt,st,use=evaluate(oos,cmap,cfg,tp,args.grid_sims)
        a=agg(mt)
        if not a or st.empty:continue
        mj=float(st["mean_jaccard"].mean());wp=float(st["p05_jaccard"].min());t1=float(st["top1_stability"].mean())
        rows.append({
            **asdict(tp),**a,
            "mean_jaccard":mj,"worst_p05_jaccard":wp,"mean_top1_stability":t1,
            "mean_names_replaced_fraction":float(st["names_replaced_fraction"].mean()),
            "worst_fold":str(st.loc[st["p05_jaccard"].idxmin(),"date"]),
            "temporal_folds":int(use["temporal_used"].sum()),
            "alpha_retention_passed":bool(alpha_pass(a,baseline,.95)),
            "stability_passed":bool(mj>=.80 and wp>=.60 and t1>=.70),
            "dd_not_worse_3pp":bool(a["mean_dd30_rate"]<=baseline["mean_dd30_rate"]+.03),
        })
    table=pd.DataFrame(rows)
    table["all_gates_passed"]=table["alpha_retention_passed"]&table["stability_passed"]&table["dd_not_worse_3pp"]
    table=table.sort_values(["stability_passed","worst_p05_jaccard","mean_jaccard","prior_weight"],
                            ascending=[False,False,False,True])

    # Stability-only policy selection: smallest prior weight that clears the
    # stability gates. Outcomes are checked only after the choice.
    stable=table[table["stability_passed"]]
    chosen_name=str(stable.sort_values(["prior_weight","mean_jaccard"],ascending=[True,False]).iloc[0]["name"]) if len(stable) else (str(table.iloc[0]["name"]) if len(table) else None)
    tp=next((x for x in POLICIES if x.name==chosen_name),None)

    confirmation=None
    if tp:
        mt,st,use=evaluate(oos,cmap,cfg,tp,args.confirm_sims)
        a=agg(mt);mj=float(st["mean_jaccard"].mean());wp=float(st["p05_jaccard"].min());t1=float(st["top1_stability"].mean())
        confirmation={
            **a,"mean_jaccard":mj,"worst_p05_jaccard":wp,"mean_top1_stability":t1,
            "mean_names_replaced_fraction":float(st["names_replaced_fraction"].mean()),
            "worst_fold":str(st.loc[st["p05_jaccard"].idxmin(),"date"]),
            "temporal_folds":int(use["temporal_used"].sum()),
            "alpha_retention_passed":bool(alpha_pass(a,baseline,.95)),
            "mean_jaccard_passed":bool(mj>=.80),"worst_p05_passed":bool(wp>=.60),
            "top1_passed":bool(t1>=.70),"dd_not_worse_3pp":bool(a["mean_dd30_rate"]<=baseline["mean_dd30_rate"]+.03),
        }
        confirmation["all_gates_passed"]=bool(
            confirmation["alpha_retention_passed"] and confirmation["mean_jaccard_passed"]
            and confirmation["worst_p05_passed"] and confirmation["top1_passed"] and confirmation["dd_not_worse_3pp"]
        )

    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    table.to_csv(out/"temporal_grid.csv",index=False)
    if tp:
        mt,st,use=evaluate(oos,cmap,cfg,tp,args.confirm_sims)
        mt.to_csv(out/"chosen_metrics_by_fold.csv",index=False)
        st.to_csv(out/"chosen_stability_by_fold.csv",index=False)
        use.to_csv(out/"chosen_temporal_usage.csv",index=False)
    summary={
        "model":"V10.2 causal temporal rank smoothing",
        "principle":"small prior-snapshot rank weight only on label-free fragile folds; no outcome information",
        "baseline":baseline,
        "chosen_policy":asdict(tp) if tp else None,
        "confirmation_500":confirmation,
        "promotion_gate":bool(confirmation and confirmation["all_gates_passed"]),
    }
    json.dump(summary,open(out/"summary.json","w"),indent=2,default=str)
    print(table.to_string(index=False))
    print(json.dumps(summary,indent=2,default=str))

if __name__=="__main__":
    main()
