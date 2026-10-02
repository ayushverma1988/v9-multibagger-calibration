from __future__ import annotations

import argparse, hashlib, json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

from v10_2_raw_rank_stability import (
    Policy as RawPolicy, spec_from_row, rank_high, rank_low,
    outcome, agg, alpha_pass, select_policy
)
from v10_2_compression_switch import perturb_all

INC=RawPolicy("incumbent","p_cal","p_dd30_cal")


@dataclass(frozen=True)
class BandPolicy:
    name:str
    rel_band:float
    fragile_only:bool=True
    internal_sims:int=50


POLICIES=[
    BandPolicy("band005",.005),
    BandPolicy("band010",.010),
    BandPolicy("band020",.020),
    BandPolicy("band030",.030),
]


def seed_for(date,name,salt):
    raw=f"{pd.Timestamp(date).date()}|{name}|{salt}".encode()
    return int(hashlib.sha256(raw).hexdigest()[:8],16)


def full_incumbent_rank(g,spec):
    q=g.dropna(subset=["p_cal","p_raw","p_dd30_cal","p_dd30_raw","model_dispersion"]).copy()
    if len(q)<10:return pd.DataFrame()

    if spec["baseline"]:
        q["selection_score_band"]=rank_high(q["p_cal"].astype(float))
        return q.sort_values(
            ["selection_score_band","p_cal","model_dispersion","symbol"],
            ascending=[False,False,True,True]
        ).copy()

    pool_n=min(int(spec["pool_n"]),len(q))
    pool=q.sort_values(["p_cal","model_dispersion","symbol"],ascending=[False,True,True]).head(pool_n).copy()
    if len(pool)<10:return pd.DataFrame()
    if float(spec["risk_drop"])>0:
        cutoff=float(pool["p_dd30_cal"].quantile(np.clip(1-float(spec["risk_drop"]),0,1)))
        pool=pool[pool["p_dd30_cal"]<=cutoff].copy()
    if len(pool)<10:return pd.DataFrame()

    ws=float(spec["w_safety"]);wc=float(spec["w_consensus"]);wa=max(0.0,1-ws-wc)
    pool["selection_score_band"]=(
        wa*rank_high(pool["p_cal"])
        +ws*rank_low(pool["p_dd30_cal"])
        +wc*rank_low(pool["model_dispersion"])
    )
    return pool.sort_values(
        ["selection_score_band","p_cal","model_dispersion","symbol"],
        ascending=[False,False,True,True]
    ).copy()


def base_select(g,spec,k=10):
    return full_incumbent_rank(g,spec).head(k).copy()


def fragility_p05(g,spec,p,k=10):
    base=base_select(g,spec,k)
    if len(base)!=k:return 0.0
    bs=set(base["symbol"].astype(str))
    rng=np.random.default_rng(seed_for(g["date"].iloc[0],p.name,"fragility"))
    js=[]
    for _ in range(p.internal_sims):
        z=perturb_all(g,.01,rng)
        s=base_select(z,spec,k)
        if len(s)!=k:continue
        ss=set(s["symbol"].astype(str))
        js.append(len(bs&ss)/len(bs|ss))
    return float(np.quantile(js,.05)) if js else 0.0


def select_band(g,spec,p,k=10,frag_override=None):
    ranked=full_incumbent_rank(g,spec)
    if len(ranked)<k:return pd.DataFrame(),False,np.nan
    frag=fragility_p05(g,spec,p,k) if frag_override is None else frag_override
    if p.fragile_only and frag>=.60:
        out=ranked.head(k).copy();out["_band_used"]=0.0
        return out,False,frag

    ranked=ranked.copy()
    cutoff=float(ranked.iloc[k-1]["selection_score_band"])
    scale=max(abs(cutoff),1e-9)
    lo=cutoff-p.rel_band*scale
    hi=cutoff+p.rel_band*scale

    above=ranked[ranked["selection_score_band"]>hi].copy()
    band=ranked[(ranked["selection_score_band"]>=lo)&(ranked["selection_score_band"]<=hi)].copy()

    # Only boundary names are re-ordered. Raw alpha is primary within the band;
    # raw downside risk and dispersion are tie-breakers.
    band["_raw_alpha_rank"]=rank_high(band["p_raw"])
    band["_raw_safety_rank"]=rank_low(band["p_dd30_raw"])
    band["_raw_consensus_rank"]=rank_low(band["model_dispersion"])
    ws=float(spec["w_safety"])*.50
    wc=float(spec["w_consensus"])*.50
    wa=max(0.0,1-ws-wc)
    band["_band_score"]=wa*band["_raw_alpha_rank"]+ws*band["_raw_safety_rank"]+wc*band["_raw_consensus_rank"]
    band=band.sort_values(["_band_score","p_raw","model_dispersion","symbol"],
                          ascending=[False,False,True,True])

    need=max(0,k-len(above))
    selected=pd.concat([above,band.head(need)],axis=0).head(k)

    if len(selected)<k:
        rest=ranked.loc[~ranked.index.isin(selected.index)].head(k-len(selected))
        selected=pd.concat([selected,rest],axis=0)
    if len(selected)!=k:
        return ranked.head(k).copy(),False,frag

    selected=selected.copy();selected["_band_used"]=1.0
    return selected,True,frag


def external_stability(g,spec,p,k,sims):
    base,_,frag=select_band(g,spec,p,k)
    if len(base)!=k:return None
    bs=set(base["symbol"].astype(str));top=str(base.iloc[0]["symbol"])
    rng=np.random.default_rng(seed_for(g["date"].iloc[0],p.name,"external"))
    js=[];tops=[];rep=[]
    for _ in range(sims):
        z=perturb_all(g,.01,rng)
        s,_,_=select_band(z,spec,p,k,frag_override=frag)
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


def incumbent_metrics(d,cmap,cfg):
    k=int(cfg.get("selection_k",10));rows=[]
    for td,r in cmap.items():
        g=d[d["date"]==td].copy()
        s=select_policy(g,spec_from_row(r),INC,k)
        m=outcome(g,s,k)
        if m:rows.append({"date":td,**m})
    return agg(pd.DataFrame(rows))


def evaluate(d,cmap,cfg,p,sims):
    k=int(cfg.get("selection_k",10));mets=[];sts=[];uses=[]
    for td,r in cmap.items():
        g=d[d["date"]==td].copy();spec=spec_from_row(r)
        s,active,frag=select_band(g,spec,p,k)
        m=outcome(g,s,k)
        if m:mets.append({"date":td,**m})
        uses.append({"date":td,"band_used":float(active),"fragility_p05":frag})
        st=external_stability(g,spec,p,k,sims)
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
    d=pd.read_parquet(args.oos);d["date"]=pd.to_datetime(d["date"])
    chosen=pd.read_csv(args.chosen931);chosen["date"]=pd.to_datetime(chosen["date"])
    cmap={pd.Timestamp(r.date):r for r in chosen.itertuples(index=False)}
    baseline=incumbent_metrics(d,cmap,cfg)

    rows=[]
    for p in POLICIES:
        mt,st,use=evaluate(d,cmap,cfg,p,args.grid_sims)
        a=agg(mt)
        if not a or st.empty:continue
        mj=float(st["mean_jaccard"].mean());wp=float(st["p05_jaccard"].min());t1=float(st["top1_stability"].mean())
        rows.append({
            **asdict(p),**a,
            "mean_jaccard":mj,"worst_p05_jaccard":wp,"mean_top1_stability":t1,
            "mean_names_replaced_fraction":float(st["names_replaced_fraction"].mean()),
            "worst_fold":str(st.loc[st["p05_jaccard"].idxmin(),"date"]),
            "band_folds":int(use["band_used"].sum()),
            "alpha_retention_passed":bool(alpha_pass(a,baseline,.95)),
            "stability_passed":bool(mj>=.80 and wp>=.60 and t1>=.70),
            "dd_not_worse_3pp":bool(a["mean_dd30_rate"]<=baseline["mean_dd30_rate"]+.03),
        })
    table=pd.DataFrame(rows)
    table["all_gates_passed"]=table["alpha_retention_passed"]&table["stability_passed"]&table["dd_not_worse_3pp"]
    table=table.sort_values(["stability_passed","rel_band","mean_jaccard"],ascending=[False,True,False])

    stable=table[table["stability_passed"]]
    chosen_name=str(stable.iloc[0]["name"]) if len(stable) else (str(table.iloc[0]["name"]) if len(table) else None)
    p=next((x for x in POLICIES if x.name==chosen_name),None)

    confirmation=None
    if p:
        mt,st,use=evaluate(d,cmap,cfg,p,args.confirm_sims)
        a=agg(mt);mj=float(st["mean_jaccard"].mean());wp=float(st["p05_jaccard"].min());t1=float(st["top1_stability"].mean())
        confirmation={
            **a,"mean_jaccard":mj,"worst_p05_jaccard":wp,"mean_top1_stability":t1,
            "mean_names_replaced_fraction":float(st["names_replaced_fraction"].mean()),
            "worst_fold":str(st.loc[st["p05_jaccard"].idxmin(),"date"]),
            "band_folds":int(use["band_used"].sum()),
            "alpha_retention_passed":bool(alpha_pass(a,baseline,.95)),
            "mean_jaccard_passed":bool(mj>=.80),"worst_p05_passed":bool(wp>=.60),
            "top1_passed":bool(t1>=.70),"dd_not_worse_3pp":bool(a["mean_dd30_rate"]<=baseline["mean_dd30_rate"]+.03),
        }
        confirmation["all_gates_passed"]=bool(
            confirmation["alpha_retention_passed"] and confirmation["mean_jaccard_passed"]
            and confirmation["worst_p05_passed"] and confirmation["top1_passed"] and confirmation["dd_not_worse_3pp"]
        )

    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    table.to_csv(out/"boundary_band_grid.csv",index=False)
    if p:
        mt,st,use=evaluate(d,cmap,cfg,p,args.confirm_sims)
        mt.to_csv(out/"chosen_metrics_by_fold.csv",index=False)
        st.to_csv(out/"chosen_stability_by_fold.csv",index=False)
        use.to_csv(out/"chosen_band_usage.csv",index=False)
    summary={
        "model":"V10.2 raw boundary-band tie-break",
        "principle":"only near-cutoff candidates on label-free fragile folds are reranked using raw alpha/risk; rest of incumbent ranking unchanged",
        "baseline":baseline,
        "chosen_policy":asdict(p) if p else None,
        "confirmation_500":confirmation,
        "promotion_gate":bool(confirmation and confirmation["all_gates_passed"]),
    }
    json.dump(summary,open(out/"summary.json","w"),indent=2,default=str)
    print(table.to_string(index=False))
    print(json.dumps(summary,indent=2,default=str))

if __name__=="__main__":
    main()
