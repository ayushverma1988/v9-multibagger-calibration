from __future__ import annotations

import argparse, hashlib, json
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import pandas as pd

from v10_2_compression_switch import SwitchPolicy, spec_from_row, select_switch, perturb_all
from v10_2_raw_rank_stability import outcome, agg, alpha_pass, select_policy, Policy as RawPolicy

BASE_SWITCH=SwitchPolicy("compress_030_raw50",.030,.50)
INC=RawPolicy("incumbent","p_cal","p_dd30_cal")

@dataclass(frozen=True)
class BagPolicy:
    name:str
    bag_sims:int
    freq_weight:float
    fragile_only:bool=True
    frag_sims:int=40

POLICIES=[
    BagPolicy("bag40_w25",40,.25),
    BagPolicy("bag40_w50",40,.50),
    BagPolicy("bag60_w25",60,.25),
    BagPolicy("bag60_w50",60,.50),
]

def seed_for(date,name,salt):
    raw=f"{pd.Timestamp(date).date()}|{name}|{salt}".encode()
    return int(hashlib.sha256(raw).hexdigest()[:8],16)

def base_select(g,spec,k=10):
    return select_switch(g,spec,BASE_SWITCH,k)

def base_full_score(g,spec):
    # use current deterministic base top-N ordering as alpha anchor
    q=g.copy()
    q["_base_rank_score"]=0.0
    # approximate full ordering by repeated k expansion, bounded by available rows
    n=min(max(30,int(spec["pool_n"])),len(q))
    s=select_switch(q,spec,BASE_SWITCH,n)
    if len(s):
        vals=np.linspace(1.0,1.0/max(2,len(s)),len(s))
        q.loc[s.index,"_base_rank_score"]=vals
    return q

def fragility(g,spec,p,k=10):
    b=base_select(g,spec,k)
    if len(b)!=k:return 0.0
    bs=set(b["symbol"].astype(str))
    rng=np.random.default_rng(seed_for(g["date"].iloc[0],p.name,"frag"))
    js=[]
    for _ in range(p.frag_sims):
        z=perturb_all(g,.01,rng)
        s=base_select(z,spec,k)
        if len(s)!=k:continue
        ss=set(s["symbol"].astype(str))
        js.append(len(bs&ss)/len(bs|ss))
    return float(np.quantile(js,.05)) if js else 0.0

def bag_select(g,spec,p,k=10,frag_override=None):
    base=base_select(g,spec,k)
    if len(base)!=k:return pd.DataFrame(),False,np.nan
    frag=fragility(g,spec,p,k) if frag_override is None else frag_override
    if p.fragile_only and frag>=.60:
        out=base.copy();out["_bag_used"]=0.0
        return out,False,frag

    # Common fixed bag stream for deterministic production selector.
    rng=np.random.default_rng(seed_for(g["date"].iloc[0],p.name,"bag"))
    counts={};rank_sum={};rank_n={}
    for _ in range(p.bag_sims):
        z=perturb_all(g,.01,rng)
        s=base_select(z,spec,k)
        if len(s)!=k:continue
        for rank,(_,r) in enumerate(s.iterrows(),1):
            sym=str(r["symbol"])
            counts[sym]=counts.get(sym,0)+1
            rank_sum[sym]=rank_sum.get(sym,0)+rank
            rank_n[sym]=rank_n.get(sym,0)+1

    base_rank={str(sym):1-(i/max(10,k+5)) for i,sym in enumerate(base["symbol"].astype(str),1)}
    syms=set(base_rank)|set(counts)
    rows=[]
    gi=g.set_index(g["symbol"].astype(str),drop=False)
    for sym in syms:
        if sym not in gi.index:continue
        row=gi.loc[sym]
        if isinstance(row,pd.DataFrame):row=row.iloc[0]
        freq=counts.get(sym,0)/max(1,p.bag_sims)
        mr=rank_sum.get(sym,0)/rank_n.get(sym,1) if rank_n.get(sym,0) else k+10
        anchor=base_rank.get(sym,0.0)
        robust_rank=1-(min(mr,k+10)-1)/(k+9)
        bag_score=(1-p.freq_weight)*anchor+p.freq_weight*(.75*freq+.25*robust_rank)
        rows.append((sym,bag_score,freq,mr,float(row.get("p_raw",np.nan)),float(row.get("p_cal",np.nan))))
    tab=pd.DataFrame(rows,columns=["symbol","bag_score","freq","mean_rank","p_raw","p_cal"]).sort_values(
        ["bag_score","freq","mean_rank","p_raw","symbol"],ascending=[False,False,True,False,True]
    ).head(k)
    out=g[g["symbol"].astype(str).isin(tab["symbol"])].copy()
    order=dict(zip(tab["symbol"],range(len(tab))))
    out["_ord"]=out["symbol"].astype(str).map(order)
    out=out.sort_values("_ord").head(k).copy()
    if len(out)!=k:return base,False,frag
    out["_bag_used"]=1.0
    return out,True,frag

def stability(g,spec,p,k,sims):
    base,_,frag=bag_select(g,spec,p,k)
    if len(base)!=k:return None
    bs=set(base["symbol"].astype(str));top=str(base.iloc[0]["symbol"])
    rng=np.random.default_rng(seed_for(g["date"].iloc[0],p.name,"external"))
    js=[];tops=[];rep=[]
    for _ in range(sims):
        z=perturb_all(g,.01,rng)
        s,_,_=bag_select(z,spec,p,k,frag_override=frag)
        if len(s)!=k:continue
        ss=set(s["symbol"].astype(str))
        js.append(len(bs&ss)/len(bs|ss)); tops.append(float(str(s.iloc[0]["symbol"])==top));rep.append(len(bs-ss)/k)
    if not js:return None
    return {
        "mean_jaccard":float(np.mean(js)),
        "p05_jaccard":float(np.quantile(js,.05)),
        "top1_stability":float(np.mean(tops)),
        "names_replaced_fraction":float(np.mean(rep)),
    }

def baseline_metrics(d,cmap,cfg):
    k=int(cfg.get("selection_k",10));rows=[]
    for td,r in cmap.items():
        g=d[d["date"]==td]
        s=select_policy(g,spec_from_row(r),INC,k);m=outcome(g,s,k)
        if m:rows.append({"date":td,**m})
    return agg(pd.DataFrame(rows))

def evaluate(d,cmap,cfg,p,sims):
    k=int(cfg.get("selection_k",10));mets=[];sts=[];uses=[]
    for td,r in cmap.items():
        g=d[d["date"]==td].copy();spec=spec_from_row(r)
        s,used,frag=bag_select(g,spec,p,k);m=outcome(g,s,k)
        if m:mets.append({"date":td,**m})
        uses.append({"date":td,"bag_used":float(used),"fragility_p05":frag})
        st=stability(g,spec,p,k,sims)
        if st:sts.append({"date":td,**st})
    return pd.DataFrame(mets),pd.DataFrame(sts),pd.DataFrame(uses)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--oos",required=True);ap.add_argument("--chosen931",required=True);ap.add_argument("--config",required=True)
    ap.add_argument("--output",required=True);ap.add_argument("--grid-sims",type=int,default=100);ap.add_argument("--confirm-sims",type=int,default=500)
    args=ap.parse_args()
    cfg=json.load(open(args.config));d=pd.read_parquet(args.oos);d["date"]=pd.to_datetime(d["date"])
    ch=pd.read_csv(args.chosen931);ch["date"]=pd.to_datetime(ch["date"]);cmap={pd.Timestamp(r.date):r for r in ch.itertuples(index=False)}
    base=baseline_metrics(d,cmap,cfg)

    rows=[]
    for p in POLICIES:
        mt,st,use=evaluate(d,cmap,cfg,p,args.grid_sims);a=agg(mt)
        if not a or st.empty:continue
        mj=float(st["mean_jaccard"].mean());wp=float(st["p05_jaccard"].min());t1=float(st["top1_stability"].mean())
        rows.append({**asdict(p),**a,"mean_jaccard":mj,"worst_p05_jaccard":wp,"mean_top1_stability":t1,
                     "mean_names_replaced_fraction":float(st["names_replaced_fraction"].mean()),
                     "worst_fold":str(st.loc[st["p05_jaccard"].idxmin(),"date"]),
                     "bag_folds":int(use["bag_used"].sum()),
                     "alpha_retention_passed":bool(alpha_pass(a,base,.95)),
                     "stability_passed":bool(mj>=.80 and wp>=.60 and t1>=.70),
                     "dd_not_worse_3pp":bool(a["mean_dd30_rate"]<=base["mean_dd30_rate"]+.03)})
    tab=pd.DataFrame(rows);tab["all_gates_passed"]=tab["alpha_retention_passed"]&tab["stability_passed"]&tab["dd_not_worse_3pp"]
    tab=tab.sort_values(["stability_passed","bag_sims","freq_weight","mean_jaccard"],ascending=[False,True,True,False])
    stable=tab[tab["stability_passed"]]
    name=str(stable.iloc[0]["name"]) if len(stable) else (str(tab.iloc[0]["name"]) if len(tab) else None)
    p=next((x for x in POLICIES if x.name==name),None)

    conf=None
    if p:
        mt,st,use=evaluate(d,cmap,cfg,p,args.confirm_sims);a=agg(mt)
        mj=float(st["mean_jaccard"].mean());wp=float(st["p05_jaccard"].min());t1=float(st["top1_stability"].mean())
        conf={**a,"mean_jaccard":mj,"worst_p05_jaccard":wp,"mean_top1_stability":t1,
              "mean_names_replaced_fraction":float(st["names_replaced_fraction"].mean()),
              "worst_fold":str(st.loc[st["p05_jaccard"].idxmin(),"date"]),
              "alpha_retention_passed":bool(alpha_pass(a,base,.95)),
              "mean_jaccard_passed":bool(mj>=.80),"worst_p05_passed":bool(wp>=.60),"top1_passed":bool(t1>=.70),
              "dd_not_worse_3pp":bool(a["mean_dd30_rate"]<=base["mean_dd30_rate"]+.03)}
        conf["all_gates_passed"]=bool(conf["alpha_retention_passed"] and conf["mean_jaccard_passed"] and conf["worst_p05_passed"] and conf["top1_passed"] and conf["dd_not_worse_3pp"])

    out=Path(args.output);out.mkdir(parents=True,exist_ok=True);tab.to_csv(out/"bagged_grid.csv",index=False)
    if p:
        mt,st,use=evaluate(d,cmap,cfg,p,args.confirm_sims);mt.to_csv(out/"chosen_metrics_by_fold.csv",index=False);st.to_csv(out/"chosen_stability_by_fold.csv",index=False);use.to_csv(out/"chosen_usage.csv",index=False)
    summary={"model":"V10.2 deterministic bagged stability selection","principle":"predeclared final fallback; deterministic selection-frequency/mean-rank aggregation only on label-free fragile folds","baseline":base,"chosen_policy":asdict(p) if p else None,"confirmation_500":conf,"promotion_gate":bool(conf and conf["all_gates_passed"])}
    json.dump(summary,open(out/"summary.json","w"),indent=2,default=str);print(tab.to_string(index=False));print(json.dumps(summary,indent=2,default=str))
if __name__=="__main__":main()
