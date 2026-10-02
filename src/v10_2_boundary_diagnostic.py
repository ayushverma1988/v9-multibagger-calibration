from __future__ import annotations

import argparse, json, hashlib
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from v10_2_compression_switch import (
    SwitchPolicy, spec_from_row, select_switch, perturb_all
)

POLICY=SwitchPolicy("compress_030_raw50",.030,.50)


def seed_for(date,salt):
    raw=f"{pd.Timestamp(date).date()}|boundary|{salt}".encode()
    return int(hashlib.sha256(raw).hexdigest()[:8],16)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--oos",required=True)
    ap.add_argument("--chosen931",required=True)
    ap.add_argument("--output",required=True)
    ap.add_argument("--date",default="2020-12-31")
    ap.add_argument("--sims",type=int,default=1000)
    args=ap.parse_args()

    d=pd.read_parquet(args.oos);d["date"]=pd.to_datetime(d["date"])
    chosen=pd.read_csv(args.chosen931);chosen["date"]=pd.to_datetime(chosen["date"])
    td=pd.Timestamp(args.date)
    rr=chosen[chosen["date"]==td]
    if rr.empty:raise SystemExit(f"No fold config for {td}")
    r=next(rr.itertuples(index=False))
    spec=spec_from_row(r)
    g=d[d["date"]==td].copy()
    base=select_switch(g,spec,POLICY,10)
    if len(base)!=10:raise SystemExit("Base selector did not return 10 names")

    base_syms=base["symbol"].astype(str).tolist()
    base_set=set(base_syms)
    inc=Counter(); rank_sum=defaultdict(float); rank_n=Counter()
    overlap=Counter(); entrants=Counter(); exits=Counter()
    rng=np.random.default_rng(seed_for(td,"external"))
    for _ in range(args.sims):
        z=perturb_all(g,.01,rng)
        s=select_switch(z,spec,POLICY,10)
        if len(s)!=10:continue
        syms=s["symbol"].astype(str).tolist(); ss=set(syms)
        common=len(base_set&ss)
        overlap[common]+=1
        for rank,sym in enumerate(syms,1):
            inc[sym]+=1;rank_sum[sym]+=rank;rank_n[sym]+=1
        for sym in ss-base_set:entrants[sym]+=1
        for sym in base_set-ss:exits[sym]+=1

    syms=set(base_syms)|set(inc)
    rows=[]
    bg=g.set_index(g["symbol"].astype(str),drop=False)
    for sym in syms:
        row=bg.loc[sym]
        if isinstance(row,pd.DataFrame):row=row.iloc[0]
        rows.append({
            "symbol":sym,
            "base_selected":sym in base_set,
            "base_rank":base_syms.index(sym)+1 if sym in base_set else np.nan,
            "selection_frequency":inc[sym]/args.sims,
            "mean_rank_when_selected":rank_sum[sym]/rank_n[sym] if rank_n[sym] else np.nan,
            "entry_frequency":entrants[sym]/args.sims,
            "exit_frequency":exits[sym]/args.sims,
            "p_cal":float(row.get("p_cal",np.nan)),
            "p_raw":float(row.get("p_raw",np.nan)),
            "p_dd30_cal":float(row.get("p_dd30_cal",np.nan)),
            "p_dd30_raw":float(row.get("p_dd30_raw",np.nan)),
            "model_dispersion":float(row.get("model_dispersion",np.nan)),
        })
    cand=pd.DataFrame(rows).sort_values(
        ["base_selected","selection_frequency","mean_rank_when_selected"],
        ascending=[False,False,True]
    )
    hist=pd.DataFrame([
        {"common_names":k,"jaccard":k/(20-k),"count":v,"fraction":v/args.sims}
        for k,v in sorted(overlap.items())
    ])

    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    cand.to_csv(out/"boundary_candidates.csv",index=False)
    hist.to_csv(out/"overlap_histogram.csv",index=False)
    summary={
        "fold":str(td.date()),
        "policy":POLICY.name,
        "simulations":args.sims,
        "base_top10":base_syms,
        "prob_overlap_le_7":float(sum(v for k,v in overlap.items() if k<=7)/args.sims),
        "prob_overlap_ge_8":float(sum(v for k,v in overlap.items() if k>=8)/args.sims),
        "p05_jaccard":float(np.quantile(
            [k/(20-k) for k,v in overlap.items() for _ in range(v)],.05
        )),
        "note":"diagnostic omits future outcome labels by design",
    }
    json.dump(summary,open(out/"summary.json","w"),indent=2)
    print(json.dumps(summary,indent=2))
    print(hist.to_string(index=False))
    print(cand.head(25).to_string(index=False))

if __name__=="__main__":
    main()
