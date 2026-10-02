from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd

from v10_2_raw_rank_stability import spec_from_row
from v10_2_robust_quantile import POLICIES as ROBUST_POLICIES, external_stability as robust_stability
from v10_2_bagged_topk import POLICIES as BAG_POLICIES, external_stability as bag_stability

BLOCKER_DATES = [pd.Timestamp("2018-12-31"), pd.Timestamp("2020-12-31")]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--oos",required=True)
    ap.add_argument("--chosen931",required=True)
    ap.add_argument("--config",required=True)
    ap.add_argument("--output",required=True)
    ap.add_argument("--sims",type=int,default=100)
    args=ap.parse_args()

    cfg=json.load(open(args.config))
    k=int(cfg.get("selection_k",10))
    oos=pd.read_parquet(args.oos); oos["date"]=pd.to_datetime(oos["date"])
    chosen=pd.read_csv(args.chosen931); chosen["date"]=pd.to_datetime(chosen["date"])
    cmap={pd.Timestamp(r.date):r for r in chosen.itertuples(index=False)}

    rows=[]
    families=[
        ("robust_quantile", ROBUST_POLICIES, robust_stability),
        ("bagged_topk", BAG_POLICIES, bag_stability),
    ]
    for family, policies, fn in families:
        for p in policies:
            fold_rows=[]
            for td in BLOCKER_DATES:
                if td not in cmap:
                    continue
                g=oos[oos["date"]==td].copy()
                if g.empty:
                    continue
                st=fn(g,spec_from_row(cmap[td]),p,k,args.sims)
                if st is None:
                    continue
                fold_rows.append({
                    "date":str(td.date()),
                    "mean_jaccard":float(st["mean_jaccard"]),
                    "p05_jaccard":float(st["p05_jaccard"]),
                    "top1_stability":float(st["top1_stability"]),
                    "names_replaced_fraction":float(st["names_replaced_fraction"]),
                })
            if not fold_rows:
                continue
            mean_j=sum(x["mean_jaccard"] for x in fold_rows)/len(fold_rows)
            worst_p05=min(x["p05_jaccard"] for x in fold_rows)
            mean_top1=sum(x["top1_stability"] for x in fold_rows)/len(fold_rows)
            rows.append({
                "family":family,
                "policy":p.name,
                "blocker_mean_jaccard":mean_j,
                "blocker_worst_p05_jaccard":worst_p05,
                "blocker_mean_top1_stability":mean_top1,
                "screen_pass":bool(mean_j>=0.80 and worst_p05>=0.60 and mean_top1>=0.70),
                "folds":fold_rows,
            })

    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    json.dump({
        "purpose":"rejection-only stability screen; no labels/outcomes used; passing does not authorize promotion",
        "blocker_dates":[str(x.date()) for x in BLOCKER_DATES],
        "sims":args.sims,
        "gates":{"mean_jaccard":0.80,"worst_p05_jaccard":0.60,"mean_top1_stability":0.70},
        "results":rows,
    },open(out/"summary.json","w"),indent=2)
    pd.DataFrame([{k:v for k,v in r.items() if k!="folds"} for r in rows]).to_csv(out/"screen.csv",index=False)
    print(pd.DataFrame([{k:v for k,v in r.items() if k!="folds"} for r in rows]).to_string(index=False))

if __name__=="__main__":
    main()
