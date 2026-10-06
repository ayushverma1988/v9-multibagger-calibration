from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

def jlist(v):
    try:
        x=json.loads(v) if isinstance(v,str) else v
        return x if isinstance(x,list) else []
    except Exception:
        return []

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--secondary",required=True)
    ap.add_argument("--primary",required=True)
    ap.add_argument("--theme-demand",required=True)
    ap.add_argument("--output",required=True)
    args=ap.parse_args()

    sec=pd.read_parquet(args.secondary)
    pri=pd.read_parquet(args.primary)
    cols=sorted(set(sec.columns)|set(pri.columns))
    for c in cols:
        if c not in sec: sec[c]=pd.NA
        if c not in pri: pri[c]=pd.NA
    allf=pd.concat([pri[cols],sec[cols]],ignore_index=True)
    allf["published_ts"]=pd.to_datetime(allf["published_ts"],utc=True,errors="coerce")
    allf=allf.dropna(subset=["symbol","published_ts"]).drop_duplicates("evidence_id")

    td=pd.read_csv(args.theme_demand)
    demand=dict(zip(td.get("theme",[]),pd.to_numeric(td.get("theme_demand_score",[]),errors="coerce").fillna(0)))

    allf["theme_demand_max"]=[
        max([demand.get(t,0.0) for t in jlist(v)] or [0.0]) for v in allf["themes"]
    ]
    allf["catalyst_type_count"]=[len(jlist(v)) for v in allf["catalyst_types"]]
    allf["magnitude_proxy"]=(
        np.clip(pd.to_numeric(allf["capacity_pct_max"],errors="coerce").fillna(0)/100.0,0,2)
        + np.clip(np.log1p(pd.to_numeric(allf["money_crore_max"],errors="coerce").fillna(0))/10.0,0,1.5)
    )
    allf["linked_evidence_score"]=(
        0.28*pd.to_numeric(allf["evidence_confidence"],errors="coerce").fillna(0)
        +0.22*pd.to_numeric(allf["stage_weight"],errors="coerce").fillna(0).clip(0,1)
        +0.16*np.clip(allf["catalyst_type_count"]/3.0,0,1)
        +0.14*allf["theme_demand_max"]
        +0.12*np.clip(allf["magnitude_proxy"]/2.0,0,1)
        +0.08*(pd.to_numeric(allf["source_tier"],errors="coerce").eq(1)).astype(float)
        -0.20*allf["negative_flag"].fillna(False).astype(float)
    )

    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
    allf.sort_values(["linked_evidence_score","published_ts"],ascending=False).to_parquet(out,index=False)

    company=[]
    for sym,g in allf.groupby("symbol"):
        recent=g.sort_values("published_ts",ascending=False)
        primary=recent[recent["source_tier"].eq(1)]
        company.append({
            "symbol":sym,
            "evidence_rows":int(len(g)),
            "primary_rows":int(len(primary)),
            "distinct_domains":int(g["domain"].nunique()),
            "max_linked_evidence_score":float(g["linked_evidence_score"].max()),
            "max_primary_score":float(primary["linked_evidence_score"].max()) if len(primary) else 0.0,
            "max_stage_weight":float(pd.to_numeric(g["stage_weight"],errors="coerce").fillna(0).max()),
            "max_theme_demand":float(g["theme_demand_max"].max()),
            "max_money_crore":float(pd.to_numeric(g["money_crore_max"],errors="coerce").max()) if pd.to_numeric(g["money_crore_max"],errors="coerce").notna().any() else None,
            "max_capacity_pct":float(pd.to_numeric(g["capacity_pct_max"],errors="coerce").max()) if pd.to_numeric(g["capacity_pct_max"],errors="coerce").notna().any() else None,
            "latest_evidence":str(g["published_ts"].max()),
            "primary_confirmed":bool(len(primary)>0),
        })
    cs=pd.DataFrame(company).sort_values(["primary_confirmed","max_linked_evidence_score","evidence_rows"],ascending=[False,False,False])
    cs.to_csv(out.parent/"company_catalyst_intelligence.csv",index=False)

    summary={
        "evidence_rows":int(len(allf)),
        "companies":int(allf["symbol"].nunique()),
        "primary_rows":int(allf["source_tier"].eq(1).sum()),
        "primary_confirmed_companies":int(cs["primary_confirmed"].sum()) if len(cs) else 0,
        "top_primary_confirmed":cs[cs["primary_confirmed"]].head(25).to_dict("records") if len(cs) else [],
    }
    json.dump(summary,open(out.parent/"merged_catalyst_summary.json","w"),indent=2,default=str)
    print(json.dumps(summary,indent=2,default=str))

if __name__=="__main__":
    main()
