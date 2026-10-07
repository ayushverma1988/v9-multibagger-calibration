from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--audit",required=True)
    ap.add_argument("--demand",required=True)
    ap.add_argument("--freeze",required=True)
    ap.add_argument("--output",required=True)
    args=ap.parse_args()

    a=pd.read_csv(args.audit)
    d=pd.read_csv(args.demand)
    cfg=json.load(open(args.freeze))
    g=cfg["chain_requirements"]

    demand=dict(zip(d["theme"].astype(str),pd.to_numeric(d["theme_demand_score"],errors="coerce").fillna(0)))
    def themes(v):
        s=str(v or "")
        if s.startswith("["):
            try:
                z=json.loads(s); return z if isinstance(z,list) else []
            except Exception:return []
        return [x for x in s.split("|") if x]
    theme_col="strongest_event_themes" if "strongest_event_themes" in a.columns else ("themes" if "themes" in a.columns else None)
    a["external_theme_demand"]=0.0
    if theme_col:
        a["external_theme_demand"]=[
            max([demand.get(t,0.0) for t in themes(v)] or [0.0]) for v in a[theme_col]
        ]

    for c in ["primary_event_count","stage_reality_score","already_priced_penalty_used",
              "earnings_inflection_score","catalyst_magnitude_score","order_probability_score",
              "promoter_accumulation_score","technical_confirmation_used"]:
        if c not in a:a[c]=0
        a[c]=pd.to_numeric(a[c],errors="coerce").fillna(0)

    a["gate_primary"]=a["primary_event_count"]>=int(g["primary_company_evidence_min"])
    a["gate_external_demand"]=a["external_theme_demand"]>=float(g["external_theme_demand_min"])
    a["gate_stage"]=a["stage_reality_score"]>=float(g["catalyst_stage_min"])
    a["gate_not_priced_in"]=a["already_priced_penalty_used"]<=float(g["priced_in_penalty_max"])
    a["gate_impact"]=(a["earnings_inflection_score"]>=0.45)|(a["catalyst_magnitude_score"]>=0.45)
    a["gate_causal_mechanism"]=(a["order_probability_score"]>=0.55)|(a["stage_reality_score"]>=0.55)
    a["causal_chain_pass"]=a[[
        "gate_primary","gate_external_demand","gate_stage","gate_not_priced_in","gate_impact","gate_causal_mechanism"
    ]].all(axis=1)

    confirmations=(
        (a["earnings_inflection_score"]>=0.55).astype(int)
        +(a["catalyst_magnitude_score"]>=0.55).astype(int)
        +(a["order_probability_score"]>=0.65).astype(int)
        +(a["promoter_accumulation_score"]>=0.35).astype(int)
    )
    a["impact_conviction_confirmations"]=confirmations
    a["causal_grade"]=np.select(
        [a["causal_chain_pass"]&(confirmations>=2),
         a["causal_chain_pass"]&(confirmations>=1)],
        ["A","B"],default=np.where(a["causal_chain_pass"],"C","REJECT")
    )
    # Transparent QA score; not a production prediction.
    a["causal_qa_score"]=(
        0.22*a["stage_reality_score"].clip(0,1)
        +0.20*a["external_theme_demand"].clip(0,1)
        +0.18*a["catalyst_magnitude_score"].clip(0,1)
        +0.16*a["earnings_inflection_score"].clip(0,1)
        +0.12*a["order_probability_score"].clip(0,1)
        +0.07*a["promoter_accumulation_score"].clip(0,1)
        +0.05*a["technical_confirmation_used"].clip(0,1)
        -0.12*a["already_priced_penalty_used"].clip(0,1)
    )
    a=a.sort_values(["causal_chain_pass","causal_grade","causal_qa_score"],ascending=[False,True,False])
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    a.to_csv(out,index=False)
    passed=a[a["causal_chain_pass"]].copy()
    passed.to_csv(out.parent/"causal_chain_pass.csv",index=False)
    summary={
        "rows":int(len(a)),"passed":int(len(passed)),
        "grade_A":int((a["causal_grade"]=="A").sum()),
        "grade_B":int((a["causal_grade"]=="B").sum()),
        "grade_C":int((a["causal_grade"]=="C").sum()),
        "rejected":int((a["causal_grade"]=="REJECT").sum()),
        "top_passed":passed.head(20)[[c for c in [
            "symbol","causal_grade","causal_qa_score","external_theme_demand",
            "stage_reality_score","catalyst_magnitude_score","earnings_inflection_score",
            "order_probability_score","promoter_accumulation_score","already_priced_penalty_used",
            "strongest_event_title"
        ] if c in passed.columns]].to_dict("records")
    }
    json.dump(summary,open(out.parent/"causal_chain_summary.json","w"),indent=2,default=str)
    print(json.dumps(summary,indent=2,default=str))

if __name__=="__main__":
    main()
