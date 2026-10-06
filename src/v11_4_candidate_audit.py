from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def pct(s):
    x=pd.to_numeric(s,errors="coerce")
    if x.notna().sum()<2:
        return pd.Series(0.5,index=s.index)
    return x.rank(pct=True,method="average").fillna(0.5)


def jlist(v):
    try:
        z=json.loads(v) if isinstance(v,str) else v
        return z if isinstance(z,list) else []
    except Exception:
        return []


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",required=True)
    ap.add_argument("--company-intelligence",required=True)
    ap.add_argument("--evidence",required=True)
    ap.add_argument("--financials",required=True)
    ap.add_argument("--market",required=True)
    ap.add_argument("--output",required=True)
    args=ap.parse_args()

    cfg=json.load(open(args.config))
    ci=pd.read_csv(args.company_intelligence)
    ev=pd.read_parquet(args.evidence)
    fin=pd.read_csv(args.financials)
    mkt=pd.read_csv(args.market)

    for d in [ci,fin,mkt]:
        d["symbol"]=d["symbol"].astype(str).str.upper().str.strip()

    # Evidence-specific audit features.
    e_rows=[]
    ev["published_ts"]=pd.to_datetime(ev["published_ts"],utc=True,errors="coerce")
    ev["stage_weight"]=pd.to_numeric(ev["stage_weight"],errors="coerce").fillna(0)
    ev["source_tier"]=pd.to_numeric(ev["source_tier"],errors="coerce")
    ev["money_crore_max"]=pd.to_numeric(ev["money_crore_max"],errors="coerce")
    ev["capacity_pct_max"]=pd.to_numeric(ev["capacity_pct_max"],errors="coerce")
    for sym,g in ev.groupby(ev["symbol"].astype(str).str.upper().str.strip()):
        g=g.sort_values(["linked_evidence_score","published_ts"],ascending=[False,False])
        prim=g[g["source_tier"].eq(1)]
        order=g[g["catalyst_types"].map(lambda x:"order" in jlist(x))]
        commission=g[g["catalyst_types"].map(lambda x:"commissioning" in jlist(x))]
        product=g[g["catalyst_types"].map(lambda x:bool(set(jlist(x))&{"product","approval"}))]
        capacity=g[g["catalyst_types"].map(lambda x:"capacity" in jlist(x))]
        neg=g[g["negative_flag"].fillna(False)]
        strongest=g.iloc[0]
        e_rows.append({
            "symbol":sym,
            "primary_event_count":int(len(prim)),
            "evidence_domain_count":int(g["domain"].nunique()),
            "order_visibility_score":float(order["stage_weight"].max()) if len(order) else 0.0,
            "commissioning_score":float(commission["stage_weight"].max()) if len(commission) else 0.0,
            "product_approval_score":float(product["stage_weight"].max()) if len(product) else 0.0,
            "capacity_stage_score":float(capacity["stage_weight"].max()) if len(capacity) else 0.0,
            "negative_event_count":int(len(neg)),
            "strongest_event_title":str(strongest.get("title",""))[:500],
            "strongest_event_url":str(strongest.get("url","")),
            "strongest_event_date":str(strongest.get("published_ts","")),
            "strongest_event_stage":str(strongest.get("stage","")),
            "strongest_event_types":str(strongest.get("catalyst_types","[]")),
            "strongest_event_themes":str(strongest.get("themes","[]")),
        })
    ea=pd.DataFrame(e_rows)

    x=ci.merge(fin,on="symbol",how="left").merge(ea,on="symbol",how="left").merge(mkt,on="symbol",how="left",suffixes=("","_mkt"))

    for c in [
        "max_stage_weight","max_linked_evidence_score","max_theme_demand_signal","max_capacity_pct",
        "max_money_crore","financial_score_raw","annualized_sales_crore","order_visibility_score",
        "commissioning_score","product_approval_score","capacity_stage_score","promoter_conviction_score",
        "accumulation_score","ownership_accumulation_score","technical_confirmation_score",
        "priced_in_penalty","risk_score","p100_anchor","v10_percentile"
    ]:
        if c not in x: x[c]=np.nan
        x[c]=pd.to_numeric(x[c],errors="coerce")

    # Magnitude: actual capacity expansion and catalyst money normalized to current sales.
    cap_mag=np.clip(x["max_capacity_pct"].fillna(0)/100.0,0,1.5)/1.5
    value_ratio=x["max_money_crore"]/x["annualized_sales_crore"].replace(0,np.nan)
    value_mag=np.clip(value_ratio.fillna(0)/1.0,0,1)
    x["catalyst_magnitude_score"]=np.maximum(cap_mag,value_mag)

    # Demand linkage: external demand signal dominates. Cross-company breadth is already capped upstream.
    x["product_demand_linkage_score"]=x["max_theme_demand_signal"].fillna(0).clip(0,1)

    x["stage_reality_score"]=x[["max_stage_weight","commissioning_score","order_visibility_score"]].max(axis=1).fillna(0).clip(0,1)
    x["earnings_inflection_score"]=x["financial_score_raw"].fillna(0).clip(0,1)
    x["order_probability_score"]=(
        0.70*x["order_visibility_score"].fillna(0).clip(0,1)
        +0.15*x["product_approval_score"].fillna(0).clip(0,1)
        +0.15*x["evidence_domain_count"].fillna(0).clip(0,3)/3.0
    ).clip(0,1)
    x["promoter_accumulation_score"]=(
        0.45*x["promoter_conviction_score"].fillna(0)
        +0.35*x["accumulation_score"].fillna(0)
        +0.20*x["ownership_accumulation_score"].fillna(0)
    ).clip(0,1)
    x["execution_quality_score"]=(
        0.45*x["stage_reality_score"]
        +0.30*x["financial_freshness"].fillna(0)
        +0.25*(x["primary_event_count"].fillna(0).clip(0,3)/3.0)
    ).clip(0,1)

    w=cfg["catalyst_families"]
    x["catalyst_intelligence_score"]=(
        float(w["catalyst_stage_and_reality"])*x["stage_reality_score"]
        +float(w["catalyst_magnitude"])*x["catalyst_magnitude_score"]
        +float(w["product_demand_linkage"])*x["product_demand_linkage_score"]
        +float(w["earnings_inflection"])*x["earnings_inflection_score"]
        +float(w["order_probability_and_visibility"])*x["order_probability_score"]
        +float(w["promoter_and_accumulation"])*x["promoter_accumulation_score"]
        +float(w["funding_and_execution_quality"])*x["execution_quality_score"]
    )

    technical=x["technical_confirmation_score"].fillna(x["v10_percentile"]).fillna(0.5).clip(0,1)
    x["technical_confirmation_used"]=technical
    x["already_priced_penalty_used"]=x["priced_in_penalty"].fillna(0).clip(0,1)
    x["v11_4_score_raw"]=(
        float(cfg["final_layers"]["catalyst_intelligence"])*x["catalyst_intelligence_score"]
        +float(cfg["final_layers"]["technical_confirmation_v10_2"])*technical
        -0.18*x["already_priced_penalty_used"]
        -0.08*x["risk_score"].fillna(0).clip(0,1)
        -0.05*np.clip(x["negative_event_count"].fillna(0),0,2)
    )

    # Eligibility requires a real primary catalyst and at least one causal mechanism.
    causal=(
        (x["stage_reality_score"]>=0.55)
        | (x["order_probability_score"]>=0.55)
        | ((x["catalyst_magnitude_score"]>=0.45)&(x["earnings_inflection_score"]>=0.45))
        | ((x["product_approval_score"]>=0.55)&(x["product_demand_linkage_score"]>=0.35))
    )
    x["v11_4_eligible"]=(
        x["primary_confirmed"].fillna(False).astype(bool)
        & (x["primary_event_count"].fillna(0)>=1)
        & causal
        & (x["already_priced_penalty_used"]<=0.70)
    )

    x["audit_grade"]=np.select(
        [
            x["v11_4_eligible"]&(x["v11_4_score_raw"]>=0.68),
            x["v11_4_eligible"]&(x["v11_4_score_raw"]>=0.58),
            x["v11_4_eligible"],
        ],
        ["A","B","C"],
        default="REJECT"
    )

    eligible=x[x["v11_4_eligible"]].sort_values("v11_4_score_raw",ascending=False).copy()
    eligible["audit_rank"]=np.arange(1,len(eligible)+1)

    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    x.sort_values("v11_4_score_raw",ascending=False).to_csv(out/"candidate_audit_all.csv",index=False)
    eligible.to_csv(out/"candidate_audit_eligible.csv",index=False)
    eligible.head(20).to_csv(out/"candidate_audit_top20.csv",index=False)

    summary={
        "companies_audited":int(len(x)),
        "eligible":int(len(eligible)),
        "grade_A":int((eligible["audit_grade"]=="A").sum()) if len(eligible) else 0,
        "grade_B":int((eligible["audit_grade"]=="B").sum()) if len(eligible) else 0,
        "grade_C":int((eligible["audit_grade"]=="C").sum()) if len(eligible) else 0,
        "financial_coverage_pct":float(x["financial_available"].fillna(False).mean()) if "financial_available" in x else 0.0,
        "top20":eligible.head(20)[[
            c for c in [
                "symbol","audit_rank","audit_grade","v11_4_score_raw","catalyst_intelligence_score",
                "stage_reality_score","catalyst_magnitude_score","product_demand_linkage_score",
                "earnings_inflection_score","order_probability_score","promoter_accumulation_score",
                "technical_confirmation_used","already_priced_penalty_used","strongest_event_title"
            ] if c in eligible
        ]].to_dict("records"),
        "note":"This is a candidate audit, not a production Top-10. Promotion requires historical validation and manual evidence QA."
    }
    json.dump(summary,open(out/"candidate_audit_summary.json","w"),indent=2,default=str)
    print(json.dumps(summary,indent=2,default=str))

if __name__=="__main__":
    main()
