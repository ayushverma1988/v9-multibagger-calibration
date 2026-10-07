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
        prim_chron=prim.sort_values("published_ts").copy()
        latest_primary_ts=prim_chron["published_ts"].max() if len(prim_chron) else pd.NaT
        first_primary_ts=prim_chron["published_ts"].min() if len(prim_chron) else pd.NaT
        recent_primary=pd.DataFrame(columns=prim_chron.columns)
        prior_primary=pd.DataFrame(columns=prim_chron.columns)
        if len(prim_chron) and pd.notna(latest_primary_ts):
            recent_primary=prim_chron[prim_chron["published_ts"]>=latest_primary_ts-pd.Timedelta(days=45)]
            prior_primary=prim_chron[prim_chron["published_ts"]<latest_primary_ts-pd.Timedelta(days=45)]
        def _types(q):
            out=set()
            for v in q.get("catalyst_types",pd.Series(dtype=object)):
                out.update(jlist(v))
            return out
        recent_types=_types(recent_primary)
        prior_types=_types(prior_primary)
        new_type_count=len(recent_types-prior_types)
        recent_stage=float(pd.to_numeric(recent_primary.get("stage_weight"),errors="coerce").max()) if len(recent_primary) else 0.0
        prior_stage=float(pd.to_numeric(prior_primary.get("stage_weight"),errors="coerce").max()) if len(prior_primary) else 0.0
        if pd.isna(recent_stage): recent_stage=0.0
        if pd.isna(prior_stage): prior_stage=0.0
        stage_upgrade=max(0.0,recent_stage-prior_stage)
        repeat_material=bool(len(prior_primary) and len(recent_primary) and recent_stage>=0.85 and bool(recent_types&{"order","commissioning","capacity","product","approval"}))
        if pd.notna(latest_primary_ts):
            age_days=max((pd.Timestamp.now(tz="UTC")-latest_primary_ts).total_seconds()/86400.0,0.0)
            recency=float(np.exp(-age_days/75.0))
        else:
            recency=0.0
        fresh_core=max(0.75 if new_type_count>0 else 0.0,min(stage_upgrade/0.30,1.0),0.65 if repeat_material else 0.0)
        fresh_generation_score=float(np.clip(recency*fresh_core,0,1))
        strongest=(prim.sort_values(["linked_evidence_score","published_ts"],ascending=[False,False]).iloc[0] if len(prim) else g.iloc[0])
        e_rows.append({
            "symbol":sym,
            "primary_event_count":int(len(prim)),
            "evidence_domain_count":int(g["domain"].nunique()),
            "order_visibility_score":float(order["stage_weight"].max()) if len(order) else 0.0,
            "commissioning_score":float(commission["stage_weight"].max()) if len(commission) else 0.0,
            "product_approval_score":float(product["stage_weight"].max()) if len(product) else 0.0,
            "capacity_stage_score":float(capacity["stage_weight"].max()) if len(capacity) else 0.0,
            "negative_event_count":int(len(neg)),
            "first_primary_event_date":str(first_primary_ts) if pd.notna(first_primary_ts) else "",
            "latest_primary_event_date":str(latest_primary_ts) if pd.notna(latest_primary_ts) else "",
            "fresh_catalyst_generation_score":fresh_generation_score,
            "new_catalyst_type_count":int(new_type_count),
            "primary_stage_upgrade":float(stage_upgrade),
            "strongest_event_title":str(strongest.get("title",""))[:500],
            "strongest_event_url":str(strongest.get("url","")),
            "strongest_event_date":str(strongest.get("published_ts","")),
            "strongest_event_stage":str(strongest.get("stage","")),
            "strongest_event_types":str(strongest.get("catalyst_types","[]")),
            "strongest_event_themes":str(strongest.get("themes","[]")),
        })
    ea=pd.DataFrame(e_rows)

    x=ci.merge(fin,on="symbol",how="left").merge(ea,on="symbol",how="left").merge(mkt,on="symbol",how="left",suffixes=("","_mkt"))
    # Support current NSE Integrated Filing XBRL financials.
    # Up to 12 quarters (~3 years) provide context, but recent quarters dominate.
    if "latest_revenue" in x.columns:
        def _sg(v,span):
            try:
                if pd.isna(v): return np.nan
                return float(np.clip(float(v)/span,-1,1))
            except Exception:
                return np.nan
        def _group_score(r,spec,turn_cols=()):
            vals=[]; ws=[]
            for col,w,span in spec:
                z=_sg(r.get(col),span)
                if pd.notna(z):
                    vals.append(w*z); ws.append(w)
            for col,w in turn_cols:
                try:z=float(np.clip(float(r.get(col) or 0),-1,1))
                except Exception:z=0.0
                vals.append(w*z); ws.append(w)
            if not ws:return np.nan
            return float(np.clip(0.5+0.5*(sum(vals)/sum(ws)),0,1))

        scores=[]; latest2=[]; previous2=[]; latest_ttm=[]; prior_ttm=[]
        for _,r in x.iterrows():
            s1=_group_score(r,[
                ("recent2_revenue_yoy",0.35,0.50),("recent2_ebitda_yoy",0.30,0.80),
                ("recent2_pat_yoy",0.15,1.00),("recent2_ebitda_margin_yoy_change",0.20,0.10),
            ],[("recent2_pat_turnaround",0.10),("recent2_ebitda_turnaround",0.08)])
            s2=_group_score(r,[
                ("prev2_revenue_yoy",0.40,0.45),("prev2_ebitda_yoy",0.30,0.75),
                ("prev2_pat_yoy",0.15,0.90),("prev2_ebitda_margin_yoy_change",0.15,0.10),
            ])
            s3=_group_score(r,[
                ("ttm_revenue_growth",0.40,0.40),("ttm_ebitda_growth",0.30,0.65),
                ("ttm_pat_growth",0.15,0.80),("ttm_ebitda_margin_change",0.15,0.10),
            ],[("ttm_pat_turnaround",0.10)])
            s4=_group_score(r,[
                ("previous_ttm_revenue_growth",0.45,0.35),("previous_ttm_ebitda_growth",0.30,0.60),
                ("previous_ttm_pat_growth",0.15,0.75),("previous_ttm_ebitda_margin_change",0.10,0.10),
            ])
            groups=[(s1,0.40),(s2,0.25),(s3,0.20),(s4,0.15)]
            num=sum(s*w for s,w in groups if pd.notna(s))
            den=sum(w for s,w in groups if pd.notna(s))
            if den:
                score=num/den
            else:
                raw=_group_score(r,[
                    ("revenue_yoy",0.35,0.50),("ebitda_yoy",0.30,0.75),
                    ("pat_yoy",0.15,1.00),("ebitda_margin_yoy_change",0.20,0.12)
                ],[("pat_turnaround_yoy",0.10),("ebitda_turnaround_yoy",0.08)])
                score=0.5 if pd.isna(raw) else raw
            scores.append(float(np.clip(score,0,1)))
            latest2.append(s1); previous2.append(s2); latest_ttm.append(s3); prior_ttm.append(s4)
        x["financial_latest2_score"]=latest2
        x["financial_previous2_score"]=previous2
        x["financial_latest_ttm_score"]=latest_ttm
        x["financial_prior_ttm_score"]=prior_ttm
        x["financial_score_raw"]=scores
        qn=pd.to_numeric(x.get("quarters_extracted"),errors="coerce").fillna(0)
        x["financial_3y_context_coverage"]=(qn/12.0).clip(0,1)
        qe=pd.to_datetime(x.get("latest_qe"),errors="coerce")
        stale=(pd.Timestamp.now().normalize()-qe).dt.days
        x["financial_stale_days"]=stale
        x["financial_freshness"]=np.exp(-stale.clip(lower=0)/240.0)
        x["financial_available"]=qe.notna()
        annual_rev=(pd.to_numeric(x["latest_ttm_revenue"],errors="coerce")
                    if "latest_ttm_revenue" in x.columns
                    else pd.Series(np.nan,index=x.index,dtype=float))
        fallback=(pd.to_numeric(x["latest_revenue"],errors="coerce")*4.0
                  if "latest_revenue" in x.columns
                  else pd.Series(np.nan,index=x.index,dtype=float))
        annual_rev=annual_rev.where(annual_rev.notna(),fallback)
        x["annualized_sales_crore"]=annual_rev/1e7

    for c in [
        "max_stage_weight","max_linked_evidence_score","max_theme_demand_signal","max_capacity_pct",
        "max_money_crore","financial_score_raw","annualized_sales_crore","order_visibility_score",
        "commissioning_score","product_approval_score","capacity_stage_score","promoter_conviction_score",
        "accumulation_score","ownership_accumulation_score","technical_confirmation_score",
        "priced_in_penalty","risk_score","p100_anchor","v10_percentile",
        "ret_60","ret_120","ret_252","fresh_catalyst_generation_score","new_catalyst_type_count",
        "primary_stage_upgrade","financial_3y_context_coverage"
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

    # Two-sleeve architecture:
    # Pre-Obvious = business change before a material rerating.
    # Second-Leg = prior 50%+ rerating is allowed only with a fresh catalyst generation.
    x["max_prior_runup_252"]=x[["ret_60","ret_120","ret_252"]].max(axis=1).fillna(0)
    x["material_prior_runup"]=x["max_prior_runup_252"]>=0.50
    x["remaining_business_upside_score"]=(
        0.30*x["catalyst_magnitude_score"]
        +0.25*x["earnings_inflection_score"]
        +0.20*x["order_probability_score"]
        +0.10*x["product_demand_linkage_score"]
        +0.10*x["execution_quality_score"]
        +0.05*x["promoter_accumulation_score"]
    ).clip(0,1)
    x["same_catalyst_priced_penalty"]=(
        x["already_priced_penalty_used"]*(1.0-0.65*x["fresh_catalyst_generation_score"].fillna(0).clip(0,1))
    ).clip(0,1)
    x["second_leg_score"]=(
        0.70*x["catalyst_intelligence_score"]
        +0.10*x["technical_confirmation_used"]
        +0.10*x["remaining_business_upside_score"]
        +0.10*x["fresh_catalyst_generation_score"].fillna(0).clip(0,1)
        -0.12*x["same_catalyst_priced_penalty"]
        -0.08*x["risk_score"].fillna(0).clip(0,1)
        -0.05*np.clip(x["negative_event_count"].fillna(0),0,2)
    )
    x["pre_obvious_score"]=x["v11_4_score_raw"]

    # Eligibility requires a real primary catalyst and at least one causal mechanism.
    causal=(
        (x["stage_reality_score"]>=0.55)
        | (x["order_probability_score"]>=0.55)
        | ((x["catalyst_magnitude_score"]>=0.45)&(x["earnings_inflection_score"]>=0.45))
        | ((x["product_approval_score"]>=0.55)&(x["product_demand_linkage_score"]>=0.35))
    )
    x["v11_4_base_eligible"]=(
        x["primary_confirmed"].fillna(False).astype(bool)
        & (x["primary_event_count"].fillna(0)>=1)
        & causal
    )
    x["pre_obvious_eligible"]=(
        x["v11_4_base_eligible"]
        & (~x["material_prior_runup"])
        & (x["already_priced_penalty_used"]<=0.70)
    )
    x["second_leg_eligible"]=(
        x["v11_4_base_eligible"]
        & x["material_prior_runup"]
        & (x["fresh_catalyst_generation_score"].fillna(0)>=0.45)
        & (x["remaining_business_upside_score"]>=0.55)
        & (x["same_catalyst_priced_penalty"]<=0.70)
    )
    x["opportunity_sleeve"]=np.select(
        [x["pre_obvious_eligible"],x["second_leg_eligible"],x["material_prior_runup"]],
        ["PRE_OBVIOUS_DISCOVERY","SECOND_LEG_REACCELERATION","OVEREXTENDED_OR_ALREADY_PRICED"],
        default="RESEARCH_OTHER"
    )
    x["v11_4_eligible"]=x["pre_obvious_eligible"]|x["second_leg_eligible"]
    x["sleeve_score"]=np.where(
        x["second_leg_eligible"],x["second_leg_score"],
        np.where(x["pre_obvious_eligible"],x["pre_obvious_score"],x["v11_4_score_raw"])
    )

    x["audit_grade"]=np.select(
        [
            x["v11_4_eligible"]&(x["sleeve_score"]>=0.68),
            x["v11_4_eligible"]&(x["sleeve_score"]>=0.58),
            x["v11_4_eligible"],
        ],
        ["A","B","C"],
        default="REJECT"
    )

    eligible=x[x["v11_4_eligible"]].sort_values("sleeve_score",ascending=False).copy()
    eligible["audit_rank"]=np.arange(1,len(eligible)+1)

    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    x.sort_values("v11_4_score_raw",ascending=False).to_csv(out/"candidate_audit_all.csv",index=False)
    eligible.to_csv(out/"candidate_audit_eligible.csv",index=False)
    eligible.head(20).to_csv(out/"candidate_audit_top20.csv",index=False)
    eligible[eligible["opportunity_sleeve"].eq("PRE_OBVIOUS_DISCOVERY")].to_csv(out/"pre_obvious_discovery.csv",index=False)
    eligible[eligible["opportunity_sleeve"].eq("SECOND_LEG_REACCELERATION")].to_csv(out/"second_leg_reacceleration.csv",index=False)
    x[x["opportunity_sleeve"].eq("OVEREXTENDED_OR_ALREADY_PRICED")].sort_values("v11_4_score_raw",ascending=False).to_csv(out/"overextended_already_priced.csv",index=False)

    summary={
        "companies_audited":int(len(x)),
        "eligible":int(len(eligible)),
        "grade_A":int((eligible["audit_grade"]=="A").sum()) if len(eligible) else 0,
        "grade_B":int((eligible["audit_grade"]=="B").sum()) if len(eligible) else 0,
        "grade_C":int((eligible["audit_grade"]=="C").sum()) if len(eligible) else 0,
        "pre_obvious_count":int(eligible["opportunity_sleeve"].eq("PRE_OBVIOUS_DISCOVERY").sum()) if len(eligible) else 0,
        "second_leg_count":int(eligible["opportunity_sleeve"].eq("SECOND_LEG_REACCELERATION").sum()) if len(eligible) else 0,
        "overextended_count":int(x["opportunity_sleeve"].eq("OVEREXTENDED_OR_ALREADY_PRICED").sum()),
        "financial_coverage_pct":float(x["financial_available"].fillna(False).mean()) if "financial_available" in x else 0.0,
        "top20":eligible.head(20)[[
            c for c in [
                "symbol","audit_rank","audit_grade","opportunity_sleeve","sleeve_score","v11_4_score_raw","second_leg_score",
                "fresh_catalyst_generation_score","remaining_business_upside_score","max_prior_runup_252",
                "catalyst_intelligence_score",
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
