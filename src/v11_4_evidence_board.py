from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def clip01(v):
    try:
        if pd.isna(v): return 0.0
        return float(np.clip(float(v),0,1))
    except Exception:
        return 0.0


def jlist(v):
    try:
        x=json.loads(v) if isinstance(v,str) else v
        return x if isinstance(x,list) else []
    except Exception:
        return []


def squash_pos(v, scale):
    try:
        if pd.isna(v) or float(v)<=0:return 0.0
        return float(1.0-math.exp(-float(v)/scale))
    except Exception:
        return 0.0


def signed_growth_score(v, center=0.0, span=0.50):
    try:
        if pd.isna(v):return 0.0
        return float(np.clip((float(v)-center)/span, -1, 1))
    except Exception:
        return 0.0


def financial_inflection(row):
    # Current numeric fundamentals are intentionally growth/margin based,
    # avoiding raw-unit comparisons until XBRL scale is independently audited.
    ry=signed_growth_score(row.get("revenue_yoy"),0,0.50)
    ey=signed_growth_score(row.get("ebitda_yoy"),0,0.75)
    py=signed_growth_score(row.get("pat_yoy"),0,1.00)
    pat_turn=clip01(max(float(row.get("pat_turnaround_yoy") or 0), float(row.get("pat_turnaround_qoq") or 0)))
    ebitda_turn=clip01(max(float(row.get("ebitda_turnaround_yoy") or 0), float(row.get("ebitda_turnaround_qoq") or 0)))
    mq=signed_growth_score(row.get("ebitda_margin_qoq_change"),0,0.08)
    my=signed_growth_score(row.get("ebitda_margin_yoy_change"),0,0.12)
    rq=signed_growth_score(row.get("revenue_qoq"),0,0.25)
    raw=0.22*ry+0.22*ey+0.12*py+0.13*mq+0.08*my+0.08*rq+0.08*pat_turn+0.07*ebitda_turn
    return float(np.clip(0.5+0.5*raw,0,1))


def evidence_company_features(e):
    rows=[]
    for sym,g in e.groupby("symbol"):
        g=g.copy()
        g["stage_weight_num"]=pd.to_numeric(g["stage_weight"],errors="coerce").fillna(0)
        g["linked_num"]=pd.to_numeric(g.get("linked_evidence_score"),errors="coerce").fillna(0)
        g["tier_num"]=pd.to_numeric(g["source_tier"],errors="coerce").fillna(3)
        demand_col="theme_demand_signal" if "theme_demand_signal" in g.columns else "theme_demand_max"
        g["demand_num"]=pd.to_numeric(g.get(demand_col),errors="coerce").fillna(0)
        g["money_num"]=pd.to_numeric(g.get("money_crore_max"),errors="coerce")
        g["cap_num"]=pd.to_numeric(g.get("capacity_pct_max"),errors="coerce")
        primary=g[g["tier_num"].eq(1)]
        high=g.sort_values(["linked_num","stage_weight_num","published_ts"],ascending=False)

        types=set()
        themes=set()
        for v in g["catalyst_types"]:
            types.update(jlist(v))
        for v in g["themes"]:
            themes.update(jlist(v))

        order_rows=g[g["catalyst_types"].map(lambda x:"order" in jlist(x))]
        capacity_rows=g[g["catalyst_types"].map(lambda x:any(t in {"capacity","commissioning"} for t in jlist(x)))]
        product_rows=g[g["catalyst_types"].map(lambda x:any(t in {"product","approval"} for t in jlist(x)))]

        def pcount(q):
            return int(pd.to_numeric(q["source_tier"],errors="coerce").eq(1).sum()) if len(q) else 0

        max_stage=float(g["stage_weight_num"].max()) if len(g) else 0
        max_primary_stage=float(primary["stage_weight_num"].max()) if len(primary) else 0
        max_conf=float(pd.to_numeric(g["evidence_confidence"],errors="coerce").fillna(0).max()) if len(g) else 0
        demand=float(g["demand_num"].max()) if len(g) else 0
        domains=int(g["domain"].nunique()) if "domain" in g else 0

        # Order probability reflects actual process stage and primary corroboration.
        order_stage=float(order_rows["stage_weight_num"].max()) if len(order_rows) else 0
        order_primary=pcount(order_rows)
        order_domains=int(order_rows["domain"].nunique()) if len(order_rows) else 0
        order_probability=clip01(
            0.55*min(order_stage,1.0)
            +0.25*min(order_primary,1)
            +0.10*min(order_domains/2.0,1)
            +0.10*demand
        ) if len(order_rows) else 0.0

        stage_reality=clip01(
            0.55*min(max_primary_stage,1.0)
            +0.20*min(max_stage,1.0)
            +0.15*max_conf
            +0.10*min(domains/3.0,1)
        )

        # Raw money amount is not compared with financial revenue here because
        # XBRL scaling needs an explicit audit first. Capacity percentage is unit-free.
        cap_pct=float(g["cap_num"].max()) if g["cap_num"].notna().any() else np.nan
        magnitude=clip01(
            0.75*squash_pos(cap_pct,50.0)
            +0.25*squash_pos(g["money_num"].max() if g["money_num"].notna().any() else 0,1000.0)
        )

        product_demand=clip01(
            0.45*demand
            +0.25*min(pcount(product_rows),1)
            +0.15*min(len(product_rows)/2.0,1)
            +0.15*(1.0 if themes else 0.0)
        ) if len(product_rows) or themes else 0.0

        funding_execution=clip01(
            0.35*min(len(primary)/3.0,1)
            +0.25*min(domains/3.0,1)
            +0.20*max_conf
            +0.20*(1.0-float(g.get("negative_flag",pd.Series(False,index=g.index)).fillna(False).astype(bool).mean()))
        )

        best=high.iloc[0] if len(high) else None
        rows.append({
            "symbol":sym,
            "evidence_rows":int(len(g)),
            "primary_rows":int(len(primary)),
            "distinct_domains":domains,
            "primary_confirmed":bool(len(primary)),
            "catalyst_types":"|".join(sorted(types)),
            "themes":"|".join(sorted(themes)),
            "max_stage":best["stage"] if best is not None else None,
            "max_stage_weight":max_stage,
            "max_primary_stage_weight":max_primary_stage,
            "max_evidence_confidence":max_conf,
            "theme_demand_score":demand,
            "capacity_pct":cap_pct,
            "money_crore_max":float(g["money_num"].max()) if g["money_num"].notna().any() else np.nan,
            "order_probability_score":order_probability,
            "catalyst_stage_score":stage_reality,
            "catalyst_magnitude_score":magnitude,
            "product_demand_linkage_score":product_demand,
            "funding_execution_quality_score":funding_execution,
            "order_primary_rows":order_primary,
            "capacity_primary_rows":pcount(capacity_rows),
            "product_primary_rows":pcount(product_rows),
            "best_title":best.get("title") if best is not None else None,
            "best_url":best.get("url") if best is not None else None,
            "latest_evidence":str(pd.to_datetime(g["published_ts"],utc=True,errors="coerce").max()),
        })
    return pd.DataFrame(rows)


def rank_pct(s):
    x=pd.to_numeric(s,errors="coerce")
    if x.notna().sum()<2:return pd.Series(0.5,index=s.index)
    return x.rank(pct=True,method="average").fillna(0.5)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--evidence",required=True)
    ap.add_argument("--financials",required=True)
    ap.add_argument("--market-transform",required=True)
    ap.add_argument("--absorption",required=False)
    ap.add_argument("--config",required=True)
    ap.add_argument("--output",required=True)
    args=ap.parse_args()

    cfg=json.load(open(args.config))
    ev=pd.read_parquet(args.evidence)
    fin=pd.read_csv(args.financials)
    mt=pd.read_csv(args.market_transform)

    board=evidence_company_features(ev)
    fin["financial_inflection_score"]=[financial_inflection(r) for _,r in fin.iterrows()]
    board=board.merge(fin,on="symbol",how="left",suffixes=("","_fin"))

    # Normalize disclosed catalyst amount to current business scale. XBRL facts
    # are parsed in INR; evidence amounts are in crore.
    annual_rev=pd.to_numeric(board.get("latest_revenue"),errors="coerce")*4.0
    money_inr=pd.to_numeric(board.get("money_crore_max"),errors="coerce")*1e7
    board["catalyst_money_to_annualized_revenue"]=money_inr/annual_rev.replace(0,np.nan)
    relative_money=board["catalyst_money_to_annualized_revenue"].map(
        lambda v: squash_pos(v,0.50) if pd.notna(v) else 0.0
    )
    cap_score=pd.to_numeric(board.get("capacity_pct"),errors="coerce").map(
        lambda v: squash_pos(v,50.0) if pd.notna(v) else 0.0
    )
    # Prefer relative magnitude once financial scale is known; retain the
    # earlier conservative proxy only where neither quantitative measure exists.
    quantified=(0.58*cap_score+0.42*relative_money).clip(0,1)
    has_quant=cap_score.gt(0)|relative_money.gt(0)
    board.loc[has_quant,"catalyst_magnitude_score"]=quantified[has_quant]

    keep=[
        "symbol","p100_cal","p100_anchor","promoter_conviction_score","accumulation_score",
        "ownership_accumulation_score","technical_confirmation_score","priced_in_penalty",
        "early_stage_score","p_dd30_cal","ret_120","ret_252",
    ]
    keep=[c for c in keep if c in mt.columns]
    board=board.merge(mt[keep].drop_duplicates("symbol"),on="symbol",how="left")
    if args.absorption:
        ab=pd.read_csv(args.absorption)
        board=board.merge(ab,on="symbol",how="left")

    board["promoter_accumulation_score"]=(
        0.45*pd.to_numeric(board.get("promoter_conviction_score"),errors="coerce").fillna(0)
        +0.55*pd.to_numeric(board.get("accumulation_score"),errors="coerce").fillna(0)
    ).clip(0,1)

    fin_score=pd.to_numeric(board["financial_inflection_score"],errors="coerce").fillna(0.35)
    layers=cfg["catalyst_families"]
    board["catalyst_intelligence_score"]=(
        float(layers["catalyst_stage_and_reality"])*board["catalyst_stage_score"]
        +float(layers["catalyst_magnitude"])*board["catalyst_magnitude_score"]
        +float(layers["product_demand_linkage"])*board["product_demand_linkage_score"]
        +float(layers["earnings_inflection"])*fin_score
        +float(layers["order_probability_and_visibility"])*board["order_probability_score"]
        +float(layers["promoter_and_accumulation"])*board["promoter_accumulation_score"]
        +float(layers["funding_and_execution_quality"])*board["funding_execution_quality_score"]
    )

    p10=pd.to_numeric(board.get("p100_anchor",board.get("p100_cal")),errors="coerce")
    board["v10_confirmation_percentile"]=rank_pct(p10)
    generic_priced=pd.to_numeric(board.get("priced_in_penalty"),errors="coerce").fillna(0.5).clip(0,1)
    if "catalyst_priced_in_penalty" in board:
        catalyst_priced=pd.to_numeric(board["catalyst_priced_in_penalty"],errors="coerce")
        priced=catalyst_priced.where(catalyst_priced.notna(),generic_priced).clip(0,1)
    else:
        priced=generic_priced
    board["effective_priced_in_penalty"]=priced
    layer_cfg=cfg["final_layers"]
    board["short_term_catalyst_score"]=(
        float(layer_cfg["catalyst_intelligence"])*board["catalyst_intelligence_score"]
        +float(layer_cfg["technical_confirmation_v10_2"])*board["v10_confirmation_percentile"]
        -0.15*priced
    )

    types=board["catalyst_types"].fillna("")
    meaningful=types.str.len()>0
    # This is an evidence-quality gate, not a forced stock count.
    board["evidence_qualified"]=(
        board["primary_confirmed"].fillna(False)
        & meaningful
        & (board["max_primary_stage_weight"]>=0.40)
        & (board["catalyst_intelligence_score"]>=0.35)
        & (priced<=0.70)
    )

    board["why_qualified"]=board.apply(
        lambda r: (
            f"types={r['catalyst_types']}; stage={r['max_stage']}; primary={int(r['primary_rows'])}; "
            f"themes={r['themes'] or '-'}; orderP={r['order_probability_score']:.2f}; "
            f"financial={r.get('financial_inflection_score',np.nan):.2f}; "
            f"promoterAccum={r.get('promoter_accumulation_score',0):.2f}; "
            f"catalystRet={r.get('return_since_catalyst',np.nan):.2f}; "
            f"pricedIn={r.get('effective_priced_in_penalty',np.nan):.2f}"
        ),axis=1
    )

    board=board.sort_values(
        ["evidence_qualified","short_term_catalyst_score","catalyst_intelligence_score"],
        ascending=[False,False,False]
    ).reset_index(drop=True)
    board["research_rank"]=np.arange(1,len(board)+1)

    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    board.to_csv(out,index=False)
    board[board["evidence_qualified"]].head(25).to_csv(out.parent/"primary_confirmed_shortlist.csv",index=False)

    summary={
        "stage":"V11.4 explainable evidence board",
        "status":"research_board_not_production_prediction",
        "companies":int(len(board)),
        "primary_confirmed":int(board["primary_confirmed"].sum()),
        "evidence_qualified":int(board["evidence_qualified"].sum()),
        "financial_coverage":float(board["quarters_extracted"].notna().mean()) if "quarters_extracted" in board else 0,
        "top_qualified":board[board["evidence_qualified"]].head(15)[
            ["symbol","short_term_catalyst_score","catalyst_intelligence_score","catalyst_types",
             "themes","max_stage","primary_rows","financial_inflection_score",
             "promoter_accumulation_score","return_since_catalyst","effective_priced_in_penalty","best_title"]
        ].to_dict("records"),
        "important_note":"No V11.4 Top-10 is promoted from this board. Primary-source linkage and financial-unit audit must pass before production ranking."
    }
    json.dump(summary,open(out.parent/"evidence_board_summary.json","w"),indent=2,default=str)
    print(json.dumps(summary,indent=2,default=str))

if __name__=="__main__":
    main()
