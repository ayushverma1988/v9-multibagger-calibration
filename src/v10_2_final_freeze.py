from __future__ import annotations

import argparse, json
from pathlib import Path
import pandas as pd


def loadj(p):
    return json.load(open(p))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--v10-manifest",required=True)
    ap.add_argument("--v101-summary",required=True)
    ap.add_argument("--v99-stage1",required=True)
    ap.add_argument("--v99-stage2",required=True)
    ap.add_argument("--freshness",required=True)
    ap.add_argument("--fresh-current",required=True)
    ap.add_argument("--output",required=True)
    args=ap.parse_args()

    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)

    v10=loadj(args.v10_manifest)
    v101=loadj(args.v101_summary)
    s1=loadj(args.v99_stage1)
    s2=loadj(args.v99_stage2)
    fresh=loadj(args.freshness)
    cur=pd.read_csv(args.fresh_current)

    tests=[]

    def add(name,passed,actual=None,requirement=None):
        tests.append({
            "test":name,"passed":bool(passed),
            "actual":actual,"requirement":requirement,
        })

    add(
        "v10_candidate_ready",
        bool(v10.get("test_run_ready",False)),
        v10.get("test_run_ready"),
        True,
    )
    add(
        "v10_selected_count",
        int(v10.get("selected_count",0))==10,
        int(v10.get("selected_count",0)),
        10,
    )

    # Adaptive fundamentals must either pass and be explicitly integrated, or
    # remain at zero. Current V10.2 supports the zero-weight fallback only.
    add(
        "v10_1_not_silently_promoted",
        not bool(v101.get("production_gate",False)),
        v101.get("production_gate"),
        False,
    )

    # Stage-1 uncertainty sanity: lower 90% bootstrap precision bound must remain
    # above zero and lift lower bound above 1x.
    p05_prec=s1.get("bootstrap_90pct_interval",{}).get("mean_precision_2x",{}).get("p05")
    p05_lift=s1.get("bootstrap_90pct_interval",{}).get("mean_capped_lift_2x",{}).get("p05")
    add("bootstrap_precision_positive",p05_prec is not None and float(p05_prec)>0,float(p05_prec) if p05_prec is not None else None,">0")
    add("bootstrap_lift_above_one",p05_lift is not None and float(p05_lift)>1,float(p05_lift) if p05_lift is not None else None,">1.0")

    stab=s2.get("selection_stability",{})
    mean_j=stab.get("mean_jaccard")
    worst_j=stab.get("worst_p05_jaccard")
    top1=stab.get("mean_top1_stability")
    add("mean_jaccard_stability",mean_j is not None and float(mean_j)>=0.80,mean_j,">=0.80")
    add("worst_fold_p05_jaccard",worst_j is not None and float(worst_j)>=0.60,worst_j,">=0.60")
    add("top1_stability",top1 is not None and float(top1)>=0.70,top1,">=0.70")

    # Pre-specified practical paper-test capacity gate at INR 10 lakh.
    cap_rows=s2.get("capacity",[])
    cap10=next((x for x in cap_rows if abs(float(x.get("portfolio_capital_inr",0))-1e6)<1),None)
    p90_part=float(cap10["p90_participation"]) if cap10 else None
    p90_slip=float(cap10["p90_slippage_bps_assumption"]) if cap10 else None
    add("10l_p90_participation",p90_part is not None and p90_part<=0.02,p90_part,"<=2% ADV")
    add("10l_p90_slippage_stress",p90_slip is not None and p90_slip<=75,p90_slip,"<=75 bps assumption")

    data_end=pd.to_datetime(fresh.get("data_end"),errors="coerce")
    today=pd.Timestamp.today().normalize()
    age=int((today-data_end.normalize()).days) if pd.notna(data_end) else None
    add("fresh_market_data",age is not None and age<=5,age,"<=5 calendar days old")
    add("fresh_selected_count",int(fresh.get("selected_rows") or 0)==10,int(fresh.get("selected_rows") or 0),10)

    # Basic current-output integrity.
    sel_col="selected_v941" if "selected_v941" in cur.columns else None
    if sel_col:
        top=cur[cur[sel_col].fillna(False).astype(bool)].copy()
    else:
        top=cur.head(10).copy()
    add("unique_selected_symbols",top["symbol"].nunique()==len(top),int(top["symbol"].nunique()),len(top))
    add("positive_selected_prices",(pd.to_numeric(top["close"],errors="coerce")>0).all(),True,True)

    all_pass=all(x["passed"] for x in tests)
    mode="frozen_paper_test_candidate" if all_pass else "paper_test_only_not_finally_frozen"

    # Preserve fresh ranking from the validated fallback; no rejected challenger
    # is reintroduced at freeze time.
    rank_col=None
    for c in ["selection_rank_v941","selection_rank_v931","selection_rank"]:
        if c in top.columns:
            rank_col=c;break
    if rank_col:
        top=top.sort_values(rank_col)
    else:
        score_col="selection_score_v941" if "selection_score_v941" in top.columns else "p_cal"
        top=top.sort_values(score_col,ascending=False)
    top.head(10).to_csv(out/"current_top10_v10_2.csv",index=False)

    manifest={
        "model":"V10.2 final freeze audit",
        "status":mode,
        "freeze_ready":all_pass,
        "tests":tests,
        "freshness":fresh,
        "v10_component_policy":v10.get("component_status",{}),
        "v10_1":v101,
        "paper_test_allowed":True,
        "note":"A failed freeze gate does not alter the validated fallback; it blocks only final freeze status.",
    }
    json.dump(manifest,open(out/"freeze_manifest.json","w"),indent=2,default=str)
    print(json.dumps(manifest,indent=2,default=str))

if __name__=="__main__":
    main()
