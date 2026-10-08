"""First STRICT read-only 18-fold V11.4 fixed catalyst sleeve vs exact frozen V10 picks.

Experiment is prespecified by V11.4 historical config: "pre-obvious" prior
120-day runup <12%, "second-leg" 12–100%, "extended" >100% excluded.
Requires exchange-document metadata for an order win OR capacity expansion
published in preceding 90 days. Ranks ONLY by the original frozen V10 OOS
p_cal probabilities; no outcome-dependent scoring or weights. Candidate
never forced to 10; exact V10 selection count matched within EACH fold.

This is a rule-overlay feasibility experiment, NOT full V11.4 promotion:
catalyst rules were studied descriptively previously, so estimates are
research/exploratory, not independent untouched final holdout proof.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd
import numpy as np

CUTOFF=pd.Timestamp("2026-10-08")
SLEEVE_1_MAX=.12
SLEEVE_2_MAX=1.0
K=10

def sleeve_mask(x):
    r=pd.to_numeric(x["ret_120"],errors="coerce")
    trigger=x["nse_capacity_expansion_90d"].gt(0)|x["nse_order_win_90d"].gt(0)
    p1=trigger&(r<SLEEVE_1_MAX)
    p2=trigger&(r>=SLEEVE_1_MAX)&(r<=SLEEVE_2_MAX)
    return p1,p2

def evaluate_slice(group):
    q=group.copy()
    y6=pd.to_numeric(q["y6"],errors="coerce")
    mature=pd.to_datetime(q["y6_mature_date"],errors="coerce")
    good=q["integrity_y6_clean"].eq(True)&mature.notna()&(mature<=CUTOFF)&y6.isin([0,1])
    valid=q[good]
    return {
        "count":len(q),
        "clean_matured":len(valid),
        "y6_doubles":int(valid["y6"].sum()),
        "y6_precision":float(valid["y6"].mean()) if len(valid) else None,
        "dd30_6m_rate":float(pd.to_numeric(valid["dd30_6m"],errors="coerce").mean()) if len(valid) else None,
        "symbols":"|".join(q["symbol"].tolist()),
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--oos",required=True)
    p.add_argument("--features",required=True)
    p.add_argument("--frozen-snapshot",required=True)
    p.add_argument("--config",required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads(Path(a.config).read_text())
    sleeves=cfg["sleeves"]
    if (sleeves["pre_obvious_discovery"]["max_prior_runup_exclusive"]!=SLEEVE_1_MAX
      or sleeves["second_leg_reacceleration"]["min_prior_runup_inclusive"]!=SLEEVE_1_MAX
      or sleeves["second_leg_reacceleration"]["max_prior_runup_inclusive"]!=SLEEVE_2_MAX
      or cfg["evaluation"]["selection"]!="do_not_force_k"):
        raise SystemExit("Frozen V11.4 sleeve/rank evaluation conditions unexpectedly changed")
    cols=[
       "date","symbol","p_cal","model_dispersion",
       "selected_v941","selection_rank_v941",
       "y6","dd30_6m"]
    oos=pd.read_parquet(a.oos,columns=cols)
    oos["date"]=pd.to_datetime(oos["date"],errors="coerce").dt.normalize()
    oos["symbol"]=oos["symbol"].astype(str).str.upper().str.strip()
    if oos.duplicated(["date","symbol"]).any():raise SystemExit("Original V10 duplicate stock date")
    f=pd.read_parquet(a.features,columns=[
        "date","symbol","nse_capacity_expansion_90d","nse_order_win_90d",
        "historical_asof_utc","integrity_feature_clean","ret_120"])
    f["date"]=pd.to_datetime(f["date"],errors="coerce").dt.normalize()
    f["symbol"]=f["symbol"].astype(str).str.upper().str.strip()
    if f.duplicated(["date","symbol"]).any():raise SystemExit("Original NSE event feature duplicate")
    original=pd.read_parquet(a.frozen_snapshot,columns=[
        "date","symbol","y6_mature_date","integrity_y6_clean"])
    original["date"]=pd.to_datetime(original["date"],errors="coerce").dt.normalize()
    original["symbol"]=original["symbol"].astype(str).str.upper().str.strip()
    if original.duplicated(["date","symbol"]).any():
        raise SystemExit("Frozen outcome maturity has duplicated company-date keys")
    z=oos.merge(f,on=["date","symbol"],how="inner",validate="1:1")
    z=z.merge(original,on=["date","symbol"],how="left",validate="1:1")
    if z["y6_mature_date"].isna().any():
        raise SystemExit("Historical V10 label maturity provenance missing")
    if len(z)!=len(f) or len(z)!=len(oos[oos["date"].isin(f["date"].unique())]):
        raise SystemExit("V10 immutable historical selection pool != 18fold catalyst source")
    times=pd.to_datetime(z["historical_asof_utc"],utc=True,format="mixed",errors="coerce")
    expected=(z["date"].dt.tz_localize("Asia/Kolkata")
              +pd.Timedelta(hours=15,minutes=30)).dt.tz_convert("UTC")
    if times.isna().any() or (times!=expected).any():
        raise SystemExit("Exchange catalyst availability after cutoff")
    folds=sorted(f["date"].unique())
    if len(folds)!=18:raise SystemExit("Expected exact 18 V10 folds")
    rows=[];picks=[]
    for td,part in z.groupby("date"):
        baseline=part[part["selected_v941"].eq(True)].sort_values(
            ["selection_rank_v941","symbol"]).copy()
        if len(baseline)!=K or baseline["selection_rank_v941"].isna().any():
            raise SystemExit(f"Frozen V10 must have exact 10 picks on {td}")
        # NO use of y6, dd30_6m, or y6 integrity flags in candidate eligibility or ranking.
        eligible=part[part["integrity_feature_clean"].eq(True)&
                      part["p_cal"].notna()&part["ret_120"].notna()].copy()
        mask1,mask2=sleeve_mask(eligible)
        bucket1=eligible[mask1].sort_values(
            ["p_cal","model_dispersion","symbol"],ascending=[False,True,True])
        bucket2=eligible[mask2].sort_values(
            ["p_cal","model_dispersion","symbol"],ascending=[False,True,True])
        candidate=pd.concat([bucket1,bucket2],ignore_index=False).head(K)
        selected_control=baseline.head(len(candidate))
        c=evaluate_slice(candidate)
        b=evaluate_slice(selected_control)
        date=str(pd.Timestamp(td).date())
        fold={ "date":date,"universe":len(part),"baseline_full_V10_k":K,
            "pre_obvious_eligible":len(bucket1),"second_leg_eligible":len(bucket2),
            "candidate_k":c["count"],"baseline_matched_k":b["count"],
            "candidate_matured_k":c["clean_matured"],
            "baseline_matched_matured_k":b["clean_matured"],
            "candidate_y6_precision":c["y6_precision"],
            "V10_count_matched_y6_precision":b["y6_precision"],
            "candidate_y6_doubles":c["y6_doubles"],
            "V10_count_matched_y6_doubles":b["y6_doubles"],
            "candidate_dd30_6m_rate":c["dd30_6m_rate"],
            "V10_count_matched_dd30_6m_rate":b["dd30_6m_rate"],
            "selected_overlap_with_real_V10":len(set(candidate["symbol"])&set(baseline["symbol"])),
            "selected_candidate_symbols":c["symbols"],"V10_matched_symbols":b["symbols"],
        }
        rows.append(fold)
        for j,(sid,subset) in enumerate([("V11_4_FIXED_CATALYST_RULE_RESEARCH_ONLY",candidate),
                                          ("FROZEN_V10_COUNT_MATCHED",selected_control)]):
            for rank,r in enumerate(subset.itertuples(index=False),start=1):
                picks.append({"date":date,"strategy":sid,"rank":rank,
                              "symbol":r.symbol,"p_cal":r.p_cal,
                              "y6":r.y6,"dd30_6m":r.dd30_6m,
                              "source":"exact_original_V10_OOS_plus_original_NSE_PIT_90d"})
    report=pd.DataFrame(rows)
    report.to_csv(out/"fixed_overlay_vs_exact_V10_matched_by_fold.csv",index=False)
    pd.DataFrame(picks).to_csv(out/"frozen_v10_and_fixed_overlay_picks_research.csv",index=False)
    comparable=report[(report["candidate_k"]>0)&
        (report["candidate_matured_k"]==report["candidate_k"])&
        (report["baseline_matched_matured_k"]==report["baseline_matched_k"])]
    if len(comparable):
        total_c=float(comparable["candidate_y6_doubles"].sum())
        total_v=float(comparable["V10_count_matched_y6_doubles"].sum())
        total_k=int(comparable["candidate_k"].sum())
    else:total_c=total_v=0.0;total_k=0
    summary={
        "scope":"EXPLORATORY_FROZEN_RULE_OVERLAY_NOT_FULL_V11_4_BACKTEST",
        "V10_control":"exact unmodified selected_v941 V10.2 OOS historical picks",
        "lookbacks":"prespecified NSE 90d order win or capacity event metadata",
        "frozen_sleeve_thresholds":{"pre_obvious_runup_max_exclusive":.12,
                                    "second_leg_runup_min_inclusive":.12,
                                    "second_leg_runup_max_inclusive":1.0},
        "candidate_model_score":"original V10 historical OOS p_cal only",
        "event_bonus_weight_learned":False,
        "frozen_original_18_folds_seen":len(folds),
        "folds_with_at_least_one_candidate":int((report["candidate_k"]>0).sum()),
        "full_label_valid_comparable_folds":len(comparable),
        "total_comparable_candidate_picks":total_k,
        "candidate_matured_successes":int(total_c),
        "matched_real_V10_matured_successes":int(total_v),
        "candidate_6mo_precision":total_c/total_k if total_k else None,
        "matched_real_V10_6mo_precision":total_v/total_k if total_k else None,
        "candidate_over_V10_precision_ratio":total_c/total_v if total_v else None,
        "full_v11_4_candidate_model_trained":False,
        "analysis_not_independent_confirmatory_holdout":True,
        "V10_production_or_weights_changed":False,
        "frozen_acceptance_evaluated_or_promoted":False,
    }
    (out/"fixed_overlay_vs_exact_v10_research_summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)
    print(report[["date","candidate_k","pre_obvious_eligible","second_leg_eligible","candidate_y6_precision","V10_count_matched_y6_precision"]].to_string(index=False),flush=True)
    if len(folds)!=18:raise SystemExit("Frozen 18-fold history missing")
    if len(comparable)<12:raise SystemExit("Under 12 folds with clean mature matched outcome; reject even exploratory acceptance")
if __name__=="__main__":main()
