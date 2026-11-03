"""Separate early/second-leg views of one verified V11.4 source snapshot.

Replay only: no new model fitting, outcome inspection, parameter search or
changes to the global observation. Unknown catalysts/fundamentals cannot
become a qualified multibagger claim. The model's global Top 10 may all be
extended; report early and second-leg candidates independently instead.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import joblib,numpy as np,pandas as pd
import v11_4_standalone_train_walkforward as core
from v11_4_forward_research_release import (
    validate_prospective_candidates,apply_frozen_calibration,score_prospective)
from v11_4_systematic_run import screen_audit,encrypt_owner_demo,CURRENT_VERSION,ADDITIONAL_CHECKS
from v11_4_tradability_risk_overlay import audit_tradability

KNOWN_SLEEVES=("PRE_OBVIOUS_LT12PCT","SECOND_LEG_12_TO_100PCT","EXTENDED_GT100PCT")

def rank_sleeve_views(scored):
    required={"date","symbol","p6_double_calibrated","discovery_sleeve"}
    if not required.issubset(scored):raise ValueError("Incomplete scored source-view identity")
    if scored[["date","symbol"]].duplicated().any():raise ValueError("Duplicate scored security")
    p=pd.to_numeric(scored["p6_double_calibrated"],errors="coerce")
    if p.isna().any() or not p.between(0,1,inclusive="neither").all():
        raise ValueError("Invalid provisional probabilities")
    if not scored["discovery_sleeve"].isin((*KNOWN_SLEEVES,"UNKNOWN")).all():
        raise ValueError("Unregistered prior-runup sleeve")
    frames=[]
    for sleeve in KNOWN_SLEEVES:
        q=scored.loc[scored["discovery_sleeve"].eq(sleeve)].sort_values(
            ["p6_double_calibrated","symbol"],ascending=[False,True]).head(10).copy()
        q.insert(0,"rank_in_sleeve",range(1,len(q)+1));frames.append(q)
    return pd.concat(frames,ignore_index=True)

def original_global_observation_agrees(features,package,asof,metadata,recorded):
    current=score_prospective(features,package,asof,metadata).sort_values("rank")
    old=recorded.sort_values("rank")
    if (len(old)!=10 or old["symbol"].duplicated().any() or
        current[["rank","symbol"]].to_dict("records")!=old[["rank","symbol"]].to_dict("records") or
        not np.allclose(current["p6_double_calibrated"],old["p6_double_calibrated"],rtol=0,atol=1e-12) or
        not np.allclose(current["close"],old["close"],rtol=0,atol=1e-9)):
        raise ValueError("Replay changed the original corrected global Top 10")
    return current

def replay(features_path,metadata_path,package_path,recorded_path,config_path,out,public_key=None):
    root=Path(out);root.mkdir(parents=True,exist_ok=True)
    metadata=json.loads(Path(metadata_path).read_text())
    digest=hashlib.sha256(Path(features_path).read_bytes()).hexdigest()
    if metadata.get("live_features_SHA256")!=digest:
        raise ValueError("Original verified snapshot hash mismatch")
    features=pd.read_parquet(features_path)
    asof=metadata["snapshot_date"]
    validate_prospective_candidates(features,asof,metadata)
    package=joblib.load(package_path)
    if (package.get("model_version")!=CURRENT_VERSION or package.get("asof")!=asof or
        package.get("features")!=list(core.MODEL_FEATURES)):
        raise ValueError("Incorrect original model package or snapshot date")
    recorded=pd.read_csv(recorded_path)
    original=original_global_observation_agrees(features,package,asof,metadata,recorded)
    config=json.loads(Path(config_path).read_text())
    table,coverage=screen_audit(features,config,root/"source_screen_audit")
    eligible=features.loc[core.eligible_asof(features)].copy()
    transformed=core.safe_featureize(eligible)
    raw=package["model"].predict_proba(transformed[list(core.MODEL_FEATURES)])[:,1]
    scored=eligible[["date","symbol","close"]].copy()
    scored["p6_double_calibrated"]=apply_frozen_calibration(raw,package["calibration"])
    scored=scored.merge(table,on=["date","symbol"],how="left",validate="1:1")
    views=rank_sleeve_views(scored)
    risk_summaries={};risk_frames=[]
    four=table[["symbol",*[k+"_status" for k in config["conditions"]]]]
    for sleeve,q in views.groupby("discovery_sleeve",sort=False):
        if len(q)!=10:
            risk_summaries[sleeve]={"status":"FEWER_THAN_TEN_CANDIDATES","stocks_cleared_for_execution":0}
            continue
        picks=q[["date","symbol","close","p6_double_calibrated","rank_in_sleeve"]].rename(
            columns={"rank_in_sleeve":"rank"})
        audit,summary=audit_tradability(features,picks,four)
        risk_summaries[sleeve]=summary
        risk_frames.append(audit[["symbol","tradability_assessment","tradability_issues","risk_flags"]])
    if risk_frames:
        views=views.merge(pd.concat(risk_frames,ignore_index=True),on="symbol",how="left",validate="1:1")
    scored.to_parquet(root/"all_sourced_sleeve_scores_PRIVATE.parquet",index=False)
    views.to_csv(root/"three_separate_sleeve_top10_views_PRIVATE.csv",index=False)
    global_overlay=original.merge(table[["date","symbol","discovery_sleeve"]],
        on=["date","symbol"],how="left",validate="1:1")
    report={"scope":"ORIGINAL_VERIFIED_SNAPSHOT_REPLAY_NOT_NEW_LIVE_OR_UNSEEN_VALIDATION",
        "source_snapshot_date_IST":asof,"model_id":CURRENT_VERSION,
        "model_refitted":False,"hyperparameters_or_probability_changed":False,
        "original_global_top10_verified_unchanged":True,
        "original_global_top10_sleeves":global_overlay["discovery_sleeve"].value_counts().to_dict(),
        "eligible_scored_companies":len(scored),
        "eligible_sleeve_counts":scored["discovery_sleeve"].value_counts().to_dict(),
        "selected_view_counts":views["discovery_sleeve"].value_counts().to_dict(),
        "unknown_lookbacks_never_assigned_early":True,
        "full_screens_and_primary_causal_chains_verified":False,
        "qualified_full_user_model_candidates":0,
        "liquidity_audits_by_sleeve":risk_summaries,
        "current_source_screen_coverage":coverage,"production_approved":False,
        "source_SHA256":{"verified_features":digest,
            "model_package":hashlib.sha256(Path(package_path).read_bytes()).hexdigest(),
            "original_global_observation":hashlib.sha256(Path(recorded_path).read_bytes()).hexdigest()}}
    (root/"sleeve_candidate_summary.json").write_text(json.dumps(report,indent=2))
    rows=[]
    for row in views.to_dict("records"):
        rows.append({"discovery_sleeve":row["discovery_sleeve"],
            "rank_in_sleeve":int(row["rank_in_sleeve"]),"symbol":row["symbol"],
            "close_INR":round(float(row["close"]),2),
            "research_6m_2x_estimate_pct":round(float(row["p6_double_calibrated"])*100,3),
            "prior_runup_max_pct":round(float(row["prior_runup_max_60_120_252_sessions"])*100,3),
            "RSI14":float(row["rsi14_wilder"]) if pd.notna(row["rsi14_wilder"]) else None,
            "family_status":{k:row[k+"_status"] for k in config["conditions"]},
            "additional_check_status":{k:row[k+"_status"] for k in ADDITIONAL_CHECKS},
            "tradability_assessment":row.get("tradability_assessment") if pd.notna(row.get("tradability_assessment")) else "UNKNOWN",
            "tradability_issues":row.get("tradability_issues") if pd.notna(row.get("tradability_issues")) else "UNKNOWN",
            "primary_causal_chain_status":"UNKNOWN"})
    if public_key:encrypt_owner_demo(rows,report,public_key,root/"owner_demo_encrypted.json")
    print(json.dumps(report,indent=2),flush=True)
    return report

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--features",required=True);p.add_argument("--metadata",required=True)
    p.add_argument("--model",required=True);p.add_argument("--recorded",required=True)
    p.add_argument("--config",default="config/v11_4_four_screener_families.json")
    p.add_argument("--output",required=True);p.add_argument("--owner-public-key",default=None)
    a=p.parse_args();replay(a.features,a.metadata,a.model,a.recorded,a.config,a.output,a.owner_public_key)
if __name__=="__main__":main()
