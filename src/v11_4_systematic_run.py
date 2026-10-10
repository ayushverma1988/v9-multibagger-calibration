"""One reproducible, owner-private V11.4 audit, model run and live demo.

Never treat successful Python execution as complete financial/news coverage,
an unseen holdout, or demonstrated prediction quality. All missing requested
inputs are explicitly reported; already frozen observations are untouched.
"""
from __future__ import annotations
import argparse,base64,hashlib,json,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
import joblib,numpy as np,pandas as pd
import v11_4_standalone_train_walkforward as core
from v11_4_forward_research_release import (
    strip_pandemic_matured_outcomes,fit_calibration_params,score_prospective)
from v11_4_four_family_live_screener import analyze,evaluate_family
from v11_4_threeFY_RSI70_promoter_source_ready import LABEL_FORBIDDEN,ORIGINAL_DATES
from v11_4_systematic_rsi_research import run as run_rsi_research
from v11_4_required_news_sources import collect as collect_news
from v11_4_tradability_risk_overlay import audit_tradability

CURRENT_VERSION="V11.4-systematic-calibration-clock-repair-20261009"
ADDITIONAL_CHECKS={
    "positive_operating_cash_flow":{"hard_rules":[["operating_cash_flow_INR",">",0]]},
    "promoter_at_least_50pct":{"hard_rules":[["promoter_holding",">=",.5]]},
    # Preserve the user's literal profit denominator; do not rewrite as sales.
    "receivables_below_10pct_profit":{"hard_rules":[["receivables_lt_10pct_profit","==",True]]},
    "chart_above_50_and_200DMA":{"hard_rules":[["price_gt_dma50_prev","==",True],
                                               ["price_gt_dma200_prev","==",True]]},
    "reserves_above_borrowings":{"hard_rules":[["reserves_gt_borrowings","==",True]]},
    "fixed_assets_rising_yoy":{"hard_rules":[["fixed_assets_up_yoy","==",True]]}}

def hash_file(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def validate_source_only(frame):
    forbidden=set(LABEL_FORBIDDEN)|{"p_cal","p_raw","p100_cal","selected_v941","selected_v94"}
    if forbidden.intersection(frame):raise ValueError("Future labels or earlier scores in source-only dataset")
    if any(str(c).startswith(("y6_","y12_","y24_")) for c in frame):
        raise ValueError("Future label metadata in source-only dataset")
    if not {"date","symbol","historical_asof_utc"}.issubset(frame):
        raise ValueError("Source-only stock/date/availability identity missing")
    if frame[["date","symbol"]].duplicated().any():raise ValueError("Duplicate source company-date")
    return frame

def screen_audit(frame,config,out):
    x=validate_source_only(frame).copy()
    if "rsi14_wilder_source" in x:
        x["rsi14_wilder"]=x["rsi14_wilder_source"]
    table=analyze(x,config)
    for name,family in ADDITIONAL_CHECKS.items():
        table[name+"_status"]=[evaluate_family(row,family)["status"]
                                     for row in x.to_dict("records")]
    # Interpret 'any of these lookbacks has rerated' using the largest move.
    # Missing windows cannot imply an early, under-discovered stock.
    returns=x[["ret_60","ret_120","ret_252"]].apply(pd.to_numeric,errors="coerce")
    move=returns.max(axis=1).where(returns.notna().all(axis=1))
    table["prior_runup_max_60_120_252_sessions"]=move
    table["discovery_sleeve"]=np.select(
        [move.isna(),move.lt(.12),move.le(1)],
        ["UNKNOWN","PRE_OBVIOUS_LT12PCT","SECOND_LEG_12_TO_100PCT"],
        default="EXTENDED_GT100PCT")
    table["verified_complete_primary_causal_chain"]="UNKNOWN"
    destination=Path(out);destination.mkdir(parents=True,exist_ok=True)
    table.to_csv(destination/"four_screens_additional_checks_sleeves_PRIVATE.csv",index=False)
    report={"rows":len(table),"screen_combination":"INDEPENDENT_FAMILIES_WITH_REPORTED_OVERLAP",
        "RSI_threshold_strictly_gt":70,
        "families":{k:table[k+"_status"].value_counts().to_dict() for k in config["conditions"]},
        "additional_user_checks":{k:table[k+"_status"].value_counts().to_dict() for k in ADDITIONAL_CHECKS},
        "discovery_sleeves":table["discovery_sleeve"].value_counts().to_dict(),
        "passing_multiple_independent_families":int(table["known_passed_condition_count"].gt(1).sum()),
        "5y7y_rules_not_replaced_with_two_year_CAGR":True,
        "quarterly_rules_not_replaced_with_annual_PAT_growth":True,
        "news_headlines_not_verified_primary_causal_chains":True,
        "full_user_model_ready":False}
    (destination/"screen_and_requirement_coverage.json").write_text(json.dumps(report,indent=2))
    return table,report

def fit_corrected_current(source,labels,asof,out):
    validate_source_only(source)
    x=source.copy();y=labels[["date","symbol","close","y6","y6_mature_date","integrity_y6_clean"]].copy()
    for z in (x,y):
        z["date"]=pd.to_datetime(z["date"],errors="raise").dt.normalize()
        z["symbol"]=z["symbol"].astype(str).str.upper().str.strip()
        if z.duplicated(["date","symbol"]).any():raise ValueError("Duplicate historical company/date")
    if len(x)!=18569 or x["date"].nunique()!=18:raise ValueError("Original historical universe changed")
    joined=x.merge(y,on=["date","symbol"],how="left",validate="1:1")
    available=pd.to_datetime(joined["historical_asof_utc"],utc=True,errors="coerce",format="mixed")
    if available.isna().any() or not available.eq(joined["date"].map(core.fold_close)).all():
        raise ValueError("Original NSE feature clock mismatch")
    keep=core.keep_train(joined,pd.Timestamp(asof))&strip_pandemic_matured_outcomes(joined)
    history=core.safe_featureize(joined.loc[keep].copy())
    previous=sorted(history["date"].unique())
    if len(previous)<5:raise ValueError("Insufficient mature historical folds")
    caldate=previous[-1];base,cal=core.partition_train_calibration(history,caldate)
    if (base["date"].nunique()<4 or len(base)<core.MIN_BASE_TRAIN_ROWS or
        base["y6"].sum()<core.MIN_BASE_POSITIVES or cal["y6"].nunique()!=2):
        raise ValueError("Insufficient separate train/calibration samples")
    model=core.make_model();model.fit(base[list(core.MODEL_FEATURES)],base["y6"].astype(int))
    params=fit_calibration_params(model.predict_proba(cal[list(core.MODEL_FEATURES)])[:,1],
                                  cal["y6"],base["y6"])
    package={"model":model,"calibration":params,"features":list(core.MODEL_FEATURES),
             "model_version":CURRENT_VERSION,"asof":asof}
    dest=Path(out);dest.mkdir(parents=True,exist_ok=True)
    joblib.dump(package,dest/"systematic_corrected_current_model_PRIVATE.joblib",compress=3)
    report={"model_id":CURRENT_VERSION,"train_rows":len(base),"train_folds":int(base["date"].nunique()),
        "calibration_date":str(pd.Timestamp(caldate).date()),"calibration_rows":len(cal),
        "training_labels_mature_before_calibration_decision":True,
        "calibration_labels_mature_before_live_decision":True,"calibration_method":params["method"],
        "COVID_initial_shock_crossing_training_labels_excluded":True,
        "live_predictor_feature_count":len(core.MODEL_FEATURES),
        "threeFY_and_RSI_variant_is_retrospective_research_separate_from_current_22input_model":True,
        "full_user_model_ready":False,"production_approved":False}
    (dest/"corrected_current_model_manifest.json").write_text(json.dumps(report,indent=2))
    return package,report

def encrypt_owner_demo(rows,summary,public_key,out):
    from cryptography.hazmat.primitives import hashes,serialization
    from cryptography.hazmat.primitives.asymmetric import padding
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    import os
    raw=json.dumps({"summary":summary,"rows":rows},allow_nan=False,separators=(",",":")).encode()
    key=AESGCM.generate_key(bit_length=256);nonce=os.urandom(12)
    aad=b"V11.4-systematic-owner-demo-20261009"
    public=serialization.load_pem_public_key(Path(public_key).read_bytes())
    wrapped=public.encrypt(key,padding.OAEP(mgf=padding.MGF1(hashes.SHA256()),algorithm=hashes.SHA256(),label=None))
    packet={k:base64.b64encode(v).decode() for k,v in {
        "key":wrapped,"nonce":nonce,"aad":aad,
        "payload":AESGCM(key).encrypt(nonce,raw,aad)}.items()}
    Path(out).write_text(json.dumps(packet))
    # Ciphertext only; no private ticker list, token or secret key in logs.
    print("OWNER_DEMO_CIPHER="+base64.b64encode(json.dumps(packet).encode()).decode(),flush=True)

def execute(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    config=json.loads(Path(args.config).read_text())
    s8=pd.read_parquet(args.source8)
    validate_source_only(s8)
    if len(s8)!=sum(ORIGINAL_DATES.values()):raise ValueError("Original eight-date source count drift")
    _,screen_report=screen_audit(s8,config,out/"historical_source_screening")
    labels=pd.read_parquet(args.labels)
    stages={"source_screening":{"status":"EXECUTED","coverage":screen_report}}
    cp=subprocess.run([sys.executable,"src/v11_4_covid_training_blackout.py",
        "--features",args.features18,"--snapshot",args.labels,"--output",str(out/"corrected_walkforward"),
        "--policy","initial_crash_2020"],capture_output=True,text=True,timeout=2400)
    (out/"corrected_walkforward_execution.log").write_text(cp.stdout+cp.stderr)
    sf=out/"corrected_walkforward/standalone_v11_4_model_summary.json"
    if sf.exists():
        wr=json.loads(sf.read_text())
        stages["corrected_standalone_walkforward"]={"status":"EXECUTED",
            "process_exit_code":cp.returncode,"acceptance_gates_pass":bool(
                wr["minimum_12_folds_gate_pass"] and wr["selection_stability_gate_pass"]),"metrics":wr}
    else:stages["corrected_standalone_walkforward"]={"status":"BLOCKED","process_exit_code":cp.returncode}
    try:
        report=run_rsi_research(s8,labels,out/"threeFY_specific_catalyst_RSI70")
        stages["threeFY_specific_catalyst_RSI70_research"]={"status":"EXECUTED","metrics":report}
    except Exception as exc:
        stages["threeFY_specific_catalyst_RSI70_research"]={"status":"BLOCKED","error":str(exc)[:500]}
    news=collect_news(out/"required_news")
    stages["required_news"]={"status":"COLLECTED" if news["both_required_sources_collected"] else "BLOCKED",
                              "source_report":news}
    if args.live_date:
        try:
            from v11_4_live_nse_market_catalyst import run as source_current
            live,metadata=source_current(args.live_date,out/"current_NSE")
            package,fit_report=fit_corrected_current(pd.read_parquet(args.features18),labels,
                                                   args.live_date,out/"current_model")
            picks=score_prospective(live,package,args.live_date,metadata)
            table,coverage=screen_audit(live,config,out/"current_screens")
            overlay=picks.merge(table,on=["date","symbol"],how="left",validate="1:1")
            if len(overlay)!=10:raise ValueError("Current ten-stock identity changed")
            four_family_columns=["date","symbol",*[k+"_status" for k in config["conditions"]]]
            risks,risk_report=audit_tradability(live,picks,table[four_family_columns])
            risks.to_csv(out/"current_tradability_audit_PRIVATE.csv",index=False)
            overlay.to_csv(out/"current_corrected_model_demo_PRIVATE.csv",index=False)
            stages["current_demo"]={"status":"EXECUTED","date":args.live_date,
                "verified_current_NSE_companies":len(live),"selected":len(picks),
                "fit_manifest":fit_report,"screen_coverage":coverage,
                "tradability_audit":risk_report,
                "news_discovery_complete":news["both_required_sources_collected"],
                "full_prompt_catalyst_chain_and_fundamental_requirements_complete":False,
                "production_approved":False}
            rows=[]
            for row in overlay.sort_values("rank").to_dict("records"):
                rows.append({"rank":int(row["rank"]),"symbol":row["symbol"],
                    "close_INR":round(float(row["close"]),2),
                    "research_6m_2x_estimate_pct":round(float(row["p6_double_calibrated"])*100,3),
                    "RSI14":float(row["rsi14_wilder"]) if pd.notna(row["rsi14_wilder"]) else None,
                    "family_status":{k:row[k+"_status"] for k in config["conditions"]},
                    "discovery_sleeve":row["discovery_sleeve"]})
            if args.owner_public_key:
                encrypt_owner_demo(rows,stages["current_demo"],args.owner_public_key,out/"owner_demo_encrypted.json")
        except Exception as exc:
            stages["current_demo"]={"status":"BLOCKED","requested_date":args.live_date,"error":str(exc)[:500]}
    report={"scope":"SYSTEMATIC_V11_4_EXECUTION_AND_REQUIREMENT_AUDIT",
        "executed_at_utc":datetime.now(timezone.utc).isoformat(),"stages":stages,
        "source_SHA256":{"original_18fold_features":hash_file(args.features18),
                         "original_mature_labels":hash_file(args.labels),
                         "threeFY_RSI70_source_only":hash_file(args.source8)},
        "latest_RSI_instruction":">70 (supersedes >80)",
        "horizon_design_preserved":{"six_month":.70,"twelve_month":.25,"twenty_four_month":.05},
        "active_prediction_target":"ARCHIVED_SIX_MONTH_2X_LABEL_ONLY",
        "full_user_model_ready":False,"production_approved":False,
        "remaining_requirements":["Exact original first-three-screener financial/valuation inputs",
            "Complete NSE+BSE listed-stock coverage and verified small/midcap classification; current collector is NSE-only",
            "Latest current three-year financial history, including quarterly context",
            "GDELT and Google News with PIT history and primary causal-chain verification",
            "Verified actual promoter purchase direction, not generic filing counts",
            "Independent 12m/24m modeling before applying 70/25/5 horizon blend",
            "Official all-session RSI source crosscheck and independent validation"],
        "original_frozen_Oct8_selections_and_history_not_overwritten":True,
        "no_V10_or_V10_4_selection_score_comparison":True}
    (out/"systematic_run_summary.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)
    return report

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--features18",required=True);p.add_argument("--labels",required=True)
    p.add_argument("--source8",required=True);p.add_argument("--output",required=True)
    p.add_argument("--config",default="config/v11_4_four_screener_families.json")
    p.add_argument("--live-date",default="");p.add_argument("--owner-public-key",default="")
    execute(p.parse_args())
if __name__=="__main__":main()
