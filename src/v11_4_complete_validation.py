"""Reproducible V11.4 validation and one preregistered robustness experiment.

The original Oct8/Oct9 observations are immutable. Previously inspected test
dates remain retrospective research. Execution success is never promotion.
Public outputs are aggregates only; securities, models and diagnostics are
saved in PRIVATE files. No V10/V10.4 predictions or selections are loaded.
"""
from __future__ import annotations
import argparse, hashlib, json
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import v11_4_standalone_train_walkforward as core
import v11_4_robust_research_model as robust
from v11_4_forward_research_release import (
    strip_pandemic_matured_outcomes, fit_calibration_params, apply_frozen_calibration,
    validate_prospective_candidates)
from v11_4_systematic_run import screen_audit, validate_source_only, CURRENT_VERSION
from v11_4_systematic_rsi_research import run as run_rsi
from v11_4_systematic_sleeve_candidates import original_global_observation_agrees
from v11_4_exchange_NSE_day_close_RSI70_source_gate import annotate as official_rsi_audit
from v11_4_threeFY_RSI70_promoter_source_ready import STRICT_STATUS
from v11_4_required_news_sources import collect as collect_news
from v11_4_validation_metrics import evaluate_ranked, summarize_folds
from v11_4_label_clock import maturity_utc

EXPECTED_HASHES = {
    "features18":"20692301a594b2d504ba80066352187500e37af2d8f762bb2cc9bb0bf87804fe",
    "labels":"2178a5f9a55c9feda7f985f22f6668ac8fd72350a8b488698f0209c8c3cf5e23",
    "source8":"8fd03f45edecbbbe3b81bd6f9bf56558b8d227fa854c9ff583da929335806eda",
}
EVALUATION_CUTOFF = pd.Timestamp("2026-10-09T00:00:00Z")


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize_keys(frame):
    x = frame.copy()
    x["date"] = pd.to_datetime(x["date"], errors="raise").dt.normalize()
    x["symbol"] = x["symbol"].astype(str).str.upper().str.strip()
    if x[["date","symbol"]].isna().any().any() or x["symbol"].eq("").any():
        raise ValueError("Unknown security/date identity")
    if x.duplicated(["date","symbol"]).any():
        raise ValueError("Duplicate original security/date")
    return x


def load_history(features_path, labels_path):
    for name, path in (("features18", features_path), ("labels", labels_path)):
        if file_hash(path) != EXPECTED_HASHES[name]:
            raise ValueError("Immutable original input hash changed: " + name)
    x = normalize_keys(validate_source_only(pd.read_parquet(features_path)))
    labels = normalize_keys(pd.read_parquet(labels_path, columns=[
        "date","symbol","close","avg_turnover_63","y6","y6_mature_date",
        "integrity_y6_clean","dd30_6m"]))
    if len(x) != 18569 or x["date"].nunique() != 18:
        raise ValueError("Original eighteen-date predictor universe changed")
    joined = x.merge(labels, on=["date","symbol"], how="left", validate="1:1",
                     suffixes=("","_label"), indicator=True)
    if not joined["_merge"].eq("both").all():
        raise ValueError("Original label-source identity missing")
    if not np.allclose(joined["avg_turnover_63"], joined["avg_turnover_63_label"],
                       rtol=1e-7, atol=1e-5, equal_nan=True):
        raise ValueError("Immutable turnover disagrees across source files")
    if "close_label" in joined:
        if not np.allclose(joined["close"],joined["close_label"],rtol=1e-7,atol=1e-5):
            raise ValueError("Immutable close disagrees across source files")
        joined = joined.drop(columns="close_label")
    actual = pd.to_datetime(joined["historical_asof_utc"], utc=True, errors="coerce", format="mixed")
    if actual.isna().any() or not actual.eq(joined["date"].map(core.fold_close)).all():
        raise ValueError("Historical feature availability clock invalid")
    counts = joined[list(core.CATALYST_PREFIXES)].apply(pd.to_numeric, errors="coerce")
    if counts.isna().any().any() or counts.lt(0).any().any():
        raise ValueError("Missing/negative exchange event counts must not become zero")
    return core.safe_featureize(joined.drop(columns=["avg_turnover_63_label","_merge"])), labels


def partitions(history, date):
    known = history.loc[core.keep_train(history, date) & strip_pandemic_matured_outcomes(history)].copy()
    past = sorted(known["date"].unique())
    if len(past) < core.MIN_BASE_TRAIN_FOLDS+1:
        return None
    base, cal = core.partition_train_calibration(known, past[-1])
    if (base["date"].nunique() < core.MIN_BASE_TRAIN_FOLDS or len(base) < core.MIN_BASE_TRAIN_ROWS
        or base["y6"].sum() < core.MIN_BASE_POSITIVES or len(cal) < 50):
        return None
    for panel, clock in ((base, core.fold_close(past[-1])), (cal, core.fold_close(date))):
        maturity = maturity_utc(panel["y6_mature_date"])
        if maturity.isna().any() or not maturity.lt(clock).all():
            raise ValueError("Outcome unavailable before train/calibration decision")
    return base, cal


def fit_predict(name, train, cal, candidates):
    if name == "fixed_training_tail_bounds_21input":
        return robust.fit_once(train, cal, candidates)
    model = core.make_model().fit(train[list(core.MODEL_FEATURES)],train["y6"].astype(int))
    params = fit_calibration_params(model.predict_proba(cal[list(core.MODEL_FEATURES)])[:,1],
                                    cal["y6"], train["y6"])
    raw = model.predict_proba(candidates[list(core.MODEL_FEATURES)])[:,1]
    return apply_frozen_calibration(raw, params), model, params


def score_with_uncertainty(name, base, cal, candidates):
    p, model, params = fit_predict(name, base, cal, candidates)
    chosen = candidates.assign(_p=p).sort_values(["_p","symbol"],ascending=[False,True]).head(10)
    samples, membership, stability, coefficient_drift = [], [], [], []
    features = list(robust.FEATURES if name == "fixed_training_tail_bounds_21input" else core.MODEL_FEATURES)
    original_coef = model.named_steps["lr"].coef_[0]
    for seed in core.STABILITY_SEEDS:
        sample = core.perturbation_training(base, seed)
        alt, fitted, _ = fit_predict(name, sample, cal, candidates)
        top = candidates.assign(_p=alt).sort_values(["_p","symbol"],ascending=[False,True]).head(10)
        samples.append(alt)
        membership.append(candidates["symbol"].isin(top["symbol"]).to_numpy())
        stability.append({"seed":seed, "jaccard":core.top10_similarity(chosen["symbol"],top["symbol"])})
        changes = np.abs(fitted.named_steps["lr"].coef_[0]-original_coef)
        coefficient_drift.append(changes)
    a = np.asarray(samples)
    q = candidates[["date","symbol","close"]].copy()
    q["probability"] = p
    q["retraining_probability_p05"] = np.quantile(a,.05,axis=0)
    q["retraining_probability_p95"] = np.quantile(a,.95,axis=0)
    q["top10_inclusion_fraction_of_12_retrains"] = np.asarray(membership).mean(axis=0)
    # Retraining ranges are sensitivity estimates, not confidence intervals.
    q["sensitivity_range_is_statistical_confidence_interval"] = False
    if name == "fixed_training_tail_bounds_21input":
        clipped, missing = model.named_steps["training_tail_bounds"].support_audit(candidates[features])
        q["market_features_bounded_to_training_tail"] = clipped
        q["missing_predictor_count"] = missing
    q = q.sort_values(["probability","symbol"],ascending=[False,True])
    q.insert(0,"rank",range(1,len(q)+1))
    gap = float(q.iloc[9]["probability"]-q.iloc[10]["probability"]) if len(q)>10 else None
    drift = pd.DataFrame({"feature":features,"mean_absolute_standardized_coefficient_change":
                          np.asarray(coefficient_drift).mean(axis=0)})
    drift = drift.sort_values("mean_absolute_standardized_coefficient_change",ascending=False)
    diagnostics = {
        "top10_vs_11_probability_gap":gap,
        "top10_members_in_all_12_retrains":int(q.head(10)["top10_inclusion_fraction_of_12_retrains"].eq(1).sum()),
        "top10_members_in_fewer_than_9_retrains":int(q.head(10)["top10_inclusion_fraction_of_12_retrains"].lt(.75).sum()),
        "largest_coefficient_drift_features":drift.head(5).to_dict("records"),
        "retraining_sensitivity_ranges_are_not_confidence_intervals":True,
    }
    return q, stability, diagnostics, model, params


def walkforward(history, out):
    root = Path(out);root.mkdir(parents=True,exist_ok=True)
    result = {}
    for name in ("corrected_22input_reference", "fixed_training_tail_bounds_21input"):
        folds, stability, diagnostics, scores = [], [], [], []
        for date in sorted(history["date"].unique()):
            split = partitions(history,date)
            if split is None:
                continue
            base, cal = split
            candidates = history.loc[history["date"].eq(date)&core.eligible_asof(history)].copy()
            q, s, d, model, params = score_with_uncertainty(name,base,cal,candidates)
            labels = candidates[["date","symbol","y6","y6_mature_date","integrity_y6_clean","dd30_6m"]]
            ranked = q.merge(labels,on=["date","symbol"],how="left",validate="1:1")
            metric = evaluate_ranked(ranked,"probability",EVALUATION_CUTOFF,float(cal["y6"].mean()))
            day = str(pd.Timestamp(date).date())
            metric.update({"date":day, "train_rows":len(base), "train_dates":int(base["date"].nunique()),
                           "calibration_date":str(pd.Timestamp(cal["date"].iloc[0]).date()),
                           "calibration_method":params["method"]})
            folds.append(metric)
            stability.extend(dict(row,date=day) for row in s)
            diagnostics.append(dict(d,date=day))
            scores.append(ranked)
        summary = summarize_folds(folds,stability)
        summary.update({"model_variant":name,"folds":folds,"selection_diagnostics":diagnostics,
                        "same_source_train_calibration_target_and_C":True,"parameter_search_performed":False})
        pd.concat(scores,ignore_index=True).to_parquet(root/(name+"_rankings_PRIVATE.parquet"),index=False)
        pd.DataFrame(stability).to_csv(root/(name+"_stability.csv"),index=False)
        result[name] = summary
    reference = result["corrected_22input_reference"]
    reproduced = (reference["confirmed_2x_hits"]==13 and reference["fully_labelled_selections"]==120
                  and abs(reference["mean_top10_jaccard"]-.8851445186286815)<1e-12
                  and abs(reference["worst_fold_p05_jaccard"]-.17647058823529413)<1e-12)
    if not reproduced:
        raise ValueError("Corrected reference did not reproduce the frozen run; investigate before comparing")
    result["original_corrected_reference_reproduced"] = reproduced
    # A variant can select an unassessable corporate-action case and lose a
    # fully-labelled date. Compare headline precision on the SAME dates only;
    # also retain bounds for every selection so coverage cannot flatter it.
    alternate=result["fixed_training_tail_bounds_21input"]
    refs={f["date"]:f for f in reference["folds"] if f["selected_clean_mature_outcomes"]==10}
    alts={f["date"]:f for f in alternate["folds"] if f["selected_clean_mature_outcomes"]==10}
    dates=sorted(refs.keys()&alts.keys())
    differences=np.array([alts[d]["top10_precision"]-refs[d]["top10_precision"] for d in dates])
    ci=[None,None]
    if len(dates)>=2:
        draws=np.random.default_rng(20261010).choice(differences,size=(4000,len(dates)),replace=True).mean(axis=1)
        ci=np.quantile(draws,[.025,.975]).tolist()
    result["paired_comparable_dates"]={"dates":dates,"date_count":len(dates),"selected_per_variant":10*len(dates),
        "reference_2x_hits":sum(refs[d]["top10_doublers"] for d in dates),
        "variant_2x_hits":sum(alts[d]["top10_doublers"] for d in dates),
        "mean_precision_difference":float(differences.mean()) if len(differences) else None,
        "paired_date_block_precision_difference95":ci,
        "retrospective_comparison_does_not_validate_superior_future_returns":True}
    for s in (reference,alternate):
        n=s["all_scored_selected"];wins=s["all_scored_known_hits"];unknown=s["all_scored_unknown_selected"]
        s["all_selected_precision_bounds"]=[wins/n,(wins+unknown)/n] if n else [None,None]
    return result


def replay_current(history, features_path, metadata_path, model_path, recorded_path, config, out):
    root = Path(out);root.mkdir(parents=True,exist_ok=True)
    source = pd.read_parquet(features_path)
    metadata = json.loads(Path(metadata_path).read_text())
    asof = metadata["snapshot_date"]
    if file_hash(features_path)!=metadata["live_features_SHA256"]:
        raise ValueError("Verified October9 source bytes changed")
    validate_prospective_candidates(source,asof,metadata)
    package = joblib.load(model_path)
    if package.get("model_version")!=CURRENT_VERSION or package.get("asof")!=asof:
        raise ValueError("Wrong original reference model identity")
    original_global_observation_agrees(source,package,asof,metadata,pd.read_csv(recorded_path))
    source = normalize_keys(source)
    table, coverage = screen_audit(source,config,root/"source_screens")
    split = partitions(history,pd.Timestamp(asof))
    if split is None:
        raise ValueError("Current train/calibration split incomplete")
    base, cal = split
    candidates = core.safe_featureize(source.loc[core.eligible_asof(source)])
    q, stability, diag, fitted, params = score_with_uncertainty(
        "fixed_training_tail_bounds_21input",base,cal,candidates)
    q = q.merge(table,on=["date","symbol"],how="left",validate="1:1")
    q.to_parquet(root/"robust_current_scores_and_uncertainty_PRIVATE.parquet",index=False)
    joblib.dump({"model":fitted,"calibration":params,"features":list(robust.FEATURES),
                 "model_version":robust.VERSION,"asof":asof,"production_approved":False},
                root/"robust_research_current_model_PRIVATE.joblib",compress=3)
    views = q.groupby("discovery_sleeve",sort=False).head(10)
    views.to_csv(root/"robust_current_sleeve_views_PRIVATE.csv",index=False)
    top = q.head(10)
    ratio = float(top["probability"].max()/top["probability"].median())
    s = np.array([row["jaccard"] for row in stability])
    return {"source_snapshot_date_IST":asof,"snapshot_replay_not_today_market_data":True,
        "original_Oct9_global_observation_reproduced_and_not_overwritten":True,
        "model_id":robust.VERSION,"feature_count":len(robust.FEATURES),
        "eligible_candidates":len(q),"mean_top10_retraining_jaccard":float(s.mean()),
        "p05_top10_retraining_jaccard":float(np.quantile(s,.05)),
        "top10_max_research_probability":float(top["probability"].max()),
        "top10_median_research_probability":float(top["probability"].median()),
        "top_probability_vs_top10_median_multiplier":ratio,
        "top10_prior_runup_sleeves":top["discovery_sleeve"].value_counts().to_dict(),
        "top10_with_features_bounded_to_training_tail":int(top["market_features_bounded_to_training_tail"].gt(0).sum()),
        "top10_with_missing_predictors":int(top["missing_predictor_count"].gt(0).sum()),
        "eligible_sleeve_counts":q["discovery_sleeve"].value_counts().to_dict(),
        "uncertainty_diagnostics":diag,"screen_coverage":coverage,
        "qualified_full_user_model_candidates":0,"production_approved":False}


def acceptance_report(walkforward_results, financial, current, news, official):
    improved = walkforward_results["fixed_training_tail_bounds_21input"]
    gates = {
        "original_source_and_reference_reproduction":walkforward_results["original_corrected_reference_reproduced"],
        "at_least_12_fully_labelled_historical_dates":improved["minimum_12_folds_gate_pass"],
        "mean_top10_jaccard_at_least_080":improved["mean_jaccard_gate_pass"],
        "worst_date_p05_jaccard_at_least_060":improved["worst_fold_p05_gate_pass"],
        # Known missing requirements remain FALSE, with evidence below. No
        # inferred size class, fiscal window, primary chain or outcome horizon.
        "unseen_independent_evaluation_complete":improved["new_blinded_evaluation_dates"]>=12,
        "exact_financial_valuation_and_quarterly_inputs_complete":False,
        "NSE_and_BSE_universe_and_small_midcap_classification_complete":False,
        "six_additional_common_checks_fully_sourced":False,
        "both_required_news_sources_collected":bool(news and news.get("both_required_sources_collected") is True),
        "primary_company_causal_chain_verification_complete":False,
        "historical_news_point_in_time_reconstruction_complete":False,
        "actual_promoter_purchase_direction_verified":False,
        "complete_official_daily_RSI_history_crosscheck":bool(official and official.get(
            "original_NSE_daily_archive_120_session_history_independently_certified") is True),
        "independently_fitted_12m_and_24m_models":False,
    }
    return {"all_required_gates":gates,"failed_gates":[k for k,v in gates.items() if not v],
        "full_user_model_ready":all(gates.values()),"production_approved":all(gates.values()),
        "financial_research_scored_dates":len(financial.get("folds",[])) if financial else 0,
        "current_source_snapshot_date":current.get("source_snapshot_date_IST") if current else None,
        "financial_screens_are_independent_no_global_AND":True,
        "promoter_and_balance_sheet_checks_have_no_fabricated_values":True}


def execute(args):
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    recipe={"version":robust.VERSION,"features":list(robust.FEATURES),
        "base_training_market_tail_quantiles":[robust.LOW_QUANTILE,robust.HIGH_QUANTILE],
        "C":core.REGULARIZATION_C,"seeds":list(core.STABILITY_SEEDS),
        "training_perturbation_retention":core.STABILITY_RETENTION,
        "future_outcomes_cannot_filter_test_candidates":True,"parameter_search_performed":False,
        "date_only_label_maturity_is_1530_IST_close_not_UTC_midnight":True,
        "already_inspected_test_dates_not_new_blind_evidence":True}
    (root/"fixed_validation_recipe.json").write_text(json.dumps(recipe,indent=2))
    history,labels=load_history(args.features18,args.labels)
    results=walkforward(history,root/"walkforward")
    config=json.loads(Path(args.config).read_text())
    financial=current=official=news=None
    if args.source8:
        if file_hash(args.source8)!=EXPECTED_HASHES["source8"]:
            raise ValueError("Immutable original financial/RSI source hash changed")
        source=pd.read_parquet(args.source8);validate_source_only(source)
        if args.official_day:
            annotated,official=official_rsi_audit(source,pd.read_parquet(args.official_day))
            known=annotated["RSI_120session_archive_available_PLUS_official_day_close"].eq(True)
            source["rsi14_wilder_source"]=source["rsi14_wilder_source"].where(known)
            source["rsi14_strict_gt70_source_verified"]=source["rsi14_strict_gt70_source_verified"].where(known,pd.NA)
            source["source_verification"]=np.where(known,STRICT_STATUS,"UNKNOWN_OFFICIAL_FOLD_DAY_PRICE_UNCONFIRMED")
            (root/"official_RSI_day_price_audit.json").write_text(json.dumps(official,indent=2))
        financial=run_rsi(source,labels,root/"corrected_financial_RSI")
        _,historical_coverage=screen_audit(source,config,root/"historical_screens")
        results["historical_full_screen_coverage"]=historical_coverage
    if args.current_features:
        current=replay_current(history,args.current_features,args.current_metadata,args.current_model,
                               args.current_recorded,config,root/"current_replay")
    if args.collect_news:
        news=collect_news(root/"required_news")
    report={"scope":"V11_4_COMPLETE_VALIDATION_AND_FIXED_ROBUSTNESS_RESEARCH",
        "executed_at_utc":datetime.now(timezone.utc).isoformat(),
        "fixed_recipe":recipe,"source_hashes":{"features18":file_hash(args.features18),
            "labels":file_hash(args.labels),**({"source8":file_hash(args.source8)} if args.source8 else {})},
        "walkforward":results,"financial_RSI_research":financial,
        "official_RSI_fold_day_price_audit":official,"current_snapshot_replay":current,"required_news":news,
        "acceptance":acceptance_report(results,financial,current,news,official),
        "no_legacy_model_predictions_rankings_or_comparison_loaded":True,
        "original_frozen_observations_preserved":True,
        "limitations":["Previously inspected retrospective dates are not independent blind validation",
            "Archive universe selection and delisted/BSE-only exclusions cannot be independently reconstructed here",
            "A 2x event label does not establish a buy/sell strategy or net portfolio return",
            "Drawdown labels do not model spreads, slippage, position sizing or trading costs",
            "RSI fold-day close reconciliation does not certify the complete historical daily series"]}
    (root/"complete_validation_summary.json").write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(report,indent=2,allow_nan=False),flush=True)
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--features18",required=True);p.add_argument("--labels",required=True)
    p.add_argument("--output",required=True);p.add_argument("--source8")
    p.add_argument("--official-day");p.add_argument("--current-features")
    p.add_argument("--current-metadata");p.add_argument("--current-model");p.add_argument("--current-recorded")
    p.add_argument("--collect-news",action="store_true")
    p.add_argument("--config",default="config/v11_4_four_screener_families.json")
    execute(p.parse_args())


if __name__=="__main__":main()
