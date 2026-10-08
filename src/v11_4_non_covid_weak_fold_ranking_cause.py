"""Root-cause analysis for V11.4 non-COVID weak folds.

Diagnostics ONLY: actual 95%-retained training reruns, exact frozen rank
margins, feature coefficient drift and disjoint Top10 swaps for 2019-06
and 2023-06; no model parameters or acceptance gates are updated.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from v11_4_standalone_train_walkforward import (
 MODEL_FEATURES,STABILITY_SEEDS,TOP_K,make_model,train_once,
 perturbation_training,keep_train,eligible_asof,safe_featureize,
 top10_similarity
)

CHECK_FOLDS=("2019-06-28","2023-06-30")

def clean(z):
    z["date"]=pd.to_datetime(z["date"]).dt.normalize()
    z["symbol"]=z["symbol"].astype(str).str.upper().str.strip()
    if z.duplicated(["date","symbol"]).any():raise SystemExit("Duplicate symbol-date")
    return z

def coefs(model):
    trained=model.named_steps["lr"]
    output=np.asarray(trained.coef_)[0]
    names=np.asarray(MODEL_FEATURES)
    if len(output)!=len(names):
        raise SystemExit("Unexpected feature column elimination during fitting")
    return output

def run():
    parser=argparse.ArgumentParser()
    parser.add_argument("--features",required=True)
    parser.add_argument("--snapshot",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    x=clean(pd.read_parquet(args.features))
    snap=clean(pd.read_parquet(args.snapshot,columns=[
        "date","symbol","close","avg_turnover_63","y6","y6_mature_date",
        "integrity_y6_clean","dd30_6m"]))
    joined=x.merge(snap,on=["date","symbol"],how="left",validate="1:1",suffixes=("","_snapshot"))
    data=safe_featureize(joined)
    records=[];features=[];candidate_rows=[]
    for selection_date in CHECK_FOLDS:
        target=pd.Timestamp(selection_date)
        known=data.loc[keep_train(data,target)]
        past=sorted(known["date"].unique())
        cal_date=past[-1]
        base=known[known["date"]<cal_date].copy()
        cal=known[known["date"]==cal_date].copy()
        test=data[(data["date"]==target)&eligible_asof(data)].copy()
        original_predictions,mode=train_once(base,cal,test,return_mode=True)
        test["score"]=original_predictions
        ordered=test.sort_values(["score","symbol"],ascending=[False,True])
        top=ordered.head(TOP_K)
        next_price_gap=float(ordered.iloc[9]["score"]-ordered.iloc[10]["score"])
        base_m=make_model().fit(base[list(MODEL_FEATURES)],base["y6"].astype(int))
        original_coefs=coefs(base_m)
        for seed in STABILITY_SEEDS:
            resampled=perturbation_training(base,seed)
            predictions=train_once(resampled,cal,test)
            revised=test.assign(p=predictions).sort_values(["p","symbol"],ascending=[False,True])
            selected=revised.head(TOP_K)
            alt_m=make_model().fit(
                resampled[list(MODEL_FEATURES)],resampled["y6"].astype(int))
            differences=np.abs(coefs(alt_m)-original_coefs)
            shift=selected["symbol"].tolist()
            record={
                "test_fold":selection_date,"training_folds":base["date"].nunique(),
                "training_rows":len(base),"train_positives":int(base["y6"].sum()),
                "seed":seed,"calibration_mode":mode,
                "top10_jaccard":top10_similarity(top["symbol"],shift),
                "baseline_top10_11_probability_margin":next_price_gap,
                "subset_top10_11_probability_margin":float(revised.iloc[9]["p"]-revised.iloc[10]["p"]),
                "stocks_entering_under_resample":"|".join(sorted(set(shift)-set(top["symbol"]))),
                "stocks_lost_under_resample":"|".join(sorted(set(top["symbol"])-set(shift))),
            }
            records.append(record)
            for idx,feature in enumerate(MODEL_FEATURES):
                features.append({"test_fold":selection_date,"seed":seed,
                                 "feature":feature,
                                 "base_standardized_logit_coefficient":float(original_coefs[idx]),
                                 "resample_coefficient":float(coefs(alt_m)[idx]),
                                 "absolute_coefficient_change":float(differences[idx])})
        for rank,row in enumerate(ordered.head(25).itertuples(index=False),start=1):
            candidate_rows.append({"test_fold":selection_date,"rank":rank,
                                   "symbol":row.symbol,"p2x_calibrated":float(row.score),
                                   "in_original_top10":rank<=10})
    re=pd.DataFrame(records)
    changes=pd.DataFrame(features)
    chosen=changes.groupby(["test_fold","feature"],as_index=False)["absolute_coefficient_change"].mean()
    chosen=chosen.sort_values(["test_fold","absolute_coefficient_change"],ascending=[True,False])
    re.to_csv(out/"weak_fold_12seed_selection_instability.csv",index=False)
    chosen.to_csv(out/"feature_parameter_instability_by_fold.csv",index=False)
    pd.DataFrame(candidate_rows).to_csv(out/"weak_fold_original_top25_probability_margins.csv",index=False)
    summary={
     "scope":"DIAGNOSE_NOT_FIX_DO_NOT_DROP_JUNE_2019_AS_COVID",
     "diagnosed_folds":list(CHECK_FOLDS),
     "seeds":len(STABILITY_SEEDS),
     "most_instability_fold":str(re.sort_values("top10_jaccard").iloc[0]["test_fold"]),
     "worst_jaccard":float(re["top10_jaccard"].min()),
     "sampled_historical_labels_only":True,
     "no_fitting_on_test_fold_label":True,
     "future_label_pit_enforced":True,
     "no_model_gate_modified":True,
    }
    (out/"weak_fold_failure_cause_summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)
    print("LOWEST_JACCARD_SEEDS",re.nsmallest(8,"top10_jaccard").to_string(index=False),flush=True)
    print("LARGEST_COEFFICIENT_DRIFT",chosen.groupby("test_fold",sort=False).head(6).to_string(index=False),flush=True)
if __name__=="__main__":run()
