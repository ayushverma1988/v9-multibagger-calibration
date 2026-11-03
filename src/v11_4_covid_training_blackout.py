"""Optional V11.4 research training with COVID-affected SIX-MONTH labels excluded.

This is a separate exploratory variant. It invokes the standalone V11.4 model
without changing hyperparameters or 12-fold/Jaccard gates. Training samples
selected before 2020-02-20 are also excluded if their forward label matures
during the COVID regime. No test fold is excluded from a full-history report.
Use exact-original point-in-time y6 maturity; do not leak test outcomes into
selection. Feature scaling may still be COVID-affected in post-crash price
histories; this is "COVID outcome blackout", NOT synthetic no-COVID prices.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import pandas as pd
import v11_4_standalone_train_walkforward as core

PERIODS={
    "initial_crash_2020":("2020-02-20","2020-06-30"),
    "extended_pandemic":("2020-02-20","2021-12-31"),
}

def training_exclusion(frame,start,end):
    date=pd.to_datetime(frame["date"]).dt.normalize()
    mat=pd.to_datetime(frame["y6_mature_date"],utc=True,errors="coerce",format="mixed").dt.tz_convert(None)
    return date.le(pd.Timestamp(end))&mat.ge(pd.Timestamp(start))

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--features",required=True)
    p.add_argument("--snapshot",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--policy",required=True,choices=sorted(PERIODS))
    a=p.parse_args()
    lo,hi=PERIODS[a.policy]
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    # Override only the existing prediction model's training fold mask.
    keep_original=core.keep_train
    audited=[]
    def covid_free_training(data,test_date):
        original_mask=keep_original(data,test_date)
        exposed=training_exclusion(data,lo,hi)
        audited.append({
            "test_fold":str(pd.Timestamp(test_date).date()),
            "original_prior_matured_training_rows":int(original_mask.sum()),
            "COVID_overlapping_matured_training_removed":int((original_mask&exposed).sum()),
            "remaining_matured_training_rows":int((original_mask&~exposed).sum())})
        return original_mask&~exposed
    core.keep_train=covid_free_training
    sys.argv=[
      "standalone_6mo_covid_training_blackout",
      "--features",a.features,"--snapshot",a.snapshot,"--output",a.output]
    blocked=None
    try:
        core.main()
    except SystemExit as exc:
        blocked=str(exc)
    finally:
        (out/"covid_training_blackout_provenance.json").write_text(json.dumps({
         "covid_label_blackout_policy":a.policy,"start":lo,"end":hi,
         "only_matured_prior_training_outcome_windows_excluded":True,
         "full_history_test_results_remain_present":True,
         "per_historical_selection_fold":audited,
         "original_unmodified_v11_4_acceptance_failure":blocked,
         "COVID_market_price_features_may_remain_in_post_2020_lookbacks":True,
         "production_approved":False
        },indent=2))
    if blocked:raise SystemExit("V11.4 COVID outcome-excluded variant NOT promoted: "+blocked)
if __name__=="__main__":main()
