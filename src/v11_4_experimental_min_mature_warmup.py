"""Experimental V11.4 minimum-history safety gate (no previous model comparison).

This investigates the June 2019 root cause: V11.4 currently scores stocks
with only 2 earlier mature training folds. A 4-fold + held-out calibration
fold "warmup" is an a priori data sufficiency guard for FUTURE use, tested
retrospectively after seeing the weak June 2019 fold. Therefore it is
exploratory, not untouched-out-of-sample proof even if gates pass.

V11.4's absolute minimum 12 tested folds and .80/.60 Jaccard thresholds stay
unchanged. Optional 2020-shock label-overlap training blackout is supported.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import pandas as pd
import v11_4_standalone_train_walkforward as core

def run():
 p=argparse.ArgumentParser()
 p.add_argument("--features",required=True)
 p.add_argument("--snapshot",required=True)
 p.add_argument("--output",required=True)
 p.add_argument("--shock-outcome-blackout",action="store_true")
 a=p.parse_args()
 out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 four_prior_training_folds=4
 core.MIN_BASE_TRAIN_FOLDS=four_prior_training_folds
 audit=[]
 if a.shock_outcome_blackout:
  original_keep=core.keep_train
  def exclude_covid(data,heldout):
   base=original_keep(data,heldout)
   date=pd.to_datetime(data["date"]).dt.normalize()
   maturity=pd.to_datetime(
       data["y6_mature_date"],utc=True,errors="coerce",format="mixed").dt.tz_convert(None)
   exposed=date.le(pd.Timestamp("2020-06-30"))&maturity.ge(pd.Timestamp("2020-02-20"))
   audit.append({"heldout":str(pd.Timestamp(heldout).date()),
                 "COVID_overlapping_matured_train_rows_removed":int((base&exposed).sum())})
   return base&~exposed
  core.keep_train=exclude_covid
 sys.argv=["v11_4_experimental_warmup_guard","--features",a.features,
           "--snapshot",a.snapshot,"--output",a.output]
 failure=None
 try:core.main()
 except SystemExit as exc:failure=str(exc)
 finally:
  (out/"minimum_mature_history_safety_gate_provenance.json").write_text(json.dumps({
   "minimum_mature_training_folds":four_prior_training_folds,
   "additional_heldout_calibration_fold_required":1,
   "COVID_training_outcomes_excluded":a.shock_outcome_blackout,
   "COVID_impact_eligibility_audit":audit,
   "past_test_fold_informed_guard_design":True,
   "not_independent_confirmatory_untouched_holdout":True,
   "future_production_promotion_allowed_from_this_exploratory_run":False,
   "original_minimum_12_fold_and_jaccard_0_80_0_60_gates_unchanged":True,
   "gate_block_message":failure
  },indent=2))
 if failure:raise SystemExit(f"Exploratory V11.4 warmup guard is NOT production accepted: {failure}")
if __name__=="__main__":run()
