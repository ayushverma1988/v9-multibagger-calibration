"""Independent four-family screening overlay for immutable standalone V11.4.

The three original user-defined screening families are preserved exactly as
earlier V11.4 config, plus RSI(14)>80 as fourth. This is an AUDIT/RESEARCH
companion, NOT an input or reranker of the previously frozen predictive
model. Missing sourced fundamentals are UNKNOWN, never fail/pass by default.

RSI(14) Wilder uses close adjusted for corporate actions, 14 daily changes
with recursive smoothing; threshold strictly >80. A high RSI is NOT evidence
of future doubling; test false positives and forward 6m outcomes separately.
"""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd

def eval_rule(value,op,threshold):
 if value is None or pd.isna(value):return None
 if isinstance(threshold,bool):
  if not isinstance(value,(bool,np.bool_)):return None
  v=bool(value)
 else:
  try:v=float(value)
  except (TypeError,ValueError):return None
  if not np.isfinite(v):return None
 t=threshold
 if op==">":return bool(v>t)
 if op=="<":return bool(v<t)
 if op==">=":return bool(v>=t)
 if op=="<=":return bool(v<=t)
 if op=="==":return bool(v==t)
 raise ValueError("Unrecognized four-family comparison")

def evaluate_family(record,family):
 rules=family["hard_rules"]
 results=[eval_rule(record.get(field),op,target) for field,op,target in rules]
 known=sum(r is not None for r in results)
 failed=sum(r is False for r in results)
 passed=sum(r is True for r in results)
 # A partial rule pass is NOT a full pass. A single hard fail is a FAIL,
 # otherwise remaining missing fields mean UNKNOWN.
 status=("FAIL" if failed else "PASS" if known==len(rules)
         else "UNKNOWN")
 return {"status":status,"rules_total":len(rules),
         "rules_known":known,"rules_pass":passed,
         "rules_fail":failed,"rules_unknown":len(rules)-known,
         "data_coverage":known/len(rules) if rules else 0.0}

def analyze(frame,config):
 if not {"date","symbol","rsi14_wilder"}.issubset(frame):
  raise ValueError("Original RSI and stock-date are required")
 if len(config["conditions"])!=4:
  raise ValueError("Exact four user screening families missing")
 if frame[["date","symbol"]].duplicated().any():
  raise ValueError("Duplicate company in screening universe")
 out=frame[["date","symbol","rsi14_wilder"]].copy()
 for key,condition in config["conditions"].items():
  assessments=[evaluate_family(r,condition) for r in frame.to_dict("records")]
  for field in ("status","rules_total","rules_known","rules_pass","rules_fail",
                "rules_unknown","data_coverage"):
   out[f"{key}_{field}"]=[a[field] for a in assessments]
 out["known_passed_condition_count"]=out[[f"{key}_status" for key in config["conditions"]]].eq("PASS").sum(axis=1)
 out["unknown_condition_count"]=out[[f"{key}_status" for key in config["conditions"]]].eq("UNKNOWN").sum(axis=1)
 out["rsi14_gt80"]=out["rsi14_wilder"].gt(80).where(
     out["rsi14_wilder"].notna(),pd.NA).astype("boolean")
 out["immutable_frozen_model_ranking_changed"]=False
 return out

def main():
 parser=argparse.ArgumentParser()
 parser.add_argument("--features",required=True)
 parser.add_argument("--config",required=True)
 parser.add_argument("--output",required=True)
 a=parser.parse_args()
 raw=pd.read_parquet(a.features)
 conf=json.loads(Path(a.config).read_text())
 table=analyze(raw,conf)
 dest=Path(a.output);dest.mkdir(parents=True,exist_ok=True)
 table.to_csv(dest/"v11_4_four_screen_families_by_stock.csv",index=False)
 keys=list(conf["conditions"])
 stat={"scope":"FOUR_SCREEN_FAMILIES_RESEARCH_NOT_FROZEN_MODEL_RERANKING",
       "companies":len(table),
       "families":{k:{"full_pass":int(table[f"{k}_status"].eq("PASS").sum()),
           "full_fail":int(table[f"{k}_status"].eq("FAIL").sum()),
           "unknown":int(table[f"{k}_status"].eq("UNKNOWN").sum())}
           for k in keys},
       "rsi14_over80_stock_count":int(table["rsi14_gt80"].fillna(False).sum()),
       "rsi14_available_stock_count":int(table["rsi14_wilder"].notna().sum()),
       "four_family_hard_AND_filter_applied":False,
       "missing_unverified_longterm_fundamentals_reported_unknown":True,
       "standalone_predictive_model_unchanged":True,
       "production_approved":False}
 (dest/"four_family_screening_audit.json").write_text(json.dumps(stat,indent=2))
 print(json.dumps(stat,indent=2),flush=True)
if __name__=="__main__":main()
