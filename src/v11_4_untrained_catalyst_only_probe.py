"""Predeclared, non-trained NSE event-only 18-fold EXPLORATORY lift check.

No optimization / feature search / weight learning using outcomes.
This is NOT V11.4 scoring or a valid model-vs-V10 acceptance backtest.
Weights reflect declared investor interest in capacity, orders, promoter buys,
regulatory and dilution, not fitted target labels.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd

FIXED_WEIGHTS={
 "nse_capacity_expansion_180d_positive":4.0,
 "nse_order_win_180d_positive":3.0,
 "nse_promoter_activity_180d_positive":2.0,
 "nse_regulatory_180d_positive":1.5,
 "nse_corporate_action_180d_positive":1.0,
 "nse_buyback_180d_positive":1.0,
 "nse_dilution_180d":-2.0,
}

def score_without_labels(features):
 missing=set(FIXED_WEIGHTS)-set(features)
 if missing:raise ValueError("Missing prescribed catalyst source columns "+str(missing))
 if any(k in features for k in ("y6","y12","y24","hit50_6m","days_to_2x")):
  raise ValueError("Forward labels must not enter fixed-weight score")
 s=pd.Series(0.0,index=features.index)
 for name,w in FIXED_WEIGHTS.items():
  val=pd.to_numeric(features[name],errors="coerce").fillna(0)
  if val.lt(0).any():raise ValueError("Negative NSE source disclosure counts")
  s+=w*np.log1p(val)
 return s

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--events",required=True)
 p.add_argument("--snapshot",required=True)
 p.add_argument("--output",required=True)
 a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 events=pd.read_parquet(a.events)
 events["date"]=pd.to_datetime(events["date"],errors="coerce").dt.normalize()
 events["symbol"]=events["symbol"].astype(str).str.upper().str.strip()
 events["catalyst_proxy_score"]=score_without_labels(events)
 snap=pd.read_parquet(a.snapshot,columns=["date","symbol","close","y6","integrity_y6_clean"])
 snap["date"]=pd.to_datetime(snap["date"],errors="coerce").dt.normalize()
 snap["symbol"]=snap["symbol"].astype(str).str.upper().str.strip()
 if events.duplicated(["date","symbol"]).any():raise SystemExit("Duplicate NSE event stock in fold")
 joined=events[["date","symbol","catalyst_proxy_score"]].merge(
  snap,on=["date","symbol"],how="left",validate="1:1",indicator=True)
 if not joined["_merge"].eq("both").all():raise SystemExit("Historical label/stock missing")
 joined["y6"]=pd.to_numeric(joined["y6"],errors="coerce")
 joined["close"]=pd.to_numeric(joined["close"],errors="coerce")
 joined["clean_label"]=joined["integrity_y6_clean"].fillna(False).astype(bool)&joined["y6"].notna()
 summaries=[];picks=[]
 for day,g in joined.groupby("date",sort=True):
  allowed=g[g["close"].between(20,2000)&g["clean_label"]].copy()
  if len(allowed)<100:raise SystemExit(f"Insufficient original-price/clean y6 stocks at {day}")
  prevalence=float(allowed["y6"].gt(0).mean())
  selected=allowed.sort_values(["catalyst_proxy_score","symbol"],ascending=[False,True]).head(10)
  hits=int(selected["y6"].gt(0).sum())
  precision=hits/len(selected)
  lift=precision/prevalence if prevalence>0 else None
  nonzero=int(selected["catalyst_proxy_score"].gt(0).sum())
  summaries.append({"date":str(day.date()),"eligible_historical_stocks":len(allowed),
                    "y6_positive":int(allowed["y6"].gt(0).sum()),
                    "prevalence":prevalence,"top10_hits":hits,
                    "top10_precision":precision,"top10_lift_raw":lift,
                    "top10_nonzero_metadata_scores":nonzero})
  for r in selected.itertuples(index=False):
   picks.append({"date":str(day.date()),"symbol":r.symbol,
                 "source_metadata_score":r.catalyst_proxy_score,
                 "known_future_y6_DIAGNOSTIC_ONLY":r.y6})
 perf=pd.DataFrame(summaries)
 perf.to_csv(out/"exploratory_event_only_untrained_fold_diagnostics.csv",index=False)
 pd.DataFrame(picks).to_csv(out/"exploratory_event_only_top10_DIAGNOSTIC.csv",index=False)
 summary={
  "status":"EXPLORATORY_UNTRAINED_NO_V10_COMPARISON",
  "original_fold_count":len(summaries),
  "top10_mean_precision":float(perf["top10_precision"].mean()),
  "unweighted_mean_top10_hits":float(perf["top10_hits"].mean()),
  "mean_prevalence":float(perf["prevalence"].mean()),
  "mean_top10_nonzero_source_scores":float(perf["top10_nonzero_metadata_scores"].mean()),
  "fixed_predeclared_source_weights":FIXED_WEIGHTS,
  "no_fitted_coefficients_or_search":True,
  "no_lookahead_in_ranking_features":True,
  "price_filter":"Rs20 to Rs2000 at historical fold",
  "not_a_V11_4_vs_V10_promotion_test":True,
  "v10_unchanged":True,
 }
 (out/"untrained_catalyst_diagnostic_summary.json").write_text(json.dumps(summary,indent=2))
 print(json.dumps(summary,indent=2),flush=True)
 print(perf.tail(7).to_string(index=False),flush=True)
 if len(summaries)!=18:raise SystemExit("Frozen fold count changed")
if __name__=="__main__":main()
