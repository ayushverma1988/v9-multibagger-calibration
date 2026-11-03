"""Independent V11.4 research tradability/probability outlier safety audit.

No new model training, no stock reordering, no probability overwrite.
A stock can have a high six-month 2x research score but insufficient
observable liquidity to buy/sell in a real market. Unknown metrics NEVER
mean an execution approval. The user's original Top 10 stays immutable.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

MIN_LAST_SESSION_VALUE_INR=500_000.0     # 5 lakh / actual session
MIN_RECENT_MEDIAN_DAILY_VALUE_INR=2_000_000.0 # 20 lakh per trading day
MIN_OBSERVED_SESSIONS_35_CALENDAR_DAYS=15
PROBABILITY_TOP_OUTLIER_MULTIPLIER=4.0

def safe_number(x):
 try:
  x=float(x)
  return x if np.isfinite(x) else None
 except (ValueError,TypeError):return None

def audit_tradability(features,picks,screeners=None):
 if len(picks)!=10 or set(picks["rank"])!=set(range(1,11)):
  raise ValueError("Must preserve original 10 rank identities")
 if picks["symbol"].duplicated().any():
  raise ValueError("Duplicate originally selected stock")
 if len(features)==0 or features["symbol"].duplicated().any():
  raise ValueError("Invalid complete original NSE observation universe")
 allowed={"symbol","date","last_session_turnover_INR","median_20_session_turnover_INR",
          "observed_trading_sessions_last_35d","avg_turnover_63",
          "integrity_feature_clean"}
 if not {"symbol","date"}.issubset(features):
  raise ValueError("Missing NSE source company/date identity")
 q=picks.merge(features[[c for c in features if c in allowed]].copy(),on=["symbol","date"],how="left",validate="1:1",indicator=True)
 if not q["_merge"].eq("both").all():raise ValueError("Original immutable prediction not in contemporaneous NSE verified universe")
 if screeners is not None:
  if screeners["symbol"].duplicated().any():raise ValueError("Screener identity duplicated")
  flags=[s for s in screeners if s.endswith("_status")]
  if len(flags)!=4:raise ValueError("Four original rule status fields must be independently audited")
  q=q.merge(screeners[["symbol",*flags]],on="symbol",how="left",validate="1:1")
 else:flags=[]
 probabilities=pd.to_numeric(q["p6_double_calibrated"],errors="coerce")
 if probabilities.isna().any() or not probabilities.between(0,1,inclusive="neither").all():
  raise ValueError("Frozen original probabilities invalid")
 ratio=float(probabilities.max()/probabilities.median()) if probabilities.median()>0 else None
 out=[]
 for row in q.sort_values("rank").to_dict("records"):
  liquidity=[]
  current=safe_number(row.get("last_session_turnover_INR"))
  median=safe_number(row.get("median_20_session_turnover_INR"))
  sessions=safe_number(row.get("observed_trading_sessions_last_35d"))
  if current is None or median is None or sessions is None:
   liquidity.append("MISSING_VERIFIED_CURRENT_LIQUIDITY_METADATA")
  else:
   if current<MIN_LAST_SESSION_VALUE_INR:liquidity.append("LAST_SESSION_TURNOVER_BELOW_5_LAKH")
   if median<MIN_RECENT_MEDIAN_DAILY_VALUE_INR:liquidity.append("MEDIAN_20_SESSION_TURNOVER_BELOW_20_LAKH")
   if sessions<MIN_OBSERVED_SESSIONS_35_CALENDAR_DAYS:
    liquidity.append("FEWER_THAN_15_RECENT_TRADING_SESSIONS")
  selected_risks=[]
  if (ratio is not None and ratio>PROBABILITY_TOP_OUTLIER_MULTIPLIER and
      float(row["p6_double_calibrated"])==float(probabilities.max())):
    selected_risks.append("TOP_PROBABILITY_EXCEEDS_4X_TOP10_MEDIAN_UNVALIDATED_TAIL")
  if flags:
   if any(str(row[f])=="UNKNOWN" for f in flags):
    selected_risks.append("MISSING_FULLY_VERIFIED_FUNDAMENTAL_RULES")
  result={
   "symbol":row["symbol"],"rank":int(row["rank"]),
   "original_frozen_probability":float(row["p6_double_calibrated"]),
   "original_frozen_close_INR":float(row["close"]),
   "last_session_turnover_INR":current,
   "median_20_session_turnover_INR":median,
   "observed_trading_sessions_last_35d":sessions,
   "tradability_assessment":"HIGH_CAUTION" if liquidity and "MISSING_VERIFIED_CURRENT_LIQUIDITY_METADATA" not in liquidity
      else "UNKNOWN" if liquidity else "SAMPLE_THRESHOLDS_ONLY_NOT_EXECUTION_APPROVED",
   "tradability_issues":"|".join(liquidity),
   "risk_flags":"|".join(selected_risks),
   "never_claim_executable_at_15_30_close":True,
   "frozen_model_ranking_unchanged":True}
  out.append(result)
 audit=pd.DataFrame(out)
 if audit[["rank","symbol"]].to_dict("records")!=picks.sort_values("rank")[["rank","symbol"]].to_dict("records"):
  raise ValueError("Source risk overlay inadvertently changed stock rank")
 report={
  "scope":"V11_4_POST_SELECTION_ANNOTATION_ONLY_NOT_A_TRADING_MODEL",
  "original_top10_stock_count":10,
  "top_probability_vs_top10_median_multiplier":round(ratio,3) if ratio is not None else None,
  "probability_outlier_alert":bool(ratio and ratio>PROBABILITY_TOP_OUTLIER_MULTIPLIER),
  "stocks_with_low_recent_tradability_evidence":int(audit["tradability_assessment"].eq("HIGH_CAUTION").sum()),
  "stocks_with_missing_tradability_evidence":int(audit["tradability_assessment"].eq("UNKNOWN").sum()),
  "stocks_cleared_for_execution":0,
  "original_rank_changed":False,"original_calibrated_probabilities_changed":False,
  "first_seen_six_month_outcome_observed":False,
  "research_not_investment_advice":True,
  "thresholds":{"minimum_latest_session_turnover_INR":MIN_LAST_SESSION_VALUE_INR,
    "minimum_median_20session_turnover_INR":MIN_RECENT_MEDIAN_DAILY_VALUE_INR,
    "minimum_traded_sessions_over_35_calendar_days":MIN_OBSERVED_SESSIONS_35_CALENDAR_DAYS,
    "top_probability_alert_multiple":PROBABILITY_TOP_OUTLIER_MULTIPLIER},
 }
 return audit,report
