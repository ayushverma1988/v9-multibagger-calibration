"""Conservative official-vintage original-market RSI(14) >70 historical source audit.

Uses public market prices for 2021-2025 and ORIGINAL frozen stock-date and
close identities, no outcome labels, no stock predictions, no ML. Requires
>=120 recent trading sessions, a verified fold-close price, no unresolved
split/bonus actions within price-history lookback, and no ISIN/series drift.
RSI(14) Wilder is computed by the same already-regression-tested function as
the 2026 forward source collector. Source-only investor research, not approval.

Missing split/bonus publication timestamps cannot be imputed as on-time.
Conservatively leave affected securities UNKNOWN rather than correcting with
an unverified PIT split factor or treating raw RSI as genuine.
"""
from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from v11_4_live_nse_market_catalyst import rsi_wilder_last

DAYS=("2022-06-30","2022-12-30","2023-06-30","2023-12-29",
      "2024-06-28","2024-12-31","2025-06-30","2025-12-31")
UNIVERSES={"2022-06-30":1087,"2022-12-30":1123,"2023-06-30":1147,
 "2023-12-29":1276,"2024-06-28":1322,"2024-12-31":1345,
 "2025-06-30":1298,"2025-12-31":1314}
SOURCE_YEARS=(2021,2022,2023,2024,2025)
CORP_TYPES={"split","bonus"}
CLOSE_TOL_REL=.002
CLOSE_TOL_MIN_INR=.10
RSI_SESSION_COUNT=120
RSI_SOURCE_WINDOW_DAYS=260
RECENT_QUOTE_WINDOW_DAYS=31
MIN_RECENT_QUOTES=15
MAX_SESSION_GAP_DAYS=12

def normalize_sources(prices,actions,original):
  market=prices.copy()
  required={"date","symbol","series","isin","close"}
  if not required.issubset(market):raise ValueError("Missing public daily NSE historical source fields")
  market["date"]=pd.to_datetime(market["date"],errors="coerce").dt.normalize()
  market["symbol"]=market["symbol"].astype(str).str.upper().str.strip()
  market["series"]=market["series"].astype(str).str.upper().str.strip()
  market["isin"]=market["isin"].fillna("").astype(str).str.upper().str.strip()
  market["close"]=pd.to_numeric(market["close"],errors="coerce")
  market=market.loc[market["date"].between(pd.Timestamp("2021-01-01"),pd.Timestamp("2025-12-31"))&
    market["series"].isin(("EQ","BE","BZ"))&market["symbol"].ne("")].copy()
  market["series_priority"]=market["series"].map({"EQ":0,"BE":1,"BZ":2})
  # Multiple observations on the same company/date with conflicting first-
  # priority quotes cannot be resolved by using future prices or outcomes.
  duplicates=market.duplicated(["symbol","date","series"],keep=False)
  problematic=market.loc[duplicates].groupby(["symbol","date","series"])["close"].nunique().gt(1)
  badkeys=set((str(a),pd.Timestamp(b)) for a,b,_ in problematic[problematic].index)
  market=market.sort_values(["symbol","date","series_priority"]).drop_duplicates(
     ["symbol","date"],keep="first")
  market["conflicting_quote_series"]=list(zip(market["symbol"],market["date"]))
  market["conflicting_quote_series"]=market["conflicting_quote_series"].isin(badkeys)
  market=market.drop(columns="series_priority")
  corp=actions.copy()
  if not {"symbol","type","ex_date"}.issubset(corp):raise ValueError("Corporate action source missing required split/bonus ex-date")
  corp["symbol"]=corp["symbol"].astype(str).str.upper().str.strip()
  corp["type"]=corp["type"].astype(str).str.lower().str.strip()
  corp=corp[corp["type"].isin(CORP_TYPES)].copy()
  corp["ex_date"]=pd.to_datetime(corp["ex_date"],errors="coerce").dt.normalize()
  unresolved=set(corp.loc[corp["ex_date"].isna(),"symbol"])
  corp=corp[corp["ex_date"].notna()].copy()
  frozen=original.copy()
  needed={"date","symbol","close"}
  if not needed.issubset(frozen):raise ValueError("Frozen market must provide only original stock-date/close")
  if any(x in frozen for x in ("y6","y12","y24","y6_mature_date","dd30_6m")):
     raise ValueError("No future stock outcomes may enter RSI source audit")
  frozen["date"]=pd.to_datetime(frozen["date"],errors="coerce").dt.strftime("%Y-%m-%d")
  frozen=frozen[frozen["date"].isin(DAYS)].copy()
  frozen["symbol"]=frozen["symbol"].astype(str).str.upper().str.strip()
  frozen["close"]=pd.to_numeric(frozen["close"],errors="coerce")
  if frozen.duplicated(["date","symbol"]).any() or len(frozen)!=sum(UNIVERSES.values()):
     raise ValueError("Expected exactly 9,912 unchanged original frozen stock-date IDs")
  for day,count in UNIVERSES.items():
   if frozen["date"].eq(day).sum()!=count:raise ValueError("Original immutable fold universe changed")
  return market,corp,unresolved,frozen

def extract_original_rsi(prices,actions,original):
 market,corp,unresolved,frozen=normalize_sources(prices,actions,original)
 interested=set(frozen["symbol"])
 market=market.loc[market["symbol"].isin(interested)].copy()
 grouped={k:z.sort_values("date").set_index("date",drop=False) for k,z in market.groupby("symbol",sort=False)}
 action_groups={k:z["ex_date"].to_numpy(dtype="datetime64[ns]") for k,z in corp.groupby("symbol",sort=False)}
 out=[];summaries=[]
 for day in DAYS:
  timestamp=pd.Timestamp(day)
  snap=frozen.loc[frozen["date"].eq(day)]
  for symbol,ref_close in snap[["symbol","close"]].itertuples(index=False,name=None):
   verdict="UNKNOWN_SOURCE_FAILURE"
   rsi=np.nan
   hist=grouped.get(symbol)
   if hist is None or timestamp not in hist.index:
    verdict="MISSING_ORIGINAL_HISTORICAL_MARKET_SESSION_CLOSE"
   else:
    curr=hist.loc[timestamp]
    source_close=float(curr["close"])
    if not np.isfinite(ref_close) or not np.isfinite(source_close) or ref_close<=0 or source_close<=0:
     verdict="INVALID_FROZEN_OR_ORIGINAL_MARKET_CLOSE"
    elif abs(ref_close-source_close)>max(CLOSE_TOL_MIN_INR,CLOSE_TOL_REL*ref_close):
     verdict="ORIGINAL_HF_SOURCE_CLOSE_NOT_EQUAL_FROZEN_FOLD_CLOSE"
    elif symbol in unresolved:
     verdict="UNRESOLVED_SPLIT_OR_BONUS_EX_DATE_IN_SOURCE"
    else:
     oldest=timestamp-pd.Timedelta(days=RSI_SOURCE_WINDOW_DAYS)
     recent=hist.loc[hist["date"].between(oldest,timestamp)].copy()
     last=recent.tail(RSI_SESSION_COUNT)
     # Do not fold over unknown corporate-action-adjusted prices. A
     # confirmed past ex-date inside rolling source is a genuine boundary.
     action_dates=action_groups.get(symbol, np.array([],dtype="datetime64[ns]"))
     in_window=((action_dates>=np.datetime64(last["date"].iloc[0]))&
                (action_dates<=np.datetime64(timestamp))).any() if len(last) else False
     if len(last)<RSI_SESSION_COUNT:
      verdict="INSUFFICIENT_120_RECENT_VALID_SESSIONS"
     elif recent["conflicting_quote_series"].any():
      verdict="MULTIPLE_CONFLICTING_PRICE_SERIES"
     elif (last["close"].isna() | last["close"].le(0)).any():
      verdict="BAD_ORIGINAL_NSE_DAILY_CLOSES"
     elif last["isin"].nunique()!=1 or not last["isin"].iloc[0].startswith("INE"):
      verdict="ISIN_CHANGED_OR_NON_EQUITY_DURING_RSI_WINDOW"
     elif last["date"].diff().dt.days.iloc[1:].gt(MAX_SESSION_GAP_DAYS).any():
      verdict="SUSPENSION_OR_LARGE_HISTORIC_DAILY_PRICE_GAP"
     elif (last["date"]>=timestamp-pd.Timedelta(days=RECENT_QUOTE_WINDOW_DAYS)).sum()<MIN_RECENT_QUOTES:
      verdict="RECENT_TRADING_NOT_CONTINUOUS_ENOUGH"
     elif in_window:
      verdict="SPLIT_OR_BONUS_IN_RSI_WINDOW_PIT_ADJUSTMENT_UNVERIFIED"
     else:
      rsi=rsi_wilder_last(last["close"].astype(float).to_numpy(),period=14)
      verdict="STRICT_ORIGINAL_120_SESSION_RSI70_AVAILABLE" if np.isfinite(rsi) else "RSI_CALCULATION_UNAVAILABLE"
   out.append({"date":day,"symbol":symbol,"rsi14_wilder_source":rsi,
               "fourth_family_rsi_strictly_gt70":bool(rsi>70) if np.isfinite(rsi) else pd.NA,
               "source_verification":verdict,"original_frozen_market_close_INR":ref_close})
  section=out[-len(snap):]
  avail=sum(rec["source_verification"]=="STRICT_ORIGINAL_120_SESSION_RSI70_AVAILABLE" for rec in section)
  passing=sum(rec["fourth_family_rsi_strictly_gt70"] is True for rec in section)
  fail_counts=pd.Series([rec["source_verification"] for rec in section]).value_counts().to_dict()
  summaries.append({"fold":day,"full_original_stock_count":len(snap),
   "120_session_original_RSI14_verified_count":avail,"strict_RSI14_gt70_count":passing,
   "verified_RSI_percent":round(100*avail/len(snap),2),
   "missing_price_or_adjustment_reasons":{str(k):int(v) for k,v in fail_counts.items()}})
 full=pd.DataFrame(out)
 if full.duplicated(["date","symbol"]).any() or len(full)!=9912:
  raise ValueError("Original 8fold RSI source-only panel key drift")
 report={"scope":"ORIGINAL_NSE_MARKET_8FOLD_ASOF_WILDER_RSI14_STRICT_GREATER_THAN_70",
   "original_stock_date_rows":len(full),
   "Wilder_RSI_period":14,"user_new_strict_threshold":70,
   "minimum_rolling_original_trading_sessions":RSI_SESSION_COUNT,
   "minimum_historic_daily_source_quality_no_forward_adjustments":True,
   "source_close_matches_immutable_frozen_market":True,
   "split_bonus_with_unverifiable_asof_adjustment_not_silently_used":True,
   "original_symbol_ISIN_instrument_continuity_checked":True,
   "all_unverified_RSI_results_are_UNKNOWN_not_FALSE":True,
   "original_2026_forward_top10_unchanged":True,
   "labels_news_or_future_prices_read":False,
   "does_not_evaluate_6month_actual_multibaggers":True,
   "production_predictor_or_historical_model_weights_modified":False,
   "per_fold":summaries}
 return full,report

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--market",required=True,nargs="+")
 p.add_argument("--actions",required=True,nargs="+")
 p.add_argument("--original-market-snapshot",required=True)
 p.add_argument("--out",required=True)
 a=p.parse_args()
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 prices=[];acts=[];hashes={}
 for f in a.market:
  path=Path(f);hashes["market/"+path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
  prices.append(pd.read_parquet(path,columns=["date","symbol","series","isin","close"]))
 for f in a.actions:
  path=Path(f);hashes["actions/"+path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
  acts.append(pd.read_parquet(path))
 snapshot=pd.read_parquet(a.original_market_snapshot,columns=["date","symbol","close"])
 private,summary=extract_original_rsi(pd.concat(prices,ignore_index=True),pd.concat(acts,ignore_index=True),snapshot)
 private.to_parquet(out/"v11_4_strict_8fold_RSI14_gt70_original_source_ONLY_PRIVATE.parquet",
    compression="zstd",index=False)
 summary["original_HF_price_and_corporate_action_source_SHA256"]=hashes
 (out/"v11_4_8fold_RSI70_strict_daily_source_quality_aggregate.json").write_text(json.dumps(summary,indent=2))
 print(json.dumps({k:v for k,v in summary.items() if k!="original_HF_price_and_corporate_action_source_SHA256"},indent=2),flush=True)
if __name__=="__main__":main()
