"""Source-only RSI(14)>70 conditioned on contemporaneous official NSE fold-day close.

Requires PRIVATE original eightfold (3 FY + RSI14 + NSE filing metadata)
and independently obtained PRIVATE original NSE two archive-family close
verification. Source-missing securities are UNKNOWN and are never removed
from the historical original denominator. NO future outcomes/ML ranks.
This is still NOT 120-session independently verified RSI or a predictor.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from v11_4_eightfold_3FY_market_catalyst_source_matrix import EXPECT_UNIVERSE,FIN_FEATS

NO_OUTCOMES={"y6","y12","y24","y6_mature_date","dd30_6m",
 "p6_double_calibrated","exploratory_p6_double_combined_research_only",
 "v11_4_p2x_calibrated","research_rank","future_close"}

def strict_pair(panel,name):
 x=panel.copy()
 if NO_OUTCOMES&set(x):
  raise ValueError(f"{name}: future target or fitted stock pick input forbidden")
 if not {"date","symbol"}.issubset(x):
  raise ValueError(f"{name}: missing frozen stock/date identity")
 x["date"]=pd.to_datetime(x["date"],errors="raise").dt.strftime("%Y-%m-%d")
 x["symbol"]=x["symbol"].astype(str).str.upper().str.strip()
 if x[["date","symbol"]].duplicated().any():
  raise ValueError(f"{name}: duplicate historical company-date")
 if len(x)!=sum(EXPECT_UNIVERSE.values()) or x.groupby("date").size().to_dict()!=EXPECT_UNIVERSE:
  raise ValueError(f"{name}: immutable original eightfold market population changed")
 return x

def annotate(source,exchange):
 f=strict_pair(source,"RSI/financial/NSE old source-only")
 q=strict_pair(exchange,"official NSE dual close original stock-specific archive")
 need_source={"close","rsi14_wilder_source","rsi14_Wilder_research_verified",
  "rsi14_strict_gt70_source_verified","promoter_buying_verified_180d",
  "has_strict_3FY_original_fiscal_source","promoter_original_180d_filing_count",
  "nse_material_specific_positive_present",*FIN_FEATS}
 if not need_source.issubset(f):raise ValueError("Prior immutable 3FY + RSI70 source quality slot missing")
 needs_exchange={"official_dual_day_close_verified","official_close_validation_status",
  "original_frozen_close_INR","official_primary_close_INR","official_full_close_INR"}
 if not needs_exchange.issubset(q):raise ValueError("Two original NSE archive-family daily close columns unavailable")
 if set(zip(f["date"],f["symbol"]))!=set(zip(q["date"],q["symbol"])):
  raise ValueError("Official NSE archived company dates do not equal 8fold source companies")
 joined=f.merge(q[["date","symbol",*sorted(needs_exchange)]],on=["date","symbol"],
               how="left",validate="1:1")
 if len(joined)!=len(f):raise ValueError("Historical source-only population filtered")
 close=pd.to_numeric(joined["close"],errors="coerce")
 quote=pd.to_numeric(joined["original_frozen_close_INR"],errors="coerce")
 if close.isna().any() or quote.isna().any() or not np.allclose(close,quote,rtol=0,atol=1e-6):
  raise ValueError("Original frozen stock prices differ across archived sources")
 old_rs=pd.to_numeric(joined["rsi14_wilder_source"],errors="coerce")
 old_good=joined["rsi14_Wilder_research_verified"].eq(True)
 official_good=joined["official_dual_day_close_verified"].eq(True)
 if official_good.sum()!=9621 and len(joined)==9912:
  raise ValueError("Exchange official strict 8fold day-close verified stock count changed")
 if joined["promoter_buying_verified_180d"].notna().any():
  raise ValueError("Unverified promoter activity must NOT become confirmed shares bought")
 if ((~old_good)&old_rs.notna()).any() or ((old_good)&(~old_rs.between(0,100))).any():
  raise ValueError("Previous Wilder RSI source provenance not trustworthy")
 raw_signal=joined["rsi14_strict_gt70_source_verified"].astype("boolean")
 if raw_signal.loc[old_good].isna().any() or raw_signal.loc[~old_good].notna().any():
  raise ValueError("Historical RSI source UNKNOWN silently encoded as pass/fail")
 if not (raw_signal.loc[old_good]==old_rs.loc[old_good].gt(70).to_numpy()).all():
  raise ValueError("User strict Wilder RSI>70 threshold changed")
 certified=old_good&official_good
 joined["fold_day_official_NSE_dual_archive_price_confirmed"]=official_good
 joined["RSI_120session_archive_available_PLUS_official_day_close"]=certified
 joined["RSI14_STRICT_GT70_when_official_NSE_DAY_close_confirmed"]=pd.Series(
    np.where(certified,old_rs.gt(70),pd.NA),index=joined.index,dtype="boolean")
 joined["fourth_screener_RSI70_official_day_close_shadow_status"]=np.where(
    certified,np.where(old_rs.gt(70),"PASS","FAIL"),"UNKNOWN")
 # Unverified NSE daily historical series is NOT solved by corroborating
 # only the last close; preserve public scientific uncertainty.
 joined["complete_120day_Wilder_RSI_source_official_NSE_certified"]=False
 if NO_OUTCOMES&set(joined):raise ValueError("Forward y6 outcome leaked into official RSI source layer")
 results=[]
 for d in EXPECT_UNIVERSE:
  x=joined.loc[joined["date"].eq(d)]
  present=x["RSI_120session_archive_available_PLUS_official_day_close"]
  hit=x["RSI14_STRICT_GT70_when_official_NSE_DAY_close_confirmed"].fillna(False)
  results.append({"frozen_fold":d,
   "all_historical_original_stock_rows":len(x),
   "threeFY_original_numeric_source_stocks":int(x["has_strict_3FY_original_fiscal_source"].eq(True).sum()),
   "two_official_NSE_archive_DAY_CLOSE_matching_stocks":int(x["fold_day_official_NSE_dual_archive_price_confirmed"].sum()),
   "Wilder_RSI_120sessions_raw_archive_also_has_official_DAY_CLOSE":int(present.sum()),
   "RSI14_GT70_in_archive_and_official_DAY_CLOSE":int(hit.sum()),
   "RSI14_GT70_3FY_available_and_official_DAY_CLOSE":int((hit&x["has_strict_3FY_original_fiscal_source"].eq(True)).sum()),
   "RSI_unknown_no_imputed_zero":int((~present).sum()),
   "no_six_month_forward_returns_or_doublers_loaded":True})
 report={"scope":"ORIGINAL_8FOLD_NSE_DAY_CLOSE_PROOF_AND_RSI14_GT70_SOURCE_ONLY_SHADOW",
  "total_original_stocks_dates":len(joined),
  "strict_3FY_original_source_company_dates":int(joined["has_strict_3FY_original_fiscal_source"].sum()),
  "NSE_original_day_close_confirmed_from_dual_exchange_archive_file_families":int(official_good.sum()),
  "Wilder_RSI_original_daily_archive_available":int(old_good.sum()),
  "Wilder_RSI_120day_archive_and_original_day_close_verified":int(certified.sum()),
  "RSI14_GT70_archived_original_source_before_official_day_recheck":int(raw_signal.fillna(False).sum()),
  "RSI14_GT70_AND_official_original_date_close_reconfirmed":int(joined["RSI14_STRICT_GT70_when_official_NSE_DAY_close_confirmed"].fillna(False).sum()),
  "original_NSE_daily_archive_120_session_history_independently_certified":False,
  "unknown_asof_inputs_not_treated_as_fail_or_dubious_RSI_70_pass":True,
  "promoter_purchase_document_direction_still_UNKNOWN":True,
  "original_frozen_V11_4_Oct2026_ranks_unchanged":True,
  "no_model_optimization_or_new_out_of_sample_accuracy_claimed":True,
  "per_fold":results}
 return joined,report

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--source",required=True)
 p.add_argument("--official-exchange-day",required=True)
 p.add_argument("--out",required=True)
 a=p.parse_args()
 f=pd.read_parquet(a.source)
 x=pd.read_parquet(a.official_exchange_day)
 assembled,report=annotate(f,x)
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 assembled.to_parquet(out/"PRIVATE_original_8fold_3FY_RSI70_official_DAY_only_no_future_labels.parquet",
                      index=False,compression="zstd")
 (out/"original_8fold_NSE_two_archive_close_RSI70_source_ready_aggregate.json").write_text(
   json.dumps(report,indent=2))
 print(json.dumps(report,indent=2),flush=True)
if __name__=="__main__":main()
