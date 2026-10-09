"""Pure source-verified eight-fold three-FY+NSE-price+catalyst feature matrix.

NO future labels, no stock predictions, no model fitting. Original eight 2022
to 2025 June/December NSE decision dates. Source-verified three-FY numbers
have only two YoY growth intervals, not user-requested literal 3y CAGR.
NSE catalysts require original as-of exchange availability and raw market
history must be original and clean. 2022 June's 69.83% is BELOW 70% gate
and remains explicitly flagged, not silently discarded or imputed.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
FOLDS={
 "2022-06-30":((2020,2021,2022),759,"june"),
 "2022-12-30":((2020,2021,2022),831,"december"),
 "2023-06-30":((2021,2022,2023),830,"june"),
 "2023-12-29":((2021,2022,2023),960,"december"),
 "2024-06-28":((2022,2023,2024),976,"june"),
 "2024-12-31":((2022,2023,2024),1068,"december"),
 "2025-06-30":((2023,2024,2025),918,"june"),
 "2025-12-31":((2023,2024,2025),1013,"december")
}
EXPECT_UNIVERSE={"2022-06-30":1087,"2022-12-30":1123,"2023-06-30":1147,
 "2023-12-29":1276,"2024-06-28":1322,"2024-12-31":1345,
 "2025-06-30":1298,"2025-12-31":1314}
PRICE=("ret_20","ret_60","ret_120","ret_252","mom_accel","vol_accel",
 "turnover_accel","off_high_252","above_low_252","trend_consistency_60",
 "volatility_60","avg_turnover_63","integrity_feature_clean")
EVENTS=("nse_capacity_expansion_180d_positive","nse_order_win_180d_positive",
 "nse_promoter_activity_180d_positive","nse_regulatory_180d_positive",
 "nse_earnings_180d_positive","nse_corporate_action_180d_positive",
 "nse_catalyst_total_180d","nse_dilution_180d")
FIN_FEATS=("f3_sales_two_year_CAGR","f3_sales_previous_yoy",
 "f3_sales_latest_yoy","f3_sales_acceleration",
 "f3_profit_two_year_CAGR","f3_profit_latest_yoy",
 "f3_profit_margin_latest","f3_profit_margin_change",
 "f3_positive_PAT_all_three_FY")
FORBIDDEN={"y6","y12","y24","y6_mature_date","dd30_6m","days_to_2x",
 "hit25_6m","hit50_6m","exploratory_p6_double_model_probability"}
def local_1530(fold):
 return pd.Timestamp(fold).tz_localize("Asia/Kolkata")+pd.Timedelta(hours=15,minutes=30)
def genuine_threeFY(source,fold,base_symbols):
 yrs,expected,typ=FOLDS[fold]
 cols=["symbol"]+[f"FY{y}_{v}_INR" for y in yrs for v in ("revenue","PAT")]
 if typ=="june":
  cols +=["verified_3FY_before_actual_June_fold_close"]
  q=pd.read_parquet(source,columns=["date",*cols])
  q=q.loc[q["verified_3FY_before_actual_June_fold_close"].eq(True),cols].copy()
 else:
  q=pd.read_csv(source)
  if not set(cols).issubset(q):raise ValueError("Annual source facts missing original fiscal year")
  t=local_1530(fold).tz_convert("UTC")
  for yy in yrs:
   pub=pd.to_datetime(q[f"FY{yy}_published_utc"],utc=True,errors="coerce")
   if pub.isna().any() or (pub>t).any():
    raise ValueError("Historical December original annual source published after decision")
   hashes=[c for c in q.columns if c.lower() in (f"fy{yy}_sha256",f"fy{yy}_source_sha256")]
   if len(hashes)!=1 or not q[hashes[0]].astype(str).str.fullmatch("[a-f0-9]{64}").all():
    raise ValueError("Original December fiscal XML source SHA256 invalid")
  q=q[cols].copy()
 q["symbol"]=q["symbol"].astype(str).str.upper().str.strip()
 if q["symbol"].duplicated().any():raise ValueError("Original issuer fiscal duplicate")
 if len(q)!=expected:raise ValueError(f"Source-verified old stocks count changed at {fold}: {len(q)} != {expected}")
 if not set(q["symbol"]).issubset(base_symbols):raise ValueError("Issuer not in historical as-of NSE market")
 for column in cols[1:]:
  if column.startswith("FY"):
   q[column]=pd.to_numeric(q[column],errors="coerce")
 if q[[c for c in cols if c.startswith("FY")]].isna().any().any():
  raise ValueError("Verified original annual fiscal company has absent number")
 for y in yrs:
  if not q[f"FY{y}_revenue_INR"].gt(0).all():
   raise ValueError("Invalid source revenue baseline")
 (a,b,c)=yrs
 sa=q[f"FY{a}_revenue_INR"];sb=q[f"FY{b}_revenue_INR"];sc=q[f"FY{c}_revenue_INR"]
 pa=q[f"FY{a}_PAT_INR"];pb=q[f"FY{b}_PAT_INR"];pc=q[f"FY{c}_PAT_INR"]
 out=pd.DataFrame({"symbol":q["symbol"]})
 out["f3_sales_two_year_CAGR"]=np.sqrt(sc/sa)-1
 out["f3_sales_previous_yoy"]=sb/sa-1
 out["f3_sales_latest_yoy"]=sc/sb-1
 out["f3_sales_acceleration"]=out["f3_sales_latest_yoy"]-out["f3_sales_previous_yoy"]
 out["f3_profit_two_year_CAGR"]=np.where((pa>0)&(pc>0),np.sqrt((pc/pa).clip(lower=0))-1,np.nan)
 out["f3_profit_latest_yoy"]=np.where(pb>0,pc/pb-1,np.nan)
 out["f3_profit_margin_latest"]=pc/sc
 out["f3_profit_margin_change"]=pc/sc-pb/sb
 out["f3_positive_PAT_all_three_FY"]=((pa>0)&(pb>0)&(pc>0)).astype(int)
 for name in FIN_FEATS:
  out[name]=pd.to_numeric(out[name],errors="coerce").replace([np.inf,-np.inf],np.nan)
 out["f3_provenance_original_fiscal_FY"]="|".join(map(str,yrs))
 out["date"]=fold
 return out
def combine(market_feature_matrix,financial_paths):
 original=market_feature_matrix.copy()
 if FORBIDDEN&set(original):raise ValueError("Future outcome label in historical market NSE feature input")
 if not {"date","symbol","historical_asof_utc",*PRICE,*EVENTS}.issubset(original):
  raise ValueError("Missing required original NSE PIT price/catalyst source features")
 original["date"]=pd.to_datetime(original["date"],errors="raise").dt.strftime("%Y-%m-%d")
 original["symbol"]=original["symbol"].astype(str).str.upper().str.strip()
 if original.duplicated(["date","symbol"]).any():raise ValueError("Duplicate original market event features")
 base=original.loc[original["date"].isin(FOLDS),["date","symbol","historical_asof_utc",*PRICE,*EVENTS]].copy()
 if len(base)!=sum(EXPECT_UNIVERSE.values()):raise ValueError("Original original 8fold stock identities lost")
 if base["integrity_feature_clean"].isna().any():
  raise ValueError("Original NSE stock price feature quality unknown")
 cut=pd.to_datetime(base["date"]).dt.tz_localize("Asia/Kolkata")+pd.Timedelta(hours=15,minutes=30)
 pub=pd.to_datetime(base["historical_asof_utc"],utc=True,errors="coerce",format="mixed")
 if pub.isna().any() or pub.gt(cut.dt.tz_convert("UTC")).any():
  raise ValueError("Original NSE catalyst metadata violates historical information clock")
 reports=[];items=[]
 for date in FOLDS:
  fold=base.loc[base["date"].eq(date)].copy()
  if len(fold)!=EXPECT_UNIVERSE[date]:raise ValueError("Historical original source stock count drift: "+date)
  f=genuine_threeFY(financial_paths[date],date,set(fold["symbol"]))
  combined=fold.merge(f,on=["date","symbol"],how="left",validate="1:1")
  if len(combined)!=len(fold):raise ValueError("Annual inner-filtering improperly dropped market stocks")
  verified=int(combined["f3_provenance_original_fiscal_FY"].notna().sum())
  if verified!=FOLDS[date][1]:raise ValueError("Strict original company 3FY annual coverage changed")
  combined["has_strict_3FY_original_fiscal_source"]=combined["f3_provenance_original_fiscal_FY"].notna()
  coverage=100*verified/len(combined)
  reports.append({"fold":date,"original_NSE_company_rows":len(combined),
   "source_verified_exactly_three_fiscal_annual_years":verified,
   "source_coverage_percent":round(coverage,2),
   "required_70percent_source_numeric_gate_met":bool(coverage>=70),
   "fiscal_vintage":FOLDS[date][0],
   "two_fiscal_growth_intervals_not_three":True})
  items.append(combined)
 matrix=pd.concat(items,ignore_index=True)
 if FORBIDDEN&set(matrix):raise ValueError("Forward outcome or prior historical model probability leaked")
 if matrix.duplicated(["date","symbol"]).any():raise ValueError("Duplicate original securities in combined research")
 missing=matrix.loc[~matrix["has_strict_3FY_original_fiscal_source"]]
 if missing[list(FIN_FEATS)].notna().any().any():
  raise ValueError("Missing historical three fiscal years improperly imputed to zero")
 report={
  "scope":"8FOLD_STRICT_THREE_FY_FINANCIAL_MARKET_PRICE_OFFICIAL_CATALYST_PIT_FEATURES_ONLY",
  "source_verified_original_8_decision_dates":len(FOLDS),
  "full_original_market_stock_date_rows":len(matrix),
  "original_NSE_strict_3FY_company_date_rows":int(matrix["has_strict_3FY_original_fiscal_source"].sum()),
  "fiscal_year_observations_per_company_date":3,
  "years_of_growth_between_fiscal_endpoints":2,
  "historical_NSE_catalyst_metadata_exposures":list(EVENTS),
  "unverified_wilder_RSI14_source_in_this_matrix_omitted":True,
  "2022_June_70pct_coverage_threshold_passed":reports[0]["required_70percent_source_numeric_gate_met"],
  "original_missing_company_financial_fields_left_NaN":True,
  "no_future_six_month_labels_or_prices_in_feature_matrix":True,
  "no_trained_new_market_financial_catalyst_model":True,
  "already_seen_2024_2025_test_company_outcomes_NOT_joined":True,
  "source_no_retroactive_prediction_tuning":True,
  "all_original_oct2026_sealed_stock_predictions_untouched":True,
  "folds":reports}
 return matrix,report
def main():
 p=argparse.ArgumentParser()
 p.add_argument("--market-catalyst",required=True)
 for date in FOLDS:p.add_argument("--financial-"+date,required=True)
 p.add_argument("--out",required=True);args=p.parse_args()
 paths={date:getattr(args,"financial_"+date.replace("-","_")) for date in FOLDS}
 raw=pd.read_parquet(args.market_catalyst)
 combined,summary=combine(raw,paths)
 out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
 combined.to_parquet(out/"NSE_original_eightfold_threeFY_PIT_market_and_catalyst_NOLABEL_RESEARCH.parquet",
  index=False,compression="zstd")
 (out/"eight_folds_original_3FY_market_catalyst_source_summary.json").write_text(json.dumps(summary,indent=2))
 print(json.dumps(summary,indent=2),flush=True)
 print(pd.DataFrame(summary["folds"]).drop(columns=["fiscal_vintage"]).to_string(index=False),flush=True)
if __name__=="__main__":main()
