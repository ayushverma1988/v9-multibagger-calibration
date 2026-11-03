"""Independent frozen 18-fold V11.4 backtest-READINESS checks.

Join NSE catalyst metadata to immutable historical V10 labels strictly by
(date,symbol). Six-month outcomes are for DIAGNOSTICS ONLY and must never
enter any derived feature or ranker. No model fitting or promotion here.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd

DROP_LABELS={"y6","y12","y24","y6_mature_date","y12_mature_date",
             "y24_mature_date","integrity_y6_clean","integrity_y12_clean",
             "integrity_y24_clean","days_to_2x","hit25_6m","hit50_6m",
             "dd30_6m","dd30_6m_mature_date","threshold_mature_date"}

def fold_close(v):
 return (pd.Timestamp(v).normalize().tz_localize("Asia/Kolkata")
         +pd.Timedelta(hours=15,minutes=30)).tz_convert("UTC")

def validate_feature_names(frame):
 forbidden=DROP_LABELS.intersection(frame.columns)
 if forbidden:raise ValueError(f"Forward outcome/label leaked into predictor features: {sorted(forbidden)}")
 if "historical_asof_utc" not in frame:raise ValueError("Missing historical source cutoff evidence")
 return True

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--snapshot",required=True)
 p.add_argument("--catalyst-features",required=True)
 p.add_argument("--folds",required=True)
 p.add_argument("--output",required=True)
 a=p.parse_args();out=Path(a.output);out.mkdir(exist_ok=True,parents=True)
 feat=pd.read_parquet(a.catalyst_features)
 validate_feature_names(feat)
 snap=pd.read_parquet(a.snapshot)
 if not {"date","symbol","y6","y6_mature_date","integrity_y6_clean"}.issubset(snap):
  raise SystemExit("Frozen snapshot missing original 6m target maturity or integrity fields")
 snap["date"]=pd.to_datetime(snap["date"],errors="coerce").dt.normalize()
 snap["symbol"]=snap["symbol"].astype(str).str.upper().str.strip()
 feat["date"]=pd.to_datetime(feat["date"],errors="coerce").dt.normalize()
 feat["symbol"]=feat["symbol"].astype(str).str.upper().str.strip()
 asof=pd.to_datetime(feat["historical_asof_utc"],utc=True,format="mixed",errors="coerce")
 expected=feat["date"].map(fold_close)
 if (asof!=expected).any():
  raise SystemExit(f"{int((asof!=expected).sum())} catalyst as-of dates do not match frozen 15:30 close")
 if feat.duplicated(["date","symbol"]).any():raise SystemExit("Duplicate catalyst stock/date feature")
 fdates=set(pd.to_datetime(pd.read_csv(a.folds)["date"],errors="coerce").dropna().dt.normalize())
 if len(fdates)!=18 or set(feat["date"])!=fdates:
  raise SystemExit("Not identical original 18 historical selection dates")
 base=snap[snap["date"].isin(fdates)].copy()
 if base.duplicated(["date","symbol"]).any():raise SystemExit("Original snapshot duplicate stock-date")
 if len(feat)!=len(base):raise SystemExit("Catalyst source dropped stocks from historical universe")
 joined=base[["date","symbol","y6","y6_mature_date","integrity_y6_clean"]].merge(
  feat,on=["date","symbol"],how="left",validate="1:1",indicator=True)
 if not joined["_merge"].eq("both").all():
  raise SystemExit("Original stock-date rows have missing catalyst source rows")
 mat=pd.to_datetime(joined["y6_mature_date"],errors="coerce",utc=True,format="mixed")
 expected_maturity=joined["date"].dt.tz_localize("UTC")
 invalid_maturity=(mat.notna())&(mat<=expected_maturity)
 if invalid_maturity.any():raise SystemExit("Six month maturity date before historical fold")
 y6=pd.to_numeric(joined["y6"],errors="coerce")
 clean=joined["integrity_y6_clean"].fillna(False).astype(bool)
 report=[]
 for d,part in joined.groupby("date"):
  label=pd.to_numeric(part["y6"],errors="coerce")
  accepted=part["integrity_y6_clean"].fillna(False).astype(bool)&label.notna()
  report.append({
   "fold":str(d.date()),"original_stocks":len(part),
   "y6_label_nonnull":int(label.notna().sum()),
   "integrity_clean_y6_labels":int(accepted.sum()),
   "original_y6_positive_label_values":int((label[accepted]>0).sum()),
   "positive_label_rate_if_binary":float((label[accepted]>0).mean()) if accepted.any() else None,
   "catalyst_180d_positive_count":int((part["nse_catalyst_total_180d"]>0).sum()),
   "nse_order_wins_180d_count":int(part["nse_order_win_180d"].sum()),
   "nse_capacity_expansion_180d_count":int(part["nse_capacity_expansion_180d"].sum()),
  })
 df=pd.DataFrame(report)
 df.to_csv(out/"original_18fold_label_and_catalyst_readiness.csv",index=False)
 summary={
  "scope":"READINESS_ONLY_NO_PREDICTION_FITTING",
  "historical_folds":len(report),"rows":len(joined),
  "total_original_y6_clean_labels":int(clean.mul(y6.notna()).sum()),
  "frozen_stock_universe_preserved":True,
  "post_close_feature_leak_rows":int((asof!=expected).sum()),
  "model_training_executed":False,
  "v11_4_accuracy_vs_v10_measured":False,
  "future_labels_used_as_features":False,
  "original_v10_not_modified":True,
  "fold_diagnostics":report,
 }
 (out/"18fold_catalyst_label_readiness_summary.json").write_text(json.dumps(summary,indent=2))
 print(json.dumps({k:v for k,v in summary.items() if k!="fold_diagnostics"},indent=2))
 print(df.tail(6).to_string(index=False),flush=True)
 if summary["total_original_y6_clean_labels"]<5000:
  raise SystemExit("Insufficient original 6-month clean validation outcomes for future performance backtest")
if __name__=="__main__":main()
