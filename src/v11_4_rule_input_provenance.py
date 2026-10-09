"""V11.4 four-family PIT financial source bridge and coverage audit.

This is a RESEARCH-ONLY companion to the immutable 22-input predictor.
It will NOT synthesize missing annual/quarterly data, silently infer market
capitalization, or use a financial filing published after exchange close.
Only original NSE/BSE source files with explicit source-time and units are
accepted. For current live feeds missing fields remain NA/UNKNOWN.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
from urllib.parse import urlparse
import numpy as np
import pandas as pd
from v11_4_standalone_train_walkforward import fold_close

HOSTS={"www.nseindia.com","nseindia.com","nsearchives.nseindia.com",
       "archives.nseindia.com","www.bseindia.com","bseindia.com"}
BOOL_FIELDS={"is_sme","pe_lt_industry_pe","price_gt_dma50_prev",
             "price_lt_dma200_prev","sales_latest_ge_yoy_quarter",
             "sales_latest_ge_2q_back","pat_latest_gt_preceding",
             "pat_preceding_gt_2q_back"}
EXCHANGE_FIELDS={"rsi14_wilder","up_from_52w_low","down_from_52w_high",
                 "price_gt_dma50_prev","price_lt_dma200_prev","is_sme"}
SOURCE_COLUMNS={"symbol","metric","value","source_available_utc",
                "source_url","source_verified","metric_units","reporting_mode"}

def allowed(url):
 try:
  p=urlparse(str(url))
  return p.scheme=="https" and p.hostname in HOSTS
 except Exception:return False

def parse_value(metric,val,units):
 if pd.isna(val):return np.nan
 if metric in BOOL_FIELDS:
  if isinstance(val,(bool,np.bool_)):return bool(val)
  v=str(val).strip().lower()
  return True if v=="true" else False if v=="false" else np.nan
 if units not in {"fraction","ratio","crore_inr","wilder_0_100"}:
  return np.nan
 try:v=float(str(val).replace(",",""))
 except (ValueError,TypeError):return np.nan
 if not np.isfinite(v):return np.nan
 if metric in {"promoter_holding","pledged_pct"} and not 0<=v<=1:return np.nan
 if metric=="market_cap_crore" and v<=0:return np.nan
 return v

def source_gate(facts,asof,eligible_metrics):
 if not SOURCE_COLUMNS.issubset(facts):
  raise ValueError("Original XBRL/NSE source metrics provenance fields missing")
 x=facts.copy()
 x["symbol"]=x["symbol"].astype(str).str.upper().str.strip()
 x["metric"]=x["metric"].astype(str).str.strip()
 x["source_available_utc"]=pd.to_datetime(
     x["source_available_utc"],utc=True,errors="coerce",format="mixed")
 cutoff=fold_close(asof)
 x["source_provenance_good"]=x["source_url"].map(allowed)&x["source_verified"].eq(True)
 x["time_good"]=x["source_available_utc"].notna()&(x["source_available_utc"]<=cutoff)
 x["metric_good"]=x["metric"].isin(eligible_metrics)
 # Fiscal ratios must tie back to a fiscal period end. No present-day XBRL
 # values can be projected into earlier selection dates.
 fiscal=x["metric"].str.contains(
     "growth|opm|roce|roe|profit|sales|pat|equity|coverage|receivables|pledged|promoter|peg",
     case=False,regex=True)
 if "source_period_end_utc" not in x:
  x["source_period_end_utc"]=pd.NaT
 x["source_period_end_utc"]=pd.to_datetime(x["source_period_end_utc"],
     utc=True,errors="coerce",format="mixed")
 period_good=(~fiscal)|(x["source_period_end_utc"].notna()&
               (x["source_period_end_utc"]<=x["source_available_utc"]))
 x["value_clean"]=[parse_value(m,v,u) for m,v,u in zip(x["metric"],x["value"],x["metric_units"])]
 x["value_good"]=x["value_clean"].notna()
 x["eligible"]=x["source_provenance_good"]&x["time_good"]&x["metric_good"]&period_good&x["value_good"]
 valid=x.loc[x["eligible"]].copy()
 # Reject contradictory values at same publication instant; never pick an
 # arbitrary financial fact among two conflicting source documents.
 keys=["symbol","metric","source_available_utc"]
 ambiguous=valid.groupby(keys)["value_clean"].nunique(dropna=False)
 if (ambiguous>1).any():raise ValueError("Ambiguous same-timestamp company fact: block rather than overwrite")
 valid=valid.sort_values(["source_available_utc","source_url"]).drop_duplicates(
     ["symbol","metric"],keep="last")
 return valid,{"supplementary_input_fact_rows":len(x),
               "valid_original_point_in_time_metrics":len(valid),
               "rejected_provenance":int((~x["source_provenance_good"]).sum()),
               "rejected_future_or_missing_publication":int((~x["time_good"]).sum()),
               "rejected_incompatible_metric":int((~x["metric_good"]).sum()),
               "rejected_missing_or_invalid_fiscal_period":int((~period_good).sum()),
               "rejected_missing_or_wrong_numeric_unit":int((~x["value_good"]).sum())}

def enrich_and_audit(live,config,asof,facts=None):
 if not {"date","symbol","historical_asof_utc"}.issubset(live):
  raise ValueError("Cannot join supplemental fields without verified stock/date PIT keys")
 x=live.copy()
 if x[["date","symbol"]].duplicated().any():
  raise ValueError("Repeated security row may mix separate quarterly filings")
 dates=pd.to_datetime(x["date"],errors="coerce").dt.normalize()
 if dates.nunique()!=1 or dates.iloc[0]!=pd.Timestamp(asof).normalize():
  raise ValueError("Live current-market snapshot cannot be merged to historical filings")
 clock=pd.to_datetime(x["historical_asof_utc"],utc=True,errors="coerce",format="mixed")
 if clock.isna().any() or (clock!=fold_close(asof)).any():
  raise ValueError("Live PIT exchange cutoff mismatch")
 family=config["conditions"]
 fields=set(field for v in family.values() for field,_,_ in v["hard_rules"])
 optional=fields-EXCHANGE_FIELDS
 srcdiag={"supplementary_input_fact_rows":0,"valid_original_point_in_time_metrics":0}
 if facts is not None:
  valid,srcdiag=source_gate(facts,asof,optional)
  if len(valid):
   for metric,group in valid.groupby("metric"):
    if metric in x:
     # The technical metric is from today's original NSE close. Never
     # allow lower-confidence supplementary data to shadow that.
     raise ValueError(f"Supplementary source attempted to overwrite {metric}")
    indexed=group.set_index("symbol")["value_clean"]
    x[metric]=x["symbol"].astype(str).str.upper().str.strip().map(indexed)
 coverage=[]
 for key,details in family.items():
  for field,op,t in details["hard_rules"]:
   if field in x:
    values=x[field]
    available=int(values.notna().sum())
   else:available=0
   coverage.append({"family":key,"metric":field,
        "company_count":len(x),"verified_values":available,
        "company_coverage_ratio":available/max(1,len(x)),
        "source_group":"NSE_official_cash_price" if field in EXCHANGE_FIELDS
              else "verified_original_NSE_BSE_disclosure_required",
        "required_operation":op,
        "metric_unavailable":available==0})
 summary={
  "scope":"NON_PREDICTIVE_4_FAMILY_PIT_DATA_COVERAGE",
  "stock_rows":len(x),"asof_IST":str(pd.Timestamp(asof).date()),
  "screen_condition_count":len(family),"distinct_requested_metrics":len(fields),
  "fields_with_any_verified_observations":len(set(
       c["metric"] for c in coverage if c["verified_values"]>0)),
  "fields_still_fully_missing":sorted({c["metric"] for c in coverage if not c["verified_values"]}),
  "supplementary_source_metrics_audit":srcdiag,
  "no_future_filings_or_unverified_annual_facts_filled":True,
  "predictive_model_weights_and_training_unchanged":True,
  "research_overlay_only":True,
  "production_approved":False}
 return x,coverage,summary

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument("--features",required=True)
 ap.add_argument("--config",required=True)
 ap.add_argument("--metrics",default="")
 ap.add_argument("--asof",required=True)
 ap.add_argument("--output",required=True)
 a=ap.parse_args()
 conf=json.loads(Path(a.config).read_text())
 d=pd.read_parquet(a.features)
 facts=(pd.read_parquet(a.metrics) if a.metrics.endswith(".parquet") else pd.read_csv(a.metrics)) if a.metrics else None
 x,report,summary=enrich_and_audit(d,conf,a.asof,facts)
 target=Path(a.output);target.mkdir(parents=True,exist_ok=True)
 x.to_parquet(target/"v11_4_pit_four_family_research_overlay.parquet",index=False)
 pd.DataFrame(report).to_csv(target/"four_family_metric_source_coverage.csv",index=False)
 (target/"four_family_filing_pit_readiness.json").write_text(json.dumps(summary,indent=2))
 print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
