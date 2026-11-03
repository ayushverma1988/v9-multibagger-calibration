"""V11.4 optional research import of authorized manual Screener CSV screens.

Screener does not provide a public API and bulk CSV export is Premium; this
tool NEVER requests/scrapes Screener, impersonates accounts, or circumvents
download restrictions. Input must be a legitimately obtained user CSV.

Screener's CURRENT 3/5/7-year aggregates may describe the world now but
do not prove the values available at historical decision timestamps.
Outputs therefore stay entirely OUT of 18-fold training and frozen scores.
"""
from __future__ import annotations
import argparse,hashlib,json,re
from pathlib import Path
from datetime import datetime,timezone
import pandas as pd

# Ingestion columns reflect Screener's existing editable screening field names.
# Different versions may label these differently; reject ambiguous columns.
MAP={
 "nse_code":("nsecode","nsecodeequity","nsecodeforcompany"),
 "bse_code":("bsecode","bsecodeequity"),
 "sales_growth_5y":("salesgrowth5years","salesgrowth5year"),
 "sales_growth_3y":("salesgrowth3years","salesgrowth3year"),
 "opm_current":("opm","opmpercent","operatingprofitmargin"),
 "opm_5y_avg":("opm5year","opm5years","opm5yr"),
 "profit_growth_5y":("profitgrowth5years","profitgrowth5year"),
 "profit_growth_7y":("profitgrowth7years","profitgrowth7year"),
 "roce_7y_avg":("averagereturnoncapitalemployed7years","averageroce7years"),
 "roce_3y_avg":("averagereturnoncapitalemployed3years","averageroce3years"),
 "roe_3y_avg":("averagereturnonequity3years","averageroe3years"),
 "roe_5y_avg":("averagereturnonequity5years","averageroe5years"),
 "debt_to_equity":("debttoequity","debttoequityratio"),
 "interest_coverage":("interestcoverage","interestcoverageratio"),
 "promoter_holding":("promoterholding","promoterholdingpercent"),
 "pledged_pct":("pledgedpercentage","pledgedpct","pledgedpercentageofpromoterholding"),
 "market_cap_crore":("marketcaprscr","marketcaprscrore","marketcapitalizationrscrore","marketcap"),
 "pe_ratio":("pe","stockpe","priceearning","priceearningsratio"),
 "peg_ratio":("peg","pegratio"),
}
FRACTION={
 "sales_growth_5y","sales_growth_3y","opm_current","opm_5y_avg",
 "profit_growth_5y","profit_growth_7y","roce_7y_avg","roce_3y_avg",
 "roe_3y_avg","roe_5y_avg","promoter_holding","pledged_pct"}
REQUIRED={"nse_code"}
ALLOWED_NAME=re.compile(r"^[A-Z0-9][A-Z0-9&_.-]{0,39}$")
def norm(s):
 return re.sub(r"[^a-z0-9]+","",str(s).lower())

def unique_columns(columns):
 groups={}
 for raw in columns:groups.setdefault(norm(raw),[]).append(raw)
 for normalized,original in groups.items():
  if len(original)>1:raise ValueError("Ambiguous duplicate normalized CSV column: "+normalized)
 return {k:v[0] for k,v in groups.items()}

def transform_csv(data,asof):
 dt=pd.Timestamp(asof).normalize()
 if dt>pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None).normalize():
  raise ValueError("Snapshot date cannot be future")
 cols=unique_columns(data.columns)
 mapped={}
 for internal,variants in MAP.items():
  found=[cols[k] for k in variants if k in cols]
  if len(found)>1:raise ValueError("Ambiguous duplicate source fields for "+internal)
  if found:mapped[internal]=found[0]
 if not REQUIRED.issubset(mapped):
  raise ValueError("Official NSE Code required; never join third-party company name to original stocks")
 out=pd.DataFrame(index=data.index)
 code=data[mapped["nse_code"]].astype("string").str.upper().str.strip()
 valid=code.str.fullmatch(ALLOWED_NAME.pattern).fillna(False)
 if not bool(valid.all()):raise ValueError("Invalid or missing NSE identifier in third-party CSV")
 if code.duplicated().any():raise ValueError("Duplicate NSE symbol from external screener")
 out["symbol"]=code.astype(str)
 out["snapshot_date_IST"]=str(dt.date())
 for k,src in mapped.items():
  if k=="nse_code":continue
  if k=="bse_code":
   out["bse_code_unverified"]=data[src].astype("string").str.strip()
   continue
  val=data[src].astype("string").str.strip().str.replace(",","",regex=False).str.replace("₹","",regex=False)
  has_pct=val.str.endswith("%").fillna(False)
  val=val.str.replace("%","",regex=False)
  numeric=pd.to_numeric(val.where(~val.isin(["-","--",""]),pd.NA),errors="coerce")
  if k in FRACTION:
   if has_pct.any() and (~has_pct&numeric.notna()).any():
    raise ValueError("Mixed percent/decimal format in "+k)
   # Screener displays percentage metrics as percent POINTS; convert to decimals.
   numeric=numeric/100.
  out[k]=numeric
 out["third_party_values_as_reported_now_NOT_PIT_verified"]=True
 out["historical_2018_to_2025_training_eligible"]=False
 out["model_retraining_or_re_ranking_authorized"]=False
 return out,mapped

def import_csv(path,asof,out):
 src=Path(path)
 if not src.is_file() or src.suffix.lower()!=".csv":raise ValueError("Only explicitly supplied CSV supported")
 data=pd.read_csv(src,dtype=str,keep_default_na=False)
 if len(data)<1:raise ValueError("Empty third-party screen")
 table,mapped=transform_csv(data,asof)
 dest=Path(out);dest.mkdir(parents=True,exist_ok=True)
 table.to_csv(dest/"screener_manual_current_fundamentals_RESEARCH_ONLY.csv",index=False)
 info={
  "scope":"MANUALLY_SUPPLIED_SCREENER_CSV_RESEARCH_NO_SCRAPING_NO_PIT_CLAIM",
  "asof_IST":str(pd.Timestamp(asof).date()),
  "company_rows":len(table),"source_csv_sha256":hashlib.sha256(src.read_bytes()).hexdigest(),
  "import_timestamp_UTC":datetime.now(timezone.utc).isoformat(),
  "mapped_fields":mapped,
  "missing_requested_fields":sorted(set(MAP)-set(mapped)),
  "third_party_source_crosschecked_with_original_exchange_FY":False,
  "historical_PIT_source_eligibility":False,
  "never_use_these_current_values_as_old_backtest_features":True,
  "standalone_frozen_model_changed":False,
  "source_restriction":"User-provided legally obtained CSV only; no automatic Screener scraping or account credentials",
 }
 (dest/"screener_manual_source_eligibility.json").write_text(json.dumps(info,indent=2))
 print(json.dumps(info,indent=2),flush=True)
 return info

def main():
 a=argparse.ArgumentParser()
 a.add_argument("--csv",required=True);a.add_argument("--asof",required=True);a.add_argument("--output",required=True)
 opts=a.parse_args();import_csv(opts.csv,opts.asof,opts.output)
if __name__=="__main__":main()
