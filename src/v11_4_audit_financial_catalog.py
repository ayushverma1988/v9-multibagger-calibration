"""Diagnose the annual-filing gap without inventing five- or seven-year fundamentals."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--financial", required=True)
    ap.add_argument("--folds", required=True)
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--output", required=True)
    a=ap.parse_args()
    out=Path(a.output)
    out.mkdir(parents=True,exist_ok=True)
    x=pd.read_parquet(a.financial)
    sym="symbol_norm" if "symbol_norm" in x else "symbol"
    dates=["broadCastDate","filingDate","toDate","fromDate"]
    for c in dates:
        if c not in x: x[c]=pd.NaT
        x["_"+c]=pd.to_datetime(x[c],errors="coerce",utc=True)
    if "period" not in x: x["period"]=""
    if "consolidated" not in x: x["consolidated"]=""
    x["_sym"]=x[sym].astype(str).str.upper().str.strip()
    x["_available"]=x["_broadCastDate"].fillna(x["_filingDate"])
    days=(x["_toDate"]-x["_fromDate"]).dt.days
    p=x["period"].astype(str).str.lower()
    x["_annual"]=p.str.contains("annual",na=False)|days.between(300,430)
    x["_quarter"]=p.str.contains("quarter",na=False)|days.between(60,120)
    x["_valid"]=x["_available"].notna()&x["_toDate"].notna()&(x["_available"]>=x["_toDate"])
    x["_period_lead_days"]=(x["_available"]-x["_toDate"]).dt.days
    folds=pd.read_csv(a.folds)
    snap=pd.read_parquet(a.snapshot,columns=["date","symbol"])
    snap["date"]=pd.to_datetime(snap["date"],errors="coerce").dt.normalize()
    snap["symbol"]=snap["symbol"].astype(str).str.upper().str.strip()
    records=[]
    for date in pd.to_datetime(folds["date"],errors="coerce").dropna().unique():
        f=pd.Timestamp(date).normalize().tz_localize("Asia/Kolkata")+pd.Timedelta(days=1)-pd.Timedelta(seconds=1)
        f=f.tz_convert("UTC")
        used=x[x["_valid"] & (x["_available"]<=f)]
        annual=used[used["_annual"]]
        quarter=used[used["_quarter"]]
        ac=annual.groupby("_sym")["_toDate"].nunique()
        qc=quarter.groupby("_sym")["_toDate"].nunique()
        # A 5-year point-in-time comparison needs 24 consecutive quarters,
        # not just 20 observations scattered across reporting periods.
        # Similarly, seven years require 32 consecutive quarters.
        per=quarter[["_sym","_toDate"]].dropna().copy()
        per["qi"]=per["_toDate"].dt.year*4+((per["_toDate"].dt.month-1)//3)
        per=per.drop_duplicates(["_sym","qi"])
        fold_universe=set(snap.loc[snap["date"].eq(f.tz_convert("Asia/Kolkata").normalize().tz_localize(None)),"symbol"])
        cutoff_period=f.year*4+(f.month-1)//3
        consecutive_24=consecutive_32=0
        for symbol, gg in per.groupby("_sym")["qi"]:
            if symbol not in fold_universe:
                continue
            vs=sorted(set(gg.astype(int)),reverse=True)
            if not vs or cutoff_period-vs[0]>2:
                continue
            count=1
            for a0,b0 in zip(vs,vs[1:]):
                if a0-b0!=1:break
                count+=1
            consecutive_24+=count>=24
            consecutive_32+=count>=32

        records.append({
            "date":f.date().isoformat(),
            "unique_fiscal_ends_annual_total":annual["_toDate"].nunique(),
            "annual_symbols":len(ac),
            "annual_6_fiscal_ends":int((ac>=6).sum()),
            "annual_8_fiscal_ends":int((ac>=8).sum()),
            "quarter_symbols":len(qc),
            "quarter_4_periods":int((qc>=4).sum()),
            "quarter_12_periods":int((qc>=12).sum()),
            "quarter_20_periods":int((qc>=20).sum()),
            "quarter_28_periods":int((qc>=28).sum()),
            "fold_universe_symbols":len(fold_universe),
            "fold_contiguous_24_quarters":consecutive_24,
            "fold_contiguous_32_quarters":consecutive_32,
            "fold_contiguous_24_share":round(consecutive_24/max(len(fold_universe),1),4),
            "fold_contiguous_32_share":round(consecutive_32/max(len(fold_universe),1),4),
        })
    pd.DataFrame(records).sort_values("date").to_csv(out/"annual_history_by_fold.csv",index=False)
    distribution={
        "source_rows":len(x),
        "symbols":int(x["_sym"].nunique()),
        "columns":list(map(str,x.columns)),
        "annual_period_label_values":x.loc[x["_annual"],"period"].astype(str).value_counts().head(12).to_dict(),
        "all_period_label_values":x["period"].astype(str).value_counts().head(12).to_dict(),
        "annual_rows":int(x["_annual"].sum()),
        "quarter_rows":int(x["_quarter"].sum()),
        "annual_distinct_symbols":int(x.loc[x["_annual"],"_sym"].nunique()),
        "valid_annual_rows":int((x["_valid"]&x["_annual"]).sum()),
        "valid_quarter_rows":int((x["_valid"]&x["_quarter"]).sum()),
        "min_filing_ts":str(x["_available"].min()),
        "min_period_end":str(x["_toDate"].min()),
        "max_filing_ts":str(x["_available"].max()),
        "unavailable_ts_rows":int(x["_available"].isna().sum()),
        "filing_before_period_end":int((x["_available"].notna()&x["_toDate"].notna()&(x["_available"]<x["_toDate"])).sum()),
        "contiguous_history_policy":"24 and 32 consecutive filed quarters; a *potential* fiscal-history proxy, NOT annual XBRL fact extraction or backtest approval",
        "retrospective_annual_min_history": "six separately published fiscal ends for five-year CAGR; eight for seven-year CAGR",
        "pit_policy": "use only filings available on or before fold, not present-day reconstructed annual values",
    }
    (out/"financial_catalog_diagnostics.json").write_text(json.dumps(distribution,indent=2))
    print(json.dumps(distribution,indent=2),flush=True)
    print(pd.DataFrame(records).tail(6).to_string(index=False),flush=True)

if __name__=="__main__": main()
