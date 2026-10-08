from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def utc(s):
    return pd.to_datetime(s,utc=True,errors="coerce")

def normsym(s):
    return s.astype(str).str.upper().str.strip()

def classify_financial_period(d):
    p=d.get("period",pd.Series("",index=d.index)).astype(str).str.lower()
    f=utc(d.get("fromDate"))
    t=utc(d.get("toDate"))
    span=(t-f).dt.days
    annual=p.str.contains("annual",na=False) | span.between(300,430)
    quarter=p.str.contains("quarter",na=False) | span.between(60,120)
    return annual,quarter

def latest_available_rows(d,symbol_col,ts_col,fold,window_days=None):
    x=d.copy()
    x["_sym"]=normsym(x[symbol_col])
    x["_ts"]=utc(x[ts_col])
    x=x[x["_ts"].notna()&(x["_ts"]<=fold)]
    if window_days is not None:
        x=x[x["_ts"]>=fold-pd.Timedelta(days=window_days)]
    return x

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--events",required=True)
    ap.add_argument("--financial-metadata",required=True)
    ap.add_argument("--shareholding",required=True)
    ap.add_argument("--insider",required=True)
    ap.add_argument("--snapshot",required=True)
    ap.add_argument("--folds",required=True)
    ap.add_argument("--output",required=True)
    args=ap.parse_args()

    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    ev=pd.read_parquet(args.events)
    fin=pd.read_parquet(args.financial_metadata)
    sh=pd.read_parquet(args.shareholding)
    ins=pd.read_parquet(args.insider)
    snap=pd.read_parquet(args.snapshot)
    folds=pd.read_csv(args.folds)

    ev["published_ts"]=utc(ev["published_ts"])
    ev["symbol"]=normsym(ev["symbol"])
    fin_sym="symbol_norm" if "symbol_norm" in fin.columns else "symbol"
    fin["_sym"]=normsym(fin[fin_sym])
    fin["_available"]=utc(fin.get("broadCastDate",fin.get("filingDate")))
    fin["_to"]=utc(fin.get("toDate"))
    annual_mask,quarter_mask=classify_financial_period(fin)
    fin["_annual"]=annual_mask
    fin["_quarter"]=quarter_mask

    sh_sym="symbol_norm" if "symbol_norm" in sh.columns else "_query_symbol" if "_query_symbol" in sh.columns else "symbol"
    sh["_sym"]=normsym(sh[sh_sym])
    sh["_available"]=utc(sh.get("broadcastDate",sh.get("submissionDate",sh.get("date"))))
    sh["_date"]=utc(sh.get("date"))

    ins_sym="symbol_norm" if "symbol_norm" in ins.columns else "symbol"
    ins["_sym"]=normsym(ins[ins_sym])
    ins["_available"]=utc(ins.get("date"))

    snap["date"]=pd.to_datetime(snap["date"],errors="coerce").dt.normalize()
    snap["symbol"]=normsym(snap["symbol"])

    fold_dates=pd.to_datetime(folds["date"],errors="coerce").dropna().sort_values().unique()
    rows=[]; violations=[]

    for fd0 in fold_dates:
        fd=pd.Timestamp(fd0).normalize()
        # Closing-price backtests must not read after-hours / next-day disclosures.
        # NSE regular trading session ends 15:30 IST; UTC midnight would leak news.
        fdu=(pd.Timestamp(fd).tz_localize("Asia/Kolkata")
             +pd.Timedelta(hours=15,minutes=30)).tz_convert("UTC")
        universe=snap[snap["date"].eq(fd)].copy()
        syms=set(universe["symbol"].dropna())
        n=max(len(syms),1)

        e=ev[
            ev["symbol"].isin(syms)
            & ev["published_ts"].notna()
            & (ev["published_ts"]<=fdu)
            & (ev["published_ts"]>=fdu-pd.Timedelta(days=180))
        ].copy()
        primary=e[
            e.get("document_url",pd.Series("",index=e.index)).astype(str).ne("")
            | e.get("source",pd.Series("",index=e.index)).astype(str).str.lower().str.contains("nse|bse|exchange|company",regex=True,na=False)
        ]

        ff=fin[fin["_sym"].isin(syms)&fin["_available"].notna()&(fin["_available"]<=fdu)
               &fin["_to"].notna()&(fin["_available"]>=fin["_to"])].copy()
        q=ff[ff["_quarter"]].copy()
        a=ff[ff["_annual"]].copy()

        q_counts=q.groupby("_sym")["_to"].nunique()
        a_counts=a.groupby("_sym")["_to"].nunique()

        ss=sh[
            sh["_sym"].isin(syms)
            & sh["_available"].notna()
            & (sh["_available"]<=fdu)
            & (sh["_date"].notna())
            & (sh["_available"]>=sh["_date"])
            & (sh["_available"]>=fdu-pd.Timedelta(days=500))
        ]
        ii=ins[
            ins["_sym"].isin(syms)
            & ins["_available"].notna()
            & (ins["_available"]<=fdu)
            & (ins["_available"]>=fdu-pd.Timedelta(days=180))
        ]

        future_event=int((e["published_ts"]>fdu).sum())
        future_fin=int((ff["_available"]>fdu).sum())
        future_sh=int((ss["_available"]>fdu).sum())
        future_ins=int((ii["_available"]>fdu).sum())
        pit_viol=future_event+future_fin+future_sh+future_ins
        if pit_viol:
            violations.append({"date":str(fd.date()),"events":future_event,"financial":future_fin,"shareholding":future_sh,"insider":future_ins})

        row={
            "date":str(fd.date()),
            "universe_symbols":len(syms),
            "event_rows_180d":len(e),
            "event_symbols_180d":e["symbol"].nunique(),
            "event_symbol_coverage":e["symbol"].nunique()/n,
            "primary_event_symbols_180d":primary["symbol"].nunique(),
            "primary_event_symbol_coverage":primary["symbol"].nunique()/n,
            "financial_symbols_any":ff["_sym"].nunique(),
            "financial_symbol_coverage":ff["_sym"].nunique()/n,
            "symbols_5q_plus":int((q_counts>=5).sum()),
            "symbols_8q_plus":int((q_counts>=8).sum()),
            "symbols_12q_plus":int((q_counts>=12).sum()),
            "symbols_5y_plus":int((a_counts>=6).sum()),
            "symbols_7y_plus":int((a_counts>=8).sum()),
            "shareholding_symbols_recent":ss["_sym"].nunique(),
            "shareholding_coverage":ss["_sym"].nunique()/n,
            "insider_symbols_180d":ii["_sym"].nunique(),
            "insider_coverage":ii["_sym"].nunique()/n,
            "pit_violations":pit_viol,
        }
        # Readiness is intentionally about reconstruction availability, not model performance.
        row["fold_reconstructable"]=bool(
            row["event_rows_180d"]>0
            and row["financial_symbol_coverage"]>=0.50
            and row["shareholding_coverage"]>=0.25
            and row["pit_violations"]==0
        )
        rows.append(row)

    df=pd.DataFrame(rows)
    df.to_csv(out/"historical_pit_readiness_by_fold.csv",index=False)
    pd.DataFrame(violations).to_csv(out/"pit_violations.csv",index=False)

    summary={
        "asof_rule":"Indian cash-market close at 15:30 Asia/Kolkata (10:00 UTC); conservative existing financial date parsing retained",
        "scope":"EXPERIMENTAL_STRICTER_PIT; production frozen rules unchanged",
        "folds":int(len(df)),
        "reconstructable_folds":int(df["fold_reconstructable"].sum()) if len(df) else 0,
        "minimum_required_folds":12,
        "all_pit_violations":int(df["pit_violations"].sum()) if len(df) else 0,
        "median_event_symbol_coverage":float(df["event_symbol_coverage"].median()) if len(df) else 0,
        "median_primary_event_symbol_coverage":float(df["primary_event_symbol_coverage"].median()) if len(df) else 0,
        "median_financial_symbol_coverage":float(df["financial_symbol_coverage"].median()) if len(df) else 0,
        "median_shareholding_coverage":float(df["shareholding_coverage"].median()) if len(df) else 0,
        "median_symbols_12q_plus":float(df["symbols_12q_plus"].median()) if len(df) else 0,
        "median_symbols_5y_plus":float(df["symbols_5y_plus"].median()) if len(df) else 0,
        "median_symbols_7y_plus":float(df["symbols_7y_plus"].median()) if len(df) else 0,
        "historical_external_demand_status":"SEPARATE_ADAPTER_REQUIRED",
        "historical_external_demand_note":"Company/event/financial/ownership PIT reconstruction is audited here. Exact historical GDELT+Google News demand reconstruction must pass separately before final V11.4 promotion.",
        "ready_for_company_level_pit_rebuild":bool(len(df)>=12 and df["fold_reconstructable"].sum()>=12 and df["pit_violations"].sum()==0),
        "ready_for_final_v11_4_backtest":False,
    }
    json.dump(summary,open(out/"historical_pit_readiness_summary.json","w"),indent=2)
    print(json.dumps(summary,indent=2))

    if len(df)<12:
        raise SystemExit("Fewer than 12 historical folds")
    if int(df["pit_violations"].sum())>0:
        raise SystemExit("Point-in-time violations detected")
    if int(df["fold_reconstructable"].sum())<12:
        raise SystemExit("Insufficient reconstructable folds")

if __name__=="__main__":
    main()
