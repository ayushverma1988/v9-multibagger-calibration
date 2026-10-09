"""Audit V11.4 standalone 18-fold financial and four-rule source readiness.

Uses only a previously archived, historical NSE source matrix. No model
training, outcomes, V10 rankings, synthetic financials, or current valuations.
One-year annual YoY is NOT interchangeable with 3y/5y/7y growth. Missing
official rules are UNKNOWN, never converted to a pass/fail.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

FUTURE_LABELS={"y6","y12","y24","dd30_6m","y6_mature_date",
               "days_to_2x","hit25_6m","hit50_6m"}
ANNUAL_SOURCE_FIELDS=("revenue_yoy_pct","pat_yoy_pct")
NEEDED_FOLDS=12
MIN_COVERAGE=0.70

def close_utc(dates):
    return (pd.to_datetime(dates).dt.tz_localize("Asia/Kolkata")+
            pd.Timedelta(hours=15,minutes=30)).dt.tz_convert("UTC")

def verified_notna(x, col):
    if col not in x: return pd.Series(False,index=x.index)
    s=x[col]
    if pd.api.types.is_bool_dtype(s): return s.notna()
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s,errors="coerce").replace([np.inf,-np.inf],np.nan).notna()
    return s.notna() & s.astype(str).str.strip().ne("")

def audit_financial_readiness(frame, config, expected_rows=None, expected_folds=18):
    if len(config.get("conditions",{}))!=4:
        raise ValueError("Expected four independent user rule families")
    if any(n in frame.columns for n in FUTURE_LABELS):
        raise ValueError("Historical future outcome is forbidden in source feature audit")
    needed={"date","symbol","source_available_utc","has_verified_annual_yoy_source",
            "historical_asof_utc",*ANNUAL_SOURCE_FIELDS}
    if not needed.issubset(frame):
        raise ValueError("Missing original historical financial provenance: "+str(needed-set(frame)))
    x=frame.copy()
    x["date"]=pd.to_datetime(x["date"],errors="raise").dt.normalize()
    x["symbol"]=x["symbol"].astype(str).str.upper().str.strip()
    if x[["date","symbol"]].duplicated().any() or not x["symbol"].str.len().gt(0).all():
        raise ValueError("Invalid original historic stock-date identities")
    if expected_rows is not None and len(x)!=expected_rows:
        raise ValueError("Changed original NSE frozen stock universe row count")
    if x["date"].nunique()!=expected_folds:
        raise ValueError("Changed original historical fold count")
    cutoff=close_utc(x["date"])
    event_clock=pd.to_datetime(x["historical_asof_utc"],utc=True,format="mixed",errors="coerce")
    if event_clock.isna().any() or (event_clock!=cutoff).any():
        raise ValueError("Wrong original historical 15:30 source cutoff")
    av=pd.to_datetime(x["source_available_utc"],utc=True,format="mixed",errors="coerce")
    state=x["has_verified_annual_yoy_source"].astype("boolean").fillna(False)
    if (state & (av.isna() | (av>cutoff))).any():
        raise ValueError("Future/unproven annual XBRL value inside historical fold")
    if (~state & x[list(ANNUAL_SOURCE_FIELDS)].notna().any(axis=1)).any():
        raise ValueError("Financial numeric value on record without verified annual source")
    # Nulls in fiscal-revenue/PAT remain null. They are not failures or zero.
    both=state & x["revenue_yoy_pct"].notna() & x["pat_yoy_pct"].notna()
    # Block surprising feature schema where a newly available source is
    # silently promoted; all rule columns must be inspected explicitly.
    rule_fields=sorted({p[0] for c in config["conditions"].values() for p in c["hard_rules"]})
    annual_obs=state & (x["revenue_yoy_pct"].notna() | x["pat_yoy_pct"].notna())
    rows=[]; backlog=[]
    for d,idx in x.groupby("date",sort=True).groups.items():
        group=x.loc[idx]
        row={"fold":d.date().isoformat(),"stocks":len(group),
             "verified_annual_source_stocks":int(state.loc[idx].sum()),
             "annual_revenue_yoy_stocks":int((state.loc[idx]&verified_notna(group,"revenue_yoy_pct")).sum()),
             "annual_pat_yoy_stocks":int((state.loc[idx]&verified_notna(group,"pat_yoy_pct")).sum()),
             "both_annual_revenue_pat_stocks":int(both.loc[idx].sum()),
             "both_annual_coverage":float(both.loc[idx].mean())}
        for family,definition in config["conditions"].items():
            relevant=[r[0] for r in definition["hard_rules"]]
            known=pd.DataFrame({r:verified_notna(group,r) for r in relevant},index=group.index)
            row[family+"_fully_known_stocks"]=int(known.all(axis=1).sum())
            row[family+"_fully_known_ratio"]=float(known.all(axis=1).mean())
            row[family+"_unknown_metrics"]="|".join([r for r in relevant if not known[r].any()])
        rows.append(row)
        if row["both_annual_coverage"]<MIN_COVERAGE:
            backlog.append({"fold":row["fold"],"stocks":len(group),
                "verified_core_annual_both":row["both_annual_revenue_pat_stocks"],
                "missing_core_annual_both":len(group)-row["both_annual_revenue_pat_stocks"],
                "prioritize":"RECONSTRUCT_ORIGINAL_PRE_CUTOFF_XBRL_WITH_UNITS_AND_PERIOD",
                "current_company_PE_as_historical_substitute_allowed":False})
    table=pd.DataFrame(rows)
    core_ok=table[table["both_annual_coverage"].ge(MIN_COVERAGE)]
    full_known_counts={k:int(table[k+"_fully_known_ratio"].ge(MIN_COVERAGE).sum())
                       for k in config["conditions"]}
    conditional={k:v>=NEEDED_FOLDS for k,v in full_known_counts.items()}
    passed=len(core_ok)>=NEEDED_FOLDS and all(conditional.values())
    report={
        "scope":"V11_4_STANDALONE_PIT_FOUR_RULE_SOURCE_READINESS",
        "historical_rows":int(len(x)),
        "historical_folds":int(x["date"].nunique()),
        "fundamental_annual_both_fields_coverage_threshold":MIN_COVERAGE,
        "minimum_folds_with_covered_fundamentals":NEEDED_FOLDS,
        "annual_revenue_and_PAT_both_covered_folds":len(core_ok),
        "annual_revenue_and_PAT_both_covered_fold_dates":core_ok["fold"].tolist(),
        "fully_known_rule_folds":full_known_counts,
        "family_data_coverage_passed":conditional,
        "historical_financial_rule_activation_approved":bool(passed),
        "fields_still_entirely_missing_in_historical_matrix":[
            r for r in rule_fields if not verified_notna(x,r).any()],
        "short_horizon_annual_yoy_cannot_replace_3y_5y_7y":True,
        "post_close_company_PE_cannot_retroactively_backfill_18_folds":True,
        "no_current_market_screener_proxies_promoted":True,
        "no_training_labels_or_existing_predictions_read":True,
        "no_missing_financial_values_imputed":True,
        "original_standalone_model_weights_or_picks_changed":False,
        "action":"HISTORICAL_ORIGINAL_NSE_BSE_FINANCIAL_SOURCE_BACKFILL_REQUIRED"
                 if not passed else "INDEPENDENT_FORWARD_TEST_REQUIRED"
    }
    return table,pd.DataFrame(backlog),report

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--features",required=True)
    p.add_argument("--config",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--expected-rows",type=int,default=18569)
    a=p.parse_args()
    frame=pd.read_parquet(a.features)
    rules=json.loads(Path(a.config).read_text())
    table,queue,report=audit_financial_readiness(frame,rules,expected_rows=a.expected_rows)
    dest=Path(a.output);dest.mkdir(parents=True,exist_ok=True)
    table.to_csv(dest/"18fold_four_rule_PIT_coverage.csv",index=False)
    queue.to_csv(dest/"historical_financial_backfill_priority.csv",index=False)
    (dest/"four_rule_historical_source_readiness.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)
    print(table.tail(8).to_string(index=False),flush=True)
    # It is correct for data coverage to FAIL promotion but succeed as an audit.
    # A source-integrity exception above is a real workflow failure.
if __name__=="__main__":main()
