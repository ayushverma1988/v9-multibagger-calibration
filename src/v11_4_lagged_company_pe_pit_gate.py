"""Strict first-observed-time eligibility for lagged original NSE company P/E.

After-close archive values can be proposed for a LATER market decision
only after an immutable, independently market-checked source was first
observed. It is not proof that valuation was known at its own 15:30 close.
Only a separate forward RESEARCH sidecar; frozen V11.4 training and picks
must not read this file until their prospective promotion gates pass.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd

def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()

def close_clock(day):
    return (pd.Timestamp(day).normalize().tz_localize("Asia/Kolkata")
            +pd.Timedelta(hours=15,minutes=30)).tz_convert("UTC")

def eligibility(manifest, csv_bytes, decision_day, max_calendar_lag=4):
    source_day=pd.Timestamp(manifest["date_IST"]).normalize()
    decision_day=pd.Timestamp(decision_day).normalize()
    first_seen=pd.to_datetime(manifest["archived_at_utc"],utc=True,errors="raise")
    source_cutoff=close_clock(source_day)
    decision_cutoff=close_clock(decision_day)
    assert_type="EXCHANGE_ORIGINAL_COMPANY_PE_RESEARCH_COMPANION_NOT_A_MODEL_PREDICTION"
    if manifest.get("type")!=assert_type:
        raise ValueError("Not the verified original company PE private archive")
    if not (manifest.get("research_only") is True and
            manifest.get("NOT_verified_at_1530_IST") is True and
            manifest.get("immutable_2026_10_08_forward_predictions_changed") is False and
            manifest.get("valuation_market_cap_and_peer_P_E_unverified") is True):
        raise ValueError("Source promises unsupported PIT/market capitalization status")
    if not str(manifest.get("original_NSE_company_PE_url","")).startswith((
            "https://archives.nseindia.com/content/equities/peDetail/PE_",
            "https://nsearchives.nseindia.com/content/equities/peDetail/PE_")):
        raise ValueError("Not an original exchange individual company PE URL")
    if first_seen<source_cutoff:
        raise ValueError("Bad chronology: archived time precedes original market close")
    if decision_day<=source_day:
        raise ValueError("Same-day company PE publication at 15:30 was never verified")
    lag=(decision_day-source_day).days
    if lag>max_calendar_lag:
        raise ValueError("Valuation observation is stale for next-session research")
    if first_seen>decision_cutoff:
        raise ValueError("Archived company PE was not available by decision cutoff")
    if sha256_bytes(csv_bytes)!=manifest.get("market_PE_stock_csv_SHA256"):
        raise ValueError("Archive input CSV bytes changed since verified immutable seal")
    from io import BytesIO
    x=pd.read_csv(BytesIO(csv_bytes))
    required={"date","symbol","isin","company_pe","company_pe_source_file_sha256",
              "pe_source_pit_at_1530_IST_verified","market_cap_crore","industry_pe_reference"}
    if not required.issubset(x):
        raise ValueError("Company PE CSV missing original research-only proof columns")
    if x.empty or x["symbol"].duplicated().any():
        raise ValueError("No companies or duplicate symbols in NSE P/E research")
    if not pd.to_datetime(x["date"],errors="coerce").dt.normalize().eq(source_day).all():
        raise ValueError("Wrong company P/E market date or mixed dates")
    if not x["company_pe_source_file_sha256"].astype(str).eq(
            manifest["original_company_PE_report_SHA256"]).all():
        raise ValueError("Original NSE reported source hash changed")
    if x["pe_source_pit_at_1530_IST_verified"].astype(str).str.lower().ne("false").any():
        raise ValueError("Historical P/E incorrectly certified known at 15:30")
    if x["market_cap_crore"].notna().any() or x["industry_pe_reference"].notna().any():
        raise ValueError("Unverified capitalization or peer P/E exposed as official")
    pe=pd.to_numeric(x["company_pe"],errors="coerce").replace([np.inf,-np.inf],np.nan)
    pe=pe.where(pe.gt(0)&pe.lt(20000))
    if pe.notna().sum()!=manifest.get("matched_company_positive_PE_count"):
        raise ValueError("Stored company P/E numeric coverage differs from source manifest")
    result=x[["symbol","isin"]].copy()
    result["company_pe_prior_original_close_only"]=pe
    result["source_market_day_IST"]=str(source_day.date())
    result["first_confirmed_archive_UTC"]=first_seen.isoformat()
    result["decision_cutoff_UTC"]=decision_cutoff.isoformat()
    result["not_verified_on_own_source_date_at_1530_IST"]=True
    # Not a historical model backfill; candidate for prospective, separate
    # research-only lagged valuation source after original first-seen timestamp.
    report={
        "scope":"FORWARD_ONLY_AFTER_CLOSE_ORIGINAL_COMPANY_PE_RESEARCH_ELIGIBILITY",
        "source_market_date":str(source_day.date()),
        "research_decision_date":str(decision_day.date()),
        "actual_first_observed_archive_UTC":first_seen.isoformat(),
        "decision_1530_UTC":decision_cutoff.isoformat(),
        "calendar_lag_days":lag,
        "company_PE_usable":int(pe.notna().sum()),
        "source_pe_SHA256":manifest["original_company_PE_report_SHA256"],
        "previously_frozen_stock_predictions_modified":False,
        "older_historical_training_folds_backfilled":False,
        "peer_PE_and_market_cap_promoted":False,
        "research_sidecar_not_approved_as_predictor":True,
        "conditional_next_market_source_availability_only":True
    }
    return result,report

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--manifest",required=True)
    ap.add_argument("--original-csv",required=True)
    ap.add_argument("--research-decision-day",required=True)
    ap.add_argument("--output",required=True)
    a=ap.parse_args()
    manifest=json.loads(Path(a.manifest).read_text())
    features,report=eligibility(manifest,Path(a.original_csv).read_bytes(),
                                a.research_decision_day)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    features.to_csv(out/"separately_eligible_lagged_company_pe_RESEARCH.csv",index=False)
    (out/"lagged_company_pe_pit_eligibility_summary.json").write_text(
        json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)

if __name__=="__main__":
    main()
