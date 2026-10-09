"""Archive independent NSE company P/E source audit privately, append-only.

Research companion for first-seen selection day, never re-rank prior stock
picks or misstate same-day P/E publication as pre-15:30 point-in-time.
Source already verified as original NSE per-company P/E and dual-price match.
"""
from __future__ import annotations
import argparse,hashlib,json,os,shutil
from pathlib import Path
from datetime import datetime,timezone
import pandas as pd
from huggingface_hub import HfApi

REPO="ayushverma1988/v10-multibagger-archive"
ROOT="v11_4/original_exchange_company_PE_companion"
SRC_CSV="nse_original_company_pe_same_day_research.csv"
SRC_SUMMARY="nse_direct_company_pe_coverage_audit.json"

def sha(p):
 return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def archive(folder,asof,token,out,api=None):
 if not token:raise ValueError("Missing HF_ARCHIVE_TOKEN, refusing to publish public market source")
 folder=Path(folder)
 report=json.loads((folder/SRC_SUMMARY).read_text())
 date=str(pd.Timestamp(asof).date())
 if report.get("asof_date_IST")!=date:
  raise ValueError("Wrong source date in original exchange company P/E companion")
 if not report.get("exact_day_bhavcopy_independently_cross_checked"):
  raise ValueError("P/E source has no two-official-market-file confirmation")
 if not report.get("model_train_or_rank_modified") is False:
  raise ValueError("P/E must not alter frozen V11.4 training or picks")
 if report.get("PE_data_publication_at_1530_IST_verified") is not False:
  raise ValueError("False original P/E publication time assumption")
 if report.get("same_day_market_and_company_PE_matches",0)<500:
  raise ValueError("Inadequate official securities date join")
 raw=folder/SRC_CSV
 x=pd.read_csv(raw)
 if len(x)!=report["independent_verified_market_rows"]:
  raise ValueError("One or more independently verified NSE market stock rows missing")
 if x[["date","symbol"]].duplicated().any():
  raise ValueError("Duplicate company/date in archived company P/E")
 if not x["date"].astype(str).str.startswith(date).all():
  raise ValueError("Source valuation file contains a future/wrong trading date")
 if int(pd.to_numeric(x["company_pe"],errors="coerce").notna().sum())!=report["individual_security_PE_usable"]:
  raise ValueError("Valuation CSV and independent NSE validation statistics disagree")
 if not x["company_pe_source_file_sha256"].eq(report["official_company_pe_original_bytes_sha256"]).all():
  raise ValueError("Original NSE individual valuation file hash mismatch")
 if x["market_cap_crore"].notna().any() or x["industry_pe_reference"].notna().any():
  raise ValueError("Unverified company market cap or peer P/E incorrectly filled")
 client=api or HfApi(token=token)
 info=client.repo_info(repo_id=REPO,repo_type="dataset")
 if not info.private:raise ValueError("NSE valuation companion target MUST remain private")
 target=f"{ROOT}/{date}"
 files=client.list_repo_files(repo_id=REPO,repo_type="dataset")
 if any(f.startswith(target+"/") or f==target for f in files):
  raise ValueError(f"Immutable private company valuation companion date already present: {target}")
 out=Path(out);out.mkdir(parents=True,exist_ok=True)
 shutil.copyfile(raw,out/SRC_CSV)
 shutil.copyfile(folder/SRC_SUMMARY,out/SRC_SUMMARY)
 manifest={
  "type":"EXCHANGE_ORIGINAL_COMPANY_PE_RESEARCH_COMPANION_NOT_A_MODEL_PREDICTION",
  "date_IST":date,"reference_model":"V11.4 standalone frozen unmodified",
  "cash_market_verification":"two original NSE UDiFF and full-security files, all price differences zero in historical trial",
  "original_NSE_company_PE_url":report["official_company_pe_url"],
  "original_company_PE_report_SHA256":report["official_company_pe_original_bytes_sha256"],
  "matched_company_positive_PE_count":report["individual_security_PE_usable"],
  "market_PE_stock_csv_SHA256":sha(out/SRC_CSV),
  "original_source_summary_SHA256":sha(out/SRC_SUMMARY),
  "NOT_verified_at_1530_IST":True,
  "valuation_market_cap_and_peer_P_E_unverified":True,
  "immutable_2026_10_08_forward_predictions_changed":False,
  "research_only":True,
  "archived_at_utc":datetime.now(timezone.utc).isoformat()}
 (out/"valuation_companion_manifest.json").write_text(json.dumps(manifest,indent=2))
 client.upload_folder(folder_path=str(out),repo_id=REPO,repo_type="dataset",
                     path_in_repo=target,parent_commit=info.sha,
                     commit_message="New original NSE company P/E independent same-day 2-source valuation research companion "+date)
 print(json.dumps({"archived_private_research_company_PE":True,"asof":date,
                   "target":target,"positive_individual_PE":report["individual_security_PE_usable"],
                   "original_top10_predictions_changed":False},indent=2),flush=True)
 return manifest
def main():
 p=argparse.ArgumentParser()
 p.add_argument("--source",required=True)
 p.add_argument("--asof",required=True)
 p.add_argument("--out",required=True)
 a=p.parse_args()
 archive(a.source,a.asof,os.environ.get("HF_ARCHIVE_TOKEN","").strip(),a.out)
if __name__=="__main__":main()
