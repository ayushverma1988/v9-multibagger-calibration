"""Annotate immutable 2026-10-08 frozen V11.4 picks with strict 3FY filings.

Original verified FY2023/24/25 financial facts are source-only, as-of
2025-12-31 annual historical snapshot. Never use as a predictive feature
for the already-frozen Oct2026 Top10; never publish private stock names.
"""
from __future__ import annotations
import argparse,hashlib,json,os
from pathlib import Path
import pandas as pd
from huggingface_hub import HfApi,hf_hub_download
REPO="ayushverma1988/v10-multibagger-archive"
SOURCE="v11_4/forward_observations/2026-10-08"
DEST="v11_4/forward_3FY_fiscal_overlay/2026-10-08_NSE_FY2023_FY2025"
ORIGINAL="V11.4-standalone-research-frozen-20261008"
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def apply(financial,token,out,runid,api=None):
 if not token:raise ValueError("Owner's HF access is missing")
 client=api or HfApi(token=token)
 info=client.repo_info(repo_id=REPO,repo_type="dataset")
 if info.private is not True:raise ValueError("Private V11.4 research overlay cannot be published")
 picks_file=Path(hf_hub_download(repo_id=REPO,filename=SOURCE+"/verified_forward_top10.csv",repo_type="dataset",token=token))
 manifest_file=Path(hf_hub_download(repo_id=REPO,filename=SOURCE+"/forward_observation_manifest.json",repo_type="dataset",token=token))
 m=json.loads(manifest_file.read_text())
 if m.get("recorded_selection_csv_SHA256")!=sha(picks_file):
  raise ValueError("Historical first-seen top10 archive source hash changed")
 if m.get("model_id")!=ORIGINAL or m.get("prediction_date_IST")!="2026-10-08":
  raise ValueError("Unfrozen/untrusted original V11.4 predictor")
 if m.get("frozen_model_ranking_unmodified_by_all_four_screens") is not True:
  raise ValueError("Frozen original rank has been modified")
 x=pd.read_csv(picks_file)
 if len(x)!=10 or x["rank"].duplicated().any() or x["symbol"].duplicated().any():
  raise ValueError("Must have 10 immutable first-seen stocks")
 q=pd.read_csv(financial)
 if q["symbol"].duplicated().any():raise ValueError("Duplicate original fiscal stock")
 fields=["symbol"]+[f"FY{y}_revenue_INR" for y in (2023,2024,2025)]+[
  f"FY{y}_PAT_INR" for y in (2023,2024,2025)]+[
  "revenue_2023_to_2025_2yr_CAGR","revenue_FY2024_yoy",
  "revenue_FY2025_yoy","profit_2023_to_2025_2yr_CAGR","profit_FY2025_yoy",
  "FY2025_source_SHA256","source_mode"]
 if not set(fields).issubset(q):raise ValueError("Incomplete strict 2025 3-FY original financial matrix")
 rank=x.sort_values("rank").reset_index(drop=True)
 overlay=rank.merge(q[fields],on="symbol",how="left",validate="1:1")
 if not rank[["rank","symbol","date","close","p6_double_calibrated"]].equals(
         overlay[["rank","symbol","date","close","p6_double_calibrated"]]):
  raise ValueError("Three-FY sidecar changed frozen V11.4 stocks or scores")
 overlay["FY2023_2025_financial_period_approved_at_2025_12_close"]=overlay["FY2025_source_SHA256"].notna()
 overlay["original_frozen_predictions_not_recomputed"]=True
 overlay["financial_3FY_since_2025_12_to_2026_10_nontrading_restatement_updates_missing"]=True
 overlay["fully_verified_four_original_screen_conditions"]=False
 record={
  "scope":"PRIVATE_3FY_FINANCIAL_ANNOTATION_OF_IMMUTABLE_FIRST_SEEN_V11_4_20261008",
  "original_prediction_date_IST":"2026-10-08",
  "last_3_consecutive_fiscal_years":"FY2023|FY2024|FY2025",
  "original_filing_latest_decision_date_IST":"2025-12-31",
  "source_original_NSE_FourD_yearly_and_legacy_XBRL_only":True,
  "actual_original_model_company_count":10,
  "fully_verified_3FY_financial_companies_in_frozen_top10":int(overlay["FY2023_2025_financial_period_approved_at_2025_12_close"].sum()),
  "missing_full_fiscal_company_count":int(overlay["FY2025_source_SHA256"].isna().sum()),
  "true_3_year_CAGR_available":False,
  "two_year_growth_intervals_available":True,
  "model_frozen_predicted_rank_changed":False,
  "original_6m_2x_probability_changed":False,
  "original_first_seen_date_rewritten":False,
  "financial_overlay_not_production_approved":True,
  "source_financial_only_not_investment_advice":True}
 root=Path(out);root.mkdir(parents=True,exist_ok=True)
 filename="original_first_seen_top10_with_verified_3FY_2025_financial_overlay_PRIVATE.csv"
 overlay.to_csv(root/filename,index=False)
 record["overlay_SHA256"]=sha(root/filename)
 (root/"original_first_seen_private_3FY_source_overlay_audit.json").write_text(json.dumps(record,indent=2))
 dest=f"{DEST}/numeric_reconciliation_run_{runid}"
 if any(path.startswith(dest+"/") for path in client.list_repo_files(repo_id=REPO,repo_type="dataset")):
  raise ValueError("Source-only sidecar cannot overwrite a prior verification")
 client.upload_folder(folder_path=str(root),repo_id=REPO,repo_type="dataset",path_in_repo=dest,
                      parent_commit=info.sha,commit_message="Add private original V11.4 first-seen FY23-25 annual source annotation")
 print(json.dumps({**{k:v for k,v in record.items() if k not in ("overlay_SHA256",)},
   "private_3FY_annotated_original_top10_saved":True,"private_source_location":dest},indent=2),flush=True)
 return record
def main():
 p=argparse.ArgumentParser();p.add_argument("--financial",required=True);p.add_argument("--out",required=True);p.add_argument("--run-id",required=True)
 a=p.parse_args();apply(a.financial,os.getenv("HF_ARCHIVE_TOKEN","").strip(),a.out,a.run_id)
if __name__=="__main__":main()
