"""V11.4 forward-only private 6-month 2x research prediction ledger.

NO historical replay and NO production signal. Scoring requires an original
verified SAME-DATE official NSE market close plus full 185d announcement feed.
The frozen research model hash must match the 2026-10-08 freeze manifest.
One YYYY-MM-DD private HF path can be written only ONCE: repeat attempts fail.
"""
from __future__ import annotations
import argparse,hashlib,json,os
from datetime import datetime,timezone
from pathlib import Path
import joblib
import pandas as pd
from huggingface_hub import HfApi
from v11_4_forward_research_release import VERSION,score_prospective,sha256
from v11_4_four_family_live_screener import analyze

HF_REPO="ayushverma1988/v10-multibagger-archive"
FORWARD_ROOT="v11_4/forward_observations"
NEEDED=["v11_4_frozen_research_model.joblib","v11_4_frozen_research_manifest.json"]

def frozen_package(folder):
 base=Path(folder)
 for name in NEEDED:
  if not (base/name).is_file():raise ValueError("Frozen model artifact is incomplete: "+name)
 manifest=json.loads((base/NEEDED[1]).read_text())
 if manifest.get("model_id")!=VERSION or manifest.get("production_approved") is not False:
  raise ValueError("Unexpected frozen V11.4 model identity/promotion")
 if manifest.get("source_SHA256",{}).get("frozen_model_joblib")!=sha256(base/NEEDED[0]):
  raise ValueError("Frozen V11.4 model SHA mismatch: cannot score")
 model=joblib.load(base/NEEDED[0])
 if model.get("model_version")!=VERSION:
  raise ValueError("Serialized model identity mismatch")
 return model,manifest

def write_forward_record(features,source_meta,frozen,asof,out,hf_token,
                         api=None):
 if not hf_token:raise ValueError("Private HF_ARCHIVE_TOKEN not configured")
 today=pd.Timestamp.now(tz="Asia/Kolkata").normalize().tz_localize(None)
 asof=pd.Timestamp(asof).normalize()
 if asof!=today:raise ValueError("Forward score must be TODAY, never backfilled later")
 source=json.loads(Path(source_meta).read_text())
 x=pd.read_parquet(features)
 if source.get("live_features_SHA256")!=sha256(Path(features)):
  raise ValueError("Live market/catalyst source parquet SHA mismatch")
 if source.get("snapshot_date")!=str(asof.date()):
  raise ValueError("Independent NSE source not a current-session snapshot")
 package,frozen_manifest=frozen_package(frozen)
 # This function also verifies 15:30 IST PIT close, full official market
 # and original NSE event catalog coverage, price freshness, etc.
 picks=score_prospective(x,package,str(asof.date()),source)
 if len(picks)!=10 or picks["symbol"].duplicated().any():
  raise ValueError("Not exactly 10 unique verified forward stock observations")
 if picks["date"].nunique()!=1 or str(picks["date"].iloc[0].date())!=str(asof.date()):
  raise ValueError("Forward picks date invalid")
 client=api or HfApi(token=hf_token)
 if not client.repo_info(repo_id=HF_REPO,repo_type="dataset").private:
  raise ValueError("Refuse to store V11.4 research shortlist in public dataset")
 prefix=f"{FORWARD_ROOT}/{asof.date()}"
 # The initial source-freeze date must not be used for an overwritten
 # historical predictions backfill.
 existing=client.list_repo_files(repo_id=HF_REPO,repo_type="dataset")
 if any(p==prefix or p.startswith(prefix+"/") for p in existing):
  raise ValueError(f"Private forward observation already exists at {prefix}: immutable; no overwrite")
 dest=Path(out);dest.mkdir(parents=True,exist_ok=True)
 raw_cfg=Path("config/v11_4_four_screener_families.json")
 if not raw_cfg.is_file():raise ValueError("Four previously requested screener families config missing")
 screener=analyze(x,json.loads(raw_cfg.read_text()))
 sc=dest/"four_screener_families_audit_ALL_ELIGIBLE.csv"
 screener.to_csv(sc,index=False)
 s4=screener[screener["symbol"].isin(picks["symbol"])].copy()
 if len(s4)!=10:raise ValueError("Four screening overlay lost a predicted stock")
 s4.to_csv(dest/"four_screener_status_original_top10_NO_RERANK.csv",index=False)
 csv=dest/"verified_forward_top10.csv"
 picks.to_csv(csv,index=False)
 manifest={
  "scope":"FIRST_SEEN_IMMUTABLE_PROSPECTIVE_RESEARCH_STOCK_SELECTION",
  "model_id":VERSION,
  "prediction_date_IST":str(asof.date()),
  "prediction_written_at_utc":datetime.now(timezone.utc).isoformat(),
  "source_cutoff_utc":str(picks["source_cutoff_utc"].iloc[0]),
  "prediction_is_after_official_15_30_close_not_executable_at_recorded_close":True,
  "independent_exchange_NSE_UDiFF_full_file_validation":source["market_source_verified"],
  "independent_NSE_disclosure_lookback_audit":source["filing_source_verified"],
  "market_universe_count":int(source["company_rows"]),
  "integrity_clean_company_count":int(source["integrity_clean_company_rows"]),
  "frozen_model_SHA256":frozen_manifest["source_SHA256"]["frozen_model_joblib"],
  "frozen_manifest_SHA256":sha256(Path(frozen)/NEEDED[1]),
  "live_PIT_market_filing_features_SHA256":source["live_features_SHA256"],
  "recorded_selection_csv_SHA256":sha256(csv),
  "independent_four_original_screening_conditions_audit_SHA256":sha256(sc),
  "fourth_RSI14_strictly_above_80_in_top10":int(s4["rsi14_gt80"].fillna(False).sum()),
  "three_original_screening_families_not_yet_fully_sourced":True,
  "frozen_model_ranking_unmodified_by_all_four_screens":True,
  "recorded_stock_count":len(picks),
  "six_month_target_evaluation_due_approximately":str((asof+pd.DateOffset(months=6)).date()),
  "no_current_2026_outcome_has_been_observed":True,
  "eligible_for_predictive_model_training_before_six_month_maturity":False,
  "production_approved":False,
  "not_user_investment_instruction":True
 }
 (dest/"forward_observation_manifest.json").write_text(json.dumps(manifest,indent=2))
 # Hugging Face private upload preserves both CSV and source manifests.
 client.upload_folder(repo_id=HF_REPO,repo_type="dataset",
                      folder_path=str(dest),path_in_repo=prefix,
                      commit_message=f"Immutable V11.4 original NSE PIT research prediction {asof.date()}")
 print(json.dumps({"stored_private_forward_observation":True,
                   "research_date":str(asof.date()),"prediction_count":len(picks),
                   "private_archive_prefix":prefix,
                   "frozen_model_SHA256":manifest["frozen_model_SHA256"],
                   "data_SHA256":manifest["live_PIT_market_filing_features_SHA256"],
                   "production_approved":False},indent=2),flush=True)
 return manifest

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--features",required=True)
 p.add_argument("--metadata",required=True)
 p.add_argument("--frozen",required=True)
 p.add_argument("--output",required=True)
 p.add_argument("--asof",required=True)
 a=p.parse_args()
 write_forward_record(a.features,a.metadata,a.frozen,a.asof,a.output,
                      os.getenv("HF_ARCHIVE_TOKEN","").strip())
if __name__=="__main__":main()
