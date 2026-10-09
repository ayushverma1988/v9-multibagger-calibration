"""Immutable PRIVATE original NSE industry-peer research audit sidecar.

Never merge CURRENT undated NSE index-member values into historic pre-15:30
predictions. Store as separate after-close screening diagnostic only.
"""
from __future__ import annotations
import argparse,json,os,hashlib,shutil
from pathlib import Path
import pandas as pd
from huggingface_hub import HfApi
HF="ayushverma1988/v10-multibagger-archive"
ROOT="v11_4/afterclose_NSE_peer_median_and_issued_capital_proxies"
REPORT="nse_valuation_peer_size_proxy_coverage.json"
DATA="official_NSE_industry_peer_and_issued_capital_proxies_RESEARCH.csv"
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(source,asof,out,token,api=None):
 if not token:raise ValueError("Private dataset authentication missing")
 src=Path(source)
 meta=json.loads((src/REPORT).read_text())
 date=str(pd.Timestamp(asof).date())
 if meta["date_of_original_closing_prices"]!=date:raise ValueError("P/E proxy date mismatch")
 if meta["current_constituents_not_reconstructable_as_at_1530_original_date"] is not True:
  raise ValueError("Do not treat undated index member source as market-close PIT")
 if meta["no_model_rank_or_training_changed"] is not True or meta["verified_official_issuer_industry_PE"] is not False:
  raise ValueError("Overstated sector industry PE evidence")
 x=pd.read_csv(src/DATA)
 if len(x)!=meta["verified_NSE_individual_PE_joined_original_market_company_rows"]:
  raise ValueError("Company valuation proxy count inconsistent with original NSE source")
 if x["symbol"].duplicated().any() or x["industry"].notna().sum()!=meta["companies_with_sector_membership_match"]:
  raise ValueError("Duplicate stock or incorrect sector membership coverage")
 client=api or HfApi(token=token)
 info=client.repo_info(repo_id=HF,repo_type="dataset")
 if not info.private:raise ValueError("Not archiving research screening figures publicly")
 prefix=f"{ROOT}/{date}"
 if any(p.startswith(prefix+"/") or p==prefix for p in client.list_repo_files(repo_id=HF,repo_type="dataset")):
  raise ValueError("Immutable industry-valuation companion already exists, do not replace")
 dest=Path(out);dest.mkdir(parents=True,exist_ok=True)
 for n in [REPORT,DATA]:shutil.copy(src/n,dest/n)
 (dest/"manifest.json").write_text(json.dumps({
    "archive_scope":"POST_CLOSE_ONLY_NSE_PEER_AND_ISSUED_CAPITAL_RESEARCH_NOT_ASOF_1530",
    "source_date":date,
    "original_price_rank_source_changed":False,
    "original_sealed_forward_stock_list_changed":False,
    "market_capitalization_certified":False,
    "official_sector_pe_certified":False,
    "nse_archive_industry_provenance_SHA256":sha(dest/REPORT),
    "research_overlay_file_SHA256":sha(dest/DATA)},indent=2))
 client.upload_folder(repo_id=HF,repo_type="dataset",folder_path=str(dest),
                     path_in_repo=prefix,parent_commit=info.sha,
                     commit_message=f"Append-only NSE after-close industry P-E peer/size proxy research {date}")
 print(json.dumps({"private_research_PE_proxy_archive":True,"path":prefix,
                   "covered_sector_peers":meta["companies_with_peer_median_PE_proxy"],
                   "original_predictor_unchanged":True},indent=2),flush=True)
def main():
 p=argparse.ArgumentParser()
 p.add_argument("--source",required=True)
 p.add_argument("--asof",required=True)
 p.add_argument("--out",required=True)
 a=p.parse_args()
 save(a.source,a.asof,a.out,os.getenv("HF_ARCHIVE_TOKEN","").strip())
if __name__=="__main__":main()
