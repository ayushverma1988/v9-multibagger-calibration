"""One-time owner-authorized research demo disclosure via envelope encryption.

This script reads already-existing private HF immutable research predictions,
verifies original source & scores, and prints ONLY strong-encrypted JSON for
decryption within authorized private ChatGPT analysis. It doesn't train,
rerank, overwrite the private dataset or add artifacts to a public repo.
"""
from __future__ import annotations
import argparse,base64,hashlib,json,os
from pathlib import Path
import pandas as pd
from huggingface_hub import hf_hub_download,HfApi
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import hashes,serialization

REPO="ayushverma1988/v10-multibagger-archive"
DATE="2026-10-08"
PREFIX=f"v11_4/forward_observations/{DATE}"
EXPECTED="V11.4-standalone-research-frozen-20261008"
FILE=("verified_forward_top10.csv","four_screener_status_original_top10_NO_RERANK.csv","forward_observation_manifest.json")
PUB_KEY="-----BEGIN PUBLIC KEY-----\nMIIBojANBgkqhkiG9w0BAQEFAAOCAY8AMIIBigKCAYEAx8+I+7bPNugMFaWxJXX3\n4zRqOXxVkHasr1PG6eB8S5jgKft02HEzEi18h0I+dMtiyNtJ9OyOPpTP82n5POM5\nUTegNMM4ghn60tjRRWnG8ilXmU9qjQ1rBZsZijxFLkC3DdK8DzyoNFKfrJm6Yfcb\nSGhAhWCC3QFgzEv/u1jAMWeV+2+MqndiJ+mdRUu0FT/DiUSchhM5bQeEDBG9OhlE\necY0IIpD5iWdRemGUwpkey83UVLM6eUqgFs8wPCvg1awj9bYTQpQAdhms71CO5o3\nkqn79HZE6bbYDS7N9rEZjNe9//h9c8D2GIgmYwHoQXrminjYhO7ofaBn2sZcMp2l\nreq9X3iGCWovmwVPk1ZLkqV/6qkJaGk7K6gW4m7f1TJW1W9L5FNQ/NRhdv2rvn9w\nPaxpCkMj2b1kRDggI7QLH1IM4i+kcH2dMWAFEJsnZH4a4juZwwy80arbqcgiE0CY\nUTrWcOK8OeOVvIbcBn8LCp6tl3JET7eLOwGDb4hlAYJJAgMBAAE=\n-----END PUBLIC KEY-----"
def file_sha(x):
 h=hashlib.sha256()
 with open(x,"rb") as f:
  for block in iter(lambda:f.read(65536),b""):h.update(block)
 return h.hexdigest()
def audit_and_encrypt(token):
 if not token:raise ValueError("Missing owner's HF token")
 client=HfApi(token=token)
 info=client.repo_info(repo_id=REPO,repo_type="dataset")
 if info.private is not True:raise ValueError("Archive no longer private, abort")
 paths={name:Path(hf_hub_download(repo_id=REPO,filename=PREFIX+"/"+name,repo_type="dataset",token=token)) for name in FILE}
 manifest=json.loads(paths[FILE[2]].read_text())
 if manifest.get("scope")!="FIRST_SEEN_IMMUTABLE_PROSPECTIVE_RESEARCH_STOCK_SELECTION":
  raise ValueError("Invalid frozen research scope")
 if manifest.get("model_id")!=EXPECTED or manifest.get("prediction_date_IST")!=DATE:
  raise ValueError("Wrong prospective frozen model or original date")
 if manifest.get("production_approved") is not False or manifest.get("recorded_stock_count")!=10:
  raise ValueError("Model declared production-approved or missing ten prospective stocks")
 if manifest.get("recorded_selection_csv_SHA256")!=file_sha(paths[FILE[0]]):
  raise ValueError("Frozen private Top 10 SHA mismatch")
 if manifest.get("frozen_model_ranking_unmodified_by_all_four_screens") is not True:
  raise ValueError("Four-screen overlay has changed original research ranking")
 picks=pd.read_csv(paths[FILE[0]])
 screens=pd.read_csv(paths[FILE[1]])
 if len(picks)!=10 or picks["symbol"].duplicated().any() or set(picks["rank"])!=set(range(1,11)):
  raise ValueError("Not exactly ranks 1-10")
 if set(picks["symbol"])!=set(screens["symbol"]):
  raise ValueError("Screener overlay company mismatch")
 if picks["model_id"].nunique()!=1 or str(picks["model_id"].iloc[0])!=EXPECTED:
  raise ValueError("Model identity in data mismatch")
 if not picks["date"].astype(str).str.startswith(DATE).all():
  raise ValueError("Historical score date mismatch")
 factors=[x for x in screens.columns if x.endswith("_status")]
 if len(factors)!=4:raise ValueError("Four original user screener states not present")
 if "rsi14_wilder" not in screens:raise ValueError("Missing exact Wilder RSI14")
 overlay=picks.merge(screens,on=["symbol","date"],how="left",validate="1:1")
 if len(overlay)!=10 or overlay[factors].isna().any().any():
  raise ValueError("Missing exact screener verdict for a selected stock")
 if any(overlay[c].isin(["PASS","FAIL","UNKNOWN"]).all() is False for c in factors):
  raise ValueError("Unrecognized screener verdict")
 pairs=[]
 for r in overlay.sort_values("rank").to_dict("records"):
  price=float(r["close"]);prob=float(r["p6_double_calibrated"])
  if price<=0 or not 0<prob<1:raise ValueError("Invalid confidential price/probability")
  pairs.append({
   "rank":int(r["rank"]),"symbol":str(r["symbol"]),"source_close_INR":round(price,2),
   "p6_double_research_model_estimate_percent":round(prob*100,2),
   "matches_20_to_2000_user_preference":bool(20<=price<=2000),
   "RSI14_Wilder":round(float(r["rsi14_wilder"]),2) if pd.notna(r["rsi14_wilder"]) else None,
   "RSI14_strictly_gt80":bool(r["rsi14_wilder"]>80) if pd.notna(r["rsi14_wilder"]) else None,
   "four_screener_family_verdicts":{k.replace("_status",""):str(r[k]) for k in factors},
   "pending_screener_family_count":sum(r[k]=="UNKNOWN" for k in factors),
  })
 result={
  "research_date_IST":DATE,
  "research_not_trading_advice":True,
  "forward_observation_original_source_is_private_HF":True,
  "immutable_original_SHA256_verified":True,
  "frozen_model":EXPECTED,
  "user_price_preference_20_2000_matches":sum(p["matches_20_to_2000_user_preference"] for p in pairs),
  "four_rule_filter_not_used_for_reranking":True,
  "six_month_outcome_not_observed_yet":True,
  "source_asof_UTC":manifest["source_cutoff_utc"],
  "rows":pairs}
 raw=json.dumps(result,separators=(",",":"),allow_nan=False).encode()
 aes_key=AESGCM.generate_key(bit_length=256)
 nonce=os.urandom(12)
 aad=b"V11.4-private-demo-original-2026-10-08"
 ciphertext=AESGCM(aes_key).encrypt(nonce,raw,aad)
 pub=serialization.load_pem_public_key(PUB_KEY.encode())
 keywrap=pub.encrypt(aes_key,padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()),algorithm=hashes.SHA256(),label=None))
 packet={"version":"RSA3072-OAEP-SHA256_AES256-GCM","key":base64.b64encode(keywrap).decode(),
        "nonce":base64.b64encode(nonce).decode(),"aad":base64.b64encode(aad).decode(),
        "payload":base64.b64encode(ciphertext).decode()}
 print("V11_DEMO_CIPHER="+base64.b64encode(json.dumps(packet,separators=(",",":")).encode()).decode(),flush=True)
 print("V11_DEMO_PRIVATE_RECORD_AUDITED=True",flush=True)
 return True
def main():
 a=argparse.ArgumentParser();a.add_argument("--run",action="store_true");x=a.parse_args()
 if not x.run:raise ValueError("One-time explicit demonstration only")
 audit_and_encrypt(os.environ.get("HF_ARCHIVE_TOKEN",""))
if __name__=="__main__":main()
