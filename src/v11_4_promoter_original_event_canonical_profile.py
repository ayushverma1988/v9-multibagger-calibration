"""Inspect canonical NSE promoter metadata provenance without prices, labels or picks."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd
SUBJECT=("headline","subject","description","details","announcement_text","title",
         "broadcast_details","event_title","event_subject","event_description")
def profile_event_source(path):
 x=pd.read_parquet(path)
 mandatory={"event_id","symbol","event_type","event_direction",
 "published_ts","exchange_received_ts","document_url"}
 if not mandatory.issubset(x):raise ValueError("Canonical NSE event fields missing")
 x=x.drop_duplicates("event_id")
 z=x.loc[x["event_type"].astype(str).str.lower().eq("promoter_activity")].copy()
 z["direction"]=pd.to_numeric(z["event_direction"],errors="coerce")
 cols=[c for c in x if c.lower() in SUBJECT]
 score={}
 for c in cols:
  words=z[c].fillna("").astype(str).str.lower()
  high=words.str.contains(r"acquisition by promoters?|purchase of shares by promoters?|promoter(s)? bought|purchase by promoter",regex=True)
  neg=words.str.contains(r"sale by promoters?|promoter(s)? sold|disposal by promoters?|promoter pledge",regex=True)
  score[c]={
   "nonblank_promoter_records":int(words.str.len().gt(5).sum()),
   "explicit_buy_headline_hits_NOT_DOCUMENT_CONFIRMED":int(high.sum()),
   "explicit_sell_headline_hits_NOT_DOCUMENT_CONFIRMED":int(neg.sum()),
   "ambiguous_both_signals":int((high&neg).sum())}
 report={
  "scope":"CANONICAL_NSE_PROMOTER_SOURCE_RAW_RECORD_CLASSIFICATION_FEASIBILITY",
  "total_deduplicated_NSE_event_records":len(x),
  "promoter_raw_records":len(z),
  "promoter_direction_values":{str(k):int(v) for k,v in z["direction"].value_counts(dropna=False).items()},
  "raw_nse_document_urls_present":int(z["document_url"].fillna("").astype(str).str.startswith("https://").sum()),
  "text_source_columns_present":cols,
  "promoter_headline_directional_signal_counts":score,
  "does_not_reinterpret_generic_promoter_disclosures_as_buying":True,
  "original_live_V11_4_rankings_untouched":True,
  "financial_market_rsi_and_outcome_not_loaded":True,
  "not_sufficient_to_confirm_actual_share_purchase":True}
 return report
def main():
 p=argparse.ArgumentParser();p.add_argument("--events",required=True);p.add_argument("--out",required=True);a=p.parse_args()
 result=profile_event_source(a.events)
 out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
 out.write_text(json.dumps(result,indent=2))
 print(json.dumps(result,indent=2),flush=True)
if __name__=="__main__":main()
