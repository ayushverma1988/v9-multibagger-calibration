"""V11.4 independent FY2023-fold historical financial expansion.

Use DIFFERENT companies than the original 24 pilot to test transferability of
strict NSE FY2022 FourD yearly values and FY2023 original standard annual facts.
Every document must be original, timestamped pre-fold, exact calendar FY and
single mode; no later restatement, no outcome labels or model predictions.
"""
from __future__ import annotations
import argparse,hashlib,json,math
from pathlib import Path
import pandas as pd
from v11_4_fy2023_historical_annual_pilot import choose_pairs,select_2023_fold
from v11_4_longterm_fundamentals import Client
from v11_4_strict_annual_numeric_features import strict_annual_facts,fold_close
from v11_4_recover_legacy_FY2022_FourD import extract_2022_legacy_fourd

RESEARCH_COHORT_SKIP=24
def annual_shard(catalog,snapshot,shard_index,shard_count,limit,out):
 if not 0<=shard_index<shard_count or not 1<=limit<=128:raise ValueError("Invalid deterministic numeric shard")
 out=Path(out);out.mkdir(parents=True,exist_ok=True)
 fold,universe=select_2023_fold(snapshot)
 pairs=choose_pairs(catalog,universe,fold)
 if len(pairs)<RESEARCH_COHORT_SKIP+limit*shard_count:
  raise ValueError("Insufficient disjoint original NSE historic source pairs")
 # Cut off initial 24 sampled pilot companies to test independently held-out cohort.
 selected=pairs[RESEARCH_COHORT_SKIP:][shard_index::shard_count][:limit]
 if len(selected)!=limit:raise ValueError("Incomplete requested historical shard")
 cutoff=fold_close(fold)
 client=Client()
 rows=[];errors=[];rawhashes=set()
 for i,(sym,mode,records) in enumerate(selected,1):
  result={"symbol":sym,"historical_2023_fold":fold,"mode":mode,
          "strict_source_full_fiscal_periods":"2022-03-31|2023-03-31",
          "source_model_prediction_approved":False}
  try:
   if len(records)!=2:raise ValueError("Not exactly two original fiscal years")
   facts={}
   for rec in records.itertuples(index=False):
    date=pd.Timestamp(rec.fy_end).date()
    url=str(rec.xbrl_url)
    pub=pd.to_datetime(rec.pub,utc=True)
    if not url.startswith("https://nsearchives.nseindia.com/") or not url.lower().endswith(".xml"):
     raise ValueError("Source not on original NSE XBRL HTTPS host")
    if pub>cutoff:raise ValueError("Filed after 2023 heldout decision cutoff")
    raw=client.get(url).content
    dig=hashlib.sha256(raw).hexdigest()
    if (sym,date) in rawhashes:raise ValueError("Duplicate historical original financial source")
    rawhashes.add((sym,date))
    if date.year==2022:
     vals,audit=extract_2022_legacy_fourd(raw,str(date),mode,source_is_annual=True)
    elif date.year==2023:
     vals,audit=strict_annual_facts(raw,date,source_is_annual=True)
     if vals["revenue"] is None or vals["pat"] is None:raise ValueError("Incomplete FY2023 strict XBRL facts")
    else:raise ValueError("Wrong fiscal-year file")
    facts[date.year]={"values":vals,"source_url":url,"source_sha256":dig,
                      "source_published_utc":pub.isoformat(),
                      "original_numeric_context_audit":json.dumps(audit,sort_keys=True)}
   if set(facts)!={2022,2023}:raise ValueError("Missing fiscal year source")
   p=facts[2022]["values"];n=facts[2023]["values"]
   if p["revenue"] is None or p["revenue"]<=0:raise ValueError("Invalid 2022 revenue baseline")
   result.update({
    "fy2022_revenue_INR":p["revenue"],"fy2023_revenue_INR":n["revenue"],
    "fy2022_PAT_INR":p["pat"],"fy2023_PAT_INR":n["pat"],
    "annual_revenue_yoy_fraction":n["revenue"]/p["revenue"]-1,
    "annual_PAT_yoy_fraction":n["pat"]/p["pat"]-1 if p["pat"]>0 else None,
    "fy2022_source_sha256":facts[2022]["source_sha256"],
    "fy2023_source_sha256":facts[2023]["source_sha256"],
    "fy2022_publication_utc":facts[2022]["source_published_utc"],
    "fy2023_publication_utc":facts[2023]["source_published_utc"],
    "fy2022_fiscal_fact_audit":facts[2022]["original_numeric_context_audit"],
    "fy2023_fiscal_fact_audit":facts[2023]["original_numeric_context_audit"],
    "status":"PAIRED_STRICT_RESEARCH_YOY"})
   rows.append(result)
  except Exception as err:
   result["status"]="UNVERIFIED_FISCAL_SOURCE"
   result["rejection_reason"]=str(err)[:200]
   errors.append(result)
  if i%8==0:
   print("shard",shard_index,"original FY2022/2023",i,"/",len(selected),
         "paired",len(rows),"rejected",len(errors),flush=True)
 pd.DataFrame(rows).to_csv(out/"independent_original_FY2022_FY2023_PIT_numeric_RESEARCH.csv",index=False)
 pd.DataFrame(errors).to_csv(out/"independent_FY2022_FY2023_PIT_rejected.csv",index=False)
 summary={"scope":"V11_4_INDEPENDENT_FY2023_ORIGINAL_NSE_ANNUAL_YOY_EXPANSION",
          "shard":shard_index,"shard_count":shard_count,
          "historical_2023_fold":fold,"historical_original_companies":len(universe),
          "eligible_historical_annual_source_pairs":len(pairs),
          "excluded_previous_tuning_pilot_companies":RESEARCH_COHORT_SKIP,
          "new_companies_requested":len(selected),"both_original_fiscal_facts_verified":len(rows),
          "FY2022_FY2023_source_rejected":len(errors),
          "annual_revenue_growth_calculable":sum(pd.notna(x["annual_revenue_yoy_fraction"]) for x in rows),
          "annual_pat_growth_calculable":sum(pd.notna(x["annual_PAT_yoy_fraction"]) for x in rows),
          "source_financial_metrics_ready_for_three_five_seven_year_growth":False,
          "no_2023_new_company_outcome_labels_used":True,
          "no_imputation":True,
          "original_standalone_model_modified":False,
          "source_results_not_approved_predictors":True}
 (out/"independent_FY2023_annual_numeric_shard_summary.json").write_text(json.dumps(summary,indent=2))
 print(json.dumps(summary,indent=2),flush=True)
 if len(rows)<max(8,int(limit*.75)):
  raise SystemExit("Held-out historical XBRL pair parser coverage <75%, no promotion")
 return summary

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--annual-index",required=True);p.add_argument("--snapshot",required=True)
 p.add_argument("--shard",type=int,required=True);p.add_argument("--shards",type=int,default=4)
 p.add_argument("--limit",type=int,default=32);p.add_argument("--out",required=True)
 a=p.parse_args()
 annual_shard(pd.read_csv(a.annual_index,dtype=str).fillna(""),
              pd.read_parquet(a.snapshot,columns=["date","symbol"]),
              a.shard,a.shards,a.limit,a.out)
if __name__=="__main__":main()
