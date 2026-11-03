"""Additional June 2022 source-clock-qualified FY2020/21/22 market cohort.

Loads 831 strictly reconciled original FY2022 fiscal company triplets. Only
uses each company's exact originally published FY2020,21,22 filing when all
were available at the original JUNE 2022 15:30 IST close.
No 2025 labels/scores/model test. Compatible with immutable FY2022 Dec archive.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd
from v11_4_threeFY_June_PIT_supplement import FOLDS,reopen_june_source

JUNE_2022=("2022-06-30",(2020,2021,2022))
def original_june_2022(fiscal,stock):
 # Reuse the previously audited common source-timing protocol without
 # altering the already completed original 2023/24/25 June research files.
 if 2022 in FOLDS and FOLDS[2022]!=JUNE_2022:
  raise ValueError("Unexpected changed 2022 original fiscal date and period")
 FOLDS[2022]=JUNE_2022
 data,summary=reopen_june_source(fiscal,stock,2022)
 if summary["historical_fold_IST"]!="2022-06-30":
  raise ValueError("June2022 historical market date altered")
 if summary["original_Dec_full_3FY_company_numerics_available"]!=831:
  raise ValueError("Original FY2020-22 private financial source identities changed")
 if not summary["asof_June_no_future_source"]:
  raise ValueError("June financial publication chronology unverified")
 summary["scope"]="V11_4_ORIGINAL_JUNE2022_STRICT_YEARLY_FY2020_2021_2022_SOURCE_AND_MATURE_LABEL"
 summary["2025_June_or_Dec_holdout_NOT_included"]=True
 summary["production_financial_model_not_changed"]=True
 return data,summary
def main():
 p=argparse.ArgumentParser()
 p.add_argument("--financial",required=True);p.add_argument("--snapshot",required=True)
 p.add_argument("--out",required=True);a=p.parse_args()
 fin=pd.read_csv(a.financial)
 snap=pd.read_parquet(a.snapshot,columns=[
  "date","symbol","close","avg_turnover_63","y6","y6_mature_date","integrity_y6_clean"])
 original,report=original_june_2022(fin,snap)
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 original.to_parquet(out/"original_2022_June_stocks_FY2020_2022_verified_pre_close_3FY_PRIVATE.parquet",index=False)
 (out/"original_June2022_3FY_source_audit.json").write_text(json.dumps(report,indent=2))
 print(json.dumps(report,indent=2),flush=True)
if __name__=="__main__":main()
