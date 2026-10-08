"""Stage NSE FY2025 Integrated FourD annual filing INDEX into older annual index.

Coverage-only. A 23/24-document FY2025 numerical pilot licenses an *index
feasibility test*, not a claim that every FY2025 company value was verified.
"""
import argparse,json
from pathlib import Path
import pandas as pd

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--original",required=True)
    p.add_argument("--integrated",required=True)
    p.add_argument("--sample-qa",required=True)
    p.add_argument("--out",required=True)
    a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    qa=json.loads(Path(a.sample_qa).read_text())
    if qa.get("verified_FourD_YTD",0)<18 or qa.get("sampled",0)<24:
        raise SystemExit("FY2025 annual YTD independent sample verification insufficient")
    old=pd.read_csv(a.original,dtype=str).fillna("")
    inte=pd.read_parquet(a.integrated)
    end=pd.to_datetime(inte["period_end"],utc=True,errors="coerce")
    pub=pd.to_datetime(inte["available_at_utc"],utc=True,errors="coerce")
    urls=inte["xbrl_url"].astype(str)
    quarter=inte[
        (end==pd.Timestamp("2025-03-31T00:00:00Z"))&
        pub.notna()&(pub>=end)&(pub<=pd.Timestamp("2025-12-31T10:00:00Z"))&
        urls.str.match(r"^https://nsearchives\.nseindia\.com/.*\.xml$",case=False)&
        ~urls.str.contains(r"/INTEGRATED_FILING_BANKING_",case=False)
    ].copy()
    quarter["pub"]=pd.to_datetime(quarter["available_at_utc"],utc=True)
    quarter["mode_priority"]=quarter["consolidated"].astype(str).str.lower().eq("consolidated").astype(int)
    quarter=quarter.sort_values(["symbol","mode_priority","pub"],ascending=[True,False,True])
    quarter=quarter.drop_duplicates(["symbol"],keep="first")
    new=pd.DataFrame({
        "symbol":quarter["symbol"].astype(str).str.upper().str.strip(),
        "fy_end":"2025-03-31",
        "available_at_utc":quarter["available_at_utc"],
        "xbrl_url":quarter["xbrl_url"],
        "consolidated":quarter["consolidated"],
        "filed_period":"Annual",
        "source":"NSE_Integrated_2025Q4_FourD_YTD_INDEX_CANDIDATE_ONLY",
    })
    old["source"]=old.get("source",pd.Series(["NSE_legacy_annual_filing_index"]*len(old)))
    comb=pd.concat([old,new],ignore_index=True,sort=False).fillna("")
    comb=comb.drop_duplicates(["symbol","fy_end","available_at_utc","xbrl_url"])
    comb.to_csv(out/"NSE_annual_index_plus_FY2025_integrated_Q4_REHEARSAL.csv",index=False)
    summary={
        "source_old_annual_index_rows":len(old),
        "FY2025_integrated_non_bank_Q4_candidates":len(new),
        "index_rows_combined":len(comb),
        "old_consolidated_labels":old["consolidated"].astype(str).value_counts().head(7).to_dict(),
        "integrated_consolidated_labels":new["consolidated"].astype(str).value_counts().head(7).to_dict(),
        "new_symbols_also_in_old_index":len(set(new["symbol"]) & set(old["symbol"])),
        "sample_verification":qa.get("verified_FourD_YTD"),
        "source_gap":"Legacy annual endpoint omits post-March-2025 financials; Q4 integrated FourD is fiscal YTD",
        "production_approved":False,
        "financial_5y7y_numerics_marketwide_verified":False,
        "note":"No financial values created; index availability source only. Bank annual taxonomy handled separately.",
    }
    (out/"FY2025_annual_index_bridge_summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)
    if len(new)<500:raise SystemExit("Insufficient FY2025 NSE integrated annual filing source")
if __name__=="__main__":main()
