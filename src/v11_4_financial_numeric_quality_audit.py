"""Report missing/implausible original NSE financial XBRL facts; never impute zeros."""
import argparse,json
from pathlib import Path
import pandas as pd

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--annual",required=True)
    p.add_argument("--integrated",required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    sources=[
        ("annual",pd.read_csv(a.annual),["revenue","pat"],"fy_end"),
        ("integrated_2025",pd.read_csv(a.integrated),["q_revenue_raw_units","q_pat_raw_units"],"quarter_end_index"),
    ]
    reports=[];missing=[]
    for name,frame,nums,period in sources:
        for n in nums:
            frame[n]=pd.to_numeric(frame[n],errors="coerce")
        bad=frame[frame[nums].isna().any(axis=1)].copy()
        negative_rev=frame[frame[nums[0]].notna()&(frame[nums[0]]<0)]
        present=frame[frame[nums].notna().all(axis=1)]
        large_pat=present[present[nums[0]].abs().gt(0)&(present[nums[1]].abs()/present[nums[0]].abs()>3)]
        for _,r in bad.iterrows():
            missing.append({
                "source":name,"symbol":str(r["symbol"]),
                "period":str(r[period]),
                "missing_fields":",".join(n for n in nums if pd.isna(r[n])),
                "original_source_document":str(r.get("source_url") or r.get("xbrl_url") or ""),
            })
        reports.append({
            "source":name,"rows":len(frame),
            "symbols":frame["symbol"].nunique(),
            "full_core_revenue_PAT_rows":len(present),
            "partial_or_missing_core_rows":len(bad),
            "negative_revenue_rows_for_review":len(negative_rev),
            "PAT_gt_3_times_revenue_rows_for_review":len(large_pat),
            "source_value_unit_normalization":"NOT_CONFIRMED",
        })
    pd.DataFrame(missing,columns=["source","symbol","period","missing_fields","original_source_document"]).to_csv(out/"source_numeric_missing_facts.csv",index=False)
    summary={
        "status":"NUMERIC_DATA_QA_DIAGNOSTICS_ONLY",
        "sources":reports,
        "no_missing_values_filled":True,
        "monetary_cross_taxonomy_comparability_approved":False,
        "suggested_next_step":"Inspect missing facts and normalize document units/financial concepts before calculating CAGR/ROCE/ROE for backtest.",
    }
    (out/"numeric_fact_quality_summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)
    print(pd.DataFrame(missing).to_string(index=False),flush=True)
if __name__=="__main__":main()
