"""PIT short-horizon annual financial feature pilot from strict validated XBRL.

Only previously confirmed company-level annual INR revenue & PAT facts are used.
Missing long-term history is missing, not zero, and historical fold selection
uses strict 15:30 Asia/Kolkata source availability.
"""
import argparse,json,math
from pathlib import Path
import pandas as pd

def close_utc(value):
    return (pd.Timestamp(value).normalize().tz_localize("Asia/Kolkata")
            +pd.Timedelta(hours=15,minutes=30)).tz_convert("UTC")

def positive_growth(new,old):
    if pd.isna(new) or pd.isna(old) or new<=0 or old<=0:return None
    return 100*(new/old-1)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--annual-numerics",required=True)
    p.add_argument("--fold-date",default="2024-12-31")
    p.add_argument("--output",required=True)
    a=p.parse_args()
    out=Path(a.output);out.mkdir(exist_ok=True,parents=True)
    data=pd.read_csv(a.annual_numerics,dtype={"symbol":str,"mode":str})
    expected={"symbol","mode","fy_end","available_at_utc","revenue","pat","xbrl_url","fact_audit","status"}
    if not expected.issubset(data):raise SystemExit("Missing strict audit fields "+str(expected-set(data)))
    data["fy"]=pd.to_datetime(data["fy_end"],utc=True,errors="coerce",format="mixed")
    data["pub"]=pd.to_datetime(data["available_at_utc"],utc=True,errors="coerce",format="mixed")
    data["revenue"]=pd.to_numeric(data["revenue"],errors="coerce")
    data["pat"]=pd.to_numeric(data["pat"],errors="coerce")
    cutoff=close_utc(a.fold_date)
    valid=(data["fy"].notna()&data["pub"].notna()&(data["fy"]<=cutoff)&
           (data["pub"]<=cutoff)&(data["pub"]>=data["fy"])&
           data["xbrl_url"].astype(str).str.match(r"^https://nsearchives\.nseindia\.com/.*\.xml$",case=False)&
           data["status"].eq("complete")&data["mode"].isin(["standalone","consolidated"])&
           data["revenue"].notna()&data["pat"].notna()&
           data["revenue"].ge(0))
    def verified_audit(row):
        try:
            audit=json.loads(row)
            keys=("revenue","pat")
            return all(
                audit[k]["unit"] and audit[k]["period_interpretation"] in
                    ("explicit_annual","annual_FourD_YTD")
                and audit[k]["tag"] and audit[k]["context"]
                and audit[k]["source_is_annual"]
                for k in keys
            )
        except (TypeError,ValueError,KeyError):return False
    valid &=data["fact_audit"].apply(verified_audit)
    clean=data[valid].sort_values("pub").drop_duplicates(
        ["symbol","mode","fy"],keep="first").copy()
    features=[];provenance=[]
    for (symbol,mode),rows in clean.groupby(["symbol","mode"]):
        rows=rows.sort_values("fy",ascending=False)
        for now in rows.itertuples(index=False):
            prior=rows[(rows["fy"]<now.fy)&
                       ((now.fy-rows["fy"]).dt.days.between(335,395))]
            if prior.empty:continue
            prev=prior.iloc[0]
            revenue_growth=positive_growth(now.revenue,prev.revenue)
            pat_growth=positive_growth(now.pat,prev.pat)
            p_now=now.pat/now.revenue if now.revenue>0 else None
            p_prev=prev.pat/prev.revenue if prev.revenue>0 else None
            row={
                "fold_date":a.fold_date,"symbol":symbol,"reporting_mode":mode,
                "latest_fy_end":str(now.fy.date()),"prior_fy_end":str(prev["fy"].date()),
                "revenue_yoy_pct":revenue_growth,
                "pat_yoy_pct":pat_growth,
                "pat_margin_latest_pct":100*p_now if p_now is not None else None,
                "pat_margin_delta_pp":100*(p_now-p_prev) if p_now is not None and p_prev is not None else None,
                "profit_turnaround":bool(prev["pat"]<=0<now.pat),
                "earlier_profit_negative":bool(prev["pat"]<=0),
                "latest_fiscal_age_days":int((cutoff-now.fy).days),
                "source_available_utc":now.available_at_utc,
                "numeric_document_count":2,
                "qa_status":"CANDIDATE_SHORT_HORIZON_ONLY",
            }
            features.append(row)
            provenance.append({
                "symbol":symbol,"reporting_mode":mode,"latest_fy_end":str(now.fy.date()),
                "source_latest":now.xbrl_url,"source_prev":prev["xbrl_url"],
                "available_latest":now.available_at_utc,"available_prev":prev["available_at_utc"],
            })
            break
    pd.DataFrame(features).to_csv(out/"short_horizon_financial_PIT_research_features.csv",index=False)
    pd.DataFrame(provenance).to_csv(out/"short_horizon_financial_source_links.csv",index=False)
    summary={
        "scope":"STAGED_ONLY_NOT_PRODUCTION",
        "original_numeric_rows":len(data),
        "strict_annual_document_rows":len(clean),
        "short_horizon_company_mode_features":len(features),
        "companies":len(set(x["symbol"] for x in features)),
        "companies_with_revenue_yoy":sum(x["revenue_yoy_pct"] is not None for x in features),
        "companies_with_PAT_yoy":sum(x["pat_yoy_pct"] is not None for x in features),
        "companies_profit_turnaround":sum(x["profit_turnaround"] for x in features),
        "asof_utc":cutoff.isoformat(),
        "partial_5y_history_imputed":False,
        "longterm_growth_calculated":False,
        "original_model_weights_unchanged":True,
        "production_approved":False,
    }
    (out/"short_horizon_financial_summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)
    if summary["companies_with_revenue_yoy"]<8:
        raise SystemExit("Insufficient source-verified short-horizon financial candidates")
if __name__=="__main__":main()
