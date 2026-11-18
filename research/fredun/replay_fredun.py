"""Replay the two V13 screens with the independently reviewed Fredun history and filings."""
from datetime import datetime, timezone
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import zipfile

import joblib
import numpy as np
import pandas as pd
import requests

import v13_available_financials as available
from v13_archive_retention import download_artifact
from v13_collection import enrich, financial_wide, merge_financial_versions
from v13_features import daily_features
from v13_financial_evidence import REQUIRED, financial_quality
from v13_model import FEATURES, probabilities
from v13_prediction_replay import snapshot
from v13_priority_research import export, rank

SYMBOL = "FREDUN"
ISIN = "INE194R01017"
ASOF = "2026-11-17"
PRICE_INPUT_SHA = "e312184aaca2fcbd605b891f9331cc1e85b1b06220072dfb34dd7e2e0ed68456"
BASE_ARTIFACT = 14421098532
BASE_DIGEST = "sha256:bf5bd323769f614405c37f9204b971f516c6c91155fbefcbcfd6074502b5dac4"
MODEL_SHA = "c9208f89c46d043ee1d74e44dee1c85b9721b7a094e40a0fabf2c5506c246571"
REFS = {
    "annual": (14438197944, "sha256:26cf3ace04ccfa2cd5e75621fd06f88392f749a2ae118a1bf3fbc34881adb2e9",
        "annual_FY26.pdf", "3acc18c150082138792a52e47da566e539f1d921cc2d8c4bcea88c25e2fe38c5"),
    "september": (14438124374, "sha256:da7c86b467b1a22ec62e69c88e2e78a156ef410da1f271bb26d44b0f496f9937",
        "q2_revised_1911.pdf", "e909d4c6cfba5abebc07a4dbea146a4c00b160d45d3ea72cde64989f5bfbaddb"),
}
HERE = Path(__file__).resolve().parent
KEYS = ["source_sha256","period_kind","period_end","security_isin","reporting_mode","available_at_utc"]
PROB = ["p6","p12","joint12","loss12"]
Q4_URLS = [
    "https://www.bseindia.com/xml-data/corpfiling/AttachHis/43005b84-44fb-41ad-aac1-968f0d9a50ab.pdf",
    "https://www.bseindia.com/xml-data/corpfiling/AttachHis/ce7def19-fdec-47f1-a196-932b995e5157.pdf",
    "https://www.bseindia.com/xml-data/corpfiling/AttachLive/43005b84-44fb-41ad-aac1-968f0d9a50ab.pdf",
    "https://www.bseindia.com/xml-data/corpfiling/AttachLive/ce7def19-fdec-47f1-a196-932b995e5157.pdf",
]

def digest(data):
    return hashlib.sha256(data).hexdigest()

def write(path,obj):
    path.write_text(json.dumps(obj,indent=2,allow_nan=False,default=str)+"\n")

def clean(x):
    if isinstance(x,dict):
        return {k:clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):
        return [clean(v) for v in x]
    if isinstance(x,np.generic):
        return clean(x.item())
    if isinstance(x,float) and not np.isfinite(x):
        return None
    return x

def original_docs(work):
    docs={}
    for key,(artifact,signature,pdf,file_sha) in REFS.items():
        payload=work/f"{key}.zip"
        download_artifact(artifact,signature,str(payload))
        with zipfile.ZipFile(payload) as z:
            body=z.read(pdf)
            proof=json.loads(z.read("source_receipts.json"))[pdf]
        if not body.startswith(b"%PDF") or digest(body)!=file_sha or proof["sha256"]!=file_sha:
            raise ValueError("An original financial PDF changed")
        (work/f"{key}.pdf").write_bytes(body)
        docs[key]=proof
    # Use the exchange's actual attachment ID, established by its original
    # announcement. The recognisable reporting tables must occur together.
    attempts=[]
    for url in Q4_URLS:
        observed=datetime.now(timezone.utc).isoformat()
        try:
            response=requests.get(url,headers={"User-Agent":"Mozilla/5.0"},timeout=(6,12))
            attempts.append({"url":url,"status":response.status_code})
            if response.status_code in (403,429):
                break
            if response.status_code!=200 or response.url!=url or not response.content.startswith(b"%PDF"):
                continue
            body=response.content
            path=work/"march_quarters.pdf"
            path.write_bytes(body)
            txt=subprocess.run(["pdftotext","-layout",str(path),"-"],check=True,capture_output=True,text=True).stdout
            compact=re.sub(r"[\s,]+","",txt).lower()
            if not all(value in compact for value in ("fredunpharmaceuticals","539730","1078.00","1047.80","31.12.2025")):
                # If the same statement is scanned, inspect only the likely
                # financial pages and retain the literal digits required.
                pdf_dir=work/"march_pages"
                pdf_dir.mkdir(exist_ok=True)
                subprocess.run(["pdftoppm","-f","2","-l","16","-r","210","-png",str(path),str(pdf_dir/"page")],
                               check=True,capture_output=True,timeout=75)
                for pic in sorted(pdf_dir.glob("*.png")):
                    ocr=subprocess.run(["tesseract",str(pic),"stdout","--psm","6"],capture_output=True,text=True,timeout=25)
                    compact+=re.sub(r"[\s,]+","",ocr.stdout).lower()
            if all(value in compact for value in ("fredunpharmaceuticals","539730","1078.00","1047.80","31.12.2025")):
                docs["march_quarters"]={"url":url,"sha256":digest(body),"observed_utc":observed,"bytes":len(body)}
                break
            attempts[-1]["validation"]="The comparative table could not be confirmed"
        except (requests.RequestException,subprocess.SubprocessError,OSError) as exc:
            attempts.append({"url":url,"error":type(exc).__name__})
    print("FINANCIAL_DOCUMENTS", sorted(docs), json.dumps(attempts), flush=True)
    return docs,attempts

def facts(docs,reviewed):
    data=json.loads((HERE/"financial_observations.json").read_text())
    confirmed=[];unrecovered=[]
    for item in data["periods"]:
        doc=docs.get(item["document"])
        if not doc:
            unrecovered.append(item)
            continue
        # The official archive never inherits a historical availability date
        # from a later review. The dated same-session comparison has its own copy.
        first=pd.Timestamp(doc["observed_utc"])
        public=pd.Timestamp(item["latest_verified_publication_utc"])
        if public>=first or item["period_end"]>str(public.date()):
            raise ValueError("Impossible source clocks")
        for metric,value in item["reported"].items():
            unit=item.get("units",{}).get(metric,"INR_lakh")
            if unit not in {"INR_lakh","INR_per_share"}:
                raise ValueError("Unsupported original units")
            v=float(value)*(100000 if unit=="INR_lakh" else 1)
            confirmed.append(dict(symbol=SYMBOL,security_isin=ISIN,reporting_mode="consolidated",
                period_kind=item["period_kind"],period_end=item["period_end"],period_start=item["period_start"],
                source_url=doc["url"],source_sha256=doc["sha256"],metric=metric,value=v,
                first_retrieved_utc=first.isoformat(),original_publication_utc=public.isoformat(),
                original_document_symbol="539730",identity_source="BSE_DISCLOSURE_CURRENT_EXCHANGE_ISIN",
                available_at_utc=max(first,reviewed).isoformat()))
    original=pd.DataFrame(confirmed)
    if original.empty:
        raise ValueError("No primary original observations were collected")
    research=original.copy()
    research["available_at_utc"]=research.original_publication_utc
    return original,research,unrecovered

def read_base(work):
    path=work/"base_worker.zip"
    download_artifact(BASE_ARTIFACT,BASE_DIGEST,str(path))
    with zipfile.ZipFile(path) as z:
        prefix="v13_state/"
        report=json.loads(z.read(prefix+"latest_run.json"))
        receipt=report["exact_prediction_replay"]["champion"]
        directory=prefix+"prediction_replay/"+receipt["input_parquet_sha256"]+"/"
        bar_input=z.read(directory+"inputs.parquet")
        score_raw=z.read(directory+"champion_"+receipt["receipt_sha256"]+".json")
        model_data=z.read(prefix+"active_heads.joblib")
        if (digest(bar_input)!=receipt["input_parquet_sha256"] or digest(score_raw)!=receipt["receipt_sha256"]
                or digest(model_data)!=MODEL_SHA or receipt["model_sha256"]!=MODEL_SHA
                or report.get("asof")!=ASOF or int(report["workflow_run"])!=44279573621):
            raise ValueError("The original model cohort could not be authenticated")
        frame=pd.read_parquet(io.BytesIO(bar_input))
        wide=pd.read_parquet(io.BytesIO(z.read(prefix+"financial_wide.parquet")))
        existing={s:json.loads(z.read(prefix+f"priority_research/latest_RSI{s}.json")) for s in (70,50)}
    score=json.loads(score_raw)
    model_path=work/"active_heads.joblib"
    model_path.write_bytes(model_data)
    heads=joblib.load(model_path)
    if frame["isin"].tolist()!=score["isins"] or ISIN in score["isins"] or len(frame)!=1634:
        raise ValueError("Unexpected reference cohort membership")
    for k,v in probabilities(heads,frame).items():
        np.testing.assert_allclose(v,score["probabilities"][k],atol=1e-12,rtol=0)
    return frame,wide,heads,model_path,existing,report

def price_features(frame,wide,report,asof,cutoff):
    p=HERE/"fredun_verified_history.csv"
    if digest(p.read_bytes())!=PRICE_INPUT_SHA:
        raise ValueError("The reviewed price sequence changed")
    bars=pd.read_csv(p,parse_dates=["date"])
    if not (bars["isin"].eq(ISIN).all() and bars["symbol"].eq(SYMBOL).all()
            and bars.date.is_unique and bars.date.is_monotonic_increasing
            and len(bars)==279 and int(bars.price_source_venue.eq("BSE").sum())==214):
        raise ValueError("Reviewed exchange source identities or coverage changed")
    features=daily_features(bars,asof)
    if len(features)<127 or str(features.date.max().date())!=asof:
        raise ValueError("The reviewed history lacks the requested date")
    target=features.iloc[[-1]].copy().reset_index(drop=True)
    if not bool(target.feature_eligible.iloc[0]):
        raise ValueError("Fredun fails the existing technical feature gate")
    src=target.iloc[0]
    if src.price_source_venue!="NSE" or (asof==ASOF and src.source_sha256!=report["steps"]["official"]["source_sha256"]):
        raise ValueError("The decision bar does not match the official NSE source")
    for c in frame.columns.difference(target.columns):
        target[c]=np.nan
    target["information_cutoff_utc"]=cutoff
    target["series"]="EQ"
    target["source_status"]="VALID_POSITIVE_VOLUME_SESSION"
    target["symbol_official"]=SYMBOL
    target["official_source_sha256"]=src.source_sha256
    target["source_close_INR"]=src.close
    target["available_sector_kind"]="NONFINANCIAL"
    target["source_volume"]=src.volume
    target["liquidity_volume_interaction"]=target.volume_confirmation*target.log_liquidity63
    markets=pd.concat([frame,target],ignore_index=True)
    for raw,factor in [("log_ret63","relative_strength63"),("log_ret20","relative_strength20"),
                       ("momentum126","relative_strength126"),("momentum252_skip21","relative_strength252_skip21")]:
        if raw in markets:
            target[factor]=markets[raw].rank(pct=True).iloc[-1]
    target["market_breadth63"]=markets.log_ret63.gt(0).mean()
    target["market_median_ret63"]=markets.log_ret63.median()
    target["market_dispersion63"]=markets.log_ret63.std()
    target["trend_regime_interaction"]=target.relative_strength63*(target.market_breadth63-.5)
    target=enrich(target,wide)
    for c in FEATURES:
        if c not in target:
            target[c]=np.nan
    return target

def candidate_probs(heads,frame):
    d=frame.copy()
    for k,v in probabilities(heads,d).items():
        d[k]=v
    return d

def archive_facts(out, work, fact_long, docs):
    from v13_checkpoint import restore, save
    from v13_frozen_integrity import FrozenGuard
    from v13_apollo_repair_run import protected
    root=work/"persistent_state"
    if not restore(root):
        raise ValueError("V13 shared state could not be restored")
    before=protected(root)
    guard=FrozenGuard(root)
    wide_path=root/"financial_wide.parquet"
    previous=pd.read_parquet(wide_path)
    current=merge_financial_versions(previous,financial_wide(fact_long))
    # Every prior row, including other Fredun vintages, remains identical.
    current_index=current.set_index(KEYS)
    old=previous.set_index(KEYS)
    old_compare=current_index.loc[old.index,old.columns]
    pd.testing.assert_frame_equal(old,old_compare,check_exact=True,check_dtype=False)
    current.to_parquet(wide_path,index=False,compression="zstd",compression_level=9)
    history=root/"data_corrections"/"fredun_20261118"
    history.mkdir(parents=True,exist_ok=True)
    for file in ("fredun_verified_history.csv","fredun_history_verification.json"):
        (history/file).write_bytes((HERE/file).read_bytes())
    for file in ("financial_observations.json","fredun_research_result.json",
                 "financial_original_receipts.json","fredun_facts_verified.parquet"):
        (history/file).write_bytes((out/file).read_bytes())
    for filename in ("latest_RSI70.json","latest_RSI50.json"):
        (history/filename).write_bytes((out/"priority_research"/filename).read_bytes())
    guard.check(pd.read_parquet(root/"prospective_observations.parquet"))
    if protected(root)!=before:
        raise ValueError("A frozen forecast or model changed")
    receipt=save(root)
    if not receipt.get("remote_publication_confirmed"):
        raise ValueError("Recovered facts were not confirmed in the research archive")
    write(out/"persistence_receipt.json",{"archive":receipt,"protected_model_and_registered_forecasts_verified":True,
          "previous_financial_rows_preserved":len(previous),"accepted_fredun_fact_versions":len(fact_long),
          "fredun_history_saved_as_reviewed_research":True})

def run(persist=False):
    work=Path("fredun_work")
    out=Path("fredun_result")
    work.mkdir(exist_ok=True);out.mkdir(exist_ok=True)
    base,wide,heads,model_path,existing,report=read_base(work)
    docs,attempts=original_docs(work)
    reviewed=pd.Timestamp.now(tz="UTC")
    original,research,not_confirmed=facts(docs,reviewed)
    write(out/"financial_original_receipts.json",{"sources":docs,"quarterly_comparative_attempts":attempts})
    original.to_csv(out/"fredun_facts_verified.csv",index=False)
    original.to_parquet(out/"fredun_facts_verified.parquet",index=False,compression="zstd")
    (out/"financial_observations.json").write_bytes((HERE/"financial_observations.json").read_bytes())
    for file in ("fredun_verified_history.csv","fredun_history_verification.json"):
        (out/file).write_bytes((HERE/file).read_bytes())
    comparison_wide=merge_financial_versions(wide,financial_wide(research))
    current_wide=merge_financial_versions(wide,financial_wide(original))
    baseline=candidate_probs(heads,base)
    for threshold,(screen,_,_) in available.screen_many(baseline,wide,thresholds=(70,50)).items():
        ranked=rank(screen,screen.available_research_eligible)
        chosen=ranked[ranked.priority_rank.notna()].sort_values("priority_rank")
        if chosen["isin"].tolist()!=[x["isin"] for x in existing[threshold]["records"]]:
            raise ValueError("Original screen rankings failed reproduction")
    cutoff=pd.Timestamp(ASOF,tz="Asia/Kolkata")+pd.Timedelta(hours=22)
    added=price_features(base,comparison_wide,report,ASOF,cutoff.tz_convert("UTC").isoformat())
    fields=list(dict.fromkeys([*base.columns,*added.columns]))
    combined=pd.concat([base.reindex(columns=fields),added.reindex(columns=fields)],ignore_index=True)
    scored=candidate_probs(heads,combined)
    for c in PROB:
        np.testing.assert_allclose(scored.iloc[:-1][c],baseline[c],atol=1e-12,rtol=0)
    receipt=snapshot(combined,scored,model_path,out,"champion")
    available.export(scored,comparison_wide,out,receipt)
    export(scored,comparison_wide,out,receipt)
    screens={}
    for threshold,(assessed,assessments,audit) in available.screen_many(scored,comparison_wide,thresholds=(70,50)).items():
        d=rank(assessed,assessed.available_research_eligible)
        pool=d[d.priority_score.notna()].sort_values(["priority_score","p6","joint12","isin"],ascending=[False,False,False,True])
        pool_ids=pool["isin"].tolist()
        row=next(r for r in assessments if r["isin"]==ISIN)
        screens[str(threshold)]={"full_rank":pool_ids.index(ISIN)+1 if ISIN in pool_ids else None,
                         "eligible_count":len(pool),"selected_top25":ISIN in pool_ids[:25],
                         "exclusion_reasons":row["exclusion_reasons"],"financial_tier":row["tier"],
                         "strict_weaknesses":row["strict_failed_checks"],"safety_failures":row["current_safety_failures"],
                         "financial_checks":row["check_states"],"approved_source_roles":row["sources"]}
    # Latest stock-only estimate is also computed at its own official close
    # and actual evidence clock. Prior cohort context is used as reference,
    # so this distinct date has no comparable same-session assigned rank.
    latest_cutoff=min(reviewed,pd.Timestamp("2026-11-18T16:30:00Z"))
    latest=price_features(base,current_wide,report,"2026-11-18",latest_cutoff.isoformat())
    latest_pred=candidate_probs(heads,latest).iloc[0]
    orig=scored.iloc[-1]
    view_keys=["source_close_INR","source_volume","rsi14","distance_ma50","distance_ma200",
               "log_ret63","median_turnover63","p6","p12","joint12","loss12",
               "eps_diluted_continuing","quarter_sales_yoy","quarter_pat_yoy","quarter_pat_margin",
               "annual_CFO_PAT","dated_debt_equity","annual_EBIT_proxy_interest_coverage","four_profitable_quarters"]
    result={"source_date":ASOF,"latest_stock_source_date":"2026-11-18","reviewed_utc":reviewed.isoformat(),
            "source_worker_run":44279573621,"original_comparison_shares":len(base),"comparison_shares":len(combined),
            "recovered_market_observations":279,"history_sources":{"BSE_before_first_NSE_print":214,"NSE":65},
            "data_availability_basis":"A retrospective same-date comparison with only then-public disclosures. Actual receipt recorded on 18 November.",
            "model_sha256":MODEL_SHA,"observed_rows":len(original),"source_models_and_peer_probabilities_reproduced":True,
            "financial_quality":financial_quality(orig.to_dict()),"screens":screens,
            "source_quality":{"primary_financial_documents":docs,"unconfirmed_source_periods":not_confirmed},
            "fredun_comparison":{k:orig.get(k) for k in view_keys},"latest_stock_only_reference":{k:latest_pred.get(k) for k in view_keys},
            "latest_stock_only_reference_uses_prior_market_reference":True,"formal_new_forecast":False,"accuracy_approved":False}
    result=clean(result)
    write(out/"fredun_research_result.json",result)
    print("FREDUN_RESULT",json.dumps({k:result[k] for k in
        ["financial_quality","fredun_comparison","latest_stock_only_reference","comparison_shares"]}),flush=True)
    if persist:
        archive_facts(out,work,original,docs)
    return result

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--persist",action="store_true")
    run(parser.parse_args().persist)
