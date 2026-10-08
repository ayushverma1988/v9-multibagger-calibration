from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def is_missing(v):
    if v is None:return True
    if isinstance(v,(float,np.floating)) and not np.isfinite(v):return True
    return pd.isna(v)

def eval_rule(v,op,t):
    if is_missing(v):return None
    if isinstance(t,bool):
        vv=bool(v)
    else:
        try:vv=float(v)
        except:return None
    if op==">":return vv>float(t)
    if op==">=":return vv>=float(t)
    if op=="<":return vv<float(t)
    if op=="<=":return vv<=float(t)
    if op=="==":return vv==t
    return None

def family_eval(row,rules):
    results=[]
    for field,op,target in rules:
        val=row.get(field,np.nan)
        passed=eval_rule(val,op,target)
        results.append((field,passed,val))
    available=sum(p is not None for _,p,_ in results)
    passed=sum(p is True for _,p,_ in results)
    failed=sum(p is False for _,p,_ in results)
    total=len(results)
    coverage=available/total if total else 0
    conditional=passed/available if available else 0
    score=conditional*math.sqrt(coverage) if coverage>0 else 0
    full_pass=(available==total and failed==0)
    signal=(coverage>=0.70 and conditional>=0.80)
    return {
        "score":float(score),"coverage":float(coverage),"conditional_pass_rate":float(conditional),
        "passed":int(passed),"failed":int(failed),"unknown":int(total-available),
        "full_pass":bool(full_pass),"signal":bool(signal),
        "detail":json.dumps([
            {"field":f,"status":"PASS" if p is True else "FAIL" if p is False else "UNKNOWN","value":None if is_missing(v) else v}
            for f,p,v in results
        ],default=str)
    }

def num_col(df,name):
    if name in df.columns:
        return pd.to_numeric(df[name],errors="coerce")
    return pd.Series(np.nan,index=df.index,dtype=float,name=name)

def bool_col(df,name):
    if name in df.columns:
        return df[name].astype("boolean")
    return pd.Series(pd.NA,index=df.index,dtype="boolean",name=name)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",required=True)
    ap.add_argument("--base-audit",required=True)
    ap.add_argument("--quarterly",required=True)
    ap.add_argument("--annual",required=True)
    ap.add_argument("--chart",required=True)
    ap.add_argument("--valuation",required=True)
    ap.add_argument("--ownership",required=True)
    ap.add_argument("--output",required=True)
    args=ap.parse_args()

    cfg=json.load(open(args.config))
    dfs=[]
    paths=[args.base_audit,args.quarterly,args.annual,args.chart,args.valuation,args.ownership]
    for idx,path in enumerate(paths):
        try:
            d=pd.read_csv(path)
        except pd.errors.EmptyDataError:
            d=pd.DataFrame(columns=["symbol"])
        if "symbol" not in d.columns:
            if idx==0:
                raise RuntimeError(f"Base audit missing symbol column: {path}")
            d=pd.DataFrame(columns=["symbol"])
        d["symbol"]=d["symbol"].astype(str).str.upper().str.strip()
        dfs.append(d)
    x=dfs[0]
    for d in dfs[1:]:
        # Avoid accidental column shadowing. Base audit keeps its own fields unless
        # a more specific multibagger collector has a different name.
        overlap=[c for c in d.columns if c!="symbol" and c in x.columns]
        d=d.drop(columns=overlap,errors="ignore")
        x=x.merge(d,on="symbol",how="left")

    # Exact/derived fields for Rule 3 and common overlays.
    # Missing optional inputs must stay UNKNOWN; never allow a scalar NaN to
    # crash the whole evaluation or to become an implicit pass/fail.
    rev_yoy=num_col(x,"revenue_yoy")
    rev_2q=num_col(x,"revenue_vs_2q_back")
    pat_yoy=num_col(x,"pat_yoy")

    x["sales_latest_ge_yoy_quarter"]=rev_yoy.ge(0).astype("boolean")
    x.loc[rev_yoy.isna(),"sales_latest_ge_yoy_quarter"]=pd.NA
    x["sales_latest_ge_2q_back"]=rev_2q.ge(0).astype("boolean")
    x.loc[rev_2q.isna(),"sales_latest_ge_2q_back"]=pd.NA
    x["yoy_quarterly_profit_growth"]=pat_yoy

    for bc in ["pat_latest_gt_preceding","pat_preceding_gt_2q_back"]:
        x[bc]=bool_col(x,bc)

    # Screener PEG is a growth-adjusted valuation. Use 5y PAT CAGR where available,
    # then 3y PAT CAGR. Keep it explicitly named a proxy.
    pg5=num_col(x,"profit_growth_5y")
    pg3=num_col(x,"profit_growth_3y")
    growth=pg5.where(pg5>0,pg3.where(pg3>0))
    pe=num_col(x,"company_pe")
    x["peg_ratio_proxy"]=pe/(growth*100.0)
    x["peg_ratio"]=x["peg_ratio_proxy"]

    # Pledge is usable as a hard rule only when XBRL confidence is adequate.
    pc=num_col(x,"pledged_pct_confidence")
    if "pledged_pct" in x:
        x.loc[pc.lt(0.70)|pc.isna(),"pledged_pct"]=np.nan

    families=[
        ("quality_compounder","family_1_quality_compounder"),
        ("recovery_value_setup","family_2_recovery_value_setup"),
        ("earnings_acceleration","family_3_earnings_acceleration"),
    ]
    for outname,key in families:
        vals=[]
        for _,r in x.iterrows():
            vals.append(family_eval(r,cfg[key]["hard_rules"]))
        for k in ["score","coverage","conditional_pass_rate","passed","failed","unknown","full_pass","signal","detail"]:
            x[f"{outname}_{k}"]=[z[k] for z in vals]

    fw=cfg["integration"]["family_weights"]
    fam=(
        fw["quality_compounder"]*x["quality_compounder_score"]
        +fw["recovery_value_setup"]*x["recovery_value_setup_score"]
        +fw["earnings_acceleration"]*x["earnings_acceleration_score"]
    )

    # User's common checkpoints. These are explicit diagnostics and a small
    # confidence component; sector-sensitive fixed assets are not a universal hard fail.
    common_tests=pd.DataFrame(index=x.index)
    ph=num_col(x,"promoter_holding")
    common_tests["promoter_50"]=ph.ge(0.50).where(ph.notna())
    cfo=num_col(x,"cfo_to_pat_last_year")
    common_tests["positive_cfo"]=cfo.gt(0).where(cfo.notna())
    rap=num_col(x,"receivables_to_pat")
    common_tests["receivable_strict"]=rap.lt(0.10).where(rap.notna())
    common_tests["above_both_dma"]=bool_col(x,"price_above_both_dma")
    common_tests["reserves_gt_borrowings"]=bool_col(x,"reserves_gt_borrowings")
    fag=num_col(x,"fixed_assets_yoy_growth")
    common_tests["fixed_assets_growing"]=fag.gt(0).where(fag.notna())

    cpass=common_tests.eq(True).sum(axis=1)
    cavail=common_tests.notna().sum(axis=1)
    ccond=(cpass/cavail.replace(0,np.nan)).fillna(0)
    ccov=cavail/common_tests.shape[1]
    x["common_quality_score"]=(ccond*np.sqrt(ccov)).clip(0,1)
    x["common_quality_coverage"]=ccov

    fam_weight=float(cfg["integration"].get("family_composite_weight_within_overlay",0.90))
    common_weight=float(cfg["integration"].get("common_quality_weight_within_overlay",0.10))
    overlay=(fam_weight*fam+common_weight*x["common_quality_score"]).clip(0,1)

    signals=(
        x["quality_compounder_signal"].astype(int)
        +x["recovery_value_setup_signal"].astype(int)
        +x["earnings_acceleration_signal"].astype(int)
    )
    x["multibagger_family_signal_count"]=signals
    bonus=np.select([signals>=3,signals>=2],[0.20,0.10],default=0.0)
    x["multibagger_rule_overlay_score"]=(overlay+bonus).clip(0,1)
    x["multibagger_overlap_bonus"]=bonus

    bw=float(cfg["integration"]["base_v11_4_weight"])
    ow=float(cfg["integration"]["multibagger_rule_overlay_weight"])
    base=num_col(x,"sleeve_score")
    if base.isna().all():
        base=num_col(x,"v11_4_score_raw")
    base=base.fillna(0)
    x["v11_4_mb_challenger_score"]=(bw*base+ow*x["multibagger_rule_overlay_score"]).clip(0,1)
    x["v11_4_mb_challenger_rank"]=x["v11_4_mb_challenger_score"].rank(ascending=False,method="first").astype(int)

    # Chart-trigger diagnostics useful for entry timing, not causal eligibility.
    x["chart_trigger_ready"]=(
        bool_col(x,"price_gt_dma50_prev").fillna(False).astype(bool)
        & bool_col(x,"dma50_slope_positive").fillna(False).astype(bool)
        & (
            bool_col(x,"price_gt_dma200_prev").fillna(False).astype(bool)
            | bool_col(x,"price_lt_dma200_prev").fillna(False).astype(bool)
        )
    )

    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    x.sort_values("v11_4_mb_challenger_rank").to_csv(out,index=False)
    x.sort_values("v11_4_mb_challenger_rank").head(30).to_csv(out.parent/"multibagger_rule_top30.csv",index=False)
    overlap=x[x["multibagger_family_signal_count"]>=2].sort_values("v11_4_mb_challenger_rank")
    overlap.to_csv(out.parent/"multi_family_overlap.csv",index=False)

    summary={
        "companies":int(len(x)),
        "family_full_pass":{
            "quality_compounder":int(x["quality_compounder_full_pass"].sum()),
            "recovery_value_setup":int(x["recovery_value_setup_full_pass"].sum()),
            "earnings_acceleration":int(x["earnings_acceleration_full_pass"].sum()),
        },
        "family_signal":{
            "quality_compounder":int(x["quality_compounder_signal"].sum()),
            "recovery_value_setup":int(x["recovery_value_setup_signal"].sum()),
            "earnings_acceleration":int(x["earnings_acceleration_signal"].sum()),
        },
        "two_or_more_family_signals":int((x["multibagger_family_signal_count"]>=2).sum()),
        "three_family_signals":int((x["multibagger_family_signal_count"]>=3).sum()),
        "median_coverage":{
            "quality_compounder":float(x["quality_compounder_coverage"].median()),
            "recovery_value_setup":float(x["recovery_value_setup_coverage"].median()),
            "earnings_acceleration":float(x["earnings_acceleration_coverage"].median()),
        },
        "note":"Challenger only. It cannot create V11.4 causal eligibility. Historical PIT validation is required before any promotion."
    }
    json.dump(summary,open(out.parent/"multibagger_rule_summary.json","w"),indent=2)
    print(json.dumps(summary,indent=2))

if __name__=="__main__":
    main()
