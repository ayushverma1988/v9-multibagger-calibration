from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import calibrate_v9_2 as base


def jlist(v):
    try:
        x=json.loads(v) if isinstance(v,str) else v
        return x if isinstance(x,list) else []
    except Exception:
        return []


def primary_escalations(ev:pd.DataFrame)->pd.DataFrame:
    x=ev.copy()
    x["published_ts"]=pd.to_datetime(x["published_ts"],utc=True,errors="coerce")
    x["stage_weight_num"]=pd.to_numeric(x["stage_weight"],errors="coerce").fillna(0)
    x["source_tier_num"]=pd.to_numeric(x["source_tier"],errors="coerce").fillna(3)
    x=x[
        x["source_tier_num"].eq(1)
        & ((x["stage_weight_num"]>=0.40) | x["catalyst_types"].fillna("[]").astype(str).ne("[]"))
        & x["published_ts"].notna()
    ].copy()

    expanded=[]
    for r in x.itertuples(index=False):
        types=jlist(getattr(r,"catalyst_types","[]"))
        if not types:
            # meaningful execution stage without a parsed family
            types=["stage_only"]
        for typ in types:
            expanded.append({
                "symbol":str(r.symbol).upper(),
                "type":typ,
                "published_ts":r.published_ts,
                "stage":getattr(r,"stage",None),
                "stage_weight":float(getattr(r,"stage_weight_num")),
                "title":getattr(r,"title",None),
                "url":getattr(r,"url",None),
            })
    z=pd.DataFrame(expanded)
    if z.empty:return z

    out=[]
    for (sym,typ),g in z.groupby(["symbol","type"]):
        g=g.sort_values("published_ts")
        prior=-1.0
        for r in g.itertuples(index=False):
            # A new type starts its own clock. Within a type, only stage
            # progression (or repeat-order stage) resets the clock.
            if r.stage_weight > prior + 0.049 or r.stage_weight>=1.10:
                out.append(r._asdict())
                prior=max(prior,r.stage_weight)
    return pd.DataFrame(out)


def penalty_from_returns(current_ret,max_ret,days):
    if not np.isfinite(current_ret):current_ret=0.0
    if not np.isfinite(max_ret):max_ret=current_ret
    absorbed=max(current_ret,0.60*max_ret)
    # Very fresh catalysts get a small grace period unless price already jumped.
    if days<=5 and absorbed<0.20:
        return 0.0
    knots=[(-0.10,0.0),(0.10,0.05),(0.25,0.18),(0.50,0.38),(0.80,0.58),(1.20,0.75),(2.00,0.90),(3.00,1.00)]
    if absorbed<=knots[0][0]:return knots[0][1]
    for (x0,y0),(x1,y1) in zip(knots[:-1],knots[1:]):
        if absorbed<=x1:
            t=(absorbed-x0)/(x1-x0)
            return float(y0+t*(y1-y0))
    return 1.0


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--evidence",required=True)
    ap.add_argument("--output",required=True)
    ap.add_argument("--start-year",type=int,default=None)
    ap.add_argument("--end-year",type=int,default=None)
    args=ap.parse_args()

    ev=pd.read_parquet(args.evidence)
    esc=primary_escalations(ev)
    if esc.empty:
        raise SystemExit("No primary catalyst escalations available")

    end_year=int(args.end_year or pd.Timestamp.today().year)
    start_year=int(args.start_year or max(end_year-2,2010))
    daily=base.load_market(start_year,end_year,None)
    daily["date"]=pd.to_datetime(daily["date"]).dt.normalize()
    current_date=pd.Timestamp(daily["date"].max()).normalize()

    results=[]
    for sym,g in esc.groupby("symbol"):
        m=daily[daily["symbol"].astype(str).str.upper().eq(sym)].sort_values("date")
        if m.empty:continue
        # Pick the latest true stage escalation across catalyst families.
        a=g.sort_values("published_ts").iloc[-1]
        ad=pd.Timestamp(a["published_ts"]).tz_convert("Asia/Kolkata").tz_localize(None).normalize()
        post=m[m["date"]>=ad]
        if post.empty:continue
        p0=float(post.iloc[0]["adj_close"])
        cur=float(m.iloc[-1]["adj_close"])
        peak=float(post["adj_close"].max())
        ret=cur/p0-1 if p0>0 else np.nan
        maxret=peak/p0-1 if p0>0 else np.nan
        days=max((current_date-ad).days,0)
        results.append({
            "symbol":sym,
            "catalyst_anchor_date":str(ad.date()),
            "catalyst_anchor_type":a["type"],
            "catalyst_anchor_stage":a["stage"],
            "catalyst_anchor_stage_weight":a["stage_weight"],
            "catalyst_anchor_title":a["title"],
            "catalyst_anchor_url":a["url"],
            "first_trade_date":str(pd.Timestamp(post.iloc[0]["date"]).date()),
            "price_at_catalyst":p0,
            "current_date":str(current_date.date()),
            "current_adj_close":cur,
            "return_since_catalyst":ret,
            "max_runup_since_catalyst":maxret,
            "days_since_catalyst":days,
            "catalyst_priced_in_penalty":penalty_from_returns(ret,maxret,days),
        })

    outdf=pd.DataFrame(results)
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    outdf.to_csv(out,index=False)
    summary={
        "stage":"V11.4 catalyst-specific price absorption",
        "current_market_date":str(current_date.date()),
        "companies":int(len(outdf)),
        "median_return_since_catalyst":float(outdf["return_since_catalyst"].median()) if len(outdf) else None,
        "median_priced_in_penalty":float(outdf["catalyst_priced_in_penalty"].median()) if len(outdf) else None,
        "low_absorption_companies":int((outdf["catalyst_priced_in_penalty"]<=0.25).sum()) if len(outdf) else 0,
        "rule":"Clock resets only on a new catalyst family or a genuine stage escalation within that family."
    }
    json.dump(summary,open(out.parent/"catalyst_absorption_summary.json","w"),indent=2,default=str)
    print(json.dumps(summary,indent=2,default=str))

if __name__=="__main__":
    main()
