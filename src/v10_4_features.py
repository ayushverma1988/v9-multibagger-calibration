from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import pandas as pd

FEATURES_V10_4=[
    "evt_opportunity_180",
    "evt_risk_180",
    "evt_promoter_180",
    "evt_intensity_90",
    "evt_recency_net_90",
    "promoter_pct",
    "promoter_delta_qoq",
    "insider_net_180",
    "insider_activity_180",
    "result_delay_days",
    "result_revision_180",
]

MATERIAL_TYPES={
    "order_win","capacity_expansion","debt_reduction","buyback","credit_rating",
    "dilution","auditor_change","litigation","regulatory","promoter_activity",
    "pledge_change","management_change","customer_supplier",
}

OPPORTUNITY_TYPES={"order_win","capacity_expansion","debt_reduction","buyback"}
RISK_TYPES={"dilution","auditor_change","litigation","regulatory"}


def norm(v):
    if pd.isna(v):return None
    s=str(v).strip().upper()
    return None if s in {"","NAN","NONE","<NA>","NULL"} else s


def security_key(isin,symbol):
    i=norm(isin)
    return "I:"+i if i else "S:"+(norm(symbol) or "")


def availability_date(s):
    z=pd.to_datetime(s,utc=True,errors="coerce")
    try:
        return z.dt.tz_convert("Asia/Kolkata").dt.tz_localize(None).dt.normalize()
    except Exception:
        return pd.to_datetime(s,dayfirst=True,errors="coerce").dt.normalize()


def _num(s):
    return pd.to_numeric(s.astype(str).str.replace(",","",regex=False),errors="coerce")


def prepare_events(path:str|Path)->pd.DataFrame:
    e=pd.read_parquet(path).copy()
    e["avail_date"]=availability_date(e["published_ts"])
    e["key"]=[security_key(i,s) for i,s in zip(e.get("isin"),e.get("symbol"))]
    e["event_type"]=e["event_type"].fillna("other").astype(str).str.lower()
    e["direction"]=pd.to_numeric(e.get("event_direction",0),errors="coerce").fillna(0.0).clip(-1,1)
    e["strength"]=pd.to_numeric(e.get("event_strength",0.5),errors="coerce").fillna(0.5).clip(0,1)

    pos=np.zeros(len(e),float)
    risk=np.zeros(len(e),float)
    typ=e["event_type"]
    st=e["strength"].to_numpy(float)
    dr=e["direction"].to_numpy(float)

    pos[np.asarray(typ.isin(OPPORTUNITY_TYPES))]=st[np.asarray(typ.isin(OPPORTUNITY_TYPES))]
    cr=np.asarray(typ.eq("credit_rating"))
    pos[cr]=np.maximum(dr[cr],0)*st[cr]
    risk[np.asarray(typ.isin(RISK_TYPES))]=st[np.asarray(typ.isin(RISK_TYPES))]
    risk[cr]=np.maximum(-dr[cr],0)*st[cr]

    e["_opp"]=pos
    e["_risk"]=risk
    prom=np.asarray(typ.isin({"promoter_activity","pledge_change"}))
    e["_prom"]=np.where(prom,dr*st,0.0)
    mat=np.asarray(typ.isin(MATERIAL_TYPES))
    e["_material"]=mat.astype(float)
    e["_signed"]=np.where(mat,dr*st,0.0)
    return e[e["avail_date"].notna() & e["key"].ne("S:")].sort_values(["key","avail_date"]).reset_index(drop=True)


def prepare_shareholding(path:str|Path)->pd.DataFrame:
    s=pd.read_parquet(path).copy()
    b=pd.to_datetime(s.get("broadcastDate",pd.Series(index=s.index,dtype=object)),dayfirst=True,errors="coerce")
    sub=pd.to_datetime(s.get("submissionDate",pd.Series(index=s.index,dtype=object)),dayfirst=True,errors="coerce")
    s["avail_date"]=b.fillna(sub).dt.normalize()
    s["key"]=[security_key(i,sy) for i,sy in zip(s.get("isin_norm",s.get("isin")),s.get("symbol_norm",s.get("symbol")))]
    s["promoter_pct"]=_num(s.get("pr_and_prgrp",pd.Series(index=s.index,dtype=object))).clip(0,100)
    s["report_date"]=pd.to_datetime(s.get("date"),dayfirst=True,errors="coerce")
    return s[s["avail_date"].notna() & s["promoter_pct"].notna()].sort_values(["key","avail_date","report_date"]).reset_index(drop=True)


def _find_col(df,names):
    low={str(c).lower():c for c in df.columns}
    for n in names:
        if n.lower() in low:return low[n.lower()]
    return None


def prepare_insider(path:str|Path)->pd.DataFrame:
    x=pd.read_parquet(path).copy()
    x["avail_date"]=pd.NaT
    for nm in ["broadcastDate","broadcastDt","intimDt","intimationDate","anex"]:
        dc=_find_col(x,[nm])
        if dc is not None:
            z=pd.to_datetime(x[dc],dayfirst=True,errors="coerce").dt.normalize()
            x["avail_date"]=pd.to_datetime(x["avail_date"],errors="coerce").fillna(z)
    ic=_find_col(x,["isin","isin_norm","secIsin"])
    sc=_find_col(x,["symbol","symbol_norm"])
    iv=x[ic] if ic else pd.Series(index=x.index,dtype=object)
    sv=x[sc] if sc else pd.Series(index=x.index,dtype=object)
    x["key"]=[security_key(i,s) for i,s in zip(iv,sv)]

    vc=_find_col(x,["secVal","securityValue","value"])
    v=_num(x[vc]) if vc else pd.Series(np.nan,index=x.index)
    qc=_find_col(x,["secAcq","securitiesAcquired","quantity"])
    q=_num(x[qc]).abs() if qc else pd.Series(np.nan,index=x.index)
    mag=np.log1p(v.abs().where(v.abs()>0,q.fillna(0)))

    tc=_find_col(x,["tdpTransactionType","transactionType","acqMode"])
    t=x[tc].fillna("").astype(str).str.lower() if tc else pd.Series("",index=x.index)
    sign=np.where(t.str.contains("sell|sale|disposal|dispose",regex=True),-1.0,
          np.where(t.str.contains("buy|purchase|acquisition|acquire",regex=True),1.0,0.0))
    x["_signed_value"]=sign*mag.to_numpy(float)
    x["_activity"]=(sign!=0).astype(float)
    return x[x["avail_date"].notna() & x["key"].ne("S:")].sort_values(["key","avail_date"]).reset_index(drop=True)


def prepare_financial(path:str|Path)->pd.DataFrame:
    f=pd.read_parquet(path).copy()
    b=pd.to_datetime(f.get("broadCastDate",pd.Series(index=f.index,dtype=object)),dayfirst=True,errors="coerce")
    fd=pd.to_datetime(f.get("filingDate",pd.Series(index=f.index,dtype=object)),dayfirst=True,errors="coerce")
    f["avail_date"]=b.fillna(fd).dt.normalize()
    f["period_end"]=pd.to_datetime(f.get("toDate"),dayfirst=True,errors="coerce")
    f["key"]=[security_key(i,s) for i,s in zip(f.get("isin_norm",f.get("isin")),f.get("symbol_norm",f.get("symbol")))]
    f["delay_days"]=(f["avail_date"]-f["period_end"].dt.normalize()).dt.days.clip(lower=0,upper=365)
    old=f.get("oldNewFlag",pd.Series("",index=f.index)).fillna("").astype(str).str.upper()
    desc=f.get("resultDescription",pd.Series("",index=f.index)).fillna("").astype(str).str.lower()
    f["_revision"]=((~old.isin(["","N","NEW"])) | desc.str.contains("revis|corrig|clarif",regex=True)).astype(float)
    return f[f["avail_date"].notna() & f["key"].ne("S:")].sort_values(["key","avail_date","period_end"]).reset_index(drop=True)


def _source_map(df,cols):
    out={}
    if df is None or df.empty:return out
    for k,g in df.groupby("key",sort=False):
        gg=g.sort_values("avail_date")
        out[k]={
            "dates":gg["avail_date"].to_numpy(dtype="datetime64[ns]"),
            **{c:gg[c].to_numpy() for c in cols}
        }
    return out


def attach_features(rows:pd.DataFrame,events:pd.DataFrame,shareholding:pd.DataFrame|None=None,
                    insider:pd.DataFrame|None=None,financial:pd.DataFrame|None=None)->pd.DataFrame:
    r=rows.copy()
    r["_key_v104"]=[security_key(i,s) for i,s in zip(r.get("isin"),r.get("symbol"))]
    r["_symbol_key_v104"]=["S:"+(norm(s) or "") for s in r.get("symbol")]
    em=_source_map(events,["_opp","_risk","_prom","_material","_signed"])
    sm=_source_map(shareholding,["promoter_pct","report_date"]) if shareholding is not None else {}
    im=_source_map(insider,["_signed_value","_activity"]) if insider is not None else {}
    fm=_source_map(financial,["delay_days","_revision","period_end"]) if financial is not None else {}

    rec=[]
    for z in r[["_key_v104","_symbol_key_v104","date"]].itertuples(index=False):
        key=z[0]; symkey=z[1]; d=np.datetime64(pd.Timestamp(z[2]).normalize())
        vals={c:np.nan for c in FEATURES_V10_4}

        e=em.get(key) or em.get(symkey)
        if e:
            dates=e["dates"]; hi=np.searchsorted(dates,d,side="right")
            lo180=np.searchsorted(dates,d-np.timedelta64(180,"D"),side="left")
            lo90=np.searchsorted(dates,d-np.timedelta64(90,"D"),side="left")
            if hi>lo180:
                age=(d-dates[lo180:hi]).astype("timedelta64[D]").astype(float)
                w=np.exp(-np.maximum(age,0)/90.0)
                vals["evt_opportunity_180"]=float(np.sum(e["_opp"][lo180:hi].astype(float)*w))
                vals["evt_risk_180"]=float(np.sum(e["_risk"][lo180:hi].astype(float)*w))
                vals["evt_promoter_180"]=float(np.sum(e["_prom"][lo180:hi].astype(float)*w))
            else:
                vals["evt_opportunity_180"]=vals["evt_risk_180"]=vals["evt_promoter_180"]=0.0
            if hi>lo90:
                age=(d-dates[lo90:hi]).astype("timedelta64[D]").astype(float)
                w=np.exp(-np.maximum(age,0)/45.0)
                vals["evt_intensity_90"]=float(np.log1p(np.sum(e["_material"][lo90:hi].astype(float))))
                vals["evt_recency_net_90"]=float(np.sum(e["_signed"][lo90:hi].astype(float)*w))
            else:
                vals["evt_intensity_90"]=vals["evt_recency_net_90"]=0.0
        else:
            vals["evt_opportunity_180"]=vals["evt_risk_180"]=vals["evt_promoter_180"]=0.0
            vals["evt_intensity_90"]=vals["evt_recency_net_90"]=0.0

        s=sm.get(key) or sm.get(symkey)
        if s:
            dates=s["dates"]; hi=np.searchsorted(dates,d,side="right")
            if hi>0:
                vals["promoter_pct"]=float(s["promoter_pct"][hi-1])
                if hi>1:
                    vals["promoter_delta_qoq"]=float(s["promoter_pct"][hi-1]-s["promoter_pct"][hi-2])

        q=im.get(key) or im.get(symkey)
        if q:
            dates=q["dates"]; hi=np.searchsorted(dates,d,side="right")
            lo=np.searchsorted(dates,d-np.timedelta64(180,"D"),side="left")
            if hi>lo:
                vals["insider_net_180"]=float(np.nansum(q["_signed_value"][lo:hi].astype(float)))
                vals["insider_activity_180"]=float(np.log1p(np.nansum(q["_activity"][lo:hi].astype(float))))
            else:
                vals["insider_net_180"]=0.0; vals["insider_activity_180"]=0.0
        else:
            vals["insider_net_180"]=0.0; vals["insider_activity_180"]=0.0

        f=fm.get(key) or fm.get(symkey)
        if f:
            dates=f["dates"]; hi=np.searchsorted(dates,d,side="right")
            if hi>0:
                vals["result_delay_days"]=float(f["delay_days"][hi-1]) if np.isfinite(float(f["delay_days"][hi-1])) else np.nan
                lo=np.searchsorted(dates,d-np.timedelta64(180,"D"),side="left")
                vals["result_revision_180"]=float(np.nansum(f["_revision"][lo:hi].astype(float)))
        rec.append(vals)

    add=pd.DataFrame(rec,index=r.index)
    for c in FEATURES_V10_4:r[c]=add[c]
    return r.drop(columns=["_key_v104","_symbol_key_v104"])


def attach_current_to_daily(daily:pd.DataFrame,events,shareholding=None,insider=None,financial=None)->pd.DataFrame:
    d=daily.copy()
    last=pd.Timestamp(d["date"].max())
    idx=d.index[d["date"].eq(last)]
    cur=attach_features(d.loc[idx,["date","symbol","isin"]],events,shareholding,insider,financial)
    for c in FEATURES_V10_4:
        d[c]=np.nan
        d.loc[idx,c]=cur[c].to_numpy()
    return d
