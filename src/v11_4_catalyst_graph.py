from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup


LEGAL_STOP={
    "limited","ltd","private","pvt","india","industries","industry","company","co",
    "corporation","corp","holdings","enterprises","enterprise"
}

STAGE_PATTERNS=[
    ("utilisation_ramp_or_repeat_order",1.10,re.compile(r"\b(utili[sz]ation ramp|ramp[- ]up|repeat order|repeat contract)\b",re.I)),
    ("commercial_production_or_purchase_order",1.00,re.compile(r"\b(commercial production|commencement of (?:commercial )?production|commenced (?:commercial )?production|production (?:has )?commenced|production commencement|fully operational|facility (?:is )?(?:now )?operational|operations (?:have )?commenced|purchase order|work order|firm order)\b",re.I)),
    ("commissioned_or_loa",0.85,re.compile(r"\b(commissioned|commissioning completed|letter of award|\bloa\b|contract awarded)\b",re.I)),
    ("trial_production_or_l1_bidder",0.70,re.compile(r"\b(trial production|trial run|l1 bidder|lowest bidder|preferred bidder|shortlisted)\b",re.I)),
    ("under_construction_or_vendor_approved",0.55,re.compile(r"\b(under construction|civil work|equipment installation|vendor approval|customer qualification|approved vendor)\b",re.I)),
    ("board_approved_or_funded",0.40,re.compile(r"\b(board approved|financial closure|funding secured|sanctioned loan|capex approved)\b",re.I)),
    ("management_intent",0.20,re.compile(r"\b(plans to|proposes to|intends to|considering|exploring|memorandum of understanding|\bmou\b)\b",re.I)),
]

CATALYST_PATTERNS={
    "capacity":re.compile(r"\b(capacity expansion|expand capacity|capacity addition|greenfield|brownfield|new plant|new facility|capex)\b",re.I),
    "commissioning":re.compile(r"\b(commissioned|commissioning|commercial production|commencement of (?:commercial )?production|commenced (?:commercial )?production|production (?:has )?commenced|production commencement|trial production|operations commenced|operations have commenced|plant operational|facility (?:is )?(?:now )?operational|fully operational)\b",re.I),
    "order":re.compile(r"\b(order win|purchase order|work order|letter of award|\bloa\b|contract awarded|l1 bidder|lowest bidder|order book)\b",re.I),
    "product":re.compile(r"\b(new product|product launch|commerciali[sz]ation|new technology|new platform)\b",re.I),
    "approval":re.compile(r"\b(regulatory approval|product approval|vendor approval|customer qualification|certification|usfda|ce marking)\b",re.I),
    "policy":re.compile(r"\b(pli|production linked incentive|government scheme|subsidy|incentive|policy approval)\b",re.I),
    "promoter":re.compile(r"\b(promoter purchase|promoter bought|open market purchase|insider purchase)\b",re.I),
}
NEGATIVE_PATTERNS=re.compile(r"\b(cancelled|canceled|termination|terminated|delay|delayed|default|insolvency|fraud|pledge invocation|auditor resignation)\b",re.I)


def norm_name(s):
    z=re.sub(r"[^a-z0-9 ]+"," ",str(s).lower())
    toks=[t for t in z.split() if t not in LEGAL_STOP and len(t)>1]
    return " ".join(toks)


def load_master(url:str):
    r=requests.get(url,timeout=60,headers={"User-Agent":"Mozilla/5.0"})
    r.raise_for_status()
    from io import StringIO
    df=pd.read_csv(StringIO(r.text))
    cols={c.strip().upper():c for c in df.columns}
    sym=cols.get("SYMBOL")
    name=cols.get("NAME OF COMPANY")
    isin=cols.get("ISIN NUMBER")
    if not sym or not name:
        raise RuntimeError(f"unexpected NSE security master columns: {list(df.columns)}")
    out=pd.DataFrame({
        "symbol":df[sym].astype(str).str.upper().str.strip(),
        "company_name":df[name].astype(str).str.strip(),
        "isin":df[isin].astype(str).str.upper().str.strip() if isin else None,
    })
    out["name_key"]=out["company_name"].map(norm_name)
    return out


def source_tier(domain,registry):
    d=str(domain).lower()
    if "nseindia.com" in d or "bseindia.com" in d or "pib.gov.in" in d or "eprocure.gov.in" in d:
        return 1,1.0
    if any(x in d for x in ["crisilratings.com","icra.in","careedge.in","indiaratings.co.in"]):
        return 1,1.0
    return 2,0.8


def fetch_text(url):
    if not url:
        return ""
    try:
        r=requests.get(url,headers={"User-Agent":"Mozilla/5.0"},timeout=25)
        if r.status_code!=200 or "text/html" not in r.headers.get("content-type",""):
            return ""
        soup=BeautifulSoup(r.text,"html.parser")
        for x in soup(["script","style","noscript","svg"]): x.decompose()
        txt=" ".join(soup.stripped_strings)
        return txt[:120000]
    except Exception:
        return ""


def match_company(text,master):
    low=" "+norm_name(text)+" "
    hits=[]
    for row in master.itertuples(index=False):
        key=row.name_key
        if len(key)<5: continue
        # Require the normalized legal-name core as a phrase.
        if f" {key} " in low:
            hits.append((row.symbol,row.company_name,row.isin,len(key)))
    if not hits:
        return None
    hits.sort(key=lambda x:x[3],reverse=True)
    return hits[0][:3]


def _keyword_hit(text,term):
    term=str(term or "").strip().lower()
    if not term:
        return False
    # Exact token/phrase matching. Short tokens such as EV or API must not
    # match inside unrelated words like revenue/capital.
    pat=re.escape(term)
    pat=pat.replace(r"\ ",r"[\\s\\-/]+")
    return re.search(r"(?<![A-Za-z0-9])"+pat+r"(?![A-Za-z0-9])",str(text),re.I) is not None

def themes(text,cfg):
    t=str(text or "")
    out=[]
    for theme,words in cfg["themes"].items():
        if any(_keyword_hit(t,w) for w in words):
            out.append(theme)
    return out


def stage(text):
    for name,w,p in STAGE_PATTERNS:
        if p.search(text):
            return name,w
    return "unspecified_or_early",0.15


def catalyst_types(text):
    return [k for k,p in CATALYST_PATTERNS.items() if p.search(text)]


def catalyst_context(text, radius=450):
    """Return local windows around catalyst/stage terms for magnitude extraction."""
    s=str(text or "")
    spans=[]
    patterns=list(CATALYST_PATTERNS.values())+[p for _,_,p in STAGE_PATTERNS]
    for p in patterns:
        for m in p.finditer(s):
            spans.append((max(0,m.start()-radius),min(len(s),m.end()+radius)))
    if not spans:
        return ""
    spans.sort()
    merged=[]
    for a,b in spans:
        if merged and a<=merged[-1][1]+80:
            merged[-1]=(merged[-1][0],max(merged[-1][1],b))
        else:
            merged.append((a,b))
    return " ".join(s[a:b] for a,b in merged[:12])


MONEY_CONTEXT_RE=re.compile(
    r"\b(order|contract|work order|purchase order|letter of award|\bloa\b|"
    r"capex|capital expenditure|investment|project cost|expansion|capacity|"
    r"new plant|new facility|facility|award value|order value|worth)\b",
    re.I,
)

def extract_catalyst_money_crore(text, radius=140):
    s=str(text or "")
    vals=[]
    pats=[
        re.compile(r"(?:rs\.?|inr|₹)\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s*(crore|cr)\b",re.I),
        re.compile(r"([0-9][0-9,]*(?:\.[0-9]+)?)\s*(crore|cr)\b",re.I),
    ]
    for p in pats:
        for m in p.finditer(s):
            a=max(0,m.start()-radius); b=min(len(s),m.end()+radius)
            if not MONEY_CONTEXT_RE.search(s[a:b]):
                continue
            try: vals.append(float(m.group(1).replace(",","")))
            except Exception: pass
    return max(vals) if vals else np.nan


def extract_money_crore(text):
    vals=[]
    pats=[
        re.compile(r"(?:rs\.?|inr|₹)\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s*(crore|cr)\b",re.I),
        re.compile(r"([0-9][0-9,]*(?:\.[0-9]+)?)\s*(crore|cr)\b",re.I),
    ]
    for p in pats:
        for m in p.finditer(text):
            try: vals.append(float(m.group(1).replace(",","")))
            except Exception: pass
    return max(vals) if vals else np.nan


def _capacity_unit(u):
    s=str(u or "").lower().replace(".","").strip()
    aliases={
        "kgs":"kg","kilogram":"kg","kilograms":"kg",
        "ton":"tonne","tons":"tonne","tonnes":"tonne","tpa":"tonne","mt":"tonne",
        "mtpa":"mtpa","mmtpa":"mmtpa",
        "mw":"mw","gw":"gw","kw":"kw",
        "units":"unit","unit":"unit",
        "kl":"kl","klpd":"klpd","mld":"mld",
    }
    return aliases.get(s,s)

def _capacity_scale(unit):
    # Normalize only within unambiguous families.
    return {
        "kg":("mass",1.0),
        "tonne":("mass",1000.0),
        "mw":("power",1.0),
        "gw":("power",1000.0),
        "kw":("power",0.001),
        "unit":("unit",1.0),
        "kl":("volume",1.0),
        "klpd":("flow",1.0),
        "mld":("flow_mld",1.0),
        "mtpa":("mtpa",1.0),
        "mmtpa":("mtpa",1000.0),
    }.get(unit,(unit,1.0))

CAP_AMOUNT=r"([0-9][0-9,]*(?:\.[0-9]+)?)\s*(kg|kgs|kilograms?|tonnes?|tons?|\bmt\b|tpa|mtpa|mmtpa|mw|gw|kw|units?|klpd|mld|kl)\b"

def extract_capacity_metrics(text):
    s=str(text or "")
    direct=extract_capacity_pct(s)
    candidates=[]

    # Explicit from X to Y is highest confidence.
    p=re.compile(r"(?:capacity[^.]{0,100}?)?from\s*"+CAP_AMOUNT+r"[^.]{0,100}?\bto\s*"+CAP_AMOUNT,re.I)
    for m in p.finditer(s):
        x=float(m.group(1).replace(",","")); ux=_capacity_unit(m.group(2))
        y=float(m.group(3).replace(",","")); uy=_capacity_unit(m.group(4))
        fx,sx=_capacity_scale(ux); fy,sy=_capacity_scale(uy)
        if fx==fy and x>0 and y>x:
            pct=(y*sy/(x*sx)-1)*100
            candidates.append(("from_to",pct,x,ux,y,uy))

    # "add X ... taking total capacity to Y" / "adds X ... total capacity Y".
    p2=re.compile(r"(?:add(?:s|ed|ition(?:al)?)?|increase(?:s|d)?|new|additional)[^.]{0,50}?"+CAP_AMOUNT+r"[^.]{0,140}?(?:total\s+capacity|capacity)[^.]{0,50}?(?:to|of|at)\s*"+CAP_AMOUNT,re.I)
    for m in p2.finditer(s):
        add=float(m.group(1).replace(",","")); ua=_capacity_unit(m.group(2))
        total=float(m.group(3).replace(",","")); ut=_capacity_unit(m.group(4))
        fa,sa=_capacity_scale(ua); ft,st=_capacity_scale(ut)
        addn=add*sa; totaln=total*st
        if fa==ft and addn>0 and totaln>addn:
            old=totaln-addn
            pct=addn/old*100
            candidates.append(("addition_to_total",pct,add,ua,total,ut))

    # Record all capacity quantities for audit even when percentage cannot be inferred.
    quantities=[]
    for m in re.finditer(CAP_AMOUNT,s,re.I):
        try:
            quantities.append({"value":float(m.group(1).replace(",","")),"unit":_capacity_unit(m.group(2))})
        except Exception:
            pass

    inferred=max([z[1] for z in candidates],default=np.nan)
    pct=max([v for v in [direct,inferred] if pd.notna(v)],default=np.nan)
    return {
        "capacity_pct":pct,
        "capacity_pct_direct":direct,
        "capacity_pct_inferred":inferred,
        "capacity_inference_method":max(candidates,key=lambda z:z[1])[0] if candidates else None,
        "capacity_quantities":quantities[:12],
    }


def extract_capacity_pct(text):
    vals=[]
    for m in re.finditer(r"([0-9]+(?:\.[0-9]+)?)\s*%\s*(?:increase|expansion|higher|additional|capacity)",text,re.I):
        try: vals.append(float(m.group(1)))
        except Exception: pass
    return max(vals) if vals else np.nan


def event_confidence(tier_trust,stage_w,types,negative,corroboration):
    base=0.25 + 0.35*stage_w + 0.20*min(len(types),2)/2 + 0.10*min(corroboration,2)/2
    base*=tier_trust
    if negative: base*=0.45
    return float(np.clip(base,0,1))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--evidence",required=True)
    ap.add_argument("--queries-config",required=True)
    ap.add_argument("--source-registry",required=True)
    ap.add_argument("--output",required=True)
    ap.add_argument("--primary-evidence",default=None)
    ap.add_argument("--master-url",default="https://archives.nseindia.com/content/equities/EQUITY_L.csv")
    ap.add_argument("--fetch-pages",action="store_true")
    args=ap.parse_args()

    qcfg=json.load(open(args.queries_config))
    registry=json.load(open(args.source_registry))
    master=load_master(args.master_url)
    ev=pd.read_parquet(args.evidence)
    primary=pd.read_parquet(args.primary_evidence) if args.primary_evidence else pd.DataFrame()

    rows=[]
    if len(primary):
        for r in primary.itertuples(index=False):
            title=str(getattr(r,"title","") or "")
            details=str(getattr(r,"details","") or "")
            url=str(getattr(r,"url","") or "")
            text=(title+" "+details).strip()
            if not text:
                continue
            cts=catalyst_types(text)
            th=themes(text,qcfg)
            st,sw=stage(text)
            neg=bool(NEGATIVE_PATTERNS.search(text))
            rows.append({
                "evidence_id":getattr(r,"evidence_id",None),
                "published_ts":getattr(r,"published_ts",None),
                "symbol":str(getattr(r,"symbol","") or "").upper(),
                "company_name":None,
                "isin":getattr(r,"isin",None),
                "domain":"nseindia.com",
                "url":url,
                "title":title,
                "query_family":"nse_corporate_announcement",
                "source_tier":1,
                "source_trust":1.0,
                "catalyst_types":json.dumps(cts),
                "themes":json.dumps(th),
                "stage":st,
                "stage_weight":sw,
                "money_crore_max":extract_money_crore(text),
                "capacity_pct_max":extract_capacity_pct(text),
                "negative_flag":neg,
            })

    for r in ev.itertuples(index=False):
        title=str(getattr(r,"title","") or "")
        url=str(getattr(r,"url","") or "")
        body=fetch_text(url) if args.fetch_pages else ""
        text=(title+" "+body).strip()
        m=match_company(text,master)
        if not m:
            continue
        symbol,company,isin=m
        domain=str(getattr(r,"domain","") or urlparse(url).netloc)
        tier,trust=source_tier(domain,registry)
        cts=catalyst_types(text)
        th=themes(text,qcfg)
        st,sw=stage(text)
        neg=bool(NEGATIVE_PATTERNS.search(text))
        capctx=catalyst_context(text)
        capm=extract_capacity_metrics(capctx)
        rows.append({
            "evidence_id":getattr(r,"evidence_id",None),
            "published_ts":getattr(r,"published_ts",None),
            "symbol":symbol,
            "company_name":company,
            "isin":isin,
            "domain":domain,
            "url":url,
            "title":title,
            "query_family":getattr(r,"query_family",None),
            "source_tier":tier,
            "source_trust":trust,
            "catalyst_types":json.dumps(cts),
            "themes":json.dumps(th),
            "stage":st,
            "stage_weight":sw,
            "money_crore_max":extract_catalyst_money_crore(capctx),
            "capacity_pct_max":capm["capacity_pct"],
            "capacity_pct_direct":capm["capacity_pct_direct"],
            "capacity_pct_inferred":capm["capacity_pct_inferred"],
            "capacity_inference_method":capm["capacity_inference_method"],
            "capacity_quantities":json.dumps(capm["capacity_quantities"]),
            "negative_flag":neg,
        })

    df=pd.DataFrame(rows)
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
    if df.empty:
        df.to_parquet(out,index=False)
        raise SystemExit("No company-linked evidence found")

    # Evidence corroboration by company/family within 30 days.
    df["published_ts"]=pd.to_datetime(df["published_ts"],utc=True,errors="coerce")
    df["corroboration_count"]=0
    for idx,row in df.iterrows():
        lo=row["published_ts"]-pd.Timedelta(days=30)
        m=(df["symbol"].eq(row["symbol"]) & df["query_family"].eq(row["query_family"]) & df["published_ts"].between(lo,row["published_ts"]))
        df.at[idx,"corroboration_count"]=int(df.loc[m,"domain"].nunique())

    df["evidence_confidence"]=[
        event_confidence(t,sw,json.loads(ct),neg,corr)
        for t,sw,ct,neg,corr in zip(df["source_trust"],df["stage_weight"],df["catalyst_types"],df["negative_flag"],df["corroboration_count"])
    ]
    df.to_parquet(out,index=False)

    company=df.groupby("symbol").agg(
        evidence_rows=("evidence_id","count"),
        primary_rows=("source_tier",lambda s:int((s==1).sum())),
        max_confidence=("evidence_confidence","max"),
        max_stage_weight=("stage_weight","max"),
        max_money_crore=("money_crore_max","max"),
        max_capacity_pct=("capacity_pct_max","max"),
        distinct_domains=("domain","nunique"),
    ).reset_index()
    company.to_csv(out.parent/"company_catalyst_evidence_summary.csv",index=False)
    summary={
        "linked_rows":int(len(df)),
        "companies":int(df["symbol"].nunique()),
        "primary_rows":int((df["source_tier"]==1).sum()),
        "families":df["query_family"].value_counts().to_dict(),
        "stages":df["stage"].value_counts().to_dict(),
        "top_companies":company.sort_values(["max_confidence","evidence_rows"],ascending=False).head(25).to_dict("records"),
    }
    json.dump(summary,open(out.parent/"catalyst_graph_summary.json","w"),indent=2,default=str)
    print(json.dumps(summary,indent=2,default=str))


if __name__=="__main__":
    main()
