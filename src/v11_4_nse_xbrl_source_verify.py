"""Verify a reproducible sample of third-party NSE ownership facts against NSE-hosted XBRL.

Scope: audit only. No trading model or training dataset is modified.
Access denial, missing XML, or parser incompatibility are reported distinctly.
"""
from __future__ import annotations
import hashlib
import json
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd
import requests

OUT = Path("outputs_v11_4_nse_ownership_probe")
CANDIDATES = OUT / "ownership_historical_pit_candidates.parquet"
PERCENT_FACT = "ShareholdingAsAPercentageOfTotalNumberOfShares"
MEMBERS = {
    "promoter": "ShareholdingOfPromoterAndPromoterGroupMember",
    "public": "PublicShareholdingMember",
}
TIMEOUT = 15
PER_YEAR = 4

def local(tag):
    return str(tag).rsplit("}", 1)[-1].rsplit(":", 1)[-1]

def parse_nse_xbrl(raw):
    root = ET.fromstring(raw)
    contexts = {}
    for node in root.iter():
        if local(node.tag) != "context":
            continue
        cid = node.attrib.get("id", "")
        members = []
        for child in node.iter():
            if local(child.tag) == "explicitMember":
                members.append((child.text or "").rsplit(":", 1)[-1])
        contexts[cid] = members
    facts = {}
    for node in root.iter():
        if local(node.tag) != PERCENT_FACT or node.text is None:
            continue
        ref = node.attrib.get("contextRef", "")
        dimensions = contexts.get(ref, [])
        if len(dimensions) > 2:
            continue
        try:
            val = float(node.text.replace(",", ""))
        except ValueError:
            continue
        if not (val == val):
            continue
        for label, member in MEMBERS.items():
            if member in dimensions and label not in facts:
                facts[label] = val
    if not all(k in facts for k in MEMBERS):
        raise ValueError(f"Cannot parse promoter/public rollups; found={list(facts)}")
    total = facts["promoter"] + facts["public"]
    scale = 0.01 if total > 1000 else 100.0 if 0 < total < 2 else 1.0
    return {k: round(v * scale, 4) for k, v in facts.items()}

def sample_records(df):
    rows = []
    used = set()
    for yr in (2018, 2019, 2020, 2021):
        group = df[df["period_end"].dt.year.eq(yr)].copy()
        if len(group) == 0:
            continue
        # Stable sort across runs; prioritize different companies.
        group["_rank"] = group.apply(
            lambda z: hashlib.sha256((str(z["ticker"]) + "|" + str(z["source_filename"])).encode()).hexdigest(),
            axis=1,
        )
        for _, item in group.sort_values("_rank").iterrows():
            sym = item["ticker"]
            if sym in used:
                continue
            rows.append(item.to_dict())
            used.add(sym)
            if sum(x["period_end"].year == yr for x in rows) >= PER_YEAR:
                break
    return rows

def main():
    OUT.mkdir(exist_ok=True, parents=True)
    df = pd.read_parquet(CANDIDATES)
    df["period_end"] = pd.to_datetime(df["period_end"], errors="coerce", utc=True)
    df = df[df["xbrl_url"].astype(str).str.match(r"^https://nsearchives\.nseindia\.com/corporate/xbrl/.*\.xml$")].copy()
    sample = sample_records(df)
    results = []
    ses = requests.Session()
    ses.headers.update({
        "User-Agent": "Mozilla/5.0 (compatible; research verification)",
        "Accept": "application/xml, text/xml, */*",
        "Referer": "https://www.nseindia.com/",
    })
    for rec in sample:
        item = {
            "ticker": rec["ticker"],
            "period": str(rec["period_end"].date()),
            "original_file": rec["source_filename"],
            "url": rec["xbrl_url"],
            "index_available_at": str(rec["available_at_utc"]),
            "candidate_promoter": float(rec["promoter_pct"]),
            "candidate_public": float(rec["public_pct"]),
        }
        try:
            resp = ses.get(rec["xbrl_url"], timeout=TIMEOUT, allow_redirects=True)
            item["http_status"] = resp.status_code
            item["response_bytes"] = len(resp.content)
            item["response_sha256"] = hashlib.sha256(resp.content).hexdigest()
            resp.raise_for_status()
            facts = parse_nse_xbrl(resp.content)
            item["original_promoter"] = facts["promoter"]
            item["original_public"] = facts["public"]
            item["promoter_abs_delta_pp"] = abs(facts["promoter"] - item["candidate_promoter"])
            item["public_abs_delta_pp"] = abs(facts["public"] - item["candidate_public"])
            item["status"] = ("verified" if max(item["promoter_abs_delta_pp"], item["public_abs_delta_pp"]) <= 0.5 else "mismatch")
        except requests.RequestException as exc:
            item["status"] = "source_unavailable"
            item["error"] = f"{type(exc).__name__}: {str(exc)[:180]}"
        except (ET.ParseError, ValueError) as exc:
            item["status"] = "parser_unsupported"
            item["error"] = f"{type(exc).__name__}: {str(exc)[:180]}"
        results.append(item)
        print(item["ticker"], item["period"], item["status"], item.get("http_status"), flush=True)
        time.sleep(0.4)
    pd.DataFrame(results).to_csv(OUT / "original_nse_xbrl_sample_verification.csv", index=False)
    status = pd.Series([x["status"] for x in results]).value_counts().to_dict()
    confirmed = [x for x in results if x["status"] == "verified"]
    years_confirmed = sorted({int(x["period"][:4]) for x in confirmed})
    summary = {
        "source": "Direct NSE-hosted XBRL document URLs taken from the indexed filings",
        "scope": "SAMPLED_SOURCE_QA_ONLY; not full-market verification",
        "sample_n": len(sample),
        "status_counts": status,
        "verified_sample_n": len(confirmed),
        "verified_years": years_confirmed,
        "source_sample_pass": len(confirmed) >= 8 and len(years_confirmed) >= 3 and status.get("mismatch", 0) == 0,
        "production_promotion_allowed": False,
        "note": "Provider blocking and unsupported taxonomy are not treated as verified. All historical availability timestamps remain unchanged.",
    }
    (OUT / "xbrl_verification_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    if not summary["source_sample_pass"]:
        raise SystemExit("Original NSE XBRL sample validation has not met verification gate")

if __name__ == "__main__":
    main()
