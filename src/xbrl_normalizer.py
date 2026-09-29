from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from lxml import etree as LET
except Exception:
    LET = None


FLOW_FIELDS = {
    "revenue", "operating_profit", "pbt", "pat", "finance_cost",
    "operating_cash_flow", "capex",
}
INSTANT_FIELDS = {
    "total_assets", "total_equity", "total_debt",
    "noncurrent_borrowings", "current_borrowings",
    "current_assets", "current_liabilities", "shares_outstanding",
    "promoter_pct", "pledged_pct",
}


def local_name(tag: str) -> str:
    tag = str(tag)
    if tag.startswith("{"):
        return tag.split("}", 1)[1]
    return tag.split(":")[-1]


def namespace_uri(tag: str) -> str:
    tag = str(tag)
    if tag.startswith("{"):
        return tag[1:].split("}", 1)[0]
    return ""


def norm_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def parse_number(text: str | None, attrs: dict[str, str]) -> float:
    nil_keys = [
        "{http://www.w3.org/2001/XMLSchema-instance}nil",
        "nil",
    ]
    if any(str(attrs.get(k, "")).lower() == "true" for k in nil_keys):
        return np.nan
    if text is None:
        return np.nan
    s = str(text).strip()
    if not s or s in {"-", "—", "NA", "N/A", "nil", "None"}:
        return np.nan

    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1]

    # iXBRL can expose unicode minus and non-breaking spaces.
    s = (
        s.replace(",", "")
        .replace("₹", "")
        .replace("%", "")
        .replace("\u2212", "-")
        .replace("\xa0", "")
        .strip()
    )

    # Strip simple currency/text wrappers while preserving exponent notation.
    m = re.search(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", s)
    if not m:
        return np.nan
    try:
        v = float(m.group(0))
    except Exception:
        return np.nan

    if neg:
        v = -abs(v)
    sign = attrs.get("sign")
    if sign == "-":
        v = -abs(v)

    scale = attrs.get("scale")
    if scale not in (None, ""):
        try:
            v *= 10.0 ** int(scale)
        except Exception:
            pass
    return v


def context_period_months(start: pd.Timestamp, end: pd.Timestamp) -> int | None:
    days = int((end - start).days) + 1
    targets = {3: 91, 6: 182, 9: 273, 12: 365}
    month = min(targets, key=lambda m: abs(days - targets[m]))
    tolerances = {3: 40, 6: 55, 9: 65, 12: 80}
    return month if abs(days - targets[month]) <= tolerances[month] else None


def load_map(path: str | Path) -> dict:
    return json.load(open(path, encoding="utf-8"))


def compile_map(mapping: dict):
    out = {}
    for canonical, spec in mapping["concepts"].items():
        aliases = {norm_name(x) for x in spec.get("aliases", [])}
        regex = [re.compile(x, re.I) for x in spec.get("regex", [])]
        out[canonical] = {
            "aliases": aliases,
            "regex": regex,
            "families": set(spec.get("families", [])),
            "exclude_regex": [
                re.compile(x, re.I) for x in spec.get("exclude_regex", [])
            ],
        }
    return out


def _heuristic_concept(n: str, family: str | None = None):
    fam = str(family or "").lower()

    # Avoid common false positives first.
    if any(x in n for x in ["segment", "perShare", "earningspershare".lower()]):
        return None, 0.0

    # Profit after tax / net profit.
    if (
        re.search(r"(profitloss|netprofit|profit).*(forperiod|aftertax|aftertaxation)", n)
        or re.search(r"netprofit(loss)?$", n)
        or n in {"profitforperiod", "profitaftertax", "netprofit"}
    ):
        return "pat", 0.62

    if re.search(r"profit.*beforetax", n):
        return "pbt", 0.62

    # Revenue / top line. For banking and insurance, TotalIncome/PremiumIncome
    # are treated as top-line analogues but family remains explicit downstream.
    if (
        "revenuefromoperations" in n
        or "totalincomefromoperations" in n
        or "netsales" in n
        or "incomefromoperations" in n
        or "salesincome" in n
    ):
        return "revenue", 0.62
    if fam in {"banking", "nbfc"} and n in {
        "totalincome", "interestearned", "interestincome", "revenue"
    }:
        return "revenue", 0.58
    if "insurance" in fam and (
        "premiumincome" in n or "grosspremium" in n or n == "totalincome"
    ):
        return "revenue", 0.58

    if (
        "ebitda" in n
        or "earningsbeforeinteresttaxdepreciation" in n
        or "operatingprofit" in n
        or re.search(r"profitlossfromoperationsbefore.*finance", n)
    ):
        return "operating_profit", 0.60
    if fam in {"banking", "nbfc"} and "operatingprofitbeforeprovision" in n:
        return "operating_profit", 0.58

    if "financecost" in n or "interestexpense" in n or "interestexpended" in n:
        return "finance_cost", 0.62

    if n in {"assets", "totalassets"} or re.fullmatch(r"totalassets.*", n):
        return "total_assets", 0.62

    if (
        n in {"equity", "totalequity", "shareholdersfunds", "networth"}
        or "equityattributabletoowners" in n
        or "totalshareholdersequity" in n
    ):
        return "total_equity", 0.62

    if n in {"borrowings", "totalborrowings", "totaldebt", "debt"}:
        return "total_debt", 0.62
    if "noncurrentborrowings" in n or "longtermborrowings" in n:
        return "noncurrent_borrowings", 0.62
    if "currentborrowings" in n or "shorttermborrowings" in n:
        return "current_borrowings", 0.62

    if n in {"currentassets", "totalcurrentassets"}:
        return "current_assets", 0.62
    if n in {"currentliabilities", "totalcurrentliabilities"}:
        return "current_liabilities", 0.62

    if (
        re.search(r"(net)?cashflows?.*operatingactivities", n)
        or "cashgeneratedfromoperations" in n
    ):
        return "operating_cash_flow", 0.62

    if (
        "paymentstoacquirepropertyplant" in n
        or "purchaseofpropertyplant" in n
        or "purchaseoffixedassets" in n
        or n == "capitalexpenditure"
    ):
        return "capex", 0.60

    if (
        "numberofsharesoutstanding" in n
        or "numberofequityshares" in n
        or "paidupnumberofequityshares" in n
        or "numberofpaidupequityshares" in n
    ):
        return "shares_outstanding", 0.60

    if "promoter" in n and "shareholding" in n and "percentage" in n:
        return "promoter_pct", 0.58
    if "promoter" in n and "pledged" in n and "percentage" in n:
        return "pledged_pct", 0.58

    return None, 0.0


def map_concept(
    concept: str,
    compiled: dict,
    family: str | None = None,
) -> tuple[str | None, float]:
    n = norm_name(concept)

    for canonical, spec in compiled.items():
        fams = spec["families"]
        if fams and family and family not in fams:
            continue
        if any(p.search(n) for p in spec["exclude_regex"]):
            continue
        if n in spec["aliases"]:
            return canonical, 1.0

    for canonical, spec in compiled.items():
        fams = spec["families"]
        if fams and family and family not in fams:
            continue
        if any(p.search(n) for p in spec["exclude_regex"]):
            continue
        if any(p.search(n) for p in spec["regex"]):
            return canonical, 0.78

    return _heuristic_concept(n, family)


def classify_taxonomy(namespaces: set[str], concepts: list[str]) -> str:
    s = (
        " ".join(sorted(namespaces)).lower()
        + " "
        + " ".join(concepts[:500]).lower()
    )
    ns = norm_name(s)

    if any(x in ns for x in ["lifeinsurance", "lifeins", "insurancelife"]):
        return "life_insurance"
    if any(x in ns for x in ["generalinsurance", "nonlifeinsurance", "insurancenonlife"]):
        return "general_insurance"
    if "insurance" in ns or "premiumincome" in ns:
        return "insurance"
    if any(x in ns for x in [
        "banking", "banktaxonomy", "interestearned", "interestexpended",
        "advances", "deposits",
    ]):
        return "banking"
    if any(x in ns for x in ["nbfc", "nonbankingfinancial", "financecompany"]):
        return "nbfc"
    if any(x in ns for x in ["indas", "indianaccountingstandards", "indasxbrl"]):
        return "ind_as"
    if any(x in ns for x in ["ingaap", "indiangaap", "gaaptaxonomy"]):
        return "indian_gaap"
    return "generic_or_unknown"


def _parse_root(xml_bytes: bytes):
    try:
        return ET.fromstring(xml_bytes)
    except Exception:
        if LET is None:
            raise
        parser = LET.XMLParser(recover=True, huge_tree=True, resolve_entities=False)
        try:
            root = LET.fromstring(xml_bytes, parser=parser)
            if root is not None:
                return root
        except Exception:
            pass
        # Last resort for malformed inline HTML/XBRL.
        parser = LET.HTMLParser(recover=True, huge_tree=True)
        root = LET.fromstring(xml_bytes, parser=parser)
        if root is None:
            raise ValueError("Unable to parse XML/iXBRL")
        return root


def _iter(root):
    return root.iter()


def _parse_contexts(root) -> dict[str, dict[str, Any]]:
    out = {}
    for el in _iter(root):
        if local_name(el.tag).lower() != "context":
            continue
        cid = el.attrib.get("id")
        if not cid:
            continue
        start = end = instant = None
        dim_count = 0
        for d in el.iter():
            ln = local_name(d.tag).lower()
            txt = "".join(d.itertext()).strip() if hasattr(d, "itertext") else (d.text or "").strip()
            if ln == "startdate":
                start = pd.to_datetime(txt, errors="coerce")
            elif ln == "enddate":
                end = pd.to_datetime(txt, errors="coerce")
            elif ln == "instant":
                instant = pd.to_datetime(txt, errors="coerce")
            elif ln in {"explicitmember", "typedmember"}:
                dim_count += 1
        out[cid] = {
            "start": start,
            "end": end,
            "instant": instant,
            "dimension_count": dim_count,
        }
    return out


def _augment_contexts_from_period_facts(root, contexts):
    """Reconstruct legacy NSE/BSE pseudo-contexts such as OneD/FourD/OneI.

    Several 2018-2021 financial-result instances reference contextRef IDs that
    are not declared as xbrli:context elements. Instead, the reporting-period
    start/end dates are themselves facts sharing that contextRef. Ignoring
    these pseudo-contexts drops the primary revenue/PAT facts and can leave only
    segment facts. This reconstruction is deterministic and uses only dates
    contained in the same filing.
    """
    period_dates = {}
    financial_dates = {}

    for el in _iter(root):
        cref = el.attrib.get("contextRef") or el.attrib.get("contextref")
        if not cref:
            continue
        concept = _fact_concept(el)
        n = norm_name(concept)
        txt = "".join(el.itertext()).strip() if hasattr(el, "itertext") else (el.text or "")
        dt = pd.to_datetime(txt, errors="coerce")
        if pd.isna(dt):
            continue

        if n == "dateofstartofreportingperiod":
            period_dates.setdefault(cref, {})["start"] = pd.Timestamp(dt)
        elif n == "dateofendofreportingperiod":
            period_dates.setdefault(cref, {})["end"] = pd.Timestamp(dt)
        elif n == "dateofstartoffinancialyear":
            financial_dates.setdefault(cref, {})["start"] = pd.Timestamp(dt)
        elif n == "dateofendoffinancialyear":
            financial_dates.setdefault(cref, {})["end"] = pd.Timestamp(dt)

    # Duration pseudo-contexts: reporting-period dates take precedence over
    # full-financial-year dates.
    for cref in set(period_dates) | set(financial_dates):
        if cref in contexts:
            continue
        d = period_dates.get(cref) or financial_dates.get(cref) or {}
        start, end = d.get("start"), d.get("end")
        if pd.notna(start) and pd.notna(end):
            contexts[cref] = {
                "start": start,
                "end": end,
                "instant": None,
                "dimension_count": 0,
                "synthetic": True,
            }

    # Instant pseudo-contexts (OneI/FourI/etc.) usually correspond to the
    # reporting end date of the same prefix's duration context.
    referenced = set()
    for el in _iter(root):
        cref = el.attrib.get("contextRef") or el.attrib.get("contextref")
        if cref:
            referenced.add(cref)

    for cref in referenced:
        if cref in contexts or not str(cref).endswith("I"):
            continue
        dref = str(cref)[:-1] + "D"
        dc = contexts.get(dref)
        if dc and pd.notna(dc.get("end")):
            contexts[cref] = {
                "start": None,
                "end": None,
                "instant": pd.Timestamp(dc["end"]),
                "dimension_count": 0,
                "synthetic": True,
            }

    return contexts


def _fact_concept(el) -> str:
    ln = local_name(el.tag)
    if ln.lower() in {"nonfraction", "nonnumeric", "fraction"}:
        name = el.attrib.get("name", "")
        return name.split(":")[-1]
    return ln


def _all_numeric_concepts(root, contexts):
    records = []
    namespaces = set()
    concepts = []
    for el in _iter(root):
        cref = el.attrib.get("contextRef") or el.attrib.get("contextref")
        if not cref or cref not in contexts:
            continue
        concept = _fact_concept(el)
        txt = "".join(el.itertext()).strip() if hasattr(el, "itertext") else (el.text or "")
        val = parse_number(txt, el.attrib)
        if not np.isfinite(val):
            continue
        ns = namespace_uri(el.tag)
        if ns:
            namespaces.add(ns)
        concepts.append(concept)
        records.append((el, cref, concept, float(val), ns))
    return records, namespaces, concepts


def _parse_facts(root, contexts, compiled):
    raw, namespaces, concept_names = _all_numeric_concepts(root, contexts)
    family = classify_taxonomy(namespaces, concept_names)

    facts = []
    for el, cref, concept, val, ns in raw:
        canonical, conf = map_concept(concept, compiled, family)
        if canonical is None:
            continue
        facts.append({
            "context": cref,
            "concept": concept,
            "canonical": canonical,
            "confidence": conf,
            "value": val,
            "unit": el.attrib.get("unitRef") or el.attrib.get("unitref"),
            "namespace": ns,
        })
    return facts, namespaces, concept_names, family, len(raw)


def _best_fact(facts, canonical, contexts, allowed_contexts):
    cand = [
        f for f in facts
        if f["canonical"] == canonical and f["context"] in allowed_contexts
    ]
    if not cand:
        return None
    cand.sort(
        key=lambda f: (
            contexts[f["context"]]["dimension_count"],
            -f["confidence"],
            len(f["concept"]),
        )
    )
    return cand[0]


def normalize_xbrl(
    xml_bytes: bytes,
    metadata: dict[str, Any],
    mapping: dict,
) -> list[dict[str, Any]]:
    try:
        root = _parse_root(xml_bytes)
    except Exception as exc:
        raise ValueError(f"XML/iXBRL parse failed: {exc}") from exc

    compiled = compile_map(mapping)
    contexts = _parse_contexts(root)
    contexts = _augment_contexts_from_period_facts(root, contexts)
    facts, namespaces, concepts, family, numeric_fact_count = _parse_facts(
        root, contexts, compiled
    )

    durations = []
    instants_by_date: dict[pd.Timestamp, list[str]] = {}
    for cid, c in contexts.items():
        if pd.notna(c["instant"]):
            d = pd.Timestamp(c["instant"]).normalize()
            instants_by_date.setdefault(d, []).append(cid)
        if pd.notna(c["start"]) and pd.notna(c["end"]):
            start, end = pd.Timestamp(c["start"]), pd.Timestamp(c["end"])
            months = context_period_months(start, end)
            if months is not None:
                durations.append(
                    (
                        cid,
                        start.normalize(),
                        end.normalize(),
                        months,
                        c["dimension_count"],
                    )
                )

    candidates: dict[tuple[pd.Timestamp, int], list[dict[str, Any]]] = {}

    for cid, start, end, months, dims in durations:
        # Include contexts with the same economic duration/end-date, then let
        # _best_fact prefer non-dimensional contexts. This is more robust across
        # old/current NSE taxonomies than requiring a single exact context id.
        flow_ctx = [
            c_id
            for c_id, c in contexts.items()
            if pd.notna(c["start"])
            and pd.notna(c["end"])
            and pd.Timestamp(c["end"]).normalize() == end
            and context_period_months(
                pd.Timestamp(c["start"]), pd.Timestamp(c["end"])
            ) == months
        ]
        instant_ctx = instants_by_date.get(end, [])

        row = {
            "symbol": str(metadata.get("symbol", "")).upper(),
            "period_end": end,
            "broadcast_ts": metadata.get("broadcast_ts"),
            "statement_scope": metadata.get("statement_scope", "unknown"),
            "period_months": months,
            "filing_id": metadata.get("filing_id"),
            "source_url": metadata.get("source_url") or metadata.get("xbrl_url"),
            "source": metadata.get("source"),
            "taxonomy_family": family,
            "context_dimension_count": dims,
            "synthetic_context_used": bool(contexts.get(cid, {}).get("synthetic", False)),
        }
        scores = []
        source_concepts = {}

        for field in FLOW_FIELDS:
            f = _best_fact(facts, field, contexts, flow_ctx)
            row[field] = f["value"] if f else np.nan
            if f:
                scores.append(f["confidence"])
                source_concepts[field] = f["concept"]

        for field in INSTANT_FIELDS:
            f = _best_fact(facts, field, contexts, instant_ctx)
            if f is None:
                f = _best_fact(facts, field, contexts, flow_ctx)
            row[field] = f["value"] if f else np.nan
            if f:
                scores.append(f["confidence"])
                source_concepts[field] = f["concept"]

        if not np.isfinite(row.get("total_debt", np.nan)):
            parts = [
                row.get("noncurrent_borrowings", np.nan),
                row.get("current_borrowings", np.nan),
            ]
            good = [x for x in parts if np.isfinite(x)]
            if good:
                row["total_debt"] = float(sum(good))
                scores.append(0.60)
                source_concepts["total_debt"] = "derived:current+noncurrent_borrowings"

        if not np.isfinite(row.get("operating_profit", np.nan)):
            pbt = row.get("pbt", np.nan)
            fc = row.get("finance_cost", np.nan)
            # Derivation is useful for non-financials/NBFCs, but not a clean
            # operating-profit proxy for banks/insurers.
            if family not in {"banking", "insurance", "life_insurance", "general_insurance"}:
                if np.isfinite(pbt) and np.isfinite(fc):
                    row["operating_profit"] = float(pbt + fc)
                    scores.append(0.50)
                    source_concepts["operating_profit"] = "derived:pbt+finance_cost"

        canonical_present = [
            c for c in [
                "revenue", "operating_profit", "pbt", "pat", "finance_cost",
                "total_assets", "total_equity", "total_debt",
                "current_assets", "current_liabilities",
                "operating_cash_flow", "capex", "shares_outstanding",
                "promoter_pct", "pledged_pct",
            ]
            if np.isfinite(row.get(c, np.nan))
        ]

        row["mapped_field_count"] = len(canonical_present)
        row["mapping_score"] = float(np.mean(scores)) if scores else 0.0
        row["namespace_count"] = len(namespaces)
        row["numeric_fact_count"] = int(numeric_fact_count)
        row["concepts_mapped"] = len(facts)
        row["concept_mapping_fraction"] = (
            float(len(facts) / numeric_fact_count) if numeric_fact_count else 0.0
        )
        row["source_concepts_json"] = json.dumps(
            source_concepts, ensure_ascii=False, sort_keys=True
        )
        candidates.setdefault((end, months), []).append(row)

    rows = []
    for _, group in candidates.items():
        group.sort(
            key=lambda r: (
                r["context_dimension_count"],
                -r["mapped_field_count"],
                -r["mapping_score"],
            )
        )
        rows.append(group[0])

    rows.sort(key=lambda r: (r["period_end"], r["period_months"]))
    return rows


def self_test(mapping_path: str | Path):
    xml = b"""<?xml version="1.0"?>
    <xbrl xmlns="http://www.xbrl.org/2003/instance" xmlns:in="http://example.com/indas">
      <context id="D3"><entity><identifier scheme="x">TEST</identifier></entity>
        <period><startDate>2025-04-01</startDate><endDate>2025-06-30</endDate></period></context>
      <context id="I"><entity><identifier scheme="x">TEST</identifier></entity>
        <period><instant>2025-06-30</instant></period></context>
      <in:RevenueFromOperationsIncludingExciseDuty contextRef="D3" unitRef="INR">140</in:RevenueFromOperationsIncludingExciseDuty>
      <in:NetProfitLossForThePeriod contextRef="D3" unitRef="INR">12</in:NetProfitLossForThePeriod>
      <in:FinanceCosts contextRef="D3" unitRef="INR">2</in:FinanceCosts>
      <in:ProfitLossBeforeTax contextRef="D3" unitRef="INR">15</in:ProfitLossBeforeTax>
      <in:Assets contextRef="I" unitRef="INR">300</in:Assets>
      <in:Equity contextRef="I" unitRef="INR">100</in:Equity>
      <in:NoncurrentBorrowings contextRef="I" unitRef="INR">30</in:NoncurrentBorrowings>
      <in:CurrentBorrowings contextRef="I" unitRef="INR">10</in:CurrentBorrowings>
    </xbrl>"""
    mapping = load_map(mapping_path)
    rows = normalize_xbrl(
        xml,
        {
            "symbol": "TEST",
            "broadcast_ts": "2025-08-10T10:00:00Z",
            "statement_scope": "consolidated",
            "filing_id": "synthetic",
            "source_url": "synthetic://test",
        },
        mapping,
    )
    assert len(rows) == 1
    r = rows[0]
    assert r["period_months"] == 3
    assert r["revenue"] == 140
    assert r["pat"] == 12
    assert r["total_debt"] == 40
    assert r["operating_profit"] == 17
    assert r["concept_mapping_fraction"] > 0
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mapping", default="config/xbrl_concept_map_v9_5.json")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--xml")
    ap.add_argument("--metadata-json")
    ap.add_argument("--output")
    args = ap.parse_args()

    if args.self_test:
        print(json.dumps(self_test(args.mapping), indent=2, default=str))
        return

    if not (args.xml and args.metadata_json and args.output):
        raise SystemExit("--xml, --metadata-json and --output are required")

    mapping = load_map(args.mapping)
    metadata = json.load(open(args.metadata_json))
    rows = normalize_xbrl(Path(args.xml).read_bytes(), metadata, mapping)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(json.dumps({"rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
