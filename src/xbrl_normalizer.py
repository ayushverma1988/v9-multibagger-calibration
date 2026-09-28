from __future__ import annotations

import argparse
import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


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
    if tag.startswith("{"):
        return tag.split("}", 1)[1]
    return tag.split(":")[-1]


def namespace_uri(tag: str) -> str:
    if tag.startswith("{"):
        return tag[1:].split("}", 1)[0]
    return ""


def norm_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def parse_number(text: str | None, attrs: dict[str, str]) -> float:
    if attrs.get("{http://www.w3.org/2001/XMLSchema-instance}nil") == "true":
        return np.nan
    if text is None:
        return np.nan
    s = str(text).strip()
    if not s or s in {"-", "—", "NA", "N/A", "nil"}:
        return np.nan
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1]
    s = s.replace(",", "").replace("₹", "").replace("%", "").strip()
    try:
        v = float(s)
    except Exception:
        return np.nan
    if neg:
        v = -v
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
    tolerances = {3: 32, 6: 45, 9: 50, 12: 65}
    return month if abs(days - targets[month]) <= tolerances[month] else None


def load_map(path: str | Path) -> dict:
    return json.load(open(path, encoding="utf-8"))


def compile_map(mapping: dict):
    out = {}
    for canonical, spec in mapping["concepts"].items():
        aliases = {norm_name(x) for x in spec.get("aliases", [])}
        regex = [re.compile(x, re.I) for x in spec.get("regex", [])]
        out[canonical] = (aliases, regex)
    return out


def map_concept(concept: str, compiled: dict) -> tuple[str | None, float]:
    n = norm_name(concept)
    for canonical, (aliases, _) in compiled.items():
        if n in aliases:
            return canonical, 1.0
    for canonical, (_, patterns) in compiled.items():
        if any(p.search(n) for p in patterns):
            return canonical, 0.70
    return None, 0.0


def classify_taxonomy(namespaces: set[str], concepts: list[str]) -> str:
    s = " ".join(sorted(namespaces)).lower() + " " + " ".join(concepts[:100]).lower()
    if "lifeinsurance" in s or "life-insurance" in s:
        return "life_insurance"
    if "generalinsurance" in s or "general-insurance" in s:
        return "general_insurance"
    if "insurance" in s:
        return "insurance"
    if "bank" in s or "banking" in s:
        return "banking"
    if "nbfc" in s:
        return "nbfc"
    if "indas" in s or "ind-as" in s:
        return "ind_as"
    return "generic_or_unknown"


def _parse_contexts(root: ET.Element) -> dict[str, dict[str, Any]]:
    out = {}
    for el in root.iter():
        if local_name(el.tag).lower() != "context":
            continue
        cid = el.attrib.get("id")
        if not cid:
            continue
        start = end = instant = None
        dim_count = 0
        for d in el.iter():
            ln = local_name(d.tag).lower()
            txt = (d.text or "").strip()
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


def _fact_concept(el: ET.Element) -> str:
    ln = local_name(el.tag)
    if ln.lower() in {"nonfraction", "nonnumeric", "fraction"}:
        name = el.attrib.get("name", "")
        return name.split(":")[-1]
    return ln


def _parse_facts(root: ET.Element, contexts: dict, compiled: dict):
    facts = []
    namespaces = set()
    concept_names = []
    for el in root.iter():
        cref = el.attrib.get("contextRef") or el.attrib.get("contextref")
        if not cref or cref not in contexts:
            continue
        concept = _fact_concept(el)
        canonical, conf = map_concept(concept, compiled)
        if canonical is None:
            continue
        text = "".join(el.itertext()).strip()
        val = parse_number(text, el.attrib)
        if not np.isfinite(val):
            continue
        ns = namespace_uri(el.tag)
        if ns:
            namespaces.add(ns)
        concept_names.append(concept)
        facts.append({
            "context": cref,
            "concept": concept,
            "canonical": canonical,
            "confidence": conf,
            "value": float(val),
            "unit": el.attrib.get("unitRef") or el.attrib.get("unitref"),
        })
    return facts, namespaces, concept_names


def _best_fact(facts, canonical, contexts, allowed_contexts):
    cand = [f for f in facts if f["canonical"] == canonical and f["context"] in allowed_contexts]
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
        root = ET.fromstring(xml_bytes)
    except Exception as exc:
        raise ValueError(f"XML parse failed: {exc}") from exc

    compiled = compile_map(mapping)
    contexts = _parse_contexts(root)
    facts, namespaces, concepts = _parse_facts(root, contexts, compiled)
    family = classify_taxonomy(namespaces, concepts)

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
                durations.append((cid, start.normalize(), end.normalize(), months, c["dimension_count"]))

    # Prefer contexts without dimensions, but retain dimensional contexts if they
    # are the only source for a period.
    candidates: dict[tuple[pd.Timestamp, int], list[dict[str, Any]]] = {}
    for cid, start, end, months, dims in durations:
        flow_ctx = [cid]
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
        }
        scores = []

        for field in FLOW_FIELDS:
            f = _best_fact(facts, field, contexts, flow_ctx)
            row[field] = f["value"] if f else np.nan
            if f:
                scores.append(f["confidence"])

        for field in INSTANT_FIELDS:
            f = _best_fact(facts, field, contexts, instant_ctx)
            if f is None:
                # Some legacy taxonomies incorrectly/loosely attach stock facts
                # to duration contexts. Use only as a fallback.
                f = _best_fact(facts, field, contexts, flow_ctx)
            row[field] = f["value"] if f else np.nan
            if f:
                scores.append(f["confidence"])

        if not np.isfinite(row.get("total_debt", np.nan)):
            parts = [
                row.get("noncurrent_borrowings", np.nan),
                row.get("current_borrowings", np.nan),
            ]
            good = [x for x in parts if np.isfinite(x)]
            if good:
                row["total_debt"] = float(sum(good))
                scores.append(0.60)

        if not np.isfinite(row.get("operating_profit", np.nan)):
            pbt = row.get("pbt", np.nan)
            fc = row.get("finance_cost", np.nan)
            if np.isfinite(pbt) and np.isfinite(fc):
                row["operating_profit"] = float(pbt + fc)
                scores.append(0.55)

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
        row["concepts_mapped"] = len(facts)
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
      <in:RevenueFromOperations contextRef="D3" unitRef="INR">140</in:RevenueFromOperations>
      <in:ProfitLossForPeriod contextRef="D3" unitRef="INR">12</in:ProfitLossForPeriod>
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
