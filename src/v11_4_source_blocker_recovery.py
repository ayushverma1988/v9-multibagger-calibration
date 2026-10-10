"""Collect official evidence to improve the three measured source blockers.

No labels, historical ranks or optimization enter this source-only workflow.
Full raw evidence and company tables are private; logs contain aggregates.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests

from v11_4_exchange_reference_universe import (
    AMFI_XLSX, NSE_MASTER, NSE_SME_MASTER, BSE_MASTER, parse_amfi, parse_nse_master, build_universe)
from v11_4_verified_current_financials import parse_financial_document, parse_promoter_document, derive_metrics, mode
from v11_4_primary_document_catalysts import classify_document, extract_pdf, CATEGORIES
from v11_4_longterm_fundamentals import HEADERS
from v11_4_integrated_2025_source_pilot import ts
from v11_4_standalone_train_walkforward import fold_close
from v11_4_four_family_live_screener import evaluate_family
from v11_4_systematic_run import ADDITIONAL_CHECKS
from v11_4_verified_promoter_transactions import collect_verified_promoter_transactions

HOSTS = {"www.nseindia.com", "nsearchives.nseindia.com", "archives.nseindia.com",
         "www.bseindia.com", "api.bseindia.com", "portal.amfiindia.com"}
API = "https://www.nseindia.com/api/"


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def shas(raw):
    return hashlib.sha256(raw).hexdigest()


class EvidenceStore:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.receipts = []

    def get(self, url, params=None):
        if urlparse(url).scheme != "https" or urlparse(url).hostname not in HOSTS:
            raise ValueError("Only approved original exchange/AMFI domains accepted")
        requested = requests.Request("GET", url, params=params).prepare().url
        key = shas(requested.encode())
        receipt_path = self.folder / (key + ".json")
        body_path = self.folder / (key + ".bin")
        if receipt_path.exists() and body_path.exists():
            receipt, raw = json.loads(receipt_path.read_text()), body_path.read_bytes()
            if shas(raw) != receipt["sha256"] or receipt["requested_url"] != requested:
                raise ValueError("Cached source identity changed")
            self.receipts.append(receipt)
            return raw, receipt
        receipt = {"requested_url": requested, "retrieved_at_utc": utcnow()}
        try:
            r = self.session.get(url, params=params, timeout=25)
            receipt.update({"status": r.status_code, "bytes": len(r.content), "content_type": r.headers.get("Content-Type", "")})
            if urlparse(r.url).hostname not in HOSTS:
                raise ValueError("Source redirected away from original provider")
            r.raise_for_status()
            raw = r.content
            if not raw:
                raise ValueError("Empty source response")
            receipt["sha256"] = shas(raw)
            receipt["cached_body"] = body_path.name
            body_path.write_bytes(raw)
            receipt_path.write_text(json.dumps(receipt, indent=2))
            self.receipts.append(receipt)
            time.sleep(.12)
            return raw, receipt
        except Exception as exc:
            receipt["error"] = type(exc).__name__ + ": " + str(exc).split(" for url:")[0][:180]
            self.receipts.append(receipt)
            raise

    def json(self, url, params=None):
        raw, receipt = self.get(url, params)
        return json.loads(raw), receipt


def current_catalog(store, symbol, cutoff):
    rows = []
    for page in range(1, 4):
        data, receipt = store.json(API + "integrated-filing-results", {
            "index": "equities", "symbol": symbol, "type": "Integrated Filing- Financials", "page": page, "size": 100})
        if not isinstance(data, dict) or not isinstance(data.get("data"), list):
            raise ValueError("Integrated filing schema changed")
        for r in data["data"]:
            if str(r.get("symbol", "")).upper() != symbol:
                raise ValueError("Integrated API returned a different issuer")
            end = pd.to_datetime(r.get("qe_Date"), format="%d-%b-%Y", errors="coerce")
            clocks = [ts(r.get(k)) for k in ("broadcast_Date", "creation_Date", "revised_Date")]
            clocks = [t for t in clocks if pd.notna(t)]
            available = max(clocks) if clocks else pd.NaT
            if pd.isna(end) or pd.isna(available) or available > cutoff or available < end.tz_localize("Asia/Kolkata"):
                continue
            rows.append({"symbol": symbol, "period_end": str(end.date()), "available_at_utc": available.isoformat(),
                         "reporting_mode": mode(r.get("consolidated")), "xbrl_url": r.get("xbrl", ""),
                         "render_url": r.get("ixbrl", ""), "filing_id": str(r.get("seq_Id", "")),
                         "catalog_sha256": receipt["sha256"], "audited": str(r.get("audited", ""))})
        if page * 100 >= int(data.get("totalCount", 0)):
            break
    return rows


def select_current_filings(catalog, max_docs=6):
    """One coherent reporting basis; latest revisions available by cutoff."""
    x = pd.DataFrame(catalog)
    if x.empty:
        return []
    x = x[x["reporting_mode"].isin(["standalone", "consolidated"])].copy()
    x["date"] = pd.to_datetime(x["period_end"])
    x["pub"] = pd.to_datetime(x["available_at_utc"], utc=True)
    options = []
    for basis, g in x.groupby("reporting_mode"):
        g = g.sort_values("pub").drop_duplicates("period_end", keep="last")
        options.append((g["date"].max(), len(g), basis == "consolidated", g))
    if not options:
        return []
    g = max(options, key=lambda z: z[:3])[3]
    latest_q = g["date"].max()
    annual = sorted(g.loc[g["date"].dt.month.eq(3), "date"], reverse=True)
    wanted = [latest_q, latest_q - pd.offsets.QuarterEnd(), latest_q - pd.offsets.QuarterEnd(2), latest_q - pd.DateOffset(years=1)] + annual[:2]
    wanted = list(dict.fromkeys(wanted))[:max_docs]
    return g[g["date"].isin(wanted)].sort_values("date", ascending=False).drop(columns=["date", "pub"]).to_dict("records")


def company_financials(store, symbol, current_isin, cutoff):
    catalog = current_catalog(store, symbol, cutoff)
    chosen = select_current_filings(catalog)
    facts, errors, docs = [], [], []
    for r in chosen:
        try:
            if "BANKING" in r["xbrl_url"].upper() or "INSURANCE" in r["xbrl_url"].upper():
                raise ValueError("Sector-specific accounting schema requires separate ratios")
            raw, proof = store.get(r["xbrl_url"])
            html, render_proof = store.get(r["render_url"])
            parsed, identity = parse_financial_document(raw, symbol, r["period_end"], r["reporting_mode"], html)
            # Older fiscal documents may precede a split/change of equity ISIN.
            # The original issuer Symbol is checked; such documents are never
            # used to silently remap a different current security's price/size.
            docs.append({**r, **identity, "source_sha256": proof["sha256"], "render_sha256": render_proof["sha256"],
                         "first_retrieved_utc": proof["retrieved_at_utc"],
                         "current_security_ISIN_matches_document": identity["document_isin"] == current_isin})
            for f in parsed:
                facts.append({**r, **f, "source_sha256": proof["sha256"], "render_sha256": render_proof["sha256"],
                              "first_retrieved_utc": proof["retrieved_at_utc"]})
        except Exception as exc:
            errors.append({"symbol": symbol, "stage": "financial_document", "source_url": r["xbrl_url"], "error": str(exc)[:180]})
    metrics = derive_metrics(facts)
    try:
        data, receipt = store.json(API + "corporate-share-holdings-master", {"index": "equities", "symbol": symbol})
        if not isinstance(data, list):
            raise ValueError("Ownership API schema changed")
        eligible = []
        for r in data:
            if str(r.get("symbol", "")).upper() != symbol or str(r.get("isin", "")).upper() != current_isin:
                continue
            dates = [ts(r.get(k)) for k in ("broadcastDate", "systemDate", "revisedDate", "revisionDate")]
            dates = [t for t in dates if pd.notna(t)]
            pub = max(dates) if dates else pd.NaT
            end = pd.to_datetime(r.get("date"), format="%d-%b-%Y", errors="coerce")
            if pd.notna(pub) and pd.notna(end) and pub <= cutoff and end.tz_localize("Asia/Kolkata") <= pub:
                eligible.append((end, pub, r))
        if eligible:
            end, pub, r = max(eligible, key=lambda z: (z[0], z[1]))
            raw, proof = store.get(r["xbrl"])
            values = parse_promoter_document(raw, symbol, current_isin, str(end.date()), r.get("pr_and_prgrp"))
            metrics.update(values)
            docs.append({"symbol": symbol, "period_end": str(end.date()), "available_at_utc": pub.isoformat(),
                         "source_kind": "shareholding", "source_sha256": proof["sha256"], "catalog_sha256": receipt["sha256"],
                         "first_retrieved_utc": proof["retrieved_at_utc"], "source_url": r["xbrl"],
                         "promoter_XML_and_index_agree": True})
    except Exception as exc:
        errors.append({"symbol": symbol, "stage": "ownership", "error": str(exc)[:180]})
    return metrics, facts, docs, errors


def company_catalysts(store, symbol, name, cutoff, max_documents=4):
    start = cutoff - pd.Timedelta(days=185)
    data, index_proof = store.json(API + "corporate-announcements", {
        "index": "equities", "symbol": symbol, "from_date": start.strftime("%d-%m-%Y"), "to_date": cutoff.strftime("%d-%m-%Y")})
    if not isinstance(data, list):
        raise ValueError("NSE announcement index schema changed")
    eligible = []
    for r in data:
        if str(r.get("symbol", "")).upper() != symbol:
            raise ValueError("Announcement issuer does not match API filter")
        dates = [ts(r.get(k)) for k in ("an_dt", "exchdisstime")]
        dates = [t for t in dates if pd.notna(t)]
        pub = max(dates) if dates else pd.NaT
        text = str(r.get("desc", "")) + " " + str(r.get("attchmntText", ""))
        if pd.isna(pub) or pub > cutoff or pub < start:
            continue
        import re
        categories = [k for k, pattern in CATEGORIES.items() if re.search(pattern, text, re.I)]
        # Generic "Updates" is an unresolved index, so inspect the attachment
        # rather than infer that it has no operating catalyst. Its contents
        # still require issuer matching and a disclosed quantified fact.
        if categories or str(r.get("desc", "")).strip().lower() in {"updates", "general updates"}:
            priority = 0 if set(categories) & {"order", "capacity", "product", "approval"} else 2 if categories else 1
            eligible.append((priority, -pub.value, str(r.get("seq_id", "")), pub, r))
    rows, errors = [], []
    seen = set()
    for _, _, eid, pub, r in sorted(eligible, key=lambda z: z[:3]):
        url = str(r.get("attchmntFile", ""))
        if url in seen:
            continue
        seen.add(url)
        if len(seen) > max_documents:
            break
        base = {"symbol": symbol, "filing_id": eid, "source_available_utc": pub.isoformat(),
                "source_url": url, "headline": str(r.get("desc", "")), "catalog_sha256": index_proof["sha256"]}
        try:
            raw, proof = store.get(url)
            text = extract_pdf(raw)
            result = classify_document(text, symbol, name)
            rows.append({**base, **result, "source_sha256": proof["sha256"], "first_retrieved_utc": proof["retrieved_at_utc"],
                         "extracted_text_sha256": shas(text.encode())})
        except Exception as exc:
            errors.append({**base, "error": str(exc)[:180]})
    return rows, errors, {"announcement_rows": len(data), "relevant_index_candidates": len(eligible)}


def execute(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    store = EvidenceStore(out / "raw_PRIVATE")
    cutoff = fold_close(args.market_date)
    raw, ap = store.get(AMFI_XLSX)
    amfi = parse_amfi(raw)
    raw, np = store.get(NSE_MASTER)
    nse = parse_nse_master(raw)
    nse["nse_master_segment"] = "MAINBOARD"
    mainboard = nse.copy()
    raw, sp = store.get(NSE_SME_MASTER)
    sme = parse_nse_master(raw)
    sme["nse_master_segment"] = "SME"
    nse = pd.concat([nse, sme], ignore_index=True)
    if nse["isin"].duplicated().any() or nse["nse_current_symbol"].duplicated().any():
        raise ValueError("Mainboard and SME current masters conflict; require migration evidence")
    universe, coverage = build_universe(amfi, nse)
    coverage["NSE_mainboard_current_master_rows"] = len(mainboard)
    coverage["NSE_SME_current_master_rows"] = len(sme)
    coverage["source_sha256"] = {"AMFI": ap["sha256"], "NSE_equity_master": np["sha256"], "NSE_SME_master": sp["sha256"]}
    try:
        raw, bp = store.get(BSE_MASTER, {"Group": "", "Scripcode": "", "segment": "Equity", "status": "Active", "scripName": ""})
        body = json.loads(raw)
        coverage["BSE_live_transport"] = {"status": "READABLE_SCHEMA_NOT_YET_VERIFIED", "source_sha256": bp["sha256"], "response_type": type(body).__name__}
    except Exception as exc:
        coverage["BSE_live_transport"] = {"status": "BLOCKED", "error": str(exc).split(" for url:")[0][:180]}
    universe.to_parquet(out / "NSE_BSE_AMFI_reference_universe_PRIVATE.parquet", index=False)
    # Source-only candidate sleeves plus deterministic controls; no outcome data.
    payload = json.loads(Path(args.candidates).read_text())
    records = payload.get("rows", payload) if isinstance(payload, dict) else payload
    symbols = list(dict.fromkeys(str(r["symbol"]).upper().strip() for r in records))
    master = nse.set_index("nse_current_symbol")
    promoter_rows, promoter_docs, promoter_errors, promoter_summary = collect_verified_promoter_transactions(store, master, cutoff)
    (out / "verified_promoter_market_transactions_PRIVATE.json").write_text(json.dumps(promoter_rows, indent=2))
    (out / "promoter_filing_group_net_quantities_PRIVATE.json").write_text(json.dumps(promoter_docs, indent=2))
    (out / "promoter_direction_source_summary.json").write_text(json.dumps(promoter_summary, indent=2))
    missing = [s for s in symbols if s not in master.index]
    if missing:
        raise ValueError("Saved candidate identity absent from current official NSE master")
    controls = sorted(set(mainboard["nse_current_symbol"]) - set(symbols), key=lambda s: shas(s.encode()))[:args.controls]
    chosen = (symbols + controls)[:args.max_companies]
    company_rows, all_facts, all_docs, all_events, errors = [], [], [], [], list(promoter_errors)
    for i, symbol in enumerate(chosen, 1):
        identity = master.loc[symbol]
        try:
            metrics, facts, docs, es = company_financials(store, symbol, identity["isin"], cutoff)
            all_facts.extend(facts); all_docs.extend(docs); errors.extend(es)
        except Exception as exc:
            metrics = {}
            errors.append({"symbol": symbol, "stage": "financial_index", "error": str(exc)[:180]})
        events = []
        try:
            events, es, event_index = company_catalysts(store, symbol, identity["nse_current_name"], cutoff, args.catalysts_per_company)
            all_events.extend(events); errors.extend(es)
        except Exception as exc:
            errors.append({"symbol": symbol, "stage": "catalyst_index", "error": str(exc)[:180]})
            event_index = {}
        reference = universe.loc[universe["isin"].eq(identity["isin"])].iloc[0]
        row = {"symbol": symbol, "isin": identity["isin"], "market_date": args.market_date,
               "historical_asof_utc": cutoff.isoformat(), "source_pilot_group": "saved_sleeves" if symbol in symbols else "source_blind_control",
               "amfi_size_category": reference["amfi_size_category"], "financial_metrics": metrics,
               "event_index": event_index, "primary_documents_read": sum(e["issuer_identity_read"] for e in events),
               "quantified_primary_evidence_candidates": sum(len(e["quantified_evidence"]) for e in events),
               "verified_promoter_purchase_evidence": "PRESENT" if any(r["symbol"] == symbol and r["direction"] == "BUY" for r in promoter_rows) else "UNKNOWN",
               "causal_chain_status": "UNKNOWN", "prospective_observation": False}
        company_rows.append(row)
        (out / "company_enrichment_PRIVATE.json").write_text(json.dumps(company_rows, indent=2, default=str))
        (out / "filing_provenance_PRIVATE.json").write_text(json.dumps(all_docs, indent=2, default=str))
        (out / "primary_catalysts_PRIVATE.json").write_text(json.dumps(all_events, indent=2, default=str))
        (out / "source_errors_PRIVATE.json").write_text(json.dumps(errors, indent=2, default=str))
        if i % 5 == 0 or i == len(chosen):
            print(json.dumps({"source_recovery_companies_completed": i, "planned": len(chosen), "parsed_financial_fact_rows": len(all_facts),
                              "primary_event_documents_read": len(all_events)}, separators=(",", ":")), flush=True)
    pd.DataFrame(all_facts).to_parquet(out / "financial_facts_exact_context_PRIVATE.parquet", index=False)
    (out / "filing_provenance_PRIVATE.json").write_text(json.dumps(all_docs, indent=2, default=str))
    (out / "primary_catalysts_PRIVATE.json").write_text(json.dumps(all_events, indent=2, default=str))
    (out / "source_errors_PRIVATE.json").write_text(json.dumps(errors, indent=2, default=str))
    (out / "source_receipts_PRIVATE.json").write_text(json.dumps(store.receipts, indent=2))
    common = {k: {"PASS": 0, "FAIL": 0, "UNKNOWN": 0} for k in ADDITIONAL_CHECKS if not k.startswith("chart_")}
    for row in company_rows:
        for k, rule in ADDITIONAL_CHECKS.items():
            if k in common:
                common[k][evaluate_family(row["financial_metrics"], rule)["status"]] += 1
    counts = {}
    for row in company_rows:
        for key in row["financial_metrics"]:
            counts[key] = counts.get(key, 0) + 1
    report = {"scope": "CURRENT_SOURCE_RECOVERY_RESEARCH_OVERLAY_ONLY", "created_utc": utcnow(), "market_snapshot_date": args.market_date,
              "coverage": coverage, "planned_saved_sleeve_symbols": len(symbols), "pilot_companies": len(company_rows),
              "source_blind_controls": len([r for r in company_rows if r["source_pilot_group"] == "source_blind_control"]),
              "financial_documents_parsed": len([r for r in all_docs if r.get("source_kind") != "shareholding"]),
              "ownership_documents_verified": len([r for r in all_docs if r.get("source_kind") == "shareholding"]),
              "financial_facts_parsed": len(all_facts), "financial_facts_agree_with_rendered_statement": sum(f["rendered_statement_agrees"] for f in all_facts),
              "derived_metric_company_coverage": counts, "five_new_common_checks_pilot_counts": common,
              "primary_event_documents_read": sum(e["issuer_identity_read"] for e in all_events),
              "quantified_primary_catalyst_candidates": sum(len(e["quantified_evidence"]) for e in all_events),
              "complete_primary_causal_chains_verified": 0, "retrieval_or_semantic_errors": len(errors),
              "actual_promoter_market_direction": promoter_summary,
              "all_requested_annual_quarterly_valuation_fields_complete": False,
              "full_NSE_BSE_scoring_coverage": False, "unchanged_model_ranking": True,
              "historical_backtest_rebuild_performed": False, "new_blind_outcomes_observed": 0,
              "inputs_first_seen_after_decision_not_new_prospective_selections": True, "production_approved": False}
    (out / "source_blocker_recovery_summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--candidates", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--market-date", default="2026-10-09")
    p.add_argument("--max-companies", type=int, default=42)
    p.add_argument("--controls", type=int, default=12)
    p.add_argument("--catalysts-per-company", type=int, default=3)
    execute(p.parse_args())


if __name__ == "__main__":
    main()
