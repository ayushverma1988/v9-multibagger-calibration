"""Source-level current NSE financial enrichment, independent of model ranks.

Every usable monetary fact has an exact period, INR unit, reporting basis,
source hash and an agreement with the exchange's rendered statement. XBRL
numbers are already rupees: LevelOfRounding is used ONLY for HTML comparison.
Missing components, banking-specific ratios and missing years stay unknown.
"""
from __future__ import annotations

import math
import re
from decimal import Decimal, InvalidOperation
from xml.etree import ElementTree as ET

import pandas as pd
from bs4 import BeautifulSoup

from v11_4_longterm_fundamentals import local, context_map
from v11_4_strict_annual_numeric_features import parse_units, dimensional_contexts

# Avoid economic substitutes: operating cash flow is after tax, PAT includes
# discontinued operations, receivables include current AND non-current.
TAGS = {
    "revenue": ["RevenueFromOperations", "RevenueFromOperationsNet"],
    "pat": ["ProfitLossForPeriod", "ProfitAfterTax", "ProfitLoss"],
    "pbt": ["ProfitBeforeTax", "ProfitLossBeforeTax"],
    "finance_cost": ["FinanceCosts", "FinanceCost"],
    "depreciation": ["DepreciationDepletionAndAmortisationExpense", "DepreciationAndAmortisationExpense"],
    "cfo": ["CashFlowsFromUsedInOperatingActivities", "NetCashFlowsFromUsedInOperatingActivities"],
    "equity": ["Equity", "TotalEquity", "ShareholdersFunds"],
    "reserves": ["OtherEquity", "ReservesAndSurplus"],
    "borrowings_current": ["BorrowingsCurrent", "CurrentBorrowings", "ShortTermBorrowings"],
    "borrowings_noncurrent": ["BorrowingsNoncurrent", "NoncurrentBorrowings", "LongTermBorrowings"],
    "receivables_current": ["TradeReceivablesCurrent", "TradeReceivables"],
    "receivables_noncurrent": ["TradeReceivablesNoncurrent", "TradeReceivablesNonCurrent"],
    "ppe": ["PropertyPlantAndEquipment", "TangibleAssets"],
    "cash": ["CashAndCashEquivalents"],
}
INSTANT = {"equity", "reserves", "borrowings_current", "borrowings_noncurrent",
           "receivables_current", "receivables_noncurrent", "ppe", "cash"}
HTML_LABELS = {
    "revenue": ["Revenue from operations"],
    "pat": ["Total profit (loss) for period", "Profit (loss) for the period", "Profit after tax"],
    "pbt": ["Total profit before tax", "Profit before tax", "Profit (loss) before tax"],
    "finance_cost": ["Finance costs"],
    "depreciation": ["Depreciation, depletion and amortisation expense", "Depreciation and amortisation expense"],
    "cfo": ["Net cash flows from (used in) operating activities"],
    "equity": ["Total equity", "Shareholders funds"],
    "reserves": ["Other equity", "Reserves and surplus"],
    "borrowings_current": ["Borrowings, current", "Current borrowings", "Short term borrowings"],
    "borrowings_noncurrent": ["Borrowings, non-current", "Noncurrent borrowings", "Long term borrowings"],
    "receivables_current": ["Trade receivables, current", "Trade receivables"],
    "receivables_noncurrent": ["Trade receivables, non-current"],
    "ppe": ["Property, plant and equipment", "Property plant and equipment", "Tangible assets"],
    "cash": ["Cash and cash equivalents"],
}
ROUNDING = {"rupees": 1, "actual": 1, "actuals": 1, "thousands": 1000,
            "lakhs": 100000, "lacs": 100000, "crores": 10000000,
            "millions": 1000000, "billions": 1000000000}


def numeric(value):
    s = str("" if value is None else value).strip().replace(",", "")
    if s.startswith("(") and s.endswith(")"):
        s = "-" + s[1:-1]
    try:
        x = float(Decimal(s))
        return x if math.isfinite(x) else None
    except (InvalidOperation, ValueError, OverflowError):
        return None


def text_normalize(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def scalar(root, tags):
    vals = {str(e.text).strip() for e in root.iter()
            if local(e.tag) in tags and e.text and e.text.strip()}
    if len(vals) > 1:
        raise ValueError("Contradictory document identity or reporting basis")
    return next(iter(vals), "")


def mode(value):
    s = str(value).lower()
    if "standalone" in s or "non-consolidated" in s:
        return "standalone"
    return "consolidated" if "consolidated" in s else "unknown"


def parse_financial_document(xml, symbol, period_end, reporting_mode, html=None):
    root = ET.fromstring(xml)
    declared_symbol = scalar(root, {"Symbol", "NSESymbol"})
    if declared_symbol.upper() != symbol.upper():
        raise ValueError("Financial document symbol differs from requested issuer")
    basis = mode(scalar(root, {"NatureOfReportStandaloneConsolidated"}))
    if basis == "unknown" or basis != mode(reporting_mode):
        raise ValueError("Financial document reporting basis differs from filing index")
    isin = scalar(root, {"ISIN"}).upper()
    if not re.fullmatch(r"INE[A-Z0-9]{8}[0-9]", isin):
        raise ValueError("Missing document equity ISIN")
    target = pd.Timestamp(period_end).date()
    ctx, units, dims = context_map(root), parse_units(root), dimensional_contexts(root)
    end_dates = {c.get("end") for c in ctx.values()}
    if target not in end_dates:
        raise ValueError("Financial document period differs from filing index")
    rounding = ROUNDING.get(scalar(root, {"LevelOfRounding"}).lower())
    rendered = []
    if html:
        soup = BeautifulSoup(html, "html.parser")
        for tr in soup.select("tr"):
            cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"], recursive=False)]
            if len(cells) < 2:
                continue
            for pos, label in enumerate(cells):
                if numeric(label) is None and label:
                    rendered.append((text_normalize(label), [numeric(z) for z in cells[pos + 1:]]))
    rows, ambiguous = [], []
    for key, aliases in TAGS.items():
        candidates = {}
        for el in root.iter():
            tag, ref = local(el.tag), el.get("contextRef")
            c = ctx.get(ref, {})
            if tag not in aliases or dims.get(ref, 0) or c.get("end") != target:
                continue
            if units.get(el.get("unitRef")) != ["INR"]:
                continue  # Reject INR/share, mixed units and dimensionless facts.
            value = numeric(el.text)
            if value is None:
                continue
            days = c.get("duration")
            kind = ("instant" if days is None else "annual" if 330 <= days <= 400
                    else "quarter" if 60 <= days <= 120 else "ytd")
            if key in INSTANT and kind != "instant":
                continue
            if key not in INSTANT and kind not in {"annual", "quarter"}:
                continue
            # CFO cannot be inferred from the quarter context of older FourD
            # schemas. Only an explicitly annual cash-flow duration is usable.
            if key == "cfo" and kind != "annual":
                continue
            candidate = {"metric": key, "value_INR": value, "tag": tag,
                         "context": ref, "period_start": str(c.get("start") or ""),
                         "period_end": str(target), "period_kind": kind,
                         "reporting_mode": basis, "document_isin": isin,
                         "currency": "INR", "unit": el.get("unitRef"),
                         "decimals": el.get("decimals", "")}
            candidates.setdefault(kind, []).append(candidate)
        for kind, options in candidates.items():
            priority = min(aliases.index(z["tag"]) for z in options)
            best = [z for z in options if aliases.index(z["tag"]) == priority]
            if len({z["value_INR"] for z in best}) != 1:
                ambiguous.append({"metric": key, "period_kind": kind})
                continue
            selected = best[0]
            labels = {text_normalize(z) for z in HTML_LABELS[key]}
            values = [v for label, vs in rendered if label in labels for v in vs if v is not None]
            # Half of one displayed last decimal is the maximum rounding error.
            # This compares representations of ONE filing, not two providers.
            tolerance = rounding * .0051 if rounding else 0
            selected["rendered_statement_agrees"] = bool(rounding and any(
                abs(v * rounding - selected["value_INR"]) <= max(tolerance, .01) for v in values))
            selected["render_scale_INR"] = rounding
            rows.append(selected)
    return rows, {"document_symbol": declared_symbol, "document_isin": isin,
                  "bse_scrip_code": scalar(root, {"ScripCode"}),
                  "reporting_mode": basis, "ambiguous_facts_rejected": ambiguous,
                  "is_banking_schema": "BANKING" in str(root.tag).upper(),
                  "monetary_values_not_rescaled": True}


def parse_promoter_document(xml, symbol, isin, period_end, index_holding_pct):
    root = ET.fromstring(xml)
    if scalar(root, {"Symbol", "NSESymbol"}).upper() != symbol.upper():
        raise ValueError("Ownership document symbol mismatch")
    if scalar(root, {"ISIN"}).upper() != isin.upper():
        raise ValueError("Ownership document current ISIN mismatch")
    target = pd.Timestamp(period_end).date()
    ctx = context_map(root)
    refs = set()
    for e in root.iter():
        if local(e.tag) != "context" or ctx.get(e.get("id"), {}).get("end") != target:
            continue
        members = [str(z.text).split(":")[-1] for z in e.iter() if local(z.tag) == "explicitMember"]
        if members == ["ShareholdingOfPromoterAndPromoterGroupMember"]:
            refs.add(e.get("id"))
    values = [numeric(e.text) for e in root.iter()
              if local(e.tag) == "ShareholdingAsAPercentageOfTotalNumberOfShares"
              and e.get("contextRef") in refs]
    values = {v for v in values if v is not None and 0 <= v <= 1}
    index = numeric(index_holding_pct)
    if len(values) != 1 or index is None or abs(next(iter(values)) * 100 - index) > .011:
        raise ValueError("Promoter aggregate does not reconcile with exchange index")
    result = {"promoter_holding": next(iter(values))}
    encumbered = scalar(root, {"WhetherAnySharesHeldByPromotersAreEncumberedUnderPledgedForPromoterAndPromoterGroup"})
    if encumbered.lower() == "false":
        result["pledged_pct"] = 0.0  # An explicit group-wide statement, not absence.
    return result


def derive_metrics(facts):
    """Only compute ratios when every required fact is rendered and coherent."""
    x = pd.DataFrame(facts)
    if x.empty:
        return {}
    x = x[x["rendered_statement_agrees"].eq(True)].copy()
    if x.empty:
        return {}
    if x["reporting_mode"].nunique() != 1:
        raise ValueError("Cannot combine standalone and consolidated ratios")
    if x.duplicated(["period_end", "period_kind", "metric"]).any():
        raise ValueError("Duplicate financial component after filing revision selection")
    annual, instant, quarter = {}, {}, {}
    for (end, kind), g in x.groupby(["period_end", "period_kind"]):
        dest = annual if kind == "annual" else instant if kind == "instant" else quarter
        dest[pd.Timestamp(end)] = dict(zip(g["metric"], g["value_INR"]))
    result = {}
    dates = sorted(annual, reverse=True)
    if not dates:
        return result
    latest = dates[0]
    f, b = annual[latest], instant.get(latest, {})
    debt = (b["borrowings_current"] + b["borrowings_noncurrent"]
            if all(k in b for k in ("borrowings_current", "borrowings_noncurrent")) else None)
    def ratio(a, denominator):
        return a / denominator if a is not None and denominator is not None and denominator > 0 else None
    def put(key, value):
        if value is not None and (isinstance(value, bool) or math.isfinite(value)):
            result[key] = value
    put("operating_cash_flow_INR", f.get("cfo"))
    put("cfo_to_pat_last_year", ratio(f.get("cfo"), f.get("pat")))
    if all(k in b for k in ("receivables_current", "receivables_noncurrent")) and f.get("pat", 0) > 0:
        put("receivables_lt_10pct_profit", b["receivables_current"] + b["receivables_noncurrent"] < .1 * f["pat"])
    put("debt_to_equity", ratio(debt, b.get("equity")))
    if debt is not None and "reserves" in b:
        put("reserves_gt_borrowings", b["reserves"] > debt)
    if all(k in f for k in ("pbt", "finance_cost")):
        ebit = f["pbt"] + f["finance_cost"]
        put("interest_coverage", ratio(ebit, f["finance_cost"]))
    prior = latest - pd.DateOffset(years=1)
    bp = instant.get(prior, {})
    if "ppe" in b and "ppe" in bp:
        put("fixed_assets_up_yoy", b["ppe"] > bp["ppe"])
    # No EBITDA-derived operating-margin or estimated-industry-PE substitutes.
    for years in (3, 5, 7):
        chain = [latest - pd.DateOffset(years=j) for j in range(years + 1)]
        if not all(d in annual for d in chain):
            continue
        old = annual[chain[-1]]
        for metric, prefix in (("revenue", "sales"), ("pat", "profit")):
            newv, oldv = f.get(metric), old.get(metric)
            if newv is not None and oldv is not None and newv > 0 and oldv > 0:
                put(f"{prefix}_growth_{years}y", (newv / oldv) ** (1 / years) - 1)
    qdates = sorted(quarter, reverse=True)
    if qdates:
        q = qdates[0]
        q1, q2, yoy = q - pd.offsets.QuarterEnd(), q - pd.offsets.QuarterEnd(2), q - pd.DateOffset(years=1)
        now, before, two, year = [quarter.get(d, {}) for d in (q, q1, q2, yoy)]
        if "revenue" in now and "revenue" in year:
            put("sales_latest_ge_yoy_quarter", now["revenue"] >= year["revenue"])
        if "revenue" in now and "revenue" in two:
            put("sales_latest_ge_2q_back", now["revenue"] >= two["revenue"])
        if "pat" in now and year.get("pat", 0) > 0:
            put("profit_latest_q_yoy_growth", now["pat"] / year["pat"] - 1)
        if "pat" in now and "pat" in before:
            put("pat_latest_gt_preceding", now["pat"] > before["pat"])
        if "pat" in before and "pat" in two:
            put("pat_preceding_gt_2q_back", before["pat"] > two["pat"])
    result["latest_annual_period_end"] = str(latest.date())
    return result
