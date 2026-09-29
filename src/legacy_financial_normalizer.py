from __future__ import annotations

import io
import json
import re
from typing import Any

import numpy as np
import pandas as pd
import requests


HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/134 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,*/*",
    "Referer": "https://www.nseindia.com/",
}


def _norm(s: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def _num(v):
    if v is None:
        return np.nan
    s = str(v).strip()
    if not s or s in {"-", "—", "nan", "None", "NA", "N/A"}:
        return np.nan
    neg = s.startswith("(") and s.endswith(")")
    if neg:
        s = s[1:-1]
    s = (
        s.replace(",", "")
        .replace("₹", "")
        .replace("%", "")
        .replace("\u2212", "-")
        .replace("\xa0", "")
    )
    m = re.search(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", s)
    if not m:
        return np.nan
    try:
        x = float(m.group(0))
        return -abs(x) if neg else x
    except Exception:
        return np.nan


def _metadata_from_tables(tables):
    meta = {}
    for t in tables:
        if t.shape[1] < 2:
            continue
        vals = t.astype(str).fillna("")
        for _, row in vals.iterrows():
            items = list(row.values)
            for i in range(0, len(items) - 1, 2):
                k = str(items[i]).strip()
                v = str(items[i + 1]).strip()
                if k and v and k.lower() != "nan" and v.lower() != "nan":
                    meta[_norm(k)] = v
    return meta


def _flatten_value_rows(tables):
    rows = []
    for t in tables:
        if t.shape[1] < 2:
            continue
        vals = t.copy().iloc[:, :2]
        for _, r in vals.iterrows():
            label = str(r.iloc[0]).strip()
            value = r.iloc[1]
            if not label or label.lower() == "nan":
                continue
            rows.append((label, value))
    return rows


def _pick(rows, exact=(), contains=(), excludes=()):
    nr = [(_norm(k), _num(v), k) for k, v in rows]
    ex = [_norm(x) for x in excludes]

    def allowed(k):
        return not any(x and x in k for x in ex)

    for key in exact:
        nk = _norm(key)
        for k, v, _ in nr:
            if k == nk and allowed(k) and np.isfinite(v):
                return float(v)
    for key in contains:
        nk = _norm(key)
        for k, v, _ in nr:
            if nk in k and allowed(k) and np.isfinite(v):
                return float(v)
    return np.nan


def _page_text(tables) -> str:
    vals = []
    for t in tables:
        try:
            vals.extend(t.astype(str).fillna("").values.ravel().tolist())
        except Exception:
            pass
    return " ".join(str(x) for x in vals).lower()


def _classify_family(tables, meta):
    txt = _page_text(tables)
    industry = str(meta.get("industry", "")).lower()
    s = txt + " " + industry

    if any(x in s for x in [
        "life insurance", "life insurance business", "policyholders account"
    ]):
        return "legacy_html_life_insurance"
    if any(x in s for x in [
        "general insurance", "non life insurance", "premium earned"
    ]):
        return "legacy_html_general_insurance"
    if "insurance" in s and any(x in s for x in ["premium", "claims incurred"]):
        return "legacy_html_insurance"
    if any(x in s for x in [
        "interest earned", "interest expended", "deposits", "advances",
        "provisions and contingencies"
    ]):
        return "legacy_html_banking"
    if any(x in s for x in [
        "non banking financial", "nbfc", "finance company"
    ]):
        return "legacy_html_nbfc"
    return "legacy_html_nonbanking"


def _period_months(meta: dict, source_hint: str = "") -> int:
    period = str(meta.get("period", "")).lower()
    relating = str(meta.get("relatingto", "")).lower()
    cumulative = str(meta.get("cumulativenoncumulative", "")).lower()

    if "annual" in period or "annual" in source_hint.lower():
        return 12

    noncum = "non-cumulative" in cumulative or "noncumulative" in cumulative
    if noncum:
        return 3

    if "first quarter" in relating:
        return 3
    if "second quarter" in relating or "half year" in relating:
        return 6
    if "third quarter" in relating or "nine month" in relating:
        return 9
    if "fourth quarter" in relating or "year ended" in relating:
        return 12

    if "quarterly" in period:
        return 3
    if "half yearly" in period or "half-yearly" in period:
        return 6
    return 3


def _scope(meta: dict, fallback: str = "") -> str:
    s = str(
        meta.get("consolidatednonconsolidated")
        or meta.get("consolidatedorstandalone")
        or fallback
    ).lower()
    return "consolidated" if "consolidated" in s and "non" not in s else "standalone"


def _map_nonfinancial(rows):
    return {
        "revenue": _pick(
            rows,
            exact=[
                "Total income from operations (net) ( a + b)",
                "Revenue from operations",
                "Total Revenue From Operations",
                "Income from operations",
                "Net Sales/Income from operations",
            ],
            contains=[
                "total income from operations",
                "revenue from operations",
                "net sales/income from operations",
                "income from operations",
                "net sales",
            ],
            excludes=["segment"],
        ),
        "operating_profit": _pick(
            rows,
            exact=[
                "Profit / (Loss) from operations before other income, finance costs and exceptional items",
                "Profit / (Loss) from operations before finance costs, exceptional items and tax",
                "Operating Profit",
                "EBITDA",
            ],
            contains=[
                "profit / (loss) from operations before other income",
                "profit / (loss) from operations before finance costs",
                "operating profit",
                "ebitda",
            ],
            excludes=["segment"],
        ),
        "pbt": _pick(
            rows,
            exact=[
                "Profit / (Loss) from ordinary activities before tax",
                "Profit before tax",
                "Profit / (loss) before tax",
                "Profit before taxation",
            ],
            contains=["before tax", "before taxation"],
            excludes=["segment"],
        ),
        "pat": _pick(
            rows,
            exact=[
                "Net Profit / (Loss) for the period",
                "Net Profit / (Loss) from ordinary activities after tax",
                "Profit for the period",
                "Profit / (loss) for the period",
                "Profit after tax",
                "Net profit after tax",
            ],
            contains=[
                "net profit / (loss) for the period",
                "profit for the period",
                "profit after tax",
                "net profit after tax",
            ],
            excludes=["earnings per share", "segment"],
        ),
        "finance_cost": _pick(
            rows,
            exact=["Finance costs", "Finance cost", "Interest and finance charges"],
            contains=["finance costs", "finance cost", "interest and finance charges"],
            excludes=["segment"],
        ),
    }


def _map_banking(rows):
    return {
        "revenue": _pick(
            rows,
            exact=["Total Income", "Interest Earned"],
            contains=["total income", "interest earned"],
            excludes=["segment"],
        ),
        "operating_profit": _pick(
            rows,
            exact=[
                "Operating Profit before Provisions and Contingencies",
                "Operating Profit",
            ],
            contains=["operating profit before provisions", "operating profit"],
            excludes=["segment"],
        ),
        "pbt": _pick(
            rows,
            exact=["Profit Before Tax", "Profit / (Loss) before tax"],
            contains=["profit before tax", "before tax"],
        ),
        "pat": _pick(
            rows,
            exact=["Net Profit", "Net Profit / (Loss) for the period", "Profit After Tax"],
            contains=["net profit", "profit after tax"],
            excludes=["earnings per share"],
        ),
        "finance_cost": _pick(
            rows,
            exact=["Interest Expended", "Interest Expense"],
            contains=["interest expended", "interest expense"],
        ),
    }


def _map_insurance(rows):
    return {
        "revenue": _pick(
            rows,
            exact=[
                "Net Premium Income", "Premium Earned (Net)",
                "Gross Premium Written", "Total Income",
            ],
            contains=[
                "net premium income", "premium earned", "gross premium written",
                "total income",
            ],
        ),
        "operating_profit": _pick(
            rows,
            exact=["Underwriting Profit / (Loss)", "Operating Profit"],
            contains=["underwriting profit", "operating profit"],
        ),
        "pbt": _pick(
            rows,
            exact=["Profit Before Tax", "Profit / (Loss) before tax"],
            contains=["profit before tax", "before tax"],
        ),
        "pat": _pick(
            rows,
            exact=["Profit After Tax", "Net Profit", "Profit for the period"],
            contains=["profit after tax", "net profit", "profit for the period"],
            excludes=["earnings per share"],
        ),
        "finance_cost": _pick(
            rows,
            exact=["Finance Costs", "Interest Expense"],
            contains=["finance cost", "interest expense"],
        ),
    }


def _map_balance_and_cash(rows):
    total_assets = _pick(
        rows,
        exact=["Total Assets", "Total Assets (A+B)", "TOTAL ASSETS"],
        contains=["total assets"],
        excludes=["turnover"],
    )
    total_equity = _pick(
        rows,
        exact=[
            "Total Equity", "Shareholders' funds", "Shareholders funds",
            "Net Worth", "Capital and Reserves", "Capital and Reserves and Surplus",
        ],
        contains=[
            "total equity", "shareholders funds", "net worth",
            "capital and reserves",
        ],
        excludes=["minority"],
    )

    noncurrent_borrowings = _pick(
        rows,
        exact=["Long-term borrowings", "Non-current borrowings"],
        contains=["long term borrowings", "non current borrowings"],
    )
    current_borrowings = _pick(
        rows,
        exact=["Short-term borrowings", "Current borrowings"],
        contains=["short term borrowings", "current borrowings"],
        excludes=["non current"],
    )
    total_debt = _pick(
        rows,
        exact=["Total Borrowings", "Total Debt", "Borrowings"],
        contains=["total borrowings", "total debt"],
    )
    if not np.isfinite(total_debt):
        parts = [x for x in [noncurrent_borrowings, current_borrowings] if np.isfinite(x)]
        if parts:
            total_debt = float(sum(parts))

    # For banks this maps borrowings, but deposits are deliberately NOT treated
    # as debt for the generic debt/equity feature.
    current_assets = _pick(
        rows,
        exact=["Current Assets", "Total Current Assets"],
        contains=["current assets"],
    )
    current_liabilities = _pick(
        rows,
        exact=["Current Liabilities", "Total Current Liabilities"],
        contains=["current liabilities"],
    )
    operating_cash_flow = _pick(
        rows,
        exact=[
            "Net cash from operating activities",
            "Net cash flow from operating activities",
            "Net Cash Generated from Operating Activities",
        ],
        contains=[
            "cash from operating activities",
            "cash flow from operating activities",
            "cash generated from operating activities",
        ],
    )
    capex = _pick(
        rows,
        exact=[
            "Purchase of fixed assets",
            "Purchase of property, plant and equipment",
            "Purchase of property plant and equipment",
            "Capital expenditure",
        ],
        contains=[
            "purchase of fixed assets",
            "purchase of property",
            "capital expenditure",
        ],
        excludes=["sale", "proceeds"],
    )

    return {
        "total_assets": total_assets,
        "total_equity": total_equity,
        "total_debt": total_debt,
        "current_assets": current_assets,
        "current_liabilities": current_liabilities,
        "operating_cash_flow": operating_cash_flow,
        "capex": capex,
        "shares_outstanding": np.nan,
        "promoter_pct": np.nan,
        "pledged_pct": np.nan,
    }


def parse_legacy_result_html(
    html: bytes,
    *,
    symbol: str,
    broadcast_ts,
    source_url: str,
    source: str,
    fallback_period_end=None,
    fallback_scope="",
    filing_id=None,
):
    tables = pd.read_html(io.BytesIO(html))
    meta = _metadata_from_tables(tables)
    rows = _flatten_value_rows(tables)
    family = _classify_family(tables, meta)

    pe = pd.to_datetime(
        meta.get("periodended") or fallback_period_end,
        errors="coerce",
        dayfirst=True,
    )

    if "banking" in family:
        flow = _map_banking(rows)
    elif "insurance" in family:
        flow = _map_insurance(rows)
    else:
        flow = _map_nonfinancial(rows)

    canon = {
        **flow,
        **_map_balance_and_cash(rows),
    }

    present = [k for k, v in canon.items() if np.isfinite(v)]
    score = 0.92 if len(present) >= 8 else 0.88 if len(present) >= 5 else 0.72

    return {
        "symbol": str(symbol).upper(),
        "period_end": pe,
        "broadcast_ts": pd.to_datetime(broadcast_ts, errors="coerce", utc=True),
        "statement_scope": _scope(meta, fallback_scope),
        "period_months": _period_months(meta, source),
        "filing_id": filing_id,
        "source_url": source_url,
        "source": source,
        "taxonomy_family": family,
        **canon,
        "mapped_field_count": len(present),
        "mapping_score": score,
        "namespace_count": 0,
        "numeric_fact_count": len(rows),
        "concepts_mapped": len(present),
        "concept_mapping_fraction": (
            float(len(present) / len(rows)) if rows else 0.0
        ),
        "context_dimension_count": 0,
        "source_concepts_json": "{}",
    }


def fetch_and_parse(url: str, **kwargs):
    r = requests.get(url, headers=HEADERS, timeout=35)
    r.raise_for_status()
    return parse_legacy_result_html(r.content, source_url=url, **kwargs)


def self_test():
    html = b"""
    <table>
      <tr><td>Symbol</td><td>TEST</td><td>Period Ended</td><td>30-Sep-2016</td></tr>
      <tr><td>Cumulative / Non-Cumulative</td><td>Non-Cumulative</td><td>Consolidated / Non-Consolidated</td><td>Non-Consolidated</td></tr>
      <tr><td>Period</td><td>Quarterly</td><td>Relating to</td><td>Second Quarter</td></tr>
    </table>
    <table>
      <tr><td>Description</td><td>Amount(Rs. in lakhs)</td></tr>
      <tr><td>Total income from operations (net) ( a + b)</td><td>198642.00</td></tr>
      <tr><td>Profit / (Loss) from operations before other income, finance costs and exceptional items</td><td>-40267.00</td></tr>
      <tr><td>Finance costs</td><td>68902.00</td></tr>
      <tr><td>Profit / (Loss) from ordinary activities before tax</td><td>-105792.00</td></tr>
      <tr><td>Net Profit / (Loss) for the period</td><td>-52771.00</td></tr>
      <tr><td>Total Assets</td><td>500000.00</td></tr>
      <tr><td>Shareholders funds</td><td>100000.00</td></tr>
    </table>"""
    r = parse_legacy_result_html(
        html,
        symbol="TEST",
        broadcast_ts="2017-01-02T18:42:04Z",
        source_url="synthetic://legacy",
        source="nse_legacy_quarterly",
    )
    assert r["period_months"] == 3
    assert r["revenue"] == 198642.0
    assert r["pat"] == -52771.0
    assert r["finance_cost"] == 68902.0
    assert r["total_assets"] == 500000.0
    return r


if __name__ == "__main__":
    print(json.dumps(self_test(), indent=2, default=str))
