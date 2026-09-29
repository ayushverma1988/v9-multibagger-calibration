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
    if not s or s in {"-", "—", "nan", "None"}:
        return np.nan
    neg = s.startswith("(") and s.endswith(")")
    if neg:
        s = s[1:-1]
    s = s.replace(",", "").replace("₹", "").replace("%", "")
    try:
        x = float(s)
        return -x if neg else x
    except Exception:
        return np.nan


def _metadata_from_tables(tables):
    meta = {}
    for t in tables:
        if t.shape[1] < 2:
            continue
        vals = t.astype(str).fillna("")
        # Common legacy header table has paired key/value columns (4 cols).
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
        vals = t.copy()
        # Focus on two-column financial tables; for wider tables use first 2 cols.
        vals = vals.iloc[:, :2]
        for _, r in vals.iterrows():
            label = str(r.iloc[0]).strip()
            value = r.iloc[1]
            if not label or label.lower() == "nan":
                continue
            rows.append((label, value))
    return rows


def _pick(rows, exact=(), contains=()):
    nr = [(_norm(k), _num(v), k) for k, v in rows]
    for key in exact:
        nk = _norm(key)
        for k, v, _ in nr:
            if k == nk and np.isfinite(v):
                return float(v)
    for key in contains:
        nk = _norm(key)
        for k, v, _ in nr:
            if nk in k and np.isfinite(v):
                return float(v)
    return np.nan


def _period_months(meta: dict, source_hint: str = "") -> int:
    period = str(meta.get("period", "")).lower()
    relating = str(meta.get("relatingto", "")).lower()
    cumulative = str(meta.get("cumulativenoncumulative", "")).lower()
    if "annual" in period or "annual" in source_hint.lower():
        return 12
    if "non-cumulative" in cumulative or "noncumulative" in cumulative:
        return 3
    if "first quarter" in relating:
        return 3
    if "second quarter" in relating or "half" in relating:
        return 6
    if "third quarter" in relating or "nine" in relating:
        return 9
    if "fourth quarter" in relating or "year" in relating:
        return 12
    return 3


def _scope(meta: dict, fallback: str = "") -> str:
    s = str(
        meta.get("consolidatednonconsolidated")
        or meta.get("consolidatedorstandalone")
        or fallback
    ).lower()
    return "consolidated" if "consolidated" in s and "non" not in s else "standalone"


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

    pe = pd.to_datetime(
        meta.get("periodended") or fallback_period_end,
        errors="coerce",
        dayfirst=True,
    )

    revenue = _pick(
        rows,
        exact=[
            "Total income from operations (net) ( a + b)",
            "Revenue from operations",
            "Total Revenue From Operations",
        ],
        contains=[
            "total income from operations",
            "revenue from operations",
            "net sales/income from operations",
        ],
    )
    operating_profit = _pick(
        rows,
        exact=[
            "Profit / (Loss) from operations before other income, finance costs and exceptional items",
            "Profit / (Loss) from operations before finance costs, exceptional items and tax",
            "Operating Profit",
        ],
        contains=[
            "profit / (loss) from operations before other income",
            "profit / (loss) from operations before finance costs",
            "operating profit",
        ],
    )
    pbt = _pick(
        rows,
        exact=[
            "Profit / (Loss) from ordinary activities before tax",
            "Profit before tax",
            "Profit / (loss) before tax",
        ],
        contains=["before tax"],
    )
    pat = _pick(
        rows,
        exact=[
            "Net Profit / (Loss) for the period",
            "Net Profit / (Loss) from ordinary activities after tax",
            "Profit for the period",
            "Profit / (loss) for the period",
        ],
        contains=["net profit / (loss) for the period", "profit for the period"],
    )
    finance_cost = _pick(
        rows,
        exact=["Finance costs", "Finance cost"],
        contains=["finance costs", "finance cost"],
    )

    total_assets = _pick(rows, exact=["Total Assets"], contains=["total assets"])
    total_equity = _pick(
        rows,
        exact=["Total Equity", "Shareholders' funds", "Shareholders funds"],
        contains=["total equity", "shareholders funds"],
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
    )
    debt_parts = [x for x in [noncurrent_borrowings, current_borrowings] if np.isfinite(x)]
    total_debt = float(sum(debt_parts)) if debt_parts else _pick(
        rows,
        exact=["Total Borrowings", "Total Debt"],
        contains=["total borrowings", "total debt"],
    )
    current_assets = _pick(rows, exact=["Current Assets", "Total Current Assets"], contains=["current assets"])
    current_liabilities = _pick(
        rows,
        exact=["Current Liabilities", "Total Current Liabilities"],
        contains=["current liabilities"],
    )
    operating_cash_flow = _pick(
        rows,
        exact=["Net cash from operating activities", "Net cash flow from operating activities"],
        contains=["cash from operating activities", "cash flow from operating activities"],
    )
    capex = _pick(
        rows,
        exact=["Purchase of fixed assets", "Purchase of property, plant and equipment"],
        contains=["purchase of fixed assets", "purchase of property"],
    )

    canon = {
        "revenue": revenue,
        "operating_profit": operating_profit,
        "pbt": pbt,
        "pat": pat,
        "finance_cost": finance_cost,
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
    present = [k for k, v in canon.items() if np.isfinite(v)]

    return {
        "symbol": str(symbol).upper(),
        "period_end": pe,
        "broadcast_ts": pd.to_datetime(broadcast_ts, errors="coerce", utc=True),
        "statement_scope": _scope(meta, fallback_scope),
        "period_months": _period_months(meta, source),
        "filing_id": filing_id,
        "source_url": source_url,
        "source": source,
        "taxonomy_family": "legacy_html_nonbanking",
        **canon,
        "mapped_field_count": len(present),
        "mapping_score": 0.90 if len(present) >= 5 else 0.70,
        "namespace_count": 0,
        "concepts_mapped": len(present),
        "context_dimension_count": 0,
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
    return r


if __name__ == "__main__":
    print(json.dumps(self_test(), indent=2, default=str))
