"""Official NSE master + AMFI exchange/size reference, joined by ISIN only.

AMFI is a six-month reference universe, not today's BSE active-security
master or a current market cap quote. These meanings remain separate.
"""
from __future__ import annotations
import io
import re
import pandas as pd

AMFI_XLSX = "https://portal.amfiindia.com/spages/AverageMarketCapitalization30Jun2026.xlsx"
AMFI_PERIOD_END = "2026-06-30"
NSE_MASTER = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
BSE_MASTER = "https://api.bseindia.com/BseIndiaAPI/api/ListofScripData_new/w"


def isin_valid(value):
    """ISO 6166 format and Luhn checksum, including expanded letters."""
    value = str(value).strip().upper()
    if not re.fullmatch(r"IN[A-Z0-9]{9}[0-9]", value):
        return False
    digits = "".join(str(ord(c) - 55) if c.isalpha() else c for c in value)
    total = 0
    for j, c in enumerate(reversed(digits)):
        v = int(c) * (2 if j % 2 else 1)
        total += v // 10 + v % 10
    return total % 10 == 0


def parse_amfi(raw):
    import openpyxl
    ws = openpyxl.load_workbook(io.BytesIO(raw), read_only=True, data_only=True).active
    it = iter(ws.values)
    title, header = next(it), next(it)
    if "30 June 2026" not in str(title[0]) or list(header)[:4] != ["Sr. No.", "Company name", "ISIN", "BSE Symbol"]:
        raise ValueError("AMFI file period or schema differs from frozen reference")
    rows = []
    for r in it:
        if not isinstance(r[0], (int, float)) or not r[2]:
            continue
        rank, isin = int(r[0]), str(r[2]).upper().strip()
        category = {"Large Cap": "LARGE", "Mid Cap": "MID", "Small Cap": "SMALL"}.get(r[10])
        expected = "LARGE" if rank <= 100 else "MID" if rank <= 250 else "SMALL"
        if category != expected or not isin_valid(isin):
            raise ValueError("Invalid AMFI rank/category or ISIN checksum")
        clean = lambda s: "" if s is None or str(s).strip() in {"-", ""} else str(s).upper().strip()
        rows.append({"isin": isin, "company_name": str(r[1]).strip(), "amfi_rank": rank,
                     "amfi_size_category": category, "bse_reference_symbol": clean(r[3]),
                     "nse_reference_symbol": clean(r[5]), "msei_reference_symbol": clean(r[7]),
                     "amfi_six_month_avg_market_cap_crore": float(r[9]) if r[9] is not None else None,
                     "amfi_period_end": AMFI_PERIOD_END})
    x = pd.DataFrame(rows)
    if len(x) < 5000 or x["isin"].duplicated().any() or x["amfi_rank"].duplicated().any():
        raise ValueError("Incomplete or ambiguous official AMFI reference")
    return x


def parse_nse_master(raw):
    x = pd.read_csv(io.BytesIO(raw))
    x.columns = x.columns.str.strip()
    cols = {"SYMBOL": "nse_current_symbol", "NAME OF COMPANY": "nse_current_name",
            "SERIES": "nse_current_series", "DATE OF LISTING": "nse_listing_date", "ISIN NUMBER": "isin"}
    if not set(cols).issubset(x):
        raise ValueError("NSE equity master schema incomplete")
    x = x[list(cols)].rename(columns=cols)
    for c in ("isin", "nse_current_symbol", "nse_current_series"):
        x[c] = x[c].str.strip().str.upper()
    if not x["isin"].map(isin_valid).all() or x["isin"].duplicated().any() or x["nse_current_symbol"].duplicated().any():
        raise ValueError("Ambiguous current NSE equity identity")
    x["nse_listing_date"] = pd.to_datetime(x["nse_listing_date"], format="%d-%b-%Y", errors="coerce")
    return x


def build_universe(amfi, nse, market=None):
    if amfi["isin"].duplicated().any() or nse["isin"].duplicated().any():
        raise ValueError("Cross-exchange identity must be one row per ISIN")
    x = amfi.merge(nse, on="isin", how="outer", validate="1:1")
    x["present_NSE_current_master"] = x["nse_current_symbol"].notna()
    x["present_BSE_AMFI_reference"] = x["bse_reference_symbol"].fillna("").ne("")
    x["bse_current_active_listing_verified"] = False
    x["cross_exchange_reference_ISIN_match"] = x["present_NSE_current_master"] & x["present_BSE_AMFI_reference"]
    x["reference_BSE_only_ISIN"] = x["present_BSE_AMFI_reference"] & ~x["present_NSE_current_master"]
    x["size_category_known_by_ISIN"] = x["amfi_size_category"].notna()
    x["verified_small_or_mid_reference"] = x["amfi_size_category"].isin(["SMALL", "MID"])
    # Same ticker, new ISIN is a coverage gap, never an implicit identity join.
    old_symbols = set(amfi["nse_reference_symbol"].dropna()) - {""}
    x["same_symbol_AMFI_ISIN_changed_requires_action_evidence"] = (
        x["present_NSE_current_master"] & ~x["size_category_known_by_ISIN"] &
        x["nse_current_symbol"].isin(old_symbols))
    report = {"official_AMFI_reference_rows": len(amfi), "current_NSE_master_rows": len(nse),
              "union_unique_ISINs": len(x),
              "BSE_referenced_ISINs": int(x["present_BSE_AMFI_reference"].sum()),
              "NSE_BSE_reference_overlap_ISINs": int(x["cross_exchange_reference_ISIN_match"].sum()),
              "BSE_only_reference_ISINs": int(x["reference_BSE_only_ISIN"].sum()),
              "current_NSE_ISINs_with_official_size_category": int((x["present_NSE_current_master"] & x["size_category_known_by_ISIN"]).sum()),
              "current_NSE_small_or_mid_reference": int((x["present_NSE_current_master"] & x["verified_small_or_mid_reference"]).sum()),
              "NSE_same_symbol_changed_ISIN_not_auto_joined": int(x["same_symbol_AMFI_ISIN_changed_requires_action_evidence"].sum()),
              "current_active_BSE_master_complete": False,
              "AMFI_average_cap_is_not_current_market_cap": True,
              "reference_first_seen_is_after_Oct9_decision": True,
              "all_listed_and_tradeable_equities_claim_supported": False}
    if market is not None:
        if not {"symbol", "isin"}.issubset(market) or market["symbol"].duplicated().any():
            raise ValueError("Same-day market identity is incomplete")
        m = market.merge(x[["isin", "amfi_size_category", "verified_small_or_mid_reference"]],
                         on="isin", how="left", validate="m:1")
        report["same_day_NSE_traded_equities"] = len(m)
        report["same_day_NSE_with_official_size_category"] = int(m["amfi_size_category"].notna().sum())
    return x, report
