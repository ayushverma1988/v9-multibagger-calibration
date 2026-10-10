"""Verify promoter market purchases/sales in original NSE insider XBRL.

Buying by one promoter is different from net accumulation by the group.
ESOPs, pledges, gifts and preferential allotments are not market purchases.
"""
from __future__ import annotations
from xml.etree import ElementTree as ET
import pandas as pd
from v11_4_longterm_fundamentals import local, context_map
from v11_4_strict_annual_numeric_features import parse_units
from v11_4_verified_current_financials import scalar, numeric, text_normalize
from v11_4_integrated_2025_source_pilot import ts


def parse_transactions(xml, expected_symbol, expected_isin):
    root = ET.fromstring(xml)
    if scalar(root, {"NSESymbol", "Symbol"}).upper() != expected_symbol.upper():
        raise ValueError("Promoter transaction issuer symbol mismatch")
    if scalar(root, {"ISIN"}).upper() != expected_isin.upper():
        raise ValueError("Promoter transaction current ISIN mismatch")
    ctx, units = context_map(root), parse_units(root)
    grouped = {}
    for e in root.iter():
        ref = e.get("contextRef")
        if not ref or not e.text or not e.text.strip():
            continue
        group = grouped.setdefault(ref, {})
        key, value = local(e.tag), (e.text.strip(), e.get("unitRef"))
        if key in group and group[key] != value:
            raise ValueError("Conflicting transaction facts in one actor context")
        group[key] = value
    rows = []
    for ref, fields in grouped.items():
        value = lambda key: fields.get(key, ("", None))[0]
        category = text_normalize(value("CategoryOfPerson"))
        if category not in {"promoter", "promoters", "promotergroup"}:
            continue
        kind = value("SecuritiesAcquiredOrDisposedTransactionType")
        acq_mode = value("ModeOfAcquisitionOrDisposal")
        direction = 1 if kind == "Buy" and acq_mode == "Market Purchase" else -1 if kind == "Sell" and acq_mode == "Market Sale" else None
        if direction is None or value("TypeOfInstrument") != "Equity Shares":
            continue
        keys = ["SecuritiesHeldPriorToAcquisitionOrDisposalNumberOfSecurity", "SecuritiesAcquiredOrDisposedNumberOfSecurity",
                "SecuritiesHeldPostAcquistionOrDisposalNumberOfSecurity"]
        if any(units.get(fields.get(k, (None, None))[1]) != ["SHARES"] for k in keys):
            raise ValueError("Promoter security counts lack share units")
        before, quantity, after = [numeric(value(k)) for k in keys]
        if any(v is None or v < 0 or not v.is_integer() for v in (before, quantity, after)) or quantity <= 0:
            raise ValueError("Invalid promoter share counts")
        if after - before != direction * quantity:
            raise ValueError("Transaction direction does not reconcile with pre/post holdings")
        trade_end = pd.to_datetime(value("DateOfAllotmentAdviceOrAcquisitionOfSharesOrSaleOfSharesSpecifyToDate"), errors="coerce")
        if pd.isna(trade_end):
            raise ValueError("Missing actual transaction date")
        rows.append({"symbol": expected_symbol, "isin": expected_isin, "context": ref,
                     "actor_category": value("CategoryOfPerson"), "actor_name": value("NameOfThePerson"),
                     "direction": "BUY" if direction == 1 else "SELL", "acquisition_mode": acq_mode,
                     "quantity_shares": int(quantity), "signed_quantity_shares": int(quantity * direction),
                     "holdings_before_shares": int(before), "holdings_after_shares": int(after),
                     "trade_end_date": str(trade_end.date()), "execution_exchange": value("ExchangeOnWhichTheTradeWasExecuted"),
                     "pre_post_holdings_reconcile": True})
    return rows


def collect_verified_promoter_transactions(store, master, cutoff):
    start = cutoff - pd.Timedelta(days=185)
    data, receipt = store.json("https://www.nseindia.com/api/corporates-pit", {
        "index": "equities", "from_date": start.strftime("%d-%m-%Y"), "to_date": cutoff.strftime("%d-%m-%Y")})
    if not isinstance(data, dict) or not isinstance(data.get("data"), list):
        raise ValueError("Official insider transaction index schema changed")
    selected = {}
    for r in data["data"]:
        symbol = str(r.get("symbol", "")).strip().upper()
        if symbol not in master.index or text_normalize(r.get("personCategory", "")) not in {"promoter", "promoters", "promotergroup"}:
            continue
        if (r.get("tdpTransactionType"), r.get("acqMode")) not in {("Buy", "Market Purchase"), ("Sell", "Market Sale")}:
            continue
        # Minute-resolution dissemination timestamps become available at the
        # END of the stated minute, never at an invented earlier second.
        pub = ts(r.get("date"))
        if pd.isna(pub):
            continue
        if len(str(r.get("date", "")).rsplit(" ", 1)[-1].split(":")) == 2:
            pub += pd.Timedelta(seconds=59)
        if pub < start or pub > cutoff:
            continue
        url = str(r.get("xbrl", ""))
        if not url:
            continue
        previous = selected.get(url)
        if previous and previous["symbol"] != symbol:
            raise ValueError("Same insider document assigned to multiple issuers")
        later = max(pub, pd.Timestamp(previous["available_at_utc"])) if previous else pub
        selected[url] = {"symbol": symbol, "available_at_utc": later.isoformat(), "index_sha256": receipt["sha256"]}
    rows, errors, documents = [], [], []
    for i, (url, meta) in enumerate(sorted(selected.items()), 1):
        try:
            raw, proof = store.get(url)
            current_isin = str(master.loc[meta["symbol"], "isin"])
            parsed = parse_transactions(raw, meta["symbol"], current_isin)
            # Full original filing may contain both buying and selling members.
            # Preserve all of them and expose the signed group sum separately.
            for row in parsed:
                if pd.Timestamp(row["trade_end_date"]).tz_localize("Asia/Kolkata") > pd.Timestamp(meta["available_at_utc"]):
                    raise ValueError("Transaction date after exchange publication")
            documents.append({**meta, "source_url": url, "source_sha256": proof["sha256"],
                              "first_retrieved_utc": proof["retrieved_at_utc"],
                              "document_net_promoter_market_quantity_shares": sum(r["signed_quantity_shares"] for r in parsed)})
            rows.extend({**meta, **r, "source_url": url, "source_sha256": proof["sha256"],
                         "first_retrieved_utc": proof["retrieved_at_utc"]} for r in parsed)
        except Exception as exc:
            errors.append({**meta, "source_url": url, "stage": "promoter_transaction_document", "error": str(exc)[:180]})
        if i % 20 == 0:
            print({"promoter_transaction_documents_checked": i, "planned": len(selected)}, flush=True)
    unique = {(r["source_sha256"], r["context"]): r for r in rows}
    rows = list(unique.values())
    summary = {"official_insider_index_rows": len(data["data"]), "filtered_market_transaction_documents": len(selected),
               "parsed_documents": len(documents), "verified_promoter_market_transaction_rows": len(rows),
               "verified_market_purchase_rows": sum(r["direction"] == "BUY" for r in rows),
               "verified_market_sale_rows": sum(r["direction"] == "SELL" for r in rows),
               "distinct_issuers_with_verified_transactions": len({r["symbol"] for r in rows}),
               "filings_with_net_positive_group_market_shares": sum(d["document_net_promoter_market_quantity_shares"] > 0 for d in documents),
               "filings_with_zero_net_group_market_shares": sum(d["document_net_promoter_market_quantity_shares"] == 0 for d in documents),
               "document_errors": len(errors), "ESOP_pledge_gift_preferential_not_market_buys": True,
               "missing_records_not_assumed_zero_purchases": True, "full_insider_feed_completeness_verified": False,
               "historical_feature_or_frozen_rank_changed": False}
    return rows, documents, errors, summary
