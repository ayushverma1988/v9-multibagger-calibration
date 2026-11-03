# API-first financial data recovery — 21 October 2026

Independent-source validation is disabled: `required_additional_sources = 0`.
A financial value can enter the master from one API or one financial website.
Identity, units, dated periods and reporting basis remain recorded so unrelated
amounts cannot silently be substituted for a missing field.

The collection order is:

1. Keep an existing original value. Fill its gaps from available Yahoo financial
   statement API fields and the user's suggested Indian Stock Market API.
2. Save the API-stage master and source receipts before continuing.
3. Search for remaining inputs and collect dated statement pages from Screener,
   Finology and Stock Analysis. A second website is not required to accept a
   value. Search snippets alone are not financial observations.
4. Save the master, per-cell source ledger, remaining gaps and query inventory.

`resolved_` columns contain the original or selected replacement. `usable_`
columns also meet the freshness rules. Optional `validated_` columns and old
source-agreement counters are diagnostics and do not control admission.

Expanded API fields include parent profit, diluted EPS from continuing
operations, equity capital and current/non-current receivables. When total and
parent profit are both present in the same API statement, their difference is
recorded as calculated non-controlling profit, with the calculation identified.
It is never labeled as an original reported amount.

If the API does not explicitly state its reporting basis, the documented
upstream consolidated-statement convention is retained and marked as not
independently confirmed. Exact optional original overlap may identify a
standalone presentation; absence of that overlap no longer rejects API data.

Google's automated endpoint returned HTTP 429 in the local collection test.
The worker stops Google requests on that response, keeps the search queue and
continues direct collection from public statement sites discovered through web
search. It does not bypass rate limits or a CAPTCHA. Other unavailable or
unreported entries remain explicitly missing rather than being filled with
invented numbers or arbitrary zeros.

A full-universe GitHub workflow collects up to 3,200 issuers, saves API data
first, and then performs website fallback. The weekday 22:00 IST worker uses
the same policy and resumes the remaining source queues.

The initial API-first change passed 674 regression tests. Targeted source tests
also cover single-source admission, API priority, original-value preservation,
units, annual/quarterly separation, derived values and optional basis evidence.
Large JSON ledgers are written incrementally and replaced atomically to avoid
losing the previous saved ledger if an export is interrupted.

This is data collection. It does not alter frozen forecasts or assert that all
publicly unreported financial fields can be recovered from an API or Google.

Sources inspected for the collection interfaces and definitions:

- https://github.com/0xramm/Indian-Stock-Market-API
- https://github.com/ranaroussi/yfinance/blob/main/yfinance/const.py
- https://morningstardirect.morningstar.com/clientcomm/DataDefinitions-EquityandExecutive.pdf
- https://stockanalysis.com/quote/nse/ALICON/financials/balance-sheet/
- https://www.screener.in/company/SINGERIND/
