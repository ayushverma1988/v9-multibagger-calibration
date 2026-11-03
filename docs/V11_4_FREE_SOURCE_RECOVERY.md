# Free source recovery: current evidence and a fixed discovery experiment

This companion repairs access and evidence coverage for the existing V11.4
research model. It does not certify a six-month doubler, replace the owner's
independent screening conditions, or retrain historical financial predictors
using today's financial pages. Four screening families remain independent;
the latest RSI rule is strictly greater than 70. Seven-year requirements stay
seven-year requirements. Missing observations remain unknown.

## Working public routes

| Requirement | Route | Evidence limits |
|---|---|---|
| BSE quotes and daily history | Yahoo public chart with the complete BSE mnemonic plus `.BO` | Verify exchange, currency, instrument, name and timestamp. Guaranteed intraday latency and historical corporate-action representation are not established. |
| Current annual, quarterly and cash flow tables | Finology public company pages; `?mode=C` requests consolidated | Read the actual checkbox state. Some pages are unavailable, old or incomplete. Reported ratios retain their own unverified evidence tier. |
| Cross-exchange identity | AMFI dated ISIN reference plus NSE current masters and dated interoperability master | A truncated `$` alias is an outage reference, not a Yahoo symbol or proof of ordinary NSE trading. |
| Company P/E | NSE `/content/equities/peDetail/PE_DDMMYY.csv` | Positive company P/E is useful; zero/loss markers, industry P/E, PEG and publication before the decision clock are not interchangeable. |
| Current market cap | Same-date NSE close × corroborated exchange share count; check provider cap on its displayed venue | Issued capital becomes a usable share count only after a per-security unit check. A BSE provider cap is checked against the separate BSE quote; the recovered NSE cap always uses the official NSE close. |
| Scanned catalysts | Local `pdftoppm` and Tesseract on original exchange PDFs | OCR permits review. Quantities, promoter purchases and complete earnings effects require their own verification. Large PDFs are explicitly partial reads. |

The public Google Finance page can assist a manual quote check. Google's
documented historical `GOOGLEFINANCE` data cannot be downloaded through the
Sheets API or Apps Script; it is not this collector's automated history feed.
StockAnalysis financial tables were tested as an additional route. Their
million-INR units and standardized operating-income/PAT definitions must be
mapped explicitly before replacing a crore-denominated provider observation.
Screener and direct BSE endpoints failed in the tested environment.

## Run from an existing source recovery checkpoint

Install `requirements_v11_4_systematic.txt`. The source-recovery folder must
contain its original official NSE market identity snapshot, ISIN universe,
company enrichment and source financial facts for local reconciliation. The
outbound financial request universe comes directly from the complete official
public NSE main and SME lists. Private model cohorts and watchlists do not
choose which company URLs to request.

```bash
PYTHONPATH=src python -m v11_4_free_source_recovery_run \
  --source-recovery /absolute/source_recovery \
  --output /absolute/current_financial_recovery --mode financial --limit 0

PYTHONPATH=src python -m v11_4_free_source_recovery_run \
  --source-recovery /absolute/source_recovery --asof 2026-10-09 \
  --output /absolute/BSE_recovery --mode bse --limit 0 --history-companies 64

PYTHONPATH=src python -m v11_4_free_financial_current_audit \
  --financial-folder /absolute/current_financial_recovery \
  --source-recovery /absolute/source_recovery --asof 2026-10-09 \
  --master-file /absolute/NSE_current_master.csv.gz \
  --master-receipt /absolute/master_receipt.json \
  --pe-file /absolute/official_company_PE_PRIVATE.parquet \
  --pe-receipt /absolute/official_company_PE_receipt.json \
  --bse-quotes /absolute/BSE_recovery/BSE_latest_quotes_source_audit_PRIVATE.parquet \
  --output /absolute/current_input_audit

PYTHONPATH=src python -m v11_4_catalyst_ocr_recovery \
  --source-recovery /absolute/source_recovery --output /absolute/OCR_recovery

PYTHONPATH=src python -m v11_4_BSE_secondary_financial_recovery \
  --bse-recovery /absolute/BSE_recovery --output /absolute/BSE_secondary_financials
```

The P/E receipt's original bytes belong in `raw_PRIVATE/<body_file>` beside
the receipt. The audit verifies their hash, trading date and normalized values.
The master's receipt verifies its original bytes and date as well. Full public
responses and first-observed receipts are cached. Normal HTTP errors remain
recorded failures; the collector never bypasses an access wall. Provider
requests are paced, and completed responses can be reused after interruption.

For PDF OCR, install the free system packages `poppler-utils` and
`tesseract-ocr`. OCR is bounded to twelve pages per failed document. Inspect
rendered pages before accepting numerical claims. Do not label a partial
annual-report scan as a fully read catalyst.

## Additional technique tested and rejected

O'Neil's published description of earnings strength, relative strength and
price-volume confirmation motivated one fixed research hypothesis. Three
available interactions were added to the current 21-input model: specific
catalyst × momentum, volume × momentum, and extended-runup × volatility.
These are our hypotheses, not a reproduction of his proprietary rankings.

The recipe was frozen before the experimental run, with the same training,
calibration, label definition and chronology. Today's recovered financial
pages were not inserted into historical observations. The current baseline
was reproduced exactly before comparing the candidate. No earlier model's
predictions or rankings were used as a comparator.

On the ten matching fully assessable dates, the current model found 14
doublers in 100 selections and the candidate found 13. The candidate also
reduced mean Top-10 Jaccard from 0.8592 to 0.8280 and worst-fold p05 from
0.4286 to 0.3857. Both fail the required worst-fold threshold of 0.60. The
candidate is rejected and its weights are not promoted. Retrospective hits
are not a calibrated prospective success rate; overlapping issuers and
outcome windows limit the uncertainty estimate.

## Remaining completion gates

1. Reproduce five/seven-year growth and return averages from correctly dated,
   same-basis observations, including old source filings and cash-flow pages.
2. Independently establish catalyst size, timing, execution and earnings impact;
   distinguish actual promoter market purchases from governance disclosures.
3. Confirm BSE instrument identity, price freshness, liquidity, listing changes
   and corporate actions before training/scoring the expanded universe.
4. Train a full source-qualified recipe with chronological evaluation and
   predeclared model selection. Recovering a current field alone does not add it
   to the already tested model.
5. Accumulate observations and matured outcomes in the immutable independent
   ledger. Existing frozen weights, source inputs and protocol are unchanged.

The source contract workflow executes software checks only and does not append
new blind predictions or overwrite the evaluation freeze.

The public StockAnalysis BSE directory supplies candidate numeric codes when
a mnemonic does not resolve on Finology. The secondary collector checks code
links and unique exact normalized issuer names, then reads income and cash-flow
tables for the same public history controls. It does not claim an official
code-to-ISIN binding or equate standardized net income with the original PAT
definition. Name collisions remain unresolved. Empty reference ticker cells
are excluded before duplicate-symbol checks.
