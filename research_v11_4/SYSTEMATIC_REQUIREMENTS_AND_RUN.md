# Systematic V11.4 correction and run — 9 October 2026

Continue `ayushverma1988/v9-multibagger-calibration`; do not restart the model
or compare its selections with V10/V10.4. The current research work is based
on `v11-4-ownership-backfill-probe`, commit
`744f8655d1741fb2e01766f6d0821d5546023039`.

Relevant instructions were recovered from NSE valuation verification,
Multibagger Algorithm Status, Show reliability improvement, V10 Six-Month
Review, V9.5 Eight-Fold Watch, Refine Algorithm with Data, and Investment
Comparison Advice. Search retrieval is selective; this ledger does not claim
that every historical chat message was exported or read verbatim.

## Authority and changes over time

| Instruction | Treatment in this run |
|---|---|
| Goal: 2x within six months, with 6–12 month emphasis | Archived integrity-checked six-month target; no invented 12/24 month calibration |
| Focus on small/midcap stocks, prices ₹20–₹2,000 | Price range enforced; complete current market-cap classification remains missing |
| Complete NSE+BSE listed-stock search | The active collector covers NSE; BSE-only coverage is still missing and cannot be claimed complete |
| Discover transformation before obvious momentum | Specific primary NSE event fields and separate prior-runup sleeves; complete product→demand→orders/capacity→earnings→timing→not-priced-in chain still requires source verification |
| Capacity expansion, new production, future products, orders, approvals, promoter buying, corporate changes | Specific NSE categories reported; no inference that generic filings establish buying or material earnings impact |
| Required GDELT and Google News | Both queried, individually reported; one cannot substitute for the other |
| Primary verification | Discovery headlines never become high-confidence corporate evidence automatically |
| Three years of financial acquisition | Exact three-fiscal-year observations retained; they provide only two annual growth intervals |
| Exact original three screens | Retained unchanged in `config/v11_4_four_screener_families.json`; unavailable inputs remain UNKNOWN |
| Fourth RSI screen, previously >80, latest >70 | Wilder RSI(14) strictly >70; 70 fails, 70.1 passes |
| Possible overlap of independent screens | Inherited independent-family interpretation; report overlap, do not require all four to pass or invent a validated bonus |
| Positive cash flow, promoter ≥50%, receivables <10% of profit, above DMAs, reserves >borrowings, rising fixed assets | Additional checks restored as independent audits; no substitution of sales for the user's literal profit denominator |
| Pre-obvious <12%, second leg 12–100%, >100% separate | Maximum of 60/120/252-session returns implements inherited any-lookback rerating thresholds; fresh causal catalyst still required for a full qualifying story |
| Leave COVID-affected data | Initial shock-crossing outcomes excluded from model fitting; historical stress results remain visible. Exact shock window is inherited implementation, not a user-specified date range |
| Prior horizon design 70%/25%/5% | Preserved as design metadata; not applied until independent 12/24-month models exist |
| ≥12 valid folds; mean Top-10 Jaccard ≥0.80 and worst-fold p05 ≥0.60 | Preserved; successful script execution cannot override a failed gate |
| Avoid tuning repeatedly on examined 2025 outcomes | New source-correction recipe fixed before execution; results are retrospective research, not unseen validation |

## Concrete defects corrected

1. A clean checkout imported `v11_4_longterm_fundamentals.py` from a different
   branch. The exact parser is now included, copied from original revision
   `59986070076affb4cef8a57f630d22755d6ce7fb`; new runs do not fetch a mutable
   branch to make imports work.
2. Base training outcomes were gated at the later test time. They must also
   mature strictly before the original calibration selection time. One shared
   partition function now enforces this in historical and current fitting.
3. Forward scoring stamped a hard-coded old model ID. It now records the
   actual package's model version, preserving model identity after a fix.
4. The required news-source acquisition and additional user checks were not
   present in the standalone runner. They now run with explicit missing or
   blocked status and no silent substitution.
5. The newer financial source-only sidecar also includes closing prices. The
   join now verifies both originals agree and restores one canonical close.
6. The live tradability audit now receives exactly the four screening
   verdicts; additional checks remain attached to the complete stock audit.
7. The additional chart check now gets a sourced above-200-DMA flag, distinct
   from the recovery screen's below-200-DMA test. Moving averages require their
   full 50/200 prior sessions; equality is neither above nor below, and missing
   observations remain UNKNOWN.

## Two model components to execute

The existing 22-feature standalone model is rerun with its original fixed
regularization, monotone calibration, twelve perturbation seeds and acceptance
gates. COVID-exposed training outcomes are excluded. Current scores use a new
model identity and newly verified closing-date inputs, preserving the Oct8
original observations.

The research variant uses nine available 3FY financial features, eleven
market features plus turnover, six specific NSE categories, and RSI fraction
plus its strict >70 indicator: 29 inputs. Generic promoter filings and total
announcement counts are excluded as substantive predictors. Missing RSI is
excluded with coverage recorded, not replaced with neutral RSI or FALSE.
Train: December2022 and June2023; calibrate: June2024; retrospective evaluation:
June and December2025. Hyperparameters and these cohorts are fixed; no search
for settings that make previously examined outcomes look successful.

This is not full delivery of the original catalyst architecture: exact long
history, quarterly, valuation and market-cap inputs, verified causal chains,
historical news, and full official daily-RSI provenance remain requirements.
The three-fiscal-year data never masquerade as literal 3/5/7-year growth,
quarterly acceleration, or today's FY2026 financial statements.

## Run

The workflow `V11.4 Systematic Repair Full Tests Research and Current Demo`
installs `requirements_v11_4_systematic.txt`, runs all regression tests,
downloads the existing immutable original sources, and runs one entry point:

```bash
PYTHONPATH=src python src/v11_4_systematic_run.py \
  --features18 ORIGINAL_18FOLD_FEATURES.parquet \
  --labels ORIGINAL_MATURE_LABELS.parquet \
  --source8 ORIGINAL_3FY_RSI70_SOURCE_ONLY.parquet \
  --output systematic_run_output \
  --live-date YYYY-MM-DD \
  --owner-public-key OWNER_PUBLIC_KEY.pem
```

The actual current-market date is derived in Asia/Kolkata. It cannot silently
substitute a prior day's quote. Outputs include source coverage, screen
verdicts, historical precision and stability, required-news status, current
demo status, and outstanding requirements. Raw stocks and model packages are
saved only into a new owner-private run path; public outputs contain aggregate
diagnostics and an encrypted owner demo. Original observations are not
overwritten.

NSE's financial-results page says filings for March2025 and later move to
Integrated Filing–Financials. Keep both source families in acquisition; this
is why legacy-only fetching misses recent periods:
https://www.nseindia.com/companies-listing/corporate-filings-financial-results
https://www.nseindia.com/companies-listing/corporate-integrated-filing

Probability calibration uses separate data from model fitting:
https://scikit-learn.org/stable/modules/calibration.html
