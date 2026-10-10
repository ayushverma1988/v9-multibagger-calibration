# V12.1: four-year, calendar 6/12-month research freeze

The owner explicitly authorized replacing the previous hard requirements on
10 October 2026. This is a new version; it does not overwrite the earlier
independent model, its protocol, its stock order, or its outcomes.

V12 is an executable, frozen research model. Its estimated probabilities are
not a promise of a 2x return, and a successful software run is not validation
of predictive accuracy. The frozen evaluation summary is authoritative for
performance and release eligibility. A failed gate stays failed.

## Changes

| Previous requirement or limitation | V12 behavior |
|---|---|
| Older historical training and long financial windows | Raw price data is limited to 9 October 2022–9 October 2026. No pre-boundary indicator warmup. Five/seven-year financial filters are retired. |
| Only a six-month target | Separate calendar six-month and twelve-month doubling-opportunity probabilities, plus twelve-month severe-loss risk. |
| Half-year historical snapshots | Monthly snapshots from newly mined daily observations. |
| RSI strictly above 70 | RSI is a continuous predictor. V12.1 accepts 45–90 with a positive 50-session trend. A positive 200-session trend identifies confirmed growth; stocks below that trend can enter an emerging-turnaround sleeve. This version uses 14-session Cutler RSI. |
| Many rigid growth/valuation conjunctions | Recent quarterly sales growth ≥10%, PAT growth ≥15%, operating margin ≥8%, debt/equity ≤1.5 and CFO/PAT ≥0.6, with source freshness and identity checks. |
| Low reported P/E could be driven by extraordinary income | Reconciled four-quarter PBT excluding reported exceptional items must be positive; unresolved earnings-quality evidence remains UNKNOWN. |
| Nominal price and size might be conflated | Current ₹20–₹2,000 price band is separate from independently corroborated ₹100–₹50,000 crore market-cap band. Actual AMFI category is displayed when available. |
| Forced Top 10 | Source-checked research views are separated from threshold-pass views; the latter may be empty. |
| Prior run-up could dominate selections | Current screen rejects a prior 63-session gain above 75%, requires liquid trading and positive moving-average trends. |
| A book-inspired signal was presumed helpful | One predeclared regularized logistic/shallow boosting ensemble is measured chronologically. No post-result parameter search or winner substitution. |
| Probability numbers might be raised to look better | Three previous matured months calibrate each head. No artificial boost or projection. Incoherent nested estimates do not pass the conviction criteria. |
| Incomplete source fields could become passes | Required missing/stale inputs remain UNKNOWN. Original-filing disagreements block a source-check pass. |

## What the target means

Signal features use a completed session. Entry is the next observed session's
open, rather than buying retrospectively at the signal close. Success means a
future **closing** total-return proxy reaches at least 2x after 0.5% modeled
entry and exit costs before the calendar deadline. Intraday high alone is not
success. Full horizon observation is required even for an early winner.

Provider adjusted close includes dividend/split adjustments. This is an
explicit total-return proxy, distinct from V11's price-only target. Closing
endpoint returns and severe drawdown are also stored; a temporary doubling
opportunity does not establish a 100% buy-and-hold return at the endpoint.
Corporate actions, circuit-limit entry fills and survivor bias are not
independently resolved, so production accuracy is unverified.

## Data and statistical controls

The outbound mining plan is the complete official public NSE main/SME list.
Private predictions and watchlists do not select provider requests. Provider
identity, venue, currency, daily OHLCV, adjusted-price availability, dates and
cached bytes are checked. Invalid observations and inadequate histories are
excluded; unavailable labels remain missing, never negative.

Market inputs include recent returns, relative strength, price/volume
confirmation, volatility, drawdown, liquidity and the contemporaneous market
regime. This is our implementation inspired by O'Neil's published discussion,
not a reproduction of his proprietary ratings. It uses no older model's
weights, probabilities, rankings or selections.

Training labels must be fully known before the **first** calibration
decision. Calibration outcomes must be fully known before each test decision.
Tail bounds, standardization and both ensemble components fit only on that
training split. Training gives each month comparable weight, with a fixed
18-month half-life. Final stability uses twelve seeded omissions of whole
issuer clusters, retaining 95% of training issuers each time.

Current financial pages are useful for current selection checks; they cannot
become historically available supervised predictors. The reported market
probability is not a separately validated conditional probability after the
financial gate. Metadata about an order, expansion or promoter filing also
does not prove its earnings effect or a promoter market purchase. Complete
causal catalyst chains remain an evidence task, not fabricated input values.

The first V12 candidate's independently calibrated probabilities were
inconsistent for 628 of 639 current rows. Its checkpoint is retained as a
failed candidate. V12.1 trains an early doubling head and a conditional late
doubling head, then computes `P12 = P6 + (1-P6)*P_late_given_no_early`.
The future early outcome only filters the late head's training/calibration
targets; it is never an input to a forecast. Twelve-month severe-loss risk
counts losses before the 2x closing target, rather than losses after a
successful target exit. The source and accuracy thresholds are unchanged.

The existing BSE quotes/history recovery remains available. V12 training and
its new source-checked forecast are NSE based. Independent validation of the
complete BSE universe, corporate-action mappings and standardized financial
definitions is not manufactured by applying NSE weights to a different venue.

## Run and replay

Install `requirements_v12_frozen.txt` to replay these exact frozen weights;
the full legacy regression suite uses `requirements_v11_4_systematic.txt`.
Use `v12_public_history.py` with
the complete public NSE request universe and an explicit as-of date. Register
the recipe with `v12_four_year_model.write_recipe` before creating the target
matrix. Run `v12_four_year_model.py` with the resulting history directory,
financial audit, AMFI reference and same-day official NSE market identity.

Outputs include frozen weights, recipe, source predictors, monthly features
and outcomes, chronological evaluations, a current source-check audit and
shortlists. `freeze_manifest.json` records per-file SHA-256 hashes.

For the structural correction, register
`v12_coherent_hazard_model.recipe` before running
`v12_coherent_hazard_model.py` on the unchanged monthly source panel.
V12.1 preserves the first candidate and does not overwrite it.

Use `v12_frozen_predict.py` for subsequent inference. It verifies the weight
and recipe hashes, rejects future outcomes and prior scores in predictor
input, prevents retroactive forecasts, and refuses an unnoticed overdue
model review after 190 days. It does not fit weights. Forecast observations
cannot be silently overwritten. Source inputs must be refreshed and match
the claimed date; the frozen model does not make an old quote live.

## Release discipline

The predeclared gates require at least six fully matured test dates for each
horizon, ≥30% selected retrospective hits, positive Brier skill against an
earlier calibration prior, and current Top-10 mean/p05 Jaccard ≥0.80/0.60.
Unseen outcomes, survivor/corporate-action checks and financial-gate
calibration remain separate unresolved gates. The freeze is for reproducible
research; it does not automatically approve trades or certify high accuracy.

Observed V12.1 results: six-month Top-10 doubling opportunities were 10/190
on 19 matured dates (5.26%); twelve-month opportunities were 3/40 on four
matured dates (7.5%). The selected-hit requirements failed. Six-month Brier
skill was negative; twelve-month skill was about +1.91% against the previous
calibration prior. Current probability ordering has zero violations.
Current source checks passed for 12 stocks, and no stock met the conviction
thresholds. Current Top-10 issuer-omission Jaccard was 1.0 in this small
12-stock pool; this does not prove historical worst-fold stability or high
prediction accuracy. The original independent weights/protocol are unchanged.

Primary method sources:

- https://www.williamoneil.com/about-us/legal/oneil-proprietary-rating-and-rankings
- https://scikit-learn.org/stable/modules/calibration.html
