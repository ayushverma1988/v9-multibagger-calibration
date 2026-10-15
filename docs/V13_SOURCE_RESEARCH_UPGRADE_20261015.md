# V13 source and research upgrade — 15 October 2026

The strict challenger intentionally abstains unless both a complete reviewed
execution chain and a source-bound net 2x valuation scenario pass. A zero
selection count is not a forecast that no stock can double. Do not pad it.

This change preserves the existing champion and formal challenger weights,
registered forecasts, prior cohorts and unknown legacy information cutoffs.
Engineering checks do not establish financial prediction accuracy.

## Order and behavior

1. Financial ingestion: official NSE symbol-change paths require the exact
   original XBRL ISIN and current master ISIN. Ambiguous paths, missing original
   ISINs and issuer typos fail closed. Newly recovered facts become available
   at the later original publication or first observed identity-proof clock.
   A rename does not prove economic comparability or adjusted price returns.
   An already valid earlier ISIN-bound fact retains its availability vintage
   when a later identity review of the same original is added.
   Reviewed PDF cells can supply same-statement reported diluted EPS comparisons
   only after original hashes, company identity, consolidated basis, quarter
   headers, column geometry and the current XBRL EPS anchor reconcile. The first
   seed is one original statement review, not general PDF coverage. Changes to
   review instructions receive a new first-review clock. No EPS subtraction or
   recurring-EPS inference is allowed.
2. Research selector: the current separately versioned available-data policy
   uses unchanged champion probabilities and RSI strictly above 50. Exactly 50
   is excluded; RSI retains the simple rolling 14-session definition. Existing
   market/risk limits and observed seven-check or owner-earnings failures still
   exclude a row. Missing financial inputs are permitted and explicitly labelled.
   Missing execution or source-supported net-2x valuation never veto a prediction.
   The ranking score combines 75% eligible-pool joint12 percentile, 10% fraction
   of seven financial inputs observed, 5% complete financial/execution support,
   5% source-supported net-2x scenario, and 5% proximity within inclusive +/-10%
   of the trailing 200-observation adjusted-close SMA. Missing or outside-band
   DMA gets no bonus and remains eligible. These are declared preference weights,
   not fitted weights, calibrated probabilities or evidence of accuracy gains.
   DMA proximity alone proves neither no prior bull run nor future upside.
   Selection caps at 20 and may abstain. Ranked records and all-name decisions
   retain eligibility, financial tier, missing evidence, DMA status, score,
   actual registration clock, selector hash and original exact replay receipt.
   Earlier RSI >70 snapshots and all frozen formal forecasts remain unchanged.
3. Execution: project-level diagnostic sidecars separate UNKNOWN from FAILED,
   retain adverse updates, and distinguish EBITDA/EBIT from owner net profit.
   An order value is not recognised revenue. Existing strict execution checks
   remain unchanged. Source owner-profit and dilution bridges remain required.
4. Valuation: reviewed forward PAT now requires a quoted annual economic period,
   an explicit six- or twelve-month target horizon and a reviewed fully diluted
   share basis. Out-of-horizon or unbound scenarios remain unknown. Existing
   hypothetical PE/EPS hurdles never become source-supported 2x signals.
5. Validation: new exact input parquet snapshots, model hashes, code hashes,
   runtime versions and all coherent probability outputs are stored immutably.
   Inference must reproduce after parquet roundtrip within absolute 1e-12.
   These receipts establish reproducibility, not accuracy. Existing examined
   folds remain development evidence; prospective promotion gates still apply.
   The frozen guard explicitly protects unknown legacy information cutoffs.
6. Coverage: public history refreshes prioritise previously technically usable
   histories while retaining the existing 25% cold expansion allocation. No
   model selections or realised outcomes enter the source queue. A coverage
   funnel reports stale bars, technical exclusions, missing official identity,
   symbol, volume and OHLC conflicts without filling stale prices forward.
   Same-day unreconciled bars can be retried with a new immutable raw-response
   vintage; the first response of the day no longer prevents later provider
   finalisation from being observed. Already exact source-aligned bars skip
   another request, and timestamped attempts rotate remaining public work.
   Current cross-sectional ranks and breadth are recomputed after official
   reconciliation so rejected provider rows cannot influence valid names.
   This context remains explicitly a partial current universe, not all NSE.

## Validation and operational use

Run `PYTHONPATH=src python -m unittest discover -s tests` in the pinned runtime.
Original-source integration checks and a full collector run are separate from
unit tests. The worker archives all new sidecars in its private checkpoint.
Publish only code, public source references and synthetic tests to this repo.
Private predictor snapshots and predictions remain in the private archive.

Measured evidence coverage must retain its actual denominator. A narrower fresh
source-date universe is not loss of historical data. Growing forecast cohorts
can reduce entry coverage percentages before next-session entries are eligible.
No test result, extra feature, or larger source count establishes predictive
improvement. Compare calibration, ranking, risk and abstention using untouched
point-in-time outcomes before approving a model.
