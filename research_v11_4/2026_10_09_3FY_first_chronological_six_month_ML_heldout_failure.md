# V11.4 independent three-fiscal-year six-month-double test — 9 October 2026

## Grounded result: historic 3-FY data now recoverable

- 2023-12-29 original NSE fold, FY2021–FY2023: 960 verified / 1,276 original equities (75.24%). Private source run 37906034342.
- 2024-12-31 original NSE fold, FY2022–FY2024: 1,068 verified / 1,345 original equities (79.41%). Private source run 37920892862.
- 2025-12-31 original NSE fold, FY2023–FY2025: 1,013 verified / 1,314 original equities (77.09%). Private source run 37907820467.
- Original NSE XML FY2025 integrated annual YTD FourD context verified. Always distinguish three fiscal-year observations from three year-on-year growth intervals: these three points support only a **2-year CAGR**.

## Independent chronological prototype test, now measured

Run: https://github.com/ayushverma1988/v9-multibagger-calibration/actions/runs/37920972818

| Experimental stage | Date | Labeled eligible stock-date observations | Six-month 2x positive |
|---|---|---:|---:|
| Fit reduced 3-FY fundamentals + original liquidity features | 2023-12-29 | 960 | 61 |
| Independently calibrate on matured data only | 2024-12-31 | 1,060 | **5** |
| Evaluate on held-out, forward-matured labels | 2025-12-31 | 1,010 | 33 |

- Heldout 2025 **Top10 true 2x winners: 0 of 10 (precision 0%)**. This is a negative finding, not production approval.
- Holdout event base rate: 33 / 1010 = 3.2673%.
- Average precision 4.1263%; ROC-AUC 0.5487; Brier 0.03195. These are weak distinctions above prevalence/chance for a six-month doubling task.
- Calibration used prior-shrunk monotone intercept because there were only 5 positive labels in 2024; fitting an independent Platt slope would be unreliable.
- Test results were seen **once** for this candidate. Do not reuse or repeatedly tune on this December 2025 test until it appears successful. Prospective future cohorts or pre-registered independent out-of-time folds are needed.
- This is a **financial-only short-history model**, NOT the original market+NSE-catalyst 22-feature V11.4 model, and it does not yet incorporate user RSI>80 or four complete screens in training.
- No comparison to V10.4 was made or authorized.
- Historical score + target file is private: `ayushverma1988/v10-multibagger-archive`, `v11_4/threeFY_chronological_single_holdout_2025/original_2023_train_2024_cal_2025_test`.
- The Oct 8, 2026 genuinely prospective frozen V11.4 Top 10 is unmodified; do not backdate these historic experimental predictions to 2025 as if recorded on that date.

## Decisions for next stage

1. **Do not deploy this finance-only 3-FY variant.** Its heldout precision 0/10 fails the user's six-month multibagger goal.
2. Keep the original three-financial-year verified data as a **financial quality / trend layer**. Rebuild on the 22 pre-cutoff stock price / volume / catalyst signals; include Wilder RSI(14)>80 as a separately tracked screen. Pre-register changes.
3. Avoid folding original 5/7-year Screener constraints into bogus 2-year replacements. Missing historical fundamental data remains UNKNOWN.
4. Design multiple chronological time folds from the rolling FY3 source, with decisions at proper market cutoffs and only matured labels. Do not call 3 fiscal years 12 independent annual folds.
5. Audit thin-liquidity/circuit/volatility, and independently validate calibration before any confidence of live 6m doubling probabilities.
6. Save per-company financial and historical test outputs privately; keep only aggregate source and performance diagnostics in public GitHub logs.

This test is evidence that NSE data collection is working and that the reduced financial-only selector is **not yet accurate enough**. Do not confuse data completeness with predictive quality.
