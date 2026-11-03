# V11.4 — NSE legacy FY2022 annual XBRL recovery and independent cohort validation

Date: 2026-10-09. This is **source research only**, not a new stock-prediction model, comparison with a legacy version, or permission to retrain on incomplete financial facts.

## Source route result

1. BSE original historical Financial Results HTML/API was examined using bounded, read-only probes. The GitHub runner received HTTP 403 on official BSE API calls and generic HTTP-200 landing pages on company-result listing requests; after fixing source-availability classification, this route deliberately fails closed. This is **not** verified BSE historical data. [First transport audit](https://github.com/ayushverma1988/v9-multibagger-calibration/actions/runs/37874628182) and [fail-closed source audit](https://github.com/ayushverma1988/v9-multibagger-calibration/actions/runs/37874698577).
2. Investigated original NSE FY2022 historical XBRL rather than substituting current financial ratios. Original raw bytes matched previously recorded SHA-256. The legacy XML has **Yearly** metadata, exact **2021-04-01 to 2022-03-31** reporting-period facts, **FourD** tagged income/profit values, INR units, and standalone/consolidated mode. Its conventional XBRL context parsing had not exposed that attached annual fiscal metadata. [Structural audit](https://github.com/ayushverma1988/v9-multibagger-calibration/actions/runs/37874875731).
3. Implemented a dedicated strict FY2022 FourD reader. It requires an Annual source index, explicit Yearly indicator, exact start/end period, same reporting mode, original source SHA-256, original 2023 historical selection 15:30 IST cutoff and INR-denominated unambiguous revenue/PAT. It **rejects** plain Q4 data, unsupported currencies, duplicate concepts, period ambiguities and future publications. All 8 new legacy parser and 6 existing financial-context tests passed.

## Verified sample and independent holdout results

| Metric | Original 24 | Separate untouched 128-company sample | Combined |
| --- | ---: | ---: | ---: |
| Requested company pairs | 24 | 128 | 152 |
| Strict FY2022/FY2023 numeric company pairs verified | 24 | 127 | **151** |
| Annual revenue YoY computable | 24 | 127 | **151** |
| Annual PAT YoY computable | 21 | 107 | **128** |
| Rejected pair(s) | 0 | 1 | 1 |

Evidence: [Original recovery](https://github.com/ayushverma1988/v9-multibagger-calibration/actions/runs/37875065460), [Independent four-shard holdout](https://github.com/ayushverma1988/v9-multibagger-calibration/actions/runs/37875228463), [Successful reconciliation](https://github.com/ayushverma1988/v9-multibagger-calibration/actions/runs/37875654855).

The 2023-12-29 original stock universe has 1,276 rows. The immutable 2023 source-only research matrix keeps **all 1,276 rows** and has **151 independently verified two-fiscal-year numerical observations (11.83% coverage)**. Every other stock has UNKNOWN financial values, and the single rejected filing is excluded with an error report. These FY2022/FY2023 values are not 3y/5y/7y financial growth and are not enough for the model's >=70% / >=12-fold activation gate.

## Next research gate

1. Use the independently tested FourD approach for an unbiased **full-original-2023-fold** numerical backfill, strictly bounded by as-of filing dates and no restatements.
2. Extend fiscal years backward into FY2021/FY2020 as original independently verified published sources, building consecutive 3/5/7-year annual series only if period contexts and reporting modes reconcile, not interpolating.
3. Separately backfill original quarterly reporting, balance sheet and cash-flow financial concepts needed for all four screening families, and independent PIT industry P/E.
4. Rerun full 18-fold coverage and source maturity gate before the first chronological experiment with these fundamental features. No benchmark against V10.4, and no retroactive edits to current frozen standalone predictions.

**Current production decision**: historical annual **source parser success**, original FY2023 model financial activation **NOT APPROVED**; no predictive training or ranking altered.
