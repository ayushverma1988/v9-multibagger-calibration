# Data correction and research output — 20 October 2026

Price source session: **2026-10-19**. Review completed: 2026-10-20T07:41:13.353451+00:00. **633/3,200 public NSE shares** have source-aligned inputs in this replay. This is partial coverage and must not replace the earlier broad watchlist.

## What was wrong

The earlier 20 October output covered only 226 of 3,200 shares. A newer completed collector expanded the current cohort. In an unranked sample of 16 stored but unscored histories, all three public price-request variants still returned blank 19 October closing-price fields. These names were unavailable for current inference, not rejected investment theses. No missing closing price or adjusted price was filled in this correction.

The financial code could also switch to standalone results when the recent consolidated quarter was absent. Mastek exposed this problem: its standalone result suggested falling sales/profit, whereas the issuer’s June consolidated release reported growth. The code now preserves consolidated continuity, marks missing group results explicitly and reserves collection capacity for their originals.

A first broad recovery attempted 1,600 filings and recovered 1,888 net numeric observations. A targeted pass recovered another 129 observations in its own snapshot. These counts overlap later collections and must not be added to claim unique new facts. The final replay unions the broad recovery, targeted recovery and the later interrupted worker’s saved collection, matching exact source hash, reporting basis, period and availability clock. No conflicting numeric extraction of the same original was found in that union.

The final pass rechecked **19 original filings**, extracted **427 observations including rechecks**, and added **0 net numeric observations** after the union. **1 identity/source error** remains visible. Frozen-model replay passed on 633 unchanged market-input rows. Original forecasts, model heads and registries were preserved.

## Financial coverage — audit of all 3,200 public NSE issuers

| Measure | Count |
| --- | ---: |
| Public NSE shares | 3,200 |
| Shares with financial observations | 2,874 |
| Original period/basis/version rows | 47,892 |
| Nonmissing numeric fact observations | 492,306 |
| Unique original document hashes | 24,770 |
| Fresh preferred-basis quarter | 2,325 |
| All seven screening ratios/checks available | 1,279 |
| Underlying required facts available, including undefined-ratio cases | 1,618 |
| Pass the seven available checks | 542 |
| Missing recent consolidated quarter | 280 |

Coverage uses information available by **2026-10-19T16:30:00+00:00**. Later-published observations can be stored but are excluded from this dated replay. Counts describe source observations, not money invested, independent confirmations or unique economic facts. Complete inputs can fail financial checks. Zero interest or a nonpositive comparison denominator can leave a ratio undefined even when original amounts are available. **Financial coverage remains incomplete.** BSE-only securities are outside this public NSE denominator.

| Available statement history | Issuers | Original rows | Earliest period | Latest period |
| --- | ---: | ---: | --- | --- |
| annual | 2,796 | 6,353 | 2025-03-31 | 2026-06-30 |
| instant | 2,806 | 12,538 | 2022-12-31 | 2026-09-30 |
| quarter | 2,400 | 22,823 | 2023-03-31 | 2026-09-30 |
| ytd | 2,551 | 6,178 | 2024-09-30 | 2026-09-30 |

These endpoints describe the collected dataset; each issuer can have gaps and duplicate original versions. Annual financial history currently starts in March 2025, so this is not a complete five- or ten-year fundamental history. Longer original annual-report coverage remains a data gap.

## Cross-validation

Compared 325 dated provider observations across 17 issuers. **20 annual revenue/operating-cash-flow field-periods agree with both Screener and reviewed StockAnalysis/S&P Global displays within their published rounding precision.** This limited check is not a claim that all fields or all companies are verified.

Only the same issuer, consolidated/standalone basis, fiscal period, metric and INR units are compared. Repeated sites using one provider count once. Net profit and EPS definitions remain separate where total, parent-attributable, basic, diluted or restated bases have not been reconciled. Agreement is corroboration, not a truth certificate; all publishers may depend on the same underlying issuer filing.

The runner fetched Screener successfully. StockAnalysis direct requests returned 403, so the third-provider observations are explicitly dated manual reviews of accessible public tables. They are not a live third-provider feed. Further conflicts include historical LTTS revenue comparisons and selected other amounts; originals remain canonical pending review.

| Comparison status | Observations |
| --- | ---: |
| NUMERIC_AGREEMENT_WITHIN_DISPLAY_ROUNDING | 148 |
| PROFIT_DEFINITION_REVIEW_REQUIRED | 135 |
| PRIMARY_FACT_MISSING_OR_AMBIGUOUS | 24 |
| NUMERIC_CONFLICT_REVIEW_REQUIRED | 18 |

Reviewed annual amounts below are normalised to **₹ crore**. Each row matches the same fiscal year-end and reporting basis. Display rounding is allowed only within the source’s recorded precision. Full original values, source URLs and retrieval times are retained in the downloadable correction data.

| Share | Year-end | Basis | Field | Original | Screener | StockAnalysis / S&P | Result |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| MASTEK | 2026-03-31 | consolidated | revenue | 3,698.75 | 3,699.00 | 3,698.80 | Agree within rounding |
| MASTEK | 2026-03-31 | consolidated | cfo | 541.69 | 542.00 | 541.70 | Agree within rounding |
| SHREEJISPG | 2026-03-31 | consolidated | revenue | 709.38 | 709.00 | 710.50 | Review required |
| SHREEJISPG | 2026-03-31 | consolidated | cfo | 190.99 | 191.00 | 191.00 | Agree within rounding |
| TANLA | 2026-03-31 | consolidated | revenue | 4,417.71 | 4,418.00 | 4,417.70 | Agree within rounding |
| TANLA | 2026-03-31 | consolidated | cfo | 574.19 | 574.00 | 574.20 | Agree within rounding |
| CUPID | 2026-03-31 | consolidated | revenue | 357.71 | 358.00 | 357.70 | Agree within rounding |
| CUPID | 2026-03-31 | consolidated | cfo | 46.12 | 46.00 | 46.12 | Agree within rounding |
| HEXT | 2025-12-31 | consolidated | revenue | 13,430.40 | 13,430.00 | 13,430.40 | Agree within rounding |
| HEXT | 2025-12-31 | consolidated | cfo | 1,739.10 | 1,739.00 | 1,739.10 | Agree within rounding |
| THELEELA | 2026-03-31 | consolidated | revenue | 1,527.29 | 1,527.00 | 1,527.30 | Agree within rounding |
| THELEELA | 2026-03-31 | consolidated | cfo | 776.71 | 777.00 | 776.70 | Agree within rounding |
| LTTS | 2026-03-31 | consolidated | revenue | 10,995.90 | 10,996.00 | 10,995.90 | Agree within rounding |
| LTTS | 2026-03-31 | consolidated | cfo | 1,455.10 | 1,455.00 | 1,455.10 | Agree within rounding |
| PERSISTENT | 2026-03-31 | consolidated | revenue | 14,748.45 | 14,748.00 | 14,748.40 | Agree within rounding |
| PERSISTENT | 2026-03-31 | consolidated | cfo | 1,767.13 | 1,767.00 | 1,767.10 | Agree within rounding |
| ZENSARTECH | 2026-03-31 | consolidated | revenue | 5,687.40 | 5,687.00 | 5,687.40 | Agree within rounding |
| ZENSARTECH | 2026-03-31 | consolidated | cfo | 770.80 | 771.00 | 770.80 | Agree within rounding |
| SUNTV | 2026-03-31 | consolidated | revenue | 4,334.82 | 4,335.00 | 4,335.40 | Review required |
| SUNTV | 2026-03-31 | consolidated | cfo | 1,801.64 | 1,802.00 | 1,801.60 | Agree within rounding |
| MPHASIS | 2026-03-31 | consolidated | revenue | 15,879.65 | 15,880.00 | 15,879.60 | Agree within rounding |
| MPHASIS | 2026-03-31 | consolidated | cfo | 1,253.28 | 1,253.00 | 1,253.30 | Agree within rounding |

## Current separate 2× research list

Same frozen heads and preferred 17 October selector: verified small/mid-cap classification, known sector, RSI strictly above 70, above 50DMA and financial/owner checks. Missing financial evidence is shown explicitly. The estimates remain unvalidated research estimates; a shortlist entry does not mean doubling is likely.

| Rank | Share | Close ₹ | Model 2× by 6m | Model 2× by 12m | Model 30% loss event | Financial status |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | GABRIEL | 1,486.10 | 6.12% | 10.55% | 83.51% | All seven available checks pass |
| 2 | CUPID | 336.30 | 9.24% | 13.85% | 82.15% | All seven available checks pass |
| 3 | SOUTHBANK | 54.26 | 5.82% | 10.41% | 60.34% | Financial-sector rules unavailable; review required |
| 4 | TANLA | 552.35 | 4.49% | 8.25% | 85.89% | All seven available checks pass |
| 5 | KARURVYSYA | 370.05 | 4.70% | 9.34% | 66.44% | Financial-sector rules unavailable; review required |
| 6 | LEMONTREE | 111.67 | 3.72% | 7.42% | 40.74% | All seven available checks pass |
| 7 | JUSTDIAL | 731.75 | 4.11% | 8.02% | 80.32% | All seven available checks pass |
| 8 | ZENSARTECH | 513.15 | 3.62% | 7.39% | 81.13% | All seven available checks pass |
| 9 | TATAELXSI | 3,790.00 | 3.55% | 7.22% | 80.47% | All seven available checks pass |

The loss head targets a 30% loss event before a 2× hit over the 12-month window. These estimates are also unvalidated and remain high for these names; a relative research rank must not be read as a favourable risk/reward conclusion.

PVRINOX was present in an intermediate partial snapshot. It is excluded after combining the available originals because the interest-coverage check fails. South Indian Bank and Karur Vysya Bank remain explicitly unverified under sector-specific financial rules; their presence is not a seven-check financial pass.

Mastek now passes all seven available financial checks after its consolidated June originals and comparison periods were recovered. It still lacks usable current market inputs in this replay. Cupid, Tanla and Zensar remain on the current list. Persistent has RSI 62.78, below this lane’s strict 70 threshold; Sun TV has a parent/minority/total-profit reconciliation conflict. Hexaware also has an observed year-on-year consolidated PAT decline. These are distinct from missing quotes.

## Earlier 11 recommendations — retained separately

The original estimates below use 15 October prices and remain historical. Current absence is not silently interpreted as lack of potential. A current momentum-screen failure also does not settle a longer-term investment thesis.

| Share | 15 Oct model 2× by 12m | Current status | Financial / owner-profit status |
| --- | ---: | --- | --- |
| MASTEK | 10.31% | AWAITING_CURRENT_DATA_RETAINED_HISTORICAL_WATCH — STORED_BUT_NOT_SCORED | PASS_AVAILABLE_SEVEN_CHECKS |
| SHREEJISPG | 15.91% | AWAITING_CURRENT_DATA_RETAINED_HISTORICAL_WATCH — STORED_BUT_NOT_SCORED | PASS_AVAILABLE_SEVEN_CHECKS; UNKNOWN_OWNER_PLUS_MINORITY_DOES_NOT_RECONCILE_TOTAL_PAT |
| TANLA | 9.15% | TOP_20 | PASS_AVAILABLE_SEVEN_CHECKS |
| CUPID | 14.59% | TOP_20 | PASS_AVAILABLE_SEVEN_CHECKS |
| HEXT | 9.11% | AWAITING_CURRENT_DATA_RETAINED_HISTORICAL_WATCH — STORED_BUT_NOT_SCORED | KNOWN_FINANCIAL_FAILURE |
| THELEELA | 10.79% | AWAITING_CURRENT_DATA_RETAINED_HISTORICAL_WATCH — STORED_BUT_NOT_SCORED | PASS_AVAILABLE_SEVEN_CHECKS |
| LTTS | 8.23% | AWAITING_CURRENT_DATA_RETAINED_HISTORICAL_WATCH — STORED_BUT_NOT_SCORED | PASS_AVAILABLE_SEVEN_CHECKS |
| PERSISTENT | 7.86% | FAILED_CURRENT_SCREEN — RSI_NOT_STRICTLY_ABOVE_70 | UNKNOWN_FINANCIAL_INPUTS |
| ZENSARTECH | 7.14% | TOP_20 | PASS_AVAILABLE_SEVEN_CHECKS |
| SUNTV | 8.41% | FAILED_CURRENT_SCREEN — OBSERVED_OWNER_EARNINGS_FAILURE_OR_CONFLICT | PASS_AVAILABLE_SEVEN_CHECKS; UNKNOWN_OWNER_PLUS_MINORITY_DOES_NOT_RECONCILE_TOTAL_PAT |
| MPHASIS | 6.97% | AWAITING_CURRENT_DATA_RETAINED_HISTORICAL_WATCH — STORED_BUT_NOT_SCORED | PASS_AVAILABLE_SEVEN_CHECKS |

## Your nine chart examples — retained research watch

The nine user-supplied chart names are tracked separately from the eleven earlier model recommendations. All nine are present in the current public NSE master, but none is in this replay’s 633 current source-aligned input rows. Their absence cannot be treated as a current technical rejection or a confirmation of upside. Their available financial evidence also differs:

| Share | Current model inputs | Financial evidence |
| --- | --- | --- |
| EMUDHRA | NO_CURRENT_SOURCE_ALIGNED_MODEL_INPUTS | PASS_AVAILABLE_SEVEN_CHECKS |
| DSSL | NO_CURRENT_SOURCE_ALIGNED_MODEL_INPUTS | KNOWN_FINANCIAL_FAILURE — quarter_sales_yoy |
| CORONA | NO_CURRENT_SOURCE_ALIGNED_MODEL_INPUTS | UNKNOWN_FINANCIAL_INPUTS — missing quarter_sales_yoy, quarter_pat_yoy |
| PARAGON | NO_CURRENT_SOURCE_ALIGNED_MODEL_INPUTS | UNKNOWN_FINANCIAL_INPUTS — missing four_profitable_quarters, quarter_sales_yoy, quarter_pat_yoy, quarter_pat_margin, annual_CFO_PAT, dated_debt_equity, annual_EBIT_proxy_interest_coverage |
| AGOL | NO_CURRENT_SOURCE_ALIGNED_MODEL_INPUTS | UNKNOWN_FINANCIAL_INPUTS — missing four_profitable_quarters, quarter_sales_yoy, quarter_pat_yoy, quarter_pat_margin, annual_CFO_PAT, dated_debt_equity, annual_EBIT_proxy_interest_coverage |
| RAJRILTD | NO_CURRENT_SOURCE_ALIGNED_MODEL_INPUTS | KNOWN_FINANCIAL_FAILURE — quarter_sales_yoy |
| RSYSTEMS | NO_CURRENT_SOURCE_ALIGNED_MODEL_INPUTS | KNOWN_FINANCIAL_FAILURE — quarter_pat_yoy |
| CHEMCRUX | NO_CURRENT_SOURCE_ALIGNED_MODEL_INPUTS | UNKNOWN_FINANCIAL_INPUTS — missing four_profitable_quarters, quarter_sales_yoy, quarter_pat_yoy, quarter_pat_margin, annual_CFO_PAT, dated_debt_equity, annual_EBIT_proxy_interest_coverage |
| HUHTAMAKI | NO_CURRENT_SOURCE_ALIGNED_MODEL_INPUTS | PASS_AVAILABLE_SEVEN_CHECKS |

These names remain a research/data-completion watch. They do not supply positive training labels or override observed revenue/profit weakness. Merging older exchange history requires verified security identity and corporate-action treatment.

## Separate early-potential chart/volume lane

57 current setups are recorded in this cohort. These are untrained 50–100% upside hypotheses, separate from the 2× model. A ranking score is not a return probability. A filing cue identifies a dated announcement to review; it does not establish completed capex or realised earnings. The first 20 are:

| Rank | Share | Close ₹ | Pattern | Financial inputs |
| ---: | --- | ---: | --- | --- |
| 1 | JAYAGROGN | 217.99 | SURGE_PULLBACK | PASS_AVAILABLE_SEVEN_CHECKS |
| 2 | NH | 1,740.90 | DEEP_RESET_BASE | PASS_AVAILABLE_SEVEN_CHECKS |
| 3 | PREMIERENE | 899.00 | DEEP_RESET_BASE | PASS_AVAILABLE_SEVEN_CHECKS |
| 4 | GRINDWELL | 1,826.80 | DEEP_RESET_BASE | UNKNOWN_FINANCIAL_INPUTS |
| 5 | VOLTAMP | 10,410.00 | DEEP_RESET_BASE | PASS_AVAILABLE_SEVEN_CHECKS |
| 6 | HSCL | 605.85 | DEEP_RESET_BASE | PASS_AVAILABLE_SEVEN_CHECKS |
| 7 | BLS | 221.03 | DEEP_RESET_BASE | PASS_AVAILABLE_SEVEN_CHECKS |
| 8 | ENDURANCE | 2,368.80 | DEEP_RESET_BASE | PASS_AVAILABLE_SEVEN_CHECKS |
| 9 | BEPL | 117.46 | DEEP_RESET_BASE | PASS_AVAILABLE_SEVEN_CHECKS |
| 10 | REFEX | 249.70 | DEEP_RESET_BASE | PASS_AVAILABLE_SEVEN_CHECKS |
| 11 | ITCHOTELS | 158.50 | DEEP_RESET_BASE | PASS_AVAILABLE_SEVEN_CHECKS |
| 12 | STAR | 1,029.30 | SURGE_PULLBACK | PASS_AVAILABLE_SEVEN_CHECKS |
| 13 | KIRIINDUS | 487.65 | SURGE_PULLBACK | PASS_AVAILABLE_SEVEN_CHECKS |
| 14 | LEMONTREE | 111.67 | DEEP_RESET_BASE | PASS_AVAILABLE_SEVEN_CHECKS |
| 15 | JKPAPER | 385.35 | DEEP_RESET_BASE | PASS_AVAILABLE_SEVEN_CHECKS |
| 16 | AWL | 180.62 | DEEP_RESET_BASE | PASS_AVAILABLE_SEVEN_CHECKS |
| 17 | KAJARIACER | 1,208.50 | DEEP_RESET_BASE | PASS_AVAILABLE_SEVEN_CHECKS |
| 18 | ARROWGREEN | 738.45 | SURGE_PULLBACK | PASS_AVAILABLE_SEVEN_CHECKS |
| 19 | GLOBUSSPR | 807.00 | DEEP_RESET_BASE | UNKNOWN_FINANCIAL_INPUTS |
| 20 | AKUMS | 709.55 | DEEP_RESET_BASE | PASS_AVAILABLE_SEVEN_CHECKS |

## Daily-worker changes

The weekday **22:00 IST** worker is pinned to tested correction commit `ad519bf2e626ff89d0d68037a704b64f79d8647f`. It requests up to 3,200 public histories and 1,600 filings per run within bounded source/time limits, retries limited source variants, counts fresh official-aligned histories separately from merely usable old histories, rotates public financial comparison work and reports full-universe financial coverage. A verified source pack retains the 24 successfully recovered original filings, including Mastek comparison periods, without seeding any desired stock rankings. Runs use the existing single-writer concurrency group. GitHub scheduling can be delayed.

The later collector timed out after spending much of its 90-minute budget restoring thousands of separate archive objects under provider rate limits. Its saved financial collection is included in this correction; its previous completed list was not treated as a new model result. Restarts now use one hash-verified bundle, preserve immutable original objects and record the actual attempt phase. Health reports distinguish a previous completed report from the present attempt.

The archive migration is running; its completion has not yet been claimed.

Validation: all **630 tests passed locally and on GitHub**, including missing consolidated results, stale source responses, original-source identity and byte-hash checks, rate limits, period/basis mismatches and duplicate-provider checks. This correction replay writes no private archive state; the normal worker persists its own verified source recovery. No model was retrained or declared accurate.

## Reproducibility and sources

- Frozen model SHA-256: `c9208f89c46d043ee1d74e44dee1c85b9721b7a094e40a0fabf2c5506c246571`
- Corrected input SHA-256: `1886f23150395b3f0904a58e2d92847d14b65b90f5bfdedac161567ec42a41c3`
- [Correction and code review](https://github.com/ayushverma1988/v9-multibagger-calibration/pull/20)
- [Final combined-financial model replay](https://github.com/ayushverma1988/v9-multibagger-calibration/actions/runs/39387830536)
- [Archive restart repair](https://github.com/ayushverma1988/v9-multibagger-calibration/actions/runs/39386777965)
- [Source diagnostic run](https://github.com/ayushverma1988/v9-multibagger-calibration/actions/runs/39367248886)
- [Contract checks](https://github.com/ayushverma1988/v9-multibagger-calibration/actions/runs/39388868606)
- [Mastek issuer consolidated June release](https://www.mastek.com/wp-content/uploads/2026/07/Press-Release-Q1FY27.pdf)
- [Mastek Screener consolidated tables](https://www.screener.in/company/MASTEK/consolidated/)
- [Mastek StockAnalysis/S&P Global financial tables](https://stockanalysis.com/quote/nse/MASTEK/financials/)
- Individual matched fields and source timestamps are retained in the correction artifact and dated source-review seed files.
