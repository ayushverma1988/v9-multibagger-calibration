# Completed collection and model outcome — 20 October 2026

The GitHub workflow completed successfully at 15:39 IST. Prices are the 19 October close. Model weights were unchanged; predictive accuracy remains unvalidated.

[Completed run and source artifact](https://github.com/ayushverma1988/v9-multibagger-calibration/actions/runs/39394075729)

## Coverage

| Metric | Previous corrected replay | Completed collector |
| --- | ---: | ---: |
| Scoreable market inputs / 3,200 | 633 | 1,636 |
| Issuers with financial observations | 2,874 | 2,874 |
| Numeric financial observations | 492,306 | 517,164 |
| Issuers with all seven financial checks | 1,279 | 1,280 |

Coverage remains partial: 1,564 issuers are outside the current scoreable cohort. Observations are numeric field values across periods, reporting bases and document vintages, not companies or reports. The run recorded 911 history-attempt errors and 267 filing-attempt errors. Successful workflow completion does not mean every source request succeeded.

## Latest 2× output

The loss event is a 30% decline before a 2× hit within 12 months, not the return at the end of 12 months. Return and loss estimates are unvalidated model outputs.

# Separate 2× research shortlist — 2026-10-19

Frozen-model estimates for doubling by 6 or 12 months. These are research estimates, not confirmed returns or trade instructions.
Existing refined selector: verified small/mid caps, known sector, strict RSI >70, financial/owner checks; 70% p6 percentile, 25% price discipline, 5% optional 200DMA band.

Coverage: 1,636/3,200 public shares. PARTIAL_DATA_DO_NOT_REPLACE_FULL_WATCHLIST. Missing data is not a failed investment thesis.

| Rank | Symbol | Close ₹ | P(2× 6m) | P(2× 12m) | 30% loss event | Financial tier | Source review |
| ---: | --- | ---: | ---: | ---: | ---: | --- | --- |
| 1 | MASTEK | 1,808.40 | 5.71% | 9.49% | 84.32% | PASS_AVAILABLE_SEVEN_CHECKS | ANNUAL_REVENUE_CFO_TWO_ADDITIONAL_PROVIDERS_AGREE |
| 2 | GABRIEL | 1,486.10 | 6.12% | 10.55% | 83.51% | PASS_AVAILABLE_SEVEN_CHECKS | ANNUAL_SOURCE_CONFLICT_REVIEW |
| 3 | THELEELA | 622.75 | 6.96% | 11.85% | 70.09% | PASS_AVAILABLE_SEVEN_CHECKS | ANNUAL_REVENUE_CFO_TWO_ADDITIONAL_PROVIDERS_AGREE |
| 4 | CUPID | 336.30 | 9.24% | 13.85% | 82.15% | PASS_AVAILABLE_SEVEN_CHECKS | ANNUAL_REVENUE_CFO_TWO_ADDITIONAL_PROVIDERS_AGREE |
| 5 | BSOFT | 312.60 | 4.93% | 8.60% | 80.33% | PASS_AVAILABLE_SEVEN_CHECKS | NO_CURRENT_ANNUAL_COMPARISON |
| 6 | SOUTHBANK | 54.26 | 5.82% | 10.41% | 60.34% | UNKNOWN_SECTOR_RULES_OR_CLASSIFICATION | SECTOR_RULES_PENDING_REVIEW |
| 7 | SHYAMMETL | 1,226.60 | 5.18% | 9.48% | 40.22% | PASS_AVAILABLE_SEVEN_CHECKS | NO_CURRENT_ANNUAL_COMPARISON |
| 8 | TANLA | 552.35 | 4.49% | 8.25% | 85.27% | PASS_AVAILABLE_SEVEN_CHECKS | ANNUAL_REVENUE_CFO_TWO_ADDITIONAL_PROVIDERS_AGREE |
| 9 | KARURVYSYA | 370.05 | 4.70% | 9.34% | 66.44% | UNKNOWN_SECTOR_RULES_OR_CLASSIFICATION | SECTOR_RULES_PENDING_REVIEW |
| 10 | BHARTIHEXA | 1,632.60 | 4.08% | 7.89% | 76.37% | PASS_AVAILABLE_SEVEN_CHECKS | NO_CURRENT_ANNUAL_COMPARISON |
| 11 | LEMONTREE | 111.67 | 3.72% | 7.42% | 36.90% | PASS_AVAILABLE_SEVEN_CHECKS | ANNUAL_SOURCE_CONFLICT_REVIEW |
| 12 | JUSTDIAL | 731.75 | 4.11% | 8.04% | 80.31% | PASS_AVAILABLE_SEVEN_CHECKS | ANNUAL_REVENUE_CFO_TWO_ADDITIONAL_PROVIDERS_AGREE |
| 13 | ZENSARTECH | 513.15 | 3.62% | 7.39% | 81.14% | PASS_AVAILABLE_SEVEN_CHECKS | ANNUAL_REVENUE_CFO_TWO_ADDITIONAL_PROVIDERS_AGREE |
| 14 | MPHASIS | 2,500.00 | 3.57% | 7.33% | 65.77% | PASS_AVAILABLE_SEVEN_CHECKS | ANNUAL_REVENUE_CFO_TWO_ADDITIONAL_PROVIDERS_AGREE |
| 15 | TATAELXSI | 3,790.00 | 3.55% | 7.22% | 80.48% | PASS_AVAILABLE_SEVEN_CHECKS | ANNUAL_SOURCE_CONFLICT_REVIEW |

## Earlier no-price-band shortlist (17 October report, 15 October source prices)

Prior membership is tracked for explanation only; it does not influence the daily selection.

| Symbol | Current result | Reason if excluded | Current P(2× 6m) | Current P(2× 12m) | 15 Oct P(2× 6m) | 15 Oct P(2× 12m) |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| MASTEK | TOP_20 | — | 5.71% | 9.49% | 6.56% | 10.31% |
| SHREEJISPG | FAILED_CURRENT_SCREEN | OBSERVED_OWNER_EARNINGS_FAILURE_OR_CONFLICT | 11.89% | 16.58% | 11.02% | 15.91% |
| TANLA | TOP_20 | — | 4.49% | 8.25% | 5.43% | 9.15% |
| CUPID | TOP_20 | — | 9.24% | 13.85% | 9.99% | 14.59% |
| HEXT | FAILED_CURRENT_SCREEN | OBSERVED_SECTOR_FINANCIAL_FAILURE, OBSERVED_OWNER_EARNINGS_FAILURE_OR_CONFLICT | 4.75% | 8.53% | 5.36% | 9.11% |
| THELEELA | TOP_20 | — | 6.96% | 11.85% | 6.02% | 10.79% |
| LTTS | FAILED_CURRENT_SCREEN | RSI_NOT_STRICTLY_ABOVE_70 | 3.81% | 7.53% | 4.52% | 8.23% |
| PERSISTENT | FAILED_CURRENT_SCREEN | RSI_NOT_STRICTLY_ABOVE_70 | 4.02% | 8.23% | 3.55% | 7.86% |
| ZENSARTECH | TOP_20 | — | 3.62% | 7.39% | 3.41% | 7.14% |
| SUNTV | FAILED_CURRENT_SCREEN | OBSERVED_OWNER_EARNINGS_FAILURE_OR_CONFLICT | 5.23% | 9.09% | 4.52% | 8.41% |
| MPHASIS | TOP_20 | — | 3.57% | 7.33% | 3.20% | 6.97% |

Scored frame: 1636 shares. Current source date: 2026-10-19.
Classification source observation dates: 2026-10-16.
A missing name cannot be reassessed from this snapshot. No 50–100% forecast is inferred from this 2× model.

## Source review

Seven current names have annual revenue/CFO corroboration against two additional provider families: MASTEK, THELEELA, CUPID, TANLA, JUSTDIAL, ZENSARTECH and MPHASIS. This covers those annual fields, not every financial input, corporate action, catalyst or prediction.

GABRIEL, LEMONTREE and TATAELXSI have annual source conflicts. BSOFT, SHYAMMETL and BHARTIHEXA lack a current additional annual comparison. SOUTHBANK and KARURVYSYA await bank-specific financial rules.

The run compared 3,884 provider observations across 129 issuers: 946 agreements within display rounding, 117 numeric conflicts, 1,709 profit-definition reviews and 1,112 missing/ambiguous primary facts. Only 25 annual field-periods passed the primary-plus-two-provider comparison; that is not 25 fully verified companies. Canonical originals were not overwritten by provider agreement.

Six of the earlier 11 names pass: MASTEK, TANLA, CUPID, THELEELA, ZENSARTECH and MPHASIS. Three absent in the 633-share replay — MASTEK, THELEELA and MPHASIS — returned after input coverage improved. No retraining was involved.

## Separate chart output

There are 175 early potentials and 117 additional chart watches. These are untrained 6–12-month research hypotheses; the scores are not return probabilities. Filing cues require original-document review and do not prove capex completion or future earnings.

# Early potential research queue — 2026-10-19

Untrained 6–12 month upside hypotheses. Rank is not a return probability.

| Rank | Symbol | Price ₹ | Pattern | Volume | Sales | Owner profit | Filing cue |
| ---: | --- | ---: | --- | --- | --- | --- | --- |
| 1 | ATUL | 6082.00 | DEEP_RESET_BASE | Yes | Yes | Yes | Review |
| 2 | NEWGEN | 479.10 | DEEP_RESET_BASE | Yes | Yes | Yes | Review |
| 3 | KEI | 4442.00 | DEEP_RESET_BASE | Yes | Yes | Yes | Review |
| 4 | TVSSRICHAK | 4295.90 | SURGE_PULLBACK | No | Yes | Yes | Review |
| 5 | QUESS | 321.85 | SURGE_PULLBACK | Yes | Yes | Yes | None observed |
| 6 | TIMETECHNO | 174.13 | DEEP_RESET_BASE | No | Yes | Yes | Review |
| 7 | KRYSTAL | 610.10 | DEEP_RESET_BASE | No | Yes | Yes | Review |
| 8 | JAYAGROGN | 217.99 | SURGE_PULLBACK | Yes | Yes | Yes | None observed |
| 9 | NH | 1740.90 | DEEP_RESET_BASE | No | Yes | Yes | Review |
| 10 | PREMIERENE | 899.00 | DEEP_RESET_BASE | No | Yes | Yes | Review |
| 11 | VENTIVE | 558.45 | DEEP_RESET_BASE | Yes | Yes | Yes | None observed |
| 12 | GRINDWELL | 1826.80 | DEEP_RESET_BASE | Yes | Yes | Yes | None observed |
| 13 | GULFOILLUB | 1061.30 | DEEP_RESET_BASE | Yes | Yes | Yes | None observed |
| 14 | RAINBOW | 1236.00 | DEEP_RESET_BASE | Yes | Yes | Yes | Review |
| 15 | VOLTAMP | 10410.00 | DEEP_RESET_BASE | Yes | Yes | Unknown/No | Review |
| 16 | ELGIEQUIP | 544.35 | DEEP_RESET_BASE | No | Yes | Yes | Review |
| 17 | HSCL | 605.85 | DEEP_RESET_BASE | Yes | Yes | Yes | None observed |
| 18 | VAIBHAVGBL | 205.90 | DEEP_RESET_BASE | Yes | Yes | Yes | None observed |
| 19 | BLS | 221.03 | DEEP_RESET_BASE | Yes | Yes | Yes | None observed |
| 20 | EXPLEOSOL | 875.55 | DEEP_RESET_BASE | Yes | Yes | Yes | None observed |

## Surge pullbacks (separate pattern queue)

| Pattern rank | Symbol | Price ₹ | Score | Volume cue | Filing cue |
| ---: | --- | ---: | ---: | --- | --- |
| 1 | TVSSRICHAK | 4295.90 | 0.678 | No | Review |
| 2 | QUESS | 321.85 | 0.672 | Yes | None observed |
| 3 | JAYAGROGN | 217.99 | 0.664 | Yes | None observed |
| 4 | UNIPARTS | 714.60 | 0.628 | Yes | None observed |
| 5 | STAR | 1029.30 | 0.610 | No | None observed |
| 6 | KIRIINDUS | 487.65 | 0.604 | No | None observed |
| 7 | ARROWGREEN | 738.45 | 0.592 | Yes | None observed |
| 8 | AVALON | 2020.50 | 0.585 | Yes | None observed |
| 9 | TIIL | 2555.20 | 0.579 | No | None observed |
| 10 | CAPLIPOINT | 2258.80 | 0.571 | No | None observed |
| 11 | VENUSPIPES | 1831.70 | 0.550 | Yes | Review |
| 12 | IVP | 210.54 | 0.533 | Yes | None observed |
| 13 | CENTENKA | 549.20 | 0.522 | Yes | None observed |
| 14 | HAPPYFORGE | 1866.80 | 0.518 | Yes | None observed |
| 15 | EMUDHRA | 540.00 | 0.497 | No | None observed |
| 16 | SAKAR | 923.20 | 0.479 | Yes | None observed |
| 17 | DIVGIITTS | 944.10 | 0.473 | Yes | None observed |
| 18 | UEL | 183.00 | 0.442 | Yes | None observed |

## Short-history bases (separate pattern queue)

| Pattern rank | Symbol | Price ₹ | Score | Volume cue | Filing cue |
| ---: | --- | ---: | ---: | --- | --- |

117 more chart watches and 681 financial conflicts appear in latest.json.

## Your nine chart examples

All nine are in the current NSE master. Their current statuses are separate from the original RSI >70 shortlist.

| Share | Current result |
| --- | --- |
| eMudhra (EMUDHRA) | Early potential, pullback after a surge; rank 119. RSI 23.96 fails the separate RSI >70 screen. |
| Dynacons Systems & Solutions (DSSL) | Chart base and volume cue present; sales YoY −4.61% triggers financial review. |
| Corona Remedies (CORONA) | Short-history chart watch; missing comparative financial growth inputs. |
| Paragon Fine and Speciality Chemical (PARAGON) | No accepted current history: public response failed instrument/venue/currency identity validation. |
| Ashapuri Gold Ornament (AGOL) | Insufficient accepted NSE observations after 12 October 2026 listing. |
| Raj Rayon Industries (RAJRILTD) | Stored history, but 20/63-session turnover fails the ₹25 lakh feature gate. |
| R Systems International (RSYSTEMS) | Pullback and volume cue present; PAT YoY failure triggers financial review. |
| Chemcrux Enterprises (CHEMCRUX) | Insufficient accepted NSE observations after 17 August 2026 listing. |
| Huhtamaki India (HUHTAMAKI) | Early potential, deep-reset base; rank 141. Volume cue present; RSI 52.68. |

The rolling-history audit begins no earlier than 19 October 2022. Complete listing-to-date histories and fully reconciled BSE/NSE continuity are not claimed.

## Operations and limitations

The archive snapshot and complete restart state were saved. Weekday capture remains scheduled for 22:00 IST. This code previously passed 636 tests. There are zero independently verified mature outcomes; this collection does not establish forecasting accuracy.

This outcome supersedes the 633-share replay for current coverage and shortlist membership. The earlier correction report remains the record of the reporting-basis repair.

