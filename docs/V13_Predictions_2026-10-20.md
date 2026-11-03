> **Superseded on 20 October 2026:** This 226-share output has known coverage and financial-basis gaps. See the [corrected 633-share replay and full-universe financial audit](https://github.com/ayushverma1988/v9-multibagger-calibration/blob/v13-data-reliability-20261020/docs/V13_Data_Correction_2026-10-20.md). The original report below is retained for traceability.

# V13 predictions — 20 October 2026

Price source: completed NSE session of **19 October 2026**. The fresh model run completed 20 October 2026 at 03:27:29 UTC / 08:57:29 IST.

**Partial coverage:** 226 of 3,200 public NSE universe members (7.06%) were freshly scoreable. Of 2,293 stored histories, 2,062 lacked the source-date bar and five failed feature eligibility. A name absent from this snapshot has not been rejected on investment merits.

The refined 2× selector is the same selector that produced the earlier 11-name no-price-band result. Its size and bank/insurance classifications use verified primary files observed on 16 October 2026; current industry membership uses the 19 October snapshot. Original observation clocks are retained after the current classification-download timeout.

The early-potential queue is separate heuristic research. Its scores are not probabilities of a 50% or 100% return.

# Separate 2× research shortlist — 2026-10-19

Frozen-model estimates for doubling by 6 or 12 months. These are research estimates, not confirmed returns or trade instructions.
Existing refined selector: verified small/mid caps, known sector, strict RSI >70, financial/owner checks; 70% p6 percentile, 25% price discipline, 5% optional 200DMA band.

| Rank | Symbol | Close ₹ | P(2× 6m) | P(2× 12m) | 30% loss event | Financial tier |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | JUSTDIAL | 731.75 | 3.68% | 7.61% | 80.71% | PASS_AVAILABLE_SEVEN_CHECKS |

## Earlier no-price-band shortlist (17 October report, 15 October source prices)

Prior membership is tracked for explanation only; it does not influence the daily selection.

| Symbol | Current result | Reason if excluded | P(2× 6m) | P(2× 12m) |
| --- | --- | --- | ---: | ---: |
| MASTEK | NOT_IN_CURRENT_SCORED_FRAME | — | — | — |
| SHREEJISPG | NOT_IN_CURRENT_SCORED_FRAME | — | — | — |
| TANLA | NOT_IN_CURRENT_SCORED_FRAME | — | — | — |
| CUPID | NOT_IN_CURRENT_SCORED_FRAME | — | — | — |
| HEXT | NOT_IN_CURRENT_SCORED_FRAME | — | — | — |
| THELEELA | NOT_IN_CURRENT_SCORED_FRAME | — | — | — |
| LTTS | NOT_IN_CURRENT_SCORED_FRAME | — | — | — |
| PERSISTENT | FAILED_CURRENT_SCREEN | RSI_NOT_STRICTLY_ABOVE_70 | 3.55% | 7.82% |
| ZENSARTECH | NOT_IN_CURRENT_SCORED_FRAME | — | — | — |
| SUNTV | NOT_IN_CURRENT_SCORED_FRAME | — | — | — |
| MPHASIS | NOT_IN_CURRENT_SCORED_FRAME | — | — | — |

Scored frame: 226 shares. Current source date: 2026-10-19.
Classification source observation dates: 2026-10-16.
A missing name cannot be reassessed from this snapshot. No 50–100% forecast is inferred from this 2× model.

---

# Early potential research queue — 2026-10-19

Untrained 6–12 month upside hypotheses. Rank is not a return probability.

| Rank | Symbol | Price ₹ | Pattern | Volume | Sales | Owner profit | Filing cue |
| ---: | --- | ---: | --- | --- | --- | --- | --- |
| 1 | VOLTAMP | 10410.00 | DEEP_RESET_BASE | Yes | Yes | Unknown/No | Review |
| 2 | ENDURANCE | 2368.80 | DEEP_RESET_BASE | No | Yes | Yes | Review |
| 3 | AWL | 180.62 | DEEP_RESET_BASE | Yes | Yes | Yes | None observed |
| 4 | KAJARIACER | 1208.50 | DEEP_RESET_BASE | Yes | Yes | Yes | None observed |
| 5 | ARROWGREEN | 738.45 | SURGE_PULLBACK | Yes | Yes | Yes | None observed |
| 6 | TIIL | 2555.20 | SURGE_PULLBACK | No | Yes | Yes | None observed |
| 7 | FORCEMOT | 17563.00 | DEEP_RESET_BASE | Yes | Yes | Yes | None observed |
| 8 | CENTURYPLY | 664.10 | DEEP_RESET_BASE | No | Yes | Yes | None observed |
| 9 | GREENPLY | 303.55 | DEEP_RESET_BASE | Yes | Yes | Yes | None observed |
| 10 | BLUEDART | 4380.00 | DEEP_RESET_BASE | Yes | Yes | Unknown/No | None observed |
| 11 | PPL | 225.85 | DEEP_RESET_BASE | Yes | Yes | Unknown/No | Review |
| 12 | GULPOLY | 163.46 | DEEP_RESET_BASE | Yes | Yes | Unknown/No | None observed |
| 13 | CANTABIL | 249.90 | DEEP_RESET_BASE | Yes | Yes | Unknown/No | None observed |
| 14 | SAKAR | 923.20 | SURGE_PULLBACK | Yes | Yes | Unknown/No | None observed |
| 15 | IXIGO | 157.08 | DEEP_RESET_BASE | Yes | Yes | Unknown/No | None observed |
| 16 | JUSTDIAL | 731.75 | DEEP_RESET_BASE | Yes | Yes | Unknown/No | None observed |
| 17 | ETHOSLTD | 2593.80 | DEEP_RESET_BASE | Yes | Yes | Unknown/No | None observed |
| 18 | KPEL | 188.62 | DEEP_RESET_BASE | Yes | Yes | Unknown/No | None observed |

## Surge pullbacks (separate pattern queue)

| Pattern rank | Symbol | Price ₹ | Score | Volume cue | Filing cue |
| ---: | --- | ---: | ---: | --- | --- |
| 1 | ARROWGREEN | 738.45 | 0.592 | Yes | None observed |
| 2 | TIIL | 2555.20 | 0.579 | No | None observed |
| 3 | SAKAR | 923.20 | 0.479 | Yes | None observed |

## Short-history bases (separate pattern queue)

| Pattern rank | Symbol | Price ₹ | Score | Volume cue | Filing cue |
| ---: | --- | ---: | ---: | --- | --- |

18 more chart watches and 76 financial conflicts appear in latest.json.

## Verification

All six frozen-model outputs were independently replayed from the exact input parquet and the saved model. Maximum absolute probability difference: 5.56e-16 (required tolerance 1e-12). ZIP, model and input hashes matched their recorded receipts. All source-date OHLC checks passed in the worker; maximum close difference against the official row was ₹0.000196. This verifies reproducibility, not investment performance.

[Completed fresh source run](https://github.com/ayushverma1988/v9-multibagger-calibration/actions/runs/39325541617). Source artifact: 12239080868; SHA-256: 83f933a3969122760c11ad65fffc69da4816be3de8310bf707ed125f24fa04d9.
