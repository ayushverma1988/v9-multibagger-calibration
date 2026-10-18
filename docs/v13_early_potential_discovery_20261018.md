# Early potential research lane — 18 October 2026

This separate branch scans the full officially reconciled NSE scored universe for **early 6–12 month upside hypotheses**. A 50–100% move is a research outcome to evaluate, **not a predicted probability, target or certainty**. The nine user screenshots define the questions to diagnose, not positive training labels. The existing 2× champion and its risk forecasts are untouched.

## Rules and evidence states

1. **Deep reset base:** at least 252 preceding observed sessions, 20–80% below the preceding 756-session peak, 6–45% range over 30 sessions, and a bounded 30-session return and low distance. This wider chart window catches a developing bottom before RSI >70 or a fresh breakout.
2. **Short-history base:** 126–251 observed sessions, 5–40% below the prior 126-session high, and a 6–30% 30-session range. This provides an honest separate lane for newly listed shares; it cannot pretend to know their pre-listing history.
3. **Surge pullback:** the independently calculated rising, high-volume surge and subsequent 8–35% pullback. All price levels use available adjusted bars; a corporate-action review is still needed before backtesting.
4. **Independent cues:** five-session volume burst, quarterly sales and owner-attributable profit growth from dated sources, and 90-day filing counts for orders/capacity/commissioning/approvals. A filing count asks for document review; it is not proof of execution or earnings.
5. **Review gates:** positive official-reconciled current price, feature quality and ₹25 lakh turnover, a dated share-count market-cap proxy ≤₹50,000 crore, and separate states for missing size, adverse filings and known financial conflict. No ₹20–₹2,000 price band, RSI threshold, positive 50-DMA, or 200-DMA band is required. Financial conflict remains visible as chart research but is **not** an early potential.

The score is a declared chart/volume/financial/news **ordering heuristic**, not a fitted probability. The daily worker exports `early_potential/latest.json` with every potential, watch and conflict, and `early_potential/latest.md` with the first 20 potentials. Every scored name has an exact-ISIN state in the nine-example diagnostic. The output includes why a company is in a queue and preserves its frozen 2× probabilities separately for context.

## Baseline snapshot before the new history feature has run

Applied locally to the completed 16 October source-date worker artifact (run 39093593689; 1,638 scored rows): 1,037 chart setups, **157 early potentials**, **146 chart watches**, **607 financial conflicts**, and 28 size-unknown reviews. These are screening counts, not observed future winners. The top ten ordering at this stage is:

| Rank | Symbol | Close ₹ | Size proxy ₹ crore | Prior-peak drawdown | Five-day max volume ratio | Evidence cue |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | SOBHA | 1,188.40 | 12,709 | −43.59% | 7.11× | Sales and owner-profit growth |
| 2 | VENTIVE | 561.20 | 13,106 | −32.03% | 3.80× | Sales and owner-profit growth |
| 3 | ELGIEQUIP | 572.20 | 18,134 | −24.74% | 0.82× | Sales, owner profit and dated filing cue |
| 4 | RAINBOW | 1,248.90 | 12,685 | −25.22% | 2.01× | Sales, owner profit and dated filing cue |
| 5 | ENDURANCE | 2,406.40 | 33,848 | −20.57% | 0.89× | Sales, owner profit and dated filing cue |
| 6 | GRINDWELL | 1,879.90 | 20,814 | −34.66% | 7.79× | Sales and owner-profit growth |
| 7 | VOLTAMP | 10,622.00 | 10,746 | −25.02% | 4.73× | Sales and dated filing cue; owner profit unresolved |
| 8 | APOLLOTYRE | 408.85 | 25,966 | −25.83% | 6.76× | Sales and owner-profit growth |
| 9 | VAIBHAVGBL | 210.88 | 3,528 | −57.54% | 7.24× | Sales and owner-profit growth |
| 10 | EXPLEOSOL | 896.65 | 1,392 | −30.53% | 3.60× | Sales and owner-profit growth |

The nominal prices above ₹2,000 are deliberately allowed. These names have not been individually cleared for valuation, execution, liquidity under a proposed position size, or catalyst quality. A volume spike may reflect selling or an event; inspect the subsequent tape and primary filing before any decision.

## User-example diagnostic on the same snapshot

| Example | New state | Why it is or is not surfaced |
| --- | --- | --- |
| eMudhra | EARLY_POTENTIAL, deep reset; rank 142/157 | Sales and owner profit grew in dated data; 30-day range is still wide (39.4%) and latest five-session volume cue absent. A developing bottom, not confirmed price turn. |
| Dynacons | FINANCIAL_CONFLICT_REVIEW | Base and volume cue observed; declining latest sales and owner/minority/total-PAT reconciliation conflict block promotion. Its two order-disclosure counts need document review. |
| Corona Remedies | Await 126-session feature; current old snapshot has no long-peak result | 208 observed bars, so a 252-session drawdown is not applicable. The daily worker will classify a short-history base from actual adjusted history. |
| Paragon, Ashapuri Gold, Chemcrux | NOT_SCORABLE_WITH_CURRENT_DATA | Current NSE master identity exists; provider history is missing or insufficient. Venue-specific backfill and corporate-action checks are required. |
| Raj Rayon | NOT_SCORABLE_WITH_CURRENT_DATA | Latest source session missing and sustained turnover below the ₹25 lakh execution gate. It remains a coverage/liquidity review. |
| R Systems | FINANCIAL_CONFLICT_REVIEW | Deep reset chart and volume cue observed, but latest owner PAT declined. |
| Huhtamaki India | EARLY_POTENTIAL, deep reset; rank 96/157 | Deep drawdown and volume plus sales growth; owner attributable PAT is unresolved. Surge-pullback status awaits the separate daily retest audit. |

These nine were **not used in any selection rule, score coefficient or fitted label**. Dates and financial fields remain point-in-time. Subsequent evaluation should follow all daily candidates, account for delistings and corporate actions, measure the first tradable +50% and +100% outcomes at 6/12 months and intervening drawdowns, then compare chronological holdouts with a random and sector/size-matched baseline. Until that evidence matures, do not promote this heuristic to production forecasts.

Primary examples of filing evidence and its limits: [Corona corporate presentation](https://nsearchives.nseindia.com/corporate/CORONAREMEDIES_03012026194656_Corporate_Presentation.pdf), [R Systems quarterly filing](https://nsearchives.nseindia.com/corporate/ixbrl/INTEGRATED_FILING_INDAS_182471_04082026212519_iXBRL_WEB.html), [eMudhra quarterly filing](https://nsearchives.nseindia.com/corporate/ixbrl/INTEGRATED_FILING_INDAS_179975_29072026181329_iXBRL_WEB.html). The numerical candidate table is from the private, dated daily worker artifact, not an issuer forecast.
