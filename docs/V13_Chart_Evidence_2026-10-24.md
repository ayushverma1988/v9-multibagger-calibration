# Chart interpretation review — 24 October 2026

Decision: add measurable daily/weekly chart diagnostics to the research system.
Keep the frozen doubling model and its probabilities intact. Treat exact shape
recognition as an experimental hypothesis until it adds value in dated tests.
The user's current inclusive RSI minimum is 50; the optional ±10% 200-DMA band
and removal of the nominal price band remain in force.

## Assessment of the eight supplied posters

| Poster or idea | Assessment | Implementation decision |
|---|---|---|
| Rounded cup and shallow handle | A recognised continuation setup; a rounded base and breakout volume are useful descriptions. The assertion of an extremely bullish outcome is unsupported by the picture. | Add a conservative rounded-base/handle proxy to a watch queue. It is not a certified cup detector or a doubling forecast. |
| Triple bottom | A potential reversal requires an actual resistance/neckline break. Three drawn lows alone do not confirm it. | Use observed prior resistance, breakout volume and failure/retest diagnostics. No separate triple-bottom accuracy claim. |
| Falling wedge | Its drawn slope does not establish the next direction. | Retain as a research description; no standalone wedge detector or bullish bonus. |
| Symmetrical triangle | Consolidation can resolve upward or downward. | Direction must be observed; compression alone is a watch condition. |
| Descending triangle | The supplied example breaks downward. | No positive multibagger credit from this picture; a failed upward breakout is a warning. |
| Bollinger settings table | There is no demonstrated universally best table for scalping through position trading. | Use the creator's 20-period, two-standard-deviation defaults as one fixed feature definition, with historical width percentiles. No tuning to these screenshots. |
| Bollinger squeeze | Low volatility is useful context; it does not specify direction. Crossing the middle band alone is insufficient confirmation. | Separate pre-breakout watch from a volume-confirmed upside resistance break. |
| One-minute order-block / 2R setup | The horizon and execution assumptions differ from a six/twelve-month equity model. | Exclude. 2R means twice the amount risked, not a doubling of the share price. |
| Position-trading breakout, pullback and MA cross | Price structure and trend agreement can help time entries; crossovers lag. | Add resistance/retest and completed-week trend diagnostics. Do not stack several correlated MA votes. |
| ATR stop | ATR describes volatility rather than direction. A stop can gap through its reference. | Add exact Wilder ATR and a labelled three-ATR-below-close reference; this is not an order, guaranteed fill or recalibrated loss probability. |
| 5/8/13, 8/15, 25/50/100/200 averages | Many overlapping price transformations can count momentum repeatedly. | Keep daily 50/200 and completed-week 10/40 trend context. Skip the fast crossover bundle. |

## What the sources establish

1. **Lo, Mamaysky and Wang (2000), Journal of Finance.** Automated patterns
   supplied incremental information in their historical US stock sample. This
   motivates objective detection, without proving our proxy, Indian-market
   profitability after costs, or six/twelve-month doubling probabilities.
   [Original paper](https://onlinelibrary.wiley.com/doi/abs/10.1111/0022-1082.00265)
2. **Jegadeesh and Titman (1993), Journal of Finance.** Their historical results
   support studying three-to-twelve-month relative momentum. They do not imply
   every high-momentum stock doubles.
   [Original paper](https://doi.org/10.1111/j.1540-6261.1993.tb04702.x)
3. **NSE Indices, Nifty200 Momentum 30.** The official method uses six- and
   twelve-month price returns adjusted for volatility. This is relevant Indian
   implementation precedent, not validation of this stock-selection model.
   [Official index method](https://www.niftyindices.com/indices/equity/strategy-indices/nifty200-momentum-30)
4. **John Bollinger's own rules.** The default is 20 periods and two deviations;
   settings are task-dependent. Band touches alone are not trade signals, and
   related indicators should not be counted as independent confirmations.
   [Creator's rules](https://www.bollingerbands.com/bollinger-band-rules)
5. **Fidelity's indicator guidance.** Cup/handle interpretation requires a prior
   advance, a rounded consolidation and a resistance break with increased volume.
   The measured move is a guideline. ATR is non-directional, includes gaps, and
   can provide volatility-aware reference levels.
   [Cup/handle](https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/cup-with-handle)
   · [ATR](https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/atr)
6. **Park and Irwin (2007).** Their review reports positive, negative and mixed
   findings and identifies selection bias, data snooping and cost/risk estimation
   problems. An attractive historical example is not an independent track record.
   [Authors' university record](https://experts.illinois.edu/en/publications/what-do-we-know-about-the-profitability-of-technical-analysis/)

These sources justify testing the concepts. None establishes a verified success
rate for this particular NSE small/midcap doubling screen.

## Added diagnostics and declared thresholds

- Exact issuer/date joins; source-date close conflicts and missing history are
  explicit. Each decision clips later bars before feature generation.
- Population-standard-deviation Bollinger width (20/2), ranked against 126
  widths. A prior-five-session width percentile at or below 20% is a squeeze cue.
- Resistance uses 63 **prior** traded sessions. A close exceeds it by 0.25 prior
  ATR with volume at least 1.5 times the **prior** 20-session median. A 20-session
  cooldown distinguishes the first qualifying event. These thresholds are
  hypotheses, not proven optimal values.
- Separate recent breakout, pullback hold, failed-breakout and extended states.
  Recent anchors last ten sessions. A close 0.5 ATR below its breakout anchor
  warns of failure; more than five ATR above the 50-session average or three
  ATR above prior resistance flags extension for review.
- Daily context requires price above the 50-session mean, that mean above the
  200-session mean, and the latter rising over 20 sessions. Completed-week
  context uses 10/40-week means with the longer mean rising over four weeks.
  Monday–Thursday observations cannot use Friday's future close. Missing weekly
  evidence is review, not a positive trend vote.
- The rounded-handle proxy examines fixed 60/90/126-session cups and
  5/10/15/20-session handles, excluding the decision bar. It requires an observed
  prior advance, approximate rim agreement, a 12–50% cup depth, time near the
  bottom, a handle in the upper half and at most one-third retracement. All
  numerical tolerances remain unvalidated. It is deliberately labelled a proxy.
- A large recent change in the provider adjustment factor triggers corporate
  action review. This does not independently certify all corporate actions.
- Financial failures and observed adverse-disclosure flags exclude the new
  candidate queue. A seven-check pass is not proof of catalyst economics.
- The queue is ordered by financial tier and setup type, with issuer identity
  breaking ties. No new hand-assigned chart score is presented as a probability.

## Evaluation before a predictive weight is earned

`v13_chart_evaluation_spec.json` freezes the comparison. Use the same dated
issuer universe, information cutoffs, entry rules and cost assumptions for
baseline and chart variants. Training/calibration must precede test dates;
overlapping twelve-month outcomes require purging at date boundaries. Exclude
COVID-affected dates according to the existing evaluation contract rather than
changing that exclusion after seeing outcomes.

Test momentum/trend, volatility contraction, structure and volume, and ATR risk
context in family-level ablations. Include failures and delisted/suspended names
where recoverable, and disclose any current-listing survivor bias. Evaluate
six/twelve-month doubling, the existing loss-first competing event, precision at
matched coverage, Brier/log loss, turnover, slippage and uncertainty by date
blocks. ATR exit policies require a separate next-session execution experiment;
they cannot rewrite frozen target labels. Improved engineering tests are not
evidence of improved predictive accuracy.

## Daily operation

The generic snapshot postprocessor reads an exact completed collector artifact,
verifies its model and checkpoint identities, restores only the required saved
histories and writes separate chart diagnostics. It has no archive save call.
This lets the chart layer run after the daily collector without refitting or
changing a frozen forecast. Its outputs describe that collector's original
cohort; the separately requested financial-input rerun retains its own lineage.
