# Multibagger Model Roadmap

This file is the canonical version roadmap for the research/backtesting pipeline.
A component is never promoted solely because its workflow ran successfully; it
must pass its documented forward, leakage, calibration, alpha-retention and
robustness gates.

## Validated / completed foundations

### V9.4.1 — Technical/downside baseline
- Frozen validated benchmark.
- Top-10 selector with calibrated 2x probability, downside probability,
  safety/consensus tie-breaks and forward configuration selection.
- Remains the exact fallback whenever a challenger does not pass its gates.

### V9.5 — Fundamental turnaround challenger
- Canonical monthly PIT archive: Jan-2015 through Sep-2026.
- Stable rank-Platt + prior-8-fold base-rate calibration.
- Fundamental standalone quality gate passes.
- Fixed global linear blend (5%-25%) is rejected because it dilutes validated
  technical Top-10 alpha; production fundamental weight remains 0%.
- Conditional fundamental confirmation/veto overlay remains a challenger only;
  it has encouraging diagnostic performance but insufficient genuinely forward
  active folds for promotion.

## V9.6 — Integrity and regime work

### V9.6A — Label/data/PIT integrity
- Immature 6-month labels must remain NaN.
- Same-day fundamentals/events are excluded.
- Revisions become available only after their historical broadcast timestamp.
- Split/bonus-normalized prices are used for labels/features.
- Ticker/ISIN continuity is under validation.
- Universal ISIN stitching is rejected.
- Conditional continuity stitching is experimental and must not replace the
  frozen baseline unless its full historical validation passes.

### V9.6B — Regime challenger
- v1 full selector switching: rejected; precision retention fell below 95%.
- v2 fixed-baseline meta-overlay: safe but selected 0% regime influence on all
  forward folds; therefore it adds no validated incremental value yet.
- No additional standalone regime tuning unless materially new PIT regime data
  becomes available. Regime interactions may be reconsidered jointly in V9.8
  or V10.1.

## V9.7 — Point-in-time news and corporate events
- Canonical V9.7 scope.
- Source order: exchange corporate announcements first, then other timestamped
  sources only if provenance is reliable.
- Append-only event archive with immutable publication/broadcast timestamps,
  source IDs, security identity and raw-document hashes.
- Initial event families: order wins, capacity expansion, debt reduction,
  rating actions, promoter/pledge changes, dilution, buybacks, management and
  auditor changes, regulatory/litigation events, earnings and corporate actions.
- Same-day event leakage is prohibited.
- Event features are challengers until strictly forward validation passes.

## V9.8 — Sector / path-shape / interaction challengers
- Sector-relative strength only with point-in-time sector/industry history.
- Path-shape challenger includes momentum acceleration, breakout proximity,
  recovery position, trend persistence, turnover-breakout and drawdown recovery.
- Standalone path-shape v1 produced no forward activation and therefore is not
  promoted.
- Sector, regime and path features may be tested as interactions, but every
  historical decision must use only information known at that decision time.

## V9.9 — Liquidity, execution realism and robustness
- Turnover/capacity constraints.
- Gap/circuit/illiquidity risk.
- Transaction-cost/slippage sensitivity.
- Bootstrap confidence intervals.
- Leave-one-fold-out and regime-stability tests.
- Perturbation / parameter-stability tests.
- Selection stability and probability reliability bands.
- Re-test V9.5 conditional fundamental overlay here with robustness diagnostics.

## V10 — Frozen base integrated ensemble
V10 is the first integrated candidate assembled only from components that have
already passed their own forward gates (or contribute through an explicit
zero-weight fallback).

Principles:
- Preserve the frozen technical baseline as fallback.
- No fixed weight is assumed merely because a signal is statistically useful.
- All ensemble weights/gates are selected using strictly prior mature folds.
- Holdout and realistic liquidity/execution tests are required before V10 is
  declared frozen.

## V10.1 — Adaptive fundamental integration
This stage explicitly revisits the V9.5 fixed linear fundamental blend **after**
the V10 base integration is available.

The rejected V9.5 approach:
- one global 5%-25% fundamental weight applied to all stocks/regimes.

V10.1 replacement:
- conditional / mixture-of-experts fundamental contribution;
- fundamental influence can depend on technical strength, market regime,
  sector-relative state, path-shape and validated event context;
- candidate mechanisms include prior-only gating, monotonic confirmation,
  veto/bonus policies, or a regularized meta-learner;
- the default/fallback fundamental contribution is 0%;
- no future fold can choose its own fundamental weight.

Promotion requirements:
- leakage-free nested forward selection;
- retain at least 95% of the validated integrated alpha metrics;
- improve pre-specified utility rather than a single cherry-picked fold;
- maintain acceptable calibration and downside behaviour;
- survive V9.9-style bootstrap, leave-one-fold-out, liquidity and perturbation
  stress tests.

The purpose of V10.1 is therefore **not** to force fundamentals into the model.
It is to test whether fundamentals add incremental value once the other
validated signals are available.

## V10.2 — Final robustness and freeze
- Untouched final validation / holdout where available.
- Full leakage audit.
- Probability calibration audit.
- Liquidity/cost/capacity stress.
- Bootstrap confidence intervals and selection stability.
- Freeze features, gates, weights and data-version manifests.
- Any component that fails reverts to its lower-version validated fallback.

## Current execution order

1. Finish V9.6A conditional identity/corporate-action diagnosis without
   promoting a performance-degrading continuity rule.
2. Build V9.7 historical PIT exchange-announcement archive month by month.
3. Build V9.8 point-in-time sector-relative data/challengers and retain
   path/regime only if genuinely incremental.
4. Run V9.9 execution/robustness layer.
5. Assemble V10 base integrated ensemble from passed components.
6. Run V10.1 adaptive fundamental integration.
7. Run V10.2 final robustness and freeze.

The model outputs calibrated probabilities/rankings under uncertainty; no
version guarantees that an individual stock will double.
