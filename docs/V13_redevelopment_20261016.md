# Extension, breakout, evidence and risk redevelopment

The existing RSI >70 research screen often ranks stocks after a large advance.
This change adds separate price-extension and first-breakout diagnostics while
retaining the frozen formal heads, original forecast registries and source clocks.

## Implemented research logic

- Price discipline uses observed 20/63-session returns and distance from 200DMA.
  Missing observations earn no stability credit. The +/-10% DMA preference stays optional.
- A first breakout crosses the prior 63-session adjusted high after a compressed
  base, with no breakout in the prior 20 sessions and volume >=1.5 times the
  prior 20-session median. Resistance, compression and volume denominators exclude
  the breakout session. An early event is <=5 sessions old and <=10% extended.
- Earnings and order evidence retain explicit missingness. Announcements are not
  execution proof; missing execution/valuation does not veto research forecasts.
- A separate evaluator compares baseline, added features and regularized conditional
  calibration with maturity-purged chronological splits and saved training clocks.
- Optional exchange-final-volume research reconciliation requires exact date/ISIN,
  matching provider/exchange OHLC, valid adjusted prices and dated original hashes.
  It changes an in-memory copy only, records both sources, and never claims two-source
  agreement on volume. The formal collector default remains unchanged.

## Evaluation decision

Three previously examined monthly cohorts (July-September 2025), 1,521 rows:

|Variant|12-month 2x Brier|Risk Brier|Selected 12-month hits|
|---|---:|---:|---:|
|Baseline|0.054937|0.243399|5/54|
|Added features|0.054778|0.239855|4/54|
|Added features + new calibration|0.060414|0.249582|4/54|

Lower Brier is better overall probability error, not proof of calibration alone.
The new calibration is rejected for operational use. Feature-only improvements
are small, and selected doubling precision did not improve. No accuracy improvement
or model promotion is approved. Historical owner-EPS/execution features have zero
usable coverage in this matrix, so their incremental predictive value is untested.
Selection comparisons change both features and ranking; all-row comparisons isolate
probability models. Repeated issuers/overlapping outcomes are not independent trials.

Use `python -m v13_redevelopment_evaluation --inputs ... --current ... --output ...
--asof YYYY-MM-DD` with the verified pinned archive inputs. It writes new experimental
heads only; it never replaces formal heads or checkpoint pointers. Daily diagnostics
reuse exact champion probabilities and save the new ranking separately.

## Known limitations

Current-listing survivor bias, revised adjusted-price vintages, only three previously
examined evaluation dates, incomplete dated fundamentals, and no mature unseen
prospective outcomes remain. Tests cover implementation and preservation, not accuracy.
