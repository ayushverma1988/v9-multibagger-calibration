# V13 discovery research — 16 October 2026

## Price-policy update — 17 October 2026

The user removed the nominal ₹20–₹2,000 share-price restriction. Current model
gates, discovery watchlists and financial-collection prioritization now require
only a finite positive price. RSI >70, small/mid-cap classification, financial
failure checks, risk limits, liquidity/history eligibility and optional DMA
preference are unchanged. Frozen probability heads are unchanged.

A controlled replay of the saved 15 October close snapshot scores the same
1,450 stocks, including 19 below ₹20 and 60 above ₹2,000. Final eligibility rises
from 8 to 11, adding LTTS, PERSISTENT and MPHASIS. Early-watch eligibility rises
from 23 to 27, retaining a ranked top20. Six probability columns reproduce
exactly; no retraining or fresh market collection occurs. Ranking can change
because its probability percentile uses the newly expanded eligible pool.

All 597 regression tests pass, including both removed price boundaries and
invalid-price rejection. Previously reported historical evaluation results and
their fixed price-band protocol are preserved as historical experiments; this
current-policy replay is not a new historical accuracy experiment. The existing
daily worker's deployment pin is not changed by this research-branch commit.

This branch adds auditable discovery research to PR16. It does not change the
operational selector, frozen models, scheduled worker, or old forecasts. The
matched six-month experiment did not improve selected doubling hits. Do not
promote it based on software tests or the more detailed evidence labels.

## What changed

- Causal breakout age, first RSI crossing, price extension and explicit failed
  breakout conditions. An initial high RSI without a prior observation is not
  an observed crossing. RSI uses the existing simple rolling calculation.
- Historical disclosure metadata bound to exact original issuer, source text,
  attachment and exchange availability clocks. Court orders, unconfirmed news,
  shareholding approvals and dematerialisation cancellations are excluded from
  business order, operating approval and project failure signals respectively.
  Metadata cues remain separate from original project facts.
- Matched baseline, breakout, catalyst and combined feature ablations, using
  the existing model fitter and calibration method. Both training and calibration
  labels must mature before the next stage. No new calibration experiment or
  selector-weight optimization is included.
- Separate bank/insurer research checks; missing regulatory and operating inputs
  remain unknown. Other financial businesses still need explicit rules.
- Current AMFI size classification and Nifty bank/insurance membership with raw
  hashes and actual observation clocks. These are not historical membership data
  or complete sector classifications.
- An early evidence watchlist and a separate strict RSI >70 research list,
  retaining all frozen probabilities and original cutoffs. Ranking preferences
  are 70% p6 percentile, 25% existing price discipline, 5% optional DMA proximity.
  Missing financial data is labelled; known failures remain excluded.
- Eight original partial project facts and six administrative counterexamples.
  No case establishes a complete execution-to-owner-earnings chain. The scenario
  bridge requires explicit revenue attribution, owner margin, diluted shares and
  valuation assumptions; it never emits a calibrated probability.

## Reproduce

Use Python 3.12, `requirements_v12_frozen.txt`,
`requirements_v11_4_systematic.txt`, and `requirements_v13_discovery.txt`.
Set `PYTHONPATH=src`, `OMP_NUM_THREADS=2`, `OPENBLAS_NUM_THREADS=2`.

```bash
python src/v13_discovery_build.py --matrix INPUT/challenger_matrix.parquet --current INPUT/expanded_primary_volume_research.parquet --events INPUT/events_all_2016_2026.parquet --history INPUT/history/securities --output OUTPUT --coverage-end 2026-09-30T18:26:18Z
python src/v13_discovery_evaluation.py --matrix OUTPUT/historical_enriched.parquet --output OUTPUT/evaluation --asof 2026-10-15
python src/v13_discovery_sources.py --output CLASSIFICATIONS
python src/v13_discovery_current.py --input OUTPUT/current_diagnostics.parquet --classification-dir CLASSIFICATIONS --evidence INPUT/redevelopment_inputs --output OUTPUT/current
python src/v13_project_evidence.py --events OUTPUT/historical_metadata.parquet --output OUTPUT/project_facts
python -m unittest discover -s tests
```

`--evidence` requires the existing `evidence/valuation_snapshot.json`,
`evidence/valuation_research_hurdles.json` and
`catalyst_evidence/document_evidence.json`. Preserve their original clocks.
Current classification downloads create a new review clock; to reproduce the
saved current list exactly, reuse saved classification receipts and review clock
through `classify_current`/`watchlists`, rather than asserting a new download was
available in the past. Failed PDF downloads remain `REVIEW_BLOCKED`; they can be
retried without adding unsupported facts.

The preserved evidence package contains enriched inputs, normalized metadata,
all paired predictions, current decisions, classification originals, project PDFs
and summaries. Evaluation can be rerun directly from `historical_enriched.parquet`.
Building that matrix from scratch also needs the original price histories and
source matrix from PR16. The full event archive came from Actions run36804202373,
artifact11137910663 (`v9-7-canonical-events-2016-2026`). Archive SHA256:
`3406e7cbb30424e8743b88770635b03ee52d6c3794d202a8f1730b5a5eec7b12`.
Input and code hashes are recorded in the build/evaluation receipts.

## Interpretation

Twenty-two previously examined development dates, June 2024–March 2026, contain
11,451 matched scored rows and 370 selections per variant. The baseline is refit
on each earlier eligible fold; applying today's frozen model to old dates would
leak future training. Each variant finds 15 six-month doublers. Combined features
reduce Brier error 0.02404015 to 0.02396606 but increase 30% loss events 115 to120.
Median six-month endpoint returns remain negative. Counts can include repeated
issuers and overlapping holding periods; they are not independent investments
or a cost-adjusted portfolio simulation. Twelve-month results use only278
mature selected rows and remain a secondary diagnostic.

This is retrospective development research, with current-listing survivor bias,
historical price-vintage limitations, incomplete source coverage and no verified
historical sector/size membership. Metadata availability reconstructed from the
original feed is not proof that the live model observed it then. Document counts
can include several updates about one project. The 14 reviewed cases are not a
representative classifier-accuracy audit and were not used as historical model
features. No untouched holdout or prospective accuracy gain is established.

Next research requirements are quantified attributable earnings and dilution,
dated sector/size histories and suitable financial inputs, followed by locked
prospective evaluation. Do not retune thresholds to force a positive result.
