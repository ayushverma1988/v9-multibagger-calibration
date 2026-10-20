# Missing-input collection and the earlier twenty shares

The user's screenshot is the broader twenty-share report in
`docs/v13_two_x_research_current_20261018.md`, using 16 October prices. It is
different from the earlier eleven-share refined shortlist. Both screens now
have separate daily exports; neither prior membership nor user preference is
used to force a share through the current rules.

## Data before analysis

`v13_master_data` exports a 3,200-issuer financial master, a long cell-level
source ledger, missing-column counts, model-column coverage and the exact
historical twenty-share comparison. The daily worker saves this inventory
before constructing current model inputs and refreshes it after scoring.

Original values and `validated_` columns remain separate. A new corroborated
input requires an original amount and two additional provider families with
the same ISIN, standalone/consolidated basis, statement period, metric and
units. Display rounding determines tolerance. Repeated or mirrored provider
pages count once. Ambiguous total/parent profit, basic/diluted EPS and
borrowings including leases are not silently equated. Missing values remain
blank. Stale periods cannot populate current validated columns.

The public-source adapter stores wire hashes, retrieval clocks and compressed
source bytes. Its queues rotate across the whole public universe without
using model ranks. It uses normal public HTTPS and stops a source on 403/429.
The October collection encountered a Finology access denial on the hosted
runner and a Screener rate limit; these remain explicit collection gaps.

## Why earlier names disappeared

The old index-member lookup lacked sectors for fourteen of the twenty
historical names. NSE's public quote metadata, Screener's four-level taxonomy
and Accord/Finology's sector labels now corroborate all twenty classifications.
The reviewed public packet preserves these receipts and their actual clocks.
Unknown classifications can be filled only when all three provider families
agree, issuer identity matches and the observations are at most 31 days old.
Existing known classifications are not overwritten.

Correcting that metadata restores eligibility for several earlier shares under
the unchanged refined selector. Six earlier names still fail its RSI rule on
19 October. The broader available-data list remains separately visible.
Financially incomplete names retain that status even when sector metadata is
recovered. Banks and insurers retain their own missing sector-specific checks.

Singer's original FY26 annual report and June-quarter statement also provide
independently reviewed original amounts. Its printed ISIN, NSE/BSE codes,
units and table locations are recorded with exact PDF hashes. Missing original
observations can supplement the research master; existing exchange amounts
are retained. Some annual-report amounts have greater displayed precision
than the exchange filing. Differences are recorded for review, without
rewriting old filing versions or historical forecasts.

An additional ingestion defect was corrected: ownership rows on an empty
consolidated page no longer count as financial statements. The collector
requests the preferred reporting basis first, tries the other public mode
only when real statements are absent, and preserves the observed basis.

## Interpretation and verification

Three-source agreement is corroboration, not a guarantee of truth or returns.
The master explicitly reports that all missing data has not been collected.
The model weights and stored prospective forecasts remain unchanged. A
classification-only reassessment of the last completed predictions is labelled
separately from a new inference run using newly collected financial inputs.

The contract suite covers identity/series matching, reporting-basis separation,
source-family duplication, rounding conflicts, future and stale observations,
sector recovery and saving the master before analysis. The archive writer is
serialized with the daily worker so a restored old state cannot overwrite a
newer collection checkpoint.
