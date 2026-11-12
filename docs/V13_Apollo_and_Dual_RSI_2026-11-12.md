# Apollo source correction and permanent dual RSI output

Apollo's September 2026 XBRL contains current-period figures but no September
2025 quarter comparison. Its original scanned NSE financial statement contains
the missing comparisons. The new reviewed-image importer binds the original PDF,
embedded page image, current master ISIN, original PDF identity, NSE attachment
and publication clock, reporting basis, economic columns and units. It imports
the missing quarter without replacing earlier observations. EPS is recorded in
rupees per share, separately from amounts reported in lakh rupees.

The original PDF's consolidated table reports September-quarter revenue of
38,814.95 versus 22,526.45 lakh rupees, total PAT of 4,615.34 versus 3,002.71 lakh,
owner-attributable PAT of 4,843.09 versus 3,111.14 lakh, and diluted EPS of 1.32
versus 0.91. Basic comparative EPS of 0.96 is a different metric. The resulting
growth observations are 72.3083%, 53.7058%, 55.6693% and 45.0549%, respectively.
Neither total PAT nor reported EPS is labelled recurring earnings.

The NSE catalogue retains an old ISIN for this attachment. Identity is therefore
bound to the current ISIN printed on the original PDF and the current master;
the catalogue discrepancy remains in the audit. The current PDF amounts and EPS
must agree with the already captured current XBRL before import is accepted.

The V13 parser now optionally retains half-year cash flow in the `ytd` period
kind. Apollo's H1 FY27 operating cash flow is INR 224,373,000. It is not substituted
for annual cash flow. The previously captured FY26 operating cash flow of
INR -1,297,635,000 remains unchanged. A complete seven-input row can still fail
the annual cash-conversion safety rule.

Newly reviewed PDF facts and EPS comparisons use the later of actual review,
retrieval and publication timestamps. They are not backdated into old forecasts.
A same-head current financial reassessment using 11 November closing-price
features is labelled separately from the original source-day model output.
Models, registered cohorts and frozen forecasts remain unchanged.

Every normal V13 priority-research export must now provide both:

1. RSI strictly greater than 70.
2. RSI strictly greater than 50.

The second screen includes eligible shares above 70; these are overlapping
screens. Both use the same probabilities, financial and identity exclusions,
no nominal price band, and unchanged 70% six-month / 25% twelve-month joint
success / 5% optional 200-day-average ranking. Each list has a maximum of 25
names. Lowering RSI does not waive observed financial weakness or data conflicts.

Canonical outputs are `priority_research/latest_RSI70.json`,
`latest_RSI50.json`, `latest_dual.json` and corresponding Markdown tables.
`latest.json` remains the RSI >70 compatibility output. When current input or
an exact replay is unavailable, both canonical lists report the explicit blocked
status; an empty blocked list is not a market-wide no-opportunity conclusion.

Original sources:

- https://nsearchives.nseindia.com/corporate/APOLLO_24102026155137_AMSresults30092026.pdf
- https://nsearchives.nseindia.com/corporate/xbrl/INTEGRATED_FILING_INDAS_1745384_24102026051130_WEB.xml

These are research estimates, not validated investment probabilities. Correcting
data or producing two screens does not establish improved prediction accuracy.
