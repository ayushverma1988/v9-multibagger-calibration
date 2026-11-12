# V13 chart null-rank repair — 12 November 2026

Chart workflow 43132457456 failed after verifying the completed collector
artifact and independently restoring the private checkpoint. The source-11
snapshot had three scored rows, no research selections and an entirely null
`research_rank`. JSON reconstructed that column as `object`, whereas CSV
reloaded it as `float64`. The exact protected-field comparison therefore
stopped on dtype despite identical null positions and forecast values.

`verify_forecast_export` now canonicalises only the nullable rank column to
Pandas `Float64` on comparison copies. It continues to compare issuer identity,
probabilities, selection, ranks and null locations exactly. It does not disable
dtype checks for the other protected fields, rewrite snapshots, run a model or
register forecasts. Invalid nonnumeric ranks and changed values still fail.

Validation includes the actual failed source and CSV bytes, all-null and mixed
rank round trips, reordered rows, and deliberate changes to every protected
field. The old assertion reproduces the reported error on the original artifact;
the corrected comparison passes without modifying those bytes. All 25 focused
chart checks pass. All 882 full-suite tests pass before the diagnostics pin is
changed. An initial local full-suite run could not find the legacy screener
configuration because the checkout was sparse; adding the unchanged tracked
`config/` directory resolved those fixture errors without modifying code.

The associated collector run 43115712102 completed successfully, but source-11
coverage was degraded: 2,294 of 2,321 saved histories had no source-date bar;
only three rows passed current source reconciliation. The existing breadth guard
blocked formal registration (minimum 337 rows), preserving the source-9 cohort,
both fixed heads and both watchlists. This repair does not relax that guard or
count stale data as current. A later independent public-provider probe received
HTTP 429 responses, so source freshness remains a collection dependency.

All 13,152 prior source-bound entry receipts remain present and unchanged; six
receipts were added, making 13,158. Entry evidence does not certify later
corporate-action intervals or returns. No promotion, changed probability,
source-verified early doubling or mature future outcome is claimed.
