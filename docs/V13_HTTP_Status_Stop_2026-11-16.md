# V13 collector HTTP-status stop correction — 16 November 2026

Worker 43516256947 completed successfully with source 13 November, but its
financial phase attempted only 1,905 of its maximum 4,000 filings. The only
error matching the old access-stop substring check was an actual HTTP 404:

`https://nsearchives.nseindia.com/corporate/xbrl/INTEGRATED_FILING_INDAS_1500429_01082025060000_WEB.xml`

The filename contains `429`. Both the bounded queue and the health summary
searched the whole error string, including the URL, and incorrectly interpreted
that filename as a provider rate-limit response. There was no actual HTTP
403/429 error in that financial receipt. The historical receipt is preserved;
it is not rewritten to conceal the original diagnostic error.

The correction shares one access-stop classifier between collection and health.
It prefers an explicit HTTP status, otherwise parses a requests/urllib status
prefix after removing URLs. Real HTTP 403/429 and explicit provider access-stop
reasons still stop submission. Ordinary 404s, short histories and identity
failures remain errors/backlog items, but URL digits cannot abandon the rest
of a queue. The existing worker limit, launch spacing, budgets and schedules
are unchanged. Missing sources do not become accepted facts or zero values.

Regression coverage reproduces the exact missing-original filename, requires
the remaining queue to run, checks typed status precedence, and verifies that
genuine 403/429 stops still abandon queued work. No forecast, model head,
probability, selected/unselected cohort, baseline, information cutoff, private
checkpoint or promotion gate is modified. A tested deployment and completed
worker audit are required before claiming any evidence-coverage improvement.
