# Malformed-master and overnight source-date recovery

Worker 42298916255, started 7 November 2026 at 03:12 IST, stopped before history,
financial or catalyst collection. Pandas raised `ParserError: Expected 1 fields
in line 9, saw 2` while reading an NSE master. That exception is not a
`ValueError` in the deployed parser, so it bypassed the prior CSV-validation
retry handler. The original failed response body is not in the retained working
archive; its exact contents and which venue failed cannot be inferred from the
old traceback alone. In particular, an HTML error page is possible but unproven.

The retry handler now explicitly covers pandas ParserError and EmptyDataError.
Every reply has its URL, actual capture clock, length and hash recorded. Each
master still has at most three fresh requests, sharing the budget with transport
and identity-validation errors. Repeated malformed replies can use only an
independently preserved original for the requested source-price date. Parsed
identity conflicts prevent recovery. Explicit access/rate-limit HTML, even with
HTTP 200, and HTTP 401/403/404/429, TLS/proxy and integrity errors remain blocked.
Unexpected programming exceptions remain visible.

The former calendar-date expiry also prevented a valid 6 November capture from
resuming its own source day after midnight. Recovery is now limited to 24 hours
from the *first* original capture and still requires exact source-date, URL,
HTTP, hash, length and timezone-aware clock matches. A later re-verification does
not reset age. A 6 November capture cannot serve a 7 or 8 November source day.
Receipts distinguish the source date, current Indian date, original retrieval,
actual recovery clock and whether this is an overnight resume. Legacy
`same_day_original_recovery` means source-date match; the added fields make that
meaning explicit. The strict original master parser and cross-master identity
checks apply to recovered bytes before any universe replacement.

At 7 November 06:39 IST both original NSE endpoints returned valid CSVs: 2,630
main-board and 583 SME identities. Their bytes exactly match the independently
retained 6 November originals. The main-board original is added to the existing
source-6 recovery seed, with its original 6 November capture and actual 7
November HTTP/hash re-verification clocks. No clock is backdated. These captures
expire 24 hours after their respective first captures; they are not timeless
fallbacks or a claim that the source cannot fail again.

Tests reproduce malformed replies for both venues, bounded exhaustion, empty
bodies, explicit access HTML, mixed format/identity/transport errors, strict
recovered identity checks, overnight same-source resume, age expiry and unchanged
frozen files. A smoke test forces all six live requests to be malformed and
recovers all 3,213 original identities from the checked source-6 bytes. Forecasts,
heads, selected and unselected cohorts, baselines, registration/cutoff clocks,
special-session policy, schedules and collection budgets are unchanged.

Full test, exact-head CI, deployment and live-run results are recorded separately
in the monitoring audit. Code checks do not certify a completed live worker,
full financial/catalyst evidence, or prediction accuracy.
