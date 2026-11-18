# NSE source-domain recovery — 18 November 2026

The completed source-17-November worker reported ten catalogue errors, each
`Source redirected off host`. Its successful workflow status did not establish
successful financial, announcement or corporate-action collection. At
02:48 UTC on 18 November, independent public requests observed HTTP 301
redirects from `www.nseindia.com` to `www.nseindia.in`, preserving the API path
and query. New financial and announcement responses also link originals on
`nsearchives.nseindia.in`.

The transport now accepts only explicit `.com`/`.in` pairs for the NSE API and
archive hosts. It validates every redirect before following it, rejects changes
of source family, insecure schemes, user information, unexpected ports,
alias path/query changes, loops and excessive hops. New receipts retain the
requested URL, final URL, redirect chain, actual retrieval time and byte hash.
Existing source-cache bytes and retrieval clocks are read without rewriting.
Financial catalogue, catalyst, action-notice and reviewed-image PDF admission
accept the exact archive alias while retaining the original literal URL.

Nine focused tests and all 917 regression tests passed locally. An isolated
live collection using the existing four-year window, 17 November 2022 through
17 November 2026, returned 66,524 financial catalogue records and 6,037
announcement records with zero catalogue errors. Normalization retained
54,155 dated fiscal document versions across 3,103 symbols. These are catalogue
and document counts, not verified financial completeness or NSE scoring breadth.
A current NSE-hosted LIBAS XML downloaded through its literal `.in` URL and
parsed 34 numeric facts; no profitability or forecast certification follows.

Only the deployed source-code reference changes after validation. The worker
schedule, concurrency, budgets, fixed model heads, existing forecasts,
probabilities, cohort membership, private checkpoint and promotion requirements
are preserved. An automatic worker started by the existing workflow-path push
trigger must complete before any new operational coverage is claimed. Stale
provider histories and missing financial/project/action evidence remain separate
backlogs. No new forecast is inferred from this source-only probe.
