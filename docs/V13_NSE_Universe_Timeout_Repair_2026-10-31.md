# Bounded retries for the public NSE universe refresh

The 30 October collector (run 41127422984) stopped after 4 minutes 31 seconds
because the SME master download timed out after 25 seconds. The main-board
master had already passed identity checks. Checkpoint restoration and the
final save succeeded, and the frozen guard passed for all 7,836 ledger records
and 37 registered immutable files. An independent comparison against the
previous source-29 audit confirms that the ledger, both heads, active metadata,
both watchlists and all twelve pair files are unchanged: eighteen exported
files compared. No source-30 forecast was produced by the failed run.

The deployed `universe` function retried invalid master contents but immediately
aborted on a network exception. The repair shares the existing three-attempt
budget between temporary fetch failures and invalid replies. Read/connect
timeouts, connection resets and HTTP 408/500/502/503/504 can retry, with one-
then two-second bounded waits. Each attempt uses a distinct cache directory
for that URL, preserving original responses and retrieval clocks. The audit
records failed attempts before waiting and records whether another attempt
will occur. General source collectors keep their existing request budgets.

HTTP 401/403/404/429, TLS and proxy errors, cache-integrity failures and unknown
exceptions are not retried. Exhausted retries stop collection. Neither a stale
master nor an incomplete one is promoted as current. The accepted universe is
replaced atomically only after both original masters and cross-master identities
pass. Existing history, probability, selection and promotion rules are unchanged.

Validation covers recovery at either endpoint, shared fetch/identity budgets,
permanent failures, finite transient HTTP retries, and preservation of prior
universe/frozen files on SME exhaustion. The full regression suite and the exact
branch CI must pass before the operational worker is repinned. Adding the repair
branch to the existing contract push trigger runs CI on the actual fix commit;
no recurring collection schedule is changed.

The public-only live probe on 31 October retrieved both original masters:
2,620 main-board rows and 581 SME rows, 3,201 distinct ISINs in total. These are
catalogue identities, not 3,201 model-eligible histories or complete financials.
The last audited source-29 model coverage remains 654/3,201 until a completed
worker supplies new dated evidence. Independent six-/twelve-month validation
still requires future market time.
