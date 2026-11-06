# Reviewed NSE special-session calendar

The current holiday master labels 8 November 2026 as a Sunday holiday, while
NSE/CMTR/76941, published 19 October 2026, explicitly schedules a live cash-market
Muhurat session that day. The original was downloaded on 6 November at
16:25:20.328716 UTC and reviewed later that day. Its exact publication time is
unknown. The reviewed PDF, hash, exact excerpts, capture and review clocks are
retained in `v13_evidence_seed/trading_session_reviews.json`.

The source specifies a normal-market open at 14:20 IST. Continuous trading for
non-CAS securities ends at 15:30; the completion guard conservatively waits
until the circular's final trade-modification cut-off at 16:15. This guard is
not a claim that continuous trading lasts until 16:15. IPO/relisted and illiquid
auction schedules remain distinct; an independently reconciled original price
and matching provider session are still needed for every entry.

The calendar now includes this reviewed Sunday, uses its actual opening clock,
and attempts its official price file when complete. A missing Sunday provider
bar or original price cannot certify Monday as the first eligible session.
Existing entry caches are rechecked against that same calendar and opening
clock. The primary PDF is copied into the action-audit raw originals, but its
presence never certifies a security's prices, action interval or returns.

The frozen entry contract remains
`FIRST_VERIFIED_NSE_SESSION_OPEN_STRICTLY_AFTER_REGISTRATION`. No registry,
head, probability, selected/unselected flag, baseline, or registration/cutoff
clock is rewritten. This later calendar review is evaluation evidence, not a
retroactively available forecasting feature. As of 6 November, no 8 November
entry or outcome is certified in advance. Earlier October/November entries are
unchanged by the future session.

The collector transport recovery remains a separate, current-day-only rule.
The 6 November SME-master capture does not become a timeless fallback.

Validation: focused cases cover the original hash/length/URL, source excerpts,
date and clocks, holiday override, weekend collection, unfinished-session
guard, before/after-open registration, missing Sunday observations, entry-only
verification and unchanged registry bytes. The full regression result is
recorded with the release audit before any operational pin changes.
