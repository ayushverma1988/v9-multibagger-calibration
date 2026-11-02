# V13 targeted capture acceleration — 2 November 2026

The source-30 October snapshot has 715 current public price-eligible rows:
472 have all seven financial checks, 78 have the required raw facts but an
undefined ratio with an observed nonpositive base, and 165 need missing or
ambiguous facts. Wider downloads alone must not turn the latter two categories
into invented growth, profitability or completed catalyst proof.

## Implemented behavior

* Basic current financial bundles run first and reserve half the filing slots.
  Current unranked issuers without an observed seven-check failure precede
  other missing-input issuers. Owner earnings reserve 20%, preferred reporting
  basis recovery 15%, and the remaining capacity rotates public-universe work.
  Unused slots are filled; no selection, probability, realised return or private
  ranked list enters request priority.
* Before/after classifications use the same public source-date cohort and exact
  current-master ISINs. Missing facts, observed nonpositive denominators and
  observed check failures stay distinct. The audit reports newly incomplete
  rows as well as recoveries. `evidence/financial_capture_acceleration.json`
  retains the field queue and the fixed-cohort comparison.
* Bounded collection supports up to eight in-flight workers; deployment uses
  six. The existing 250 ms launch spacing and per-store host spacing remain,
  and a 403/429 stops new submissions. In-flight requests can finish; this is
  not a promise of higher provider request-rate limits.
* A definite wrong instrument/identity or HTTP 404 skips redundant full-history
  variants. Stale/short histories, missing accepted sessions and timeouts still
  get the existing bounded variants. No raw official price is appended into an
  adjusted provider series, and no accepted session/identity guard is relaxed.
* Cached catalyst-field extraction considers up to 12,000 unique originals
  instead of 5,000. This removes the former cap at the current inventory size.
  Candidate fields still require source hash, text hash and issuer checks;
  a text candidate does not certify project execution, profit or net 2×.

## Operational configuration

The separate default-branch workflow update will pin only the tested revision:

| Parameter | Previous | Revised |
|---|---:|---:|
| History security budget | 3,200 | 3,200 |
| Filing-version budget | 2,400 | 4,000 |
| Financial network phase budget | 900 s | 1,500 s |
| Catalyst download budget | 360 | 600 |
| Catalyst network phase budget | 360 s | 600 s |
| In-flight source workers | 4 | 6 |
| Cached catalyst extraction cap | 5,000 | 12,000 |

These are bounds, not claims of completed coverage or proportional speedup.
At 4,000 versions/pass, 16,516 pending catalogue versions need at least five
ideal passes for one attempt each; new filings, unavailable sources and reviews
can require more. The 180-minute workflow ceiling and existing schedule remain.

## Validation and release limits

Full local suite: **742 tests passed** (731 existing + 11 new), with the frozen
runtime dependency versions. Tests cover real financial readiness/ISIN/date
filtering, nonpositive bases, fixed-cohort recoveries and regressions, request
priority, six simultaneous jobs, access-limit stopping and terminal-history
request suppression. GitHub exact-commit CI and live recovery are separate
release checks; their receipts must be recorded before claiming deployment or
increased verified coverage.

No model head, probability, selection rule, baseline, old registry, entry rule
or promotion gate is changed. BSE is optional. Source-supported catalyst and
net-2× evidence remain unknown where required fields are absent. Independent
six/twelve-month evaluation still requires elapsed market time. The source-30
cohort cannot be enlarged or rescored, even if this batch recovers more inputs.
