# V13 dated comparison completion repair — 13 October 2026

The completion queue used to require a recent pending filing. Once that
filing had been parsed, the same issuer's older missing comparison periods
could leave the completion queue. An observed, available, recent quarter now
anchors completion even when only older filings remain pending. Stale and
not-yet-available quarters cannot anchor a bundle, and one reporting basis
is preserved. Public fresh-price coverage remains the collection priority;
forecast ranking and future returns are not used to select requests.

The first deployed run completed successfully but exposed a second queue
issue: accepted filings awaiting a parser upgrade consumed completion slots
ahead of missing economic periods. Completion now targets absent revenue/PAT
quarters and absent annual/balance facts, rather than parser-version status
alone. A nine-month upgrade remains eligible where it is needed to reconstruct
an absent balancing quarter. Broader parser upgrades continue in the rotated
remainder. A known zero/negative denominator is not fabricated into a ratio.

Live validation also distinguished raw-price coverage from the actual public
market-eligible population. Completion now prioritises every stock passing
the existing public price, liquidity and history eligibility checks before
other fresh price matches. This includes unselected names and does not read
model probabilities, rankings or future labels. Raw matches and wider public
coverage remain in the rotated collection queue.

The last live check still lacked March/June 2025 originals that were present
in an earlier captured primary NSE catalogue. A hash-checked recovery seed
now restores 8,365 public catalogue rows across 2,681 issuers for these fiscal
periods. It contains source URLs/publication metadata, not numeric financial
facts, model forecasts or outcomes. Current versions are retained and the
four-year boundary is enforced. The collector must retrieve the originals
and pass the existing identity/basis/unit/hash checks before adding facts.
Recovery is idempotent and survives a thinner later source response.

A bounded public-source engineering probe retrieved 300 older originals
for 62 issuers in the unranked 163-stock price-matched population. Every
request was accepted with exact issuer/basis checks and original source
hashes. Required financial completeness increased from zero to 42; 24 of
those passed the existing seven-field quality conditions. This is a source
and parser check, not a new forecast, live-worker receipt, or unseen accuracy
evaluation. Registered probabilities and cohorts are unchanged.

Catalyst numeric reviews now require the numeric quantity and explicit
currency scale to match an exact original excerpt. Percentage fields need
an explicit percentage unit. Original issued share counts must reconcile
with the new allotment before derived dilution arithmetic is exported.
Share-count arithmetic never becomes a provider price adjustment factor.

Two additional original NSE disclosures were reviewed:

- SWASTIKAIN: received LOA for INR 6,157,600,000 including taxes, with a
  24-month execution period from the LOA date. Start date, phased twelve-month
  value, margins and sufficient funding remain unknown. The full order is
  not included as a twelve-month executable value.
- NAZARA: issued shares increased from 384,696,024 to 408,577,735 through
  23,881,711 preferentially allotted shares. Arithmetic gives 6.207943%
  share-count growth and 5.845084% existing-holder ownership dilution.
  Listing/trading approval remains pending. This is not weighted diluted EPS,
  a split/bonus, or proof of secured subsidiary funding or acquisition profit.

Review availability starts at actual source review, not the older publication
clock. Frozen forecasts remain immutable. Six/twelve-month fair evaluation,
all unselected observations, versioned policy contracts, monthly research
refits and independently verified future-only promotion gates remain intact.
