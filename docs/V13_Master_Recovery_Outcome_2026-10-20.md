# Completed collection and current master — 20 October 2026

[The three-pass original-filing collection completed successfully](https://github.com/ayushverma1988/v9-multibagger-calibration/actions/runs/39432733098).
The collector processed 114,036 financial fact observations across the three
passes. The deduplicated financial-history increase is 96,843 numeric facts.

| Original financial history | Before | After |
| --- | ---: | ---: |
| Issuers with observations | 2,874 | 2,876 |
| Numeric fact observations | 517,164 | 614,007 |
| Distinct source document hashes | 25,653 | 29,704 |
| Period / reporting basis / version rows | 50,429 | 60,345 |
| Issuers with all seven research checks complete | 1,280 | 1,280 |

These history totals use the audit's 19 October 2026, 16:30 UTC information
cutoff. The master source review took place on 20 October. More historical
data did not automatically complete the latest required checks.

## Latest master snapshot

- 3,200 public issuers; 27 tracked financial fields; 86,400 issuer/field cells.
- 64,789 current-section original values observed.
- 21,611 cells remain unobserved or ambiguous. Some fields may not be
  reported or applicable; they are not silently replaced with zero.
- 158 current-section values match an original and two additional provider
  families; 153 also meet the current-period freshness rule.
- 11,525 distinct comparison observations in the trailing three-year window.
- All 20 user-supplied earlier shares have three-source sector classifications.

The main current exclusions included a missing sector lookup for fourteen of
the earlier twenty. Correcting the classifications makes fourteen eligible
under the existing refined screen, with thirteen ranked in its current top
twenty. One eligible share ranks below the top twenty. Six fail the existing
RSI condition. This is a classification reassessment of the last completed
19 October model inputs and probabilities, not new financial inference.

The broader available-data twenty-share screen, the refined sector/size
screen and early chart/catalyst potentials remain separate outputs. Historical
membership does not force a current pass, and incomplete financials remain
explicitly labelled.

## Source limitations and daily continuation

The hosted additional-source pass accepted 2,389 NSE metadata responses and
54 Screener page parses. The older parser could accept an ownership-only
page; this has been corrected in the deployed adapter. Finology returned
403 on the hosted runner and Screener reached a rate limit. Those collectors
stopped rather than treating unsuccessful requests as complete data.

The independent reviewed twenty-share source packet and public web-table
reviews retain their actual observation clocks, reporting bases and URLs.
Raw-page hashes are recorded where original wire bytes were available; a
web-table review is explicitly not represented as an original-byte hash.

The weekday daily worker remains scheduled for 16:30 UTC / 22:00 IST and is
pinned on main to tested code `7880092a4c572186b05eb262b6c94b76dc1d00ed`.
It saves the missing-input audit before constructing model inputs and rotates
the collection queues across the public universe. The source/archive merge
is serialized with the original collector to preserve newer observations.

Validation: 655 contract tests passed. Workbook totals, all 86,400 input-cell
mappings, native chart export and archive integrity were checked. Model
weights and historical prospective forecasts were not rewritten. The data
recovery is incomplete; source agreement does not certify a prediction.
