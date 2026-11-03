# Missing financial input recovery — 21 October 2026

The validation requirement is now one additional independent provider, or two
providers in total when the original source has no usable value. Period,
reporting basis, issuer identity, currency, definitions and rounding still
have to match. A corroborated substitute is stored separately from originals.

The audited snapshot uses market data dated **19 October 2026** and a source
review clock of **21 October 2026, 07:09:57 UTC (12:39:57 IST)**.

| Measure | Audited snapshot |
| --- | ---: |
| Public issuers | 3,200 |
| Tracked financial fields per issuer | 27 |
| Required issuer/field cells | 86,400 |
| Original source values retained | 64,789 |
| Original gaps before alternative-source recovery | 21,611 |
| Missing values recovered from two providers, all observed periods | 225 |
| Recovered values meeting freshness requirements | 210 |
| Remaining missing or unresolved values | 21,386 |
| Current cells corroborated by at least two sources total | 17,392 |
| Current cells meeting the former three-source requirement | 7,312 |

Verification of an already known original value is different from recovering
a missing value. The 17,392 corroborated cells include the 210 current recovered
values. Fifteen further recovered values refer to older periods and are excluded
from current validated inputs. Unreported, ambiguous or incompatible values
remain blank; none were invented or replaced with zero.

## Sources and daily collection

- The user's suggested Indian Stock Market API was tested successfully using
  upstream commit `147cd5c92a7888104e0be63eef7ea72a5b61ad3e`. Its implementation
  provides quote, valuation and sector fields. It is MIT-licensed and uses
  Yahoo Finance; it is not an MIT research or institutional data product.
- Dated income-statement, balance-sheet and cash-flow lines use the public
  Yahoo statement endpoint. Yahoo's wrapper, Yahoo pages and yfinance are one
  source family and cannot corroborate each other.
- Screener, Accord/Finology and original exchange/company statements provide
  the other financial observations. Standalone Net Profit can identify PAT;
  consolidated owner-versus-total profit remains unresolved without evidence.
- Yahoo reporting basis is never assumed. Exact name/ticker/currency checks
  and multiple original-fact anchors establish a unique reporting basis.
- A provider's newer isolated line cannot erase the rest of a dated original
  statement section. Conflicts and ambiguous owner/EPS/debt definitions remain
  visible in the source ledger.
- The weekday worker remains scheduled for **22:00 IST**. It saves the master
  before analysis, collects the new sources, and publishes two-source coverage.
- A separate whole-universe recovery run is active:
  https://github.com/ayushverma1988/v9-multibagger-calibration/actions/runs/39552051031

The initial current-section source packet retains **40,796 numeric provider
observations** with URLs, hashes, identity and information clocks. Its decoded
JSON SHA-256 is
`02182d6bce6f63f0b27aa43778265282c35de0819057aa266470f86fe8a3900f`.
A supplementary numeric packet retains later observations without duplicating the initial packet.
Full collected statement observations are retained in the downloadable source
data package. Provider access limits stop collection from that host.

## Validation and forecasts

The initial implementation passed 666 regression tests and the GitHub learning
contract. Subsequent standalone-profit and source changes passed 217 V13 tests.
The master audit covers all 3,200 issuers independently of private model ranks.

The earlier twenty-name comparison and broader screen remain separate. This
data-recovery snapshot does not retrain the model, replace frozen forecasts or
issue a new validated stock prediction. Source agreement is corroboration,
not proof of future returns. Recovery remains incomplete.

Upstream API: https://github.com/0xramm/Indian-Stock-Market-API
Yahoo financial statement interface: https://ranaroussi.github.io/yfinance/reference/yfinance.financials.html
