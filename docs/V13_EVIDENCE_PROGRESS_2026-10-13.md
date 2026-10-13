# V13 evidence progress — 13 October 2026

The [completed worker 38370447460](https://github.com/ayushverma1988/v9-multibagger-calibration/actions/runs/38370447460) succeeded. Computation finished **13 October 2026, 14:20:14 IST**; aggregate health/artifact publication completed at **14:20:24 IST**. Source prices remain the **12 October 2026 NSE close**, not live 13 October quotes. The default worker is pinned to [a6f92a510c5b784c561f6e7794a948d231fc37d6](https://github.com/ayushverma1988/v9-multibagger-calibration/commit/a6f92a510c5b784c561f6e7794a948d231fc37d6) through research PR 14.

**493 local regression tests passed**, and all four GitHub contracts passed: 38370450853, 38370450731, 38370450667, 38370444348. No fixed model head was promoted or replaced.

## Verified completed-run results

| Evidence | Completed worker result |
|---|---|
| Financial-input completeness | **62 / 164 current price-matched stocks** have all seven required numerical inputs; the preceding run had three complete sets |
| Financial quality | Completeness is separate from profitability, cash-conversion, leverage and interest-cover checks; no claim that all 62 pass quality or constitute recommendations |
| Public inventory / current research coverage | **3,186 / 164**; partial NSE coverage |
| Original financial filings | **4,972** accepted filing versions; 400 requests, 7,311 new facts and two unresolved issuer-symbol mismatches in the final batch |
| Recovered earlier source inventory | 8,365 public March/June 2025 catalogue records across 2,681 issuers; **7,152 newly recovered versions**, 56,778 retained versions |
| Remaining financial queue | **49,995** versions; increased inventory reflects recovery of previously omitted originals |
| Completion priority | 300 completion requests, including **248** for the entire unranked public market-eligible population of 143 names before the history refresh |
| Stored usable histories | **2,276**; 271 usable / 400 attempted in this batch, with 129 unavailable, insufficient or rejected-identity histories |
| Catalyst inventory | **399** original issuer-matched documents, 101 extraction/review candidates, six accepted source-bound reviews |
| Execution / valuation support | Zero current fully funded, approved, positive-margin execution chains; zero fully source-bound forward earnings/sector-valuation scenarios |
| Paired registration | Same **82 stocks per model**, including unselected stocks; original source date, input hash, registration time and registry hashes reused |
| Full observation ledger | **834** records: 670 TRACKING and 164 UNKNOWN_ENTRY; **145** independently verified entries |
| Performance evidence | Zero confirmed early doubles, zero raw-price doubling alerts and zero independently verified mature outcomes |
| Model approval | **False**; promotion remains rejected for source/action coverage and insufficient mature future evidence |

The seven completeness fields are four profitable-quarter observations, current sales/PAT year-on-year growth, PAT margin, annual CFO/PAT, dated debt/equity and annual EBIT-proxy interest cover. A zero profitability flag or negative ratio can be observed data: completeness alone is not a quality pass. Undefined ratios and absent facts remain unknown.

## Repairs made

1. An already parsed, available, recent quarter now anchors collection of older missing comparison periods. Issuers no longer disappear from completion simply because their newest filing has been parsed.
2. Completion requests target missing economic facts before redundant parser-version upgrades. Nine-month inputs remain eligible when needed for source-bound annual-minus-nine-month balancing quarters; EPS, share counts and cash flows are not subtracted.
3. The full unranked population passing the existing public price, liquidity and history checks is prioritised before other raw-price matches. It includes unselected names and does not use model probabilities, prediction ranks or future outcomes.
4. Earlier captured primary NSE catalogue metadata restores missing March/June 2025 originals when current dynamic responses omit them. Recovery is hash checked, idempotent and clipped to the trailing four-year window. It contains source URLs/publication metadata, not copied financial numbers, forecasts or outcomes. Originals are retrieved and validated for exact issuer, reporting basis, period, unit and source hash before facts are added.
5. Quantified catalyst values must match an exact original excerpt and its currency scale. Percentage units are explicit. Original issued share counts must reconcile with allotted shares before derived dilution arithmetic is exported. Share-count arithmetic does not become a provider price-adjustment factor.

A separate earlier 300-original engineering probe completed 42 of its fixed 163-stock input population, with 24 passing the basic financial-quality checks. Those are **probe results**, not a quality count for the final live run, a candidate list, or independent prediction validation. The live source inventory and population were explicitly checked rather than substituted with those probe counts.

## New quantified source facts

| Issuer | Original publication, IST | Verified fact and restriction |
|---|---|---|
| SWASTIKAIN | 7 October 2026, 10:29:06 | [Original NSE LOA disclosure](https://nsearchives.nseindia.com/corporate/SWASTIKAINFRA_07102026102824_Disclosure_of_Order_win.pdf): ₹615.76 crore including taxes, with a 24-month execution term from the LOA date. Exact start, phased twelve-month value, margins and sufficient funding remain unknown; full order value is not certified due within twelve months |
| NAZARA | 12 October 2026, 18:46:42 | [Original NSE allotment notice](https://nsearchives.nseindia.com/corporate/NAZARA_12102026184557_BMOutcomeFinal.pdf): 23,881,711 preferential shares, issued shares 384,696,024 → 408,577,735. Arithmetic gives 6.207943% share-count growth and 5.845084% existing-holder ownership dilution. Listing/trading approval remained pending; planned subsidiary investment is not proof of secured funding or acquisition profit |

Original SHA-256 values: SWASTIKAIN `5c70f0aac068ba5cdbd32b2f4afd6ac3196d51f6336007f26545ebd5acf65125`; NAZARA `f8af6246b57b82d938ae48abbcc79456a502e2b667275869be8880c73aa35655`. These originals were inspected and matched before the workspace disconnected. Accepted review availability is **13 October at 13:04:31–13:04:32 IST**, not backdated to publication.

Earlier accepted JNPR, ARSSINFRA, TATAPOWER and AEROFLEX reviews remain. Physical commissioning, pending certificates, unknown contract starts, subsidiary/third-party scope, unknown margins and unknown funding remain distinct. No announced intent is treated as completed capacity, recognised revenue or profit.

At unchanged valuation, Nazara's reported share-count increase would require approximately 114.55% growth in profit attributable to common shareholders to support a net 2× relative to a pre-allotment share-count baseline after the registered 0.5% cost per side. This is conditional arithmetic, not an earnings forecast, price target or probability. No ≥30% price-loss event is inferred from the allotment.

## Fair comparison and remaining evidence

The newest frozen paired forecast remains sourced to **12 October**, registered **2026-10-12T17:15:26.499188+00:00 / 22:45:26.499188 IST**. Its intended entry is the next verified session open on **13 October**, rather than the preceding close. The final job ran before that session finished; completed primary 13 October prices were not yet available to it.

The price range remains **₹20–₹2,000**, NSE coverage remains partial, and verified/discovery tiers remain separate. Later-collected financial inputs do not retrospectively change an old forecast's tier or probabilities. The latest registered selection remains the previously published discovery cohort; this run reused the old registries and did not register an invented revised forecast.

Both horizons are measured by frozen financial tier, observed-cohort market breadth and liquidity, using complete cohorts and unselected names. Immature Brier, precision and paired gains remain null. Changed input/selection-policy contracts cannot be pooled for promotion. Older unpaired forecasts are not given retrospective challenger predictions.

Monthly experimental refits and small predeclared market, market+financial and market+financial+catalyst comparisons against a maturity-purged prior-rate baseline continue inside the trailing four-year boundary. Previously examined historical diagnostics remain development evidence. The formal champion and challenger stay fixed; replacement requires independently verified future paired outcomes and every promotion gate. The stricter evidence policy can select zero stocks.

Remaining gaps include 102 current stocks without all seven numerical inputs, recurring diluted-EPS definitions, source-bound forward earnings/sector valuation, quantified executable profit/funding, bank-specific definitions, independently verified identity transitions, complete corporate-action intervals and genuine six/twelve-month maturity. Missing data is not interpreted as profitability. BSE is optional. No trades are executed.

## Audit trail and exact local blocker

The following remote results were checked: default worker pin, all four successful CI checks, successful worker/job steps, full worker JSON in the completed job log, aggregate health, reused model/cohort/registry hashes and artifact metadata. Three preceding completed artifacts were downloaded and compared locally: all 834 frozen probability/selection records and both watchlists matched their earlier successful versions.

The final artifact is **11799521587**, with GitHub-reported ZIP SHA-256 `a6523fe12418775305e2c51194400bb370ada0f615837ea42380bc8e37c7de5f`. **The final ZIP has not been independently re-hashed and its checkpoint receipt/manifest has not been compared locally.** The workspace disconnected with `409 Conflict, environment_offline: Environment is not connected`. That also blocks replacement of the original downloadable monitoring report; its last saved version remains unchanged. This Git-backed progress report preserves the new findings without claiming those pending checks succeeded.

Frozen paired identifiers reported by the successful worker:
- Cohort: `88ffeb6e860a66a5d5df1040940fc74a2f0ab2aa829a9dad7688473552f7bfa8`
- Input: `9c37dadbf83e4293581d99d9b09d083043d4d52f51e6e5783a84e108d6f2c129`
- Champion model: `c9208f89c46d043ee1d74e44dee1c85b9721b7a094e40a0fabf2c5506c246571`
- Challenger model: `f6facda12b9fcc757a15cb9e7cdd54b84ebaabe3dff0213b313f3f6d6471f291`
- Champion registry: `aef769b131a81c90c099406044ea22a581c666fb466a20b4183b6d6dfe58985b.json`
- Challenger registry: `4d7ce6938176ed85d49bfc3a58e6f8d88d5cb8b212d71d305bb8bb0f3b1d8973.json`

Public recovery subset SHA-256: `8ce2f124ddd03797d9c26ed2e3782b8ca33de8b095a84f717e4ff13a862d6e15`; original earlier full normalized catalogue SHA-256: `9782c9bacd9433e8b054d04b68934eeb27830427948ea5340d1123c05e0a23fb`. Primary 12 October bhavcopy SHA-256: `818f673b3cdb96ae8ec4e466c7c4fd3a28d035482173a675dd24f9437a3ecd99`.

**Financial data coverage has materially improved; prediction accuracy is not established.** The deployed collector continues independently of the disconnected local workspace.
