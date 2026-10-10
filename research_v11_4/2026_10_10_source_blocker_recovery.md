# Financial, catalyst, exchange and independent evaluation improvements

The October 10 source recovery ran successfully against the October 9 market snapshot. It improves source coverage and evaluation controls; it does not approve the complete multibagger model. The original observations, scores and acceptance thresholds remain unchanged.

## Measured source improvements

The numeric pilot contains the 30 saved sleeve candidates and 12 deterministic source-only controls selected by SHA256 order from the mainboard master. It reads no outcomes or rankings from other model versions. Forty pilot companies are present in the original 2,101-row current scoring source cohort.

| Recovered field | Verified companies in 42-company pilot |
|---|---:|
| Annual operating cash flow | 35 |
| Promoter holding, reconciled between exchange index and ownership XBRL | 31 |
| Both receivables components and positive annual profit | 21 |
| Reserves and both borrowing components | 31 |
| Property, plant and equipment rising/falling year over year | 25 |
| Latest quarterly sales compared with year-ago quarter | 23 |
| Latest quarterly profit growth with a positive year-ago denominator | 18 |

XBRL monetary values are already rupees. The exchange's generated HTML can display raw rupees or scaled values. The new parser chooses one document-wide rendering interpretation supported by at least two nonzero labelled facts, then compares each value. It rejects conflicting scales, wrong periods, segmented facts, unsupported monetary units, conflicting issuer/basis and missing components. Quarter-only flows cannot become annual totals. A missing borrowing component cannot become zero. Negative/zero prior profit does not become an ordinary positive growth denominator.

Four previously completely missing non-chart common checks now have verified values for 35, 29, 21 and 31 companies respectively in the same 2,101-row source cohort. The 25 PPE-growth observations are reported separately: a component of fixed assets cannot silently pass a broader fixed-assets rule. That rule remains unknown. Most of the cohort remains unknown. The six chart/common checks and four family thresholds remain exactly as requested. All three financial/valuation families still have zero full passes because their complete inputs remain unavailable.

## Primary catalyst and actual promoter direction

The operational-document pilot reads 109 issuer-matched original attachments and extracts 11 quantified disclosure candidates. They require assessment of timing, conditions, materiality and earnings impact. They are not eleven verified complete transformations. Generic Updates filings are inspected rather than inferred to have no catalyst. Scanned documents requiring OCR stay unresolved.

The separate official insider-XBRL recovery verifies 47 actor transaction rows: 15 market purchases and 32 market sales, across 11 issuers. Share units, person category, current ISIN and pre/post-holding arithmetic are checked. ESOPs, gifts, pledges and preferential allotments are excluded from market purchases. A filing containing a buy and a corresponding promoter-group sale can have zero group accumulation; the group sum is recorded separately. Missing index records cannot prove zero buying. Complete insider-feed coverage is not established.

## NSE and BSE coverage

| Source/reference | Identifiers |
|---|---:|
| Current official NSE mainboard master | 2,603 |
| Current official NSE SME master | 579 |
| Official AMFI June 2026 reference | 5,427 |
| Union, deduplicated by security ISIN | 6,103 |
| AMFI reference with a BSE symbol | 5,107 |
| Original current source cohort with an official AMFI size classification | 2,088 / 2,101 |
| Classified small/mid reference in that source cohort | 1,996 |

These are security identifiers, not a certified count of current unique issuers. Reference-period and present-day exchange membership are reported separately. AMFI includes 2,954 rows with a BSE symbol and no NSE symbol at its reference period; this is not proof of current BSE-only listing. Twelve same-symbol changed-ISIN cases are not silently linked. SME identities are retained even though the long-history model does not score them and the AMFI size category is unavailable for those SME master matches.

Direct BSE transport returned HTTP 403. Thus the expanded reference is not a verified current BSE active master, price history or BSE-only prediction universe. AMFI's six-month average market cap is not inserted into a rule requiring current market cap. New listings and unsupported exchange/size histories remain visible source gaps.

## Verified invariant and independent evaluation

The enrichment was applied to the original 2,101-row current source cohort. Every original predictor value, and every actual raw and calibrated probability from the tested fixed 21-input research model, is identical before and after enrichment. No source-qualified new ranking or future hit-rate improvement is claimed.

The independent evaluation infrastructure now freezes the actual previously tested weight file and an explicit protocol, including code/source fingerprints, 126 actual market sessions, ten selections per predeclared view and the existing acceptance gates. Records cannot be overwritten; changed weights, late sources, invalid security identities, post-open/backfilled observations and outcome columns are rejected. Unknown selected outcomes remain in the denominator. Missing or infinite prices cannot become successes or failures. Dates from the already examined archive or recovered October 9 source are not new blind evidence.

The tested weight file SHA256 is `ee36de6e3efcacde47908405b0aa80d2b720767fb4bb59ce4dae196a470dfe37`. The private protocol is pinned by a public execution receipt. The observation ledger is intentionally empty: zero new prospective observations and zero mature blind outcomes. Infrastructure validation is not independent prediction validation. The full financial/catalyst recipe is still source-incomplete; changing the recipe requires a separate freeze.

## Remaining gaps

Full-universe financial recovery; exact five/seven-year histories, operating margins and return ratios; independently sourced current valuation/PEG; direct BSE current price/history coverage; SME/new-listing modelling; complete verified primary causal chains; full promoter disclosure coverage; historical point-in-time reconstruction; and mature prospective outcomes remain incomplete. The six-, twelve- and twenty-four-month models have not been replaced by invented probabilities. Production approval remains false.

## Reproduction

- `src/v11_4_source_blocker_recovery.py` collects and retains original raw evidence, hashes, publication times and first-retrieval clocks.
- `src/v11_4_enriched_current_source_audit.py` compares the same source cohort before/after and verifies actual probability invariance.
- `src/v11_4_verified_current_financials.py` and `src/v11_4_verified_promoter_transactions.py` implement the monetary and direction contracts.
- `src/v11_4_exchange_reference_universe.py` separates current masters and the dated AMFI reference.
- `src/v11_4_independent_evaluation_ledger.py` and `src/v11_4_evaluation_protocol_run.py` implement the immutable prospective protocol.
- `.github/workflows/v11_4_source_blocker_contract.yml` runs the full test suite and stores actual research weights/protocol privately without registering past observations.

Primary routes: [NSE financial integrated filings](https://www.nseindia.com/companies-listing/corporate-integrated-filing), [NSE trading security lists](https://www.nseindia.com/static/market-data/securities-available-for-trading), and [AMFI stock categorisation](https://www.amfiindia.com/otherdata/categorisation-of-stocks).
