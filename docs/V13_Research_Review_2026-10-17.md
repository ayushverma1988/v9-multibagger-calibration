# V13 research review — 17 October 2026

## Material finding and correction

The scheduled worker run `38956446692` completed on 16 October 2026 at
20:02:05 UTC and its artifact ZIP matches GitHub's published SHA-256
`72ee08456f64285679f4402762175231d796efc4702af7a7c7f4dd5e36aedafb`.
The private checkpoint receipt identifies snapshot
`20261016T200154Z_4beb5a732c42` and manifest
`4beb5a732c426ed9ad340ddbef32dcefa09dd33ccc4c2bc0719cd2589ae51e19`.
Frozen integrity passed for all 1,898 existing forecast records, 13 immutable
registered files and the fixed heads.

The run was nevertheless a coverage-quality failure: only 3 of 3,185 public
NSE listings had same-source-date, independently matched market inputs. The
funnel records 2,276 stored histories without a 16 October provider bar. The
worker registered a three-stock paired cohort even though the recent committed
cohorts contained 229, 228 and 3 stocks. That cohort is retained unchanged as
an immutable observation, but it must not be interpreted as a breadth advance
or promotion-quality comparison.

A prospective integrity guard now blocks a new paired registration when its
verified row count is below 50% of the median of the three latest committed
cohorts, with a 20-row floor. On the audited artifact the threshold is 114 rows,
so 3 rows are blocked. The guard changes neither probabilities, selection
rules, model heads, old registries nor outcomes. The first-ever paired cohort
may still establish its baseline. The collector's `source_rate_limited` field
now reports an observed 429/403 condition instead of always returning true.

The operational price range remains **₹20–₹2,000** in the pinned worker. The
17 October no-price-filter experiment is confined to draft PR 17 and is not
deployed by the default-branch worker. PR 14 remains open and draft; its branch
is unchanged. Promotion remains false.

## Six-area evidence coverage

Source date: **16 October 2026**. These bars measure evidence coverage, not
engineering completion or predictive accuracy. Denominators are intentionally
different and must not be averaged.

| Evidence area | Exact coverage | Percent | Bar |
|---|---:|---:|---|
| Financial complete rows / current verified market rows | 1/3 | 33.33% | `███████░░░░░░░░░░░░░` |
| Execution-chain supported rows / current verified market rows | 0/3 | 0.00% | `░░░░░░░░░░░░░░░░░░░░` |
| Source-bound net-2× scenario rows / current verified market rows | 0/3 | 0.00% | `░░░░░░░░░░░░░░░░░░░░` |
| Independently verified entries / immutable ledger records | 1,434/1,904 | 75.32% | `███████████████░░░░░` |
| Independently verified mature 12-month outcomes / immutable ledger records | 0/1,904 | 0.00% | `░░░░░░░░░░░░░░░░░░░░` |
| Current verified market rows / public NSE universe | 3/3,185 | 0.09% | `░░░░░░░░░░░░░░░░░░░░` |

## Engineering, evidence and timing

| Item | Tested/deployed state | Qualified ETA or limitation |
|---|---|---|
| Collapsed-cohort registration guard | 37 focused and 572 full local tests pass | GitHub checks required before operational repin; normally one CI cycle |
| Provider request-state reporting | Corrected to observed 429/403 evidence | Same CI dependency |
| Official NSE price cross-check | 2,949/2,949 overlaps; zero close difference on 16 October | Complete for that source date only |
| Dated recurring financial inputs | 1/3 complete in the collapsed 16 October scoring frame | Source availability and rotating queues prevent a guaranteed completion date |
| Quantified execution chains | 0/3 | Requires compatible original evidence for the same project/order identity |
| Source-supported net-2× cases | 0/3 | Reported P/E and hypothetical hurdles do not satisfy this gate |
| Entry/action reconciliation | 1,434/1,904 entries verified | Remaining action intervals and identity transitions stay unknown |
| Mature unseen outcomes | 0/1,904 | Requires elapsed market time from each actual entry; software cannot accelerate 6/12-month maturity |

## Preserved boundaries

- NSE-first collection, the four-year history boundary, verified/discovery
  tiers, known-failure exclusions, 95% research risk ceiling, 95% upside/5%
  inverse-risk ranking and the ₹20–₹2,000 price range remain unchanged.
- Champion and formal challenger heads, probabilities, baselines, selected and
  unselected cohorts, private checkpoints and old registries remain immutable.
- The 16 October three-stock cohort is not deleted or enlarged. Future runs can
  be blocked from repeating the failure, but later evidence cannot be backdated.
- No Brier, PR-AUC, precision or calibration approval is asserted: there are no
  independently verified mature future outcomes. Source/action reconciliation
  is incomplete and `accuracy_approved` remains false.
- The daily collector schedule remains active. Run 38956446692 was delayed but
  completed; the material problem was source-date coverage, not checkpoint loss.
