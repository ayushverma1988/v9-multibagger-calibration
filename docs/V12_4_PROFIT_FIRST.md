# Profit-first selection experiment — not promoted

The prior research table ranked upside scores without requiring financial completeness. It could therefore display loss-making or financially unverified issuers. The strict V12.3 selector already rejected them, but that distinction is unsuitable for a profit-qualified candidate list.

V12.4 adds source-backed evidence of four consecutive positive reported quarterly PAT values and positive annual PAT/CFO, on one reporting basis, using only facts available before each decision. Missing, stale, conflicting or wrong-identity evidence blocks admission. Existing positive growth, cash conversion and debt checks remain. RSI is a model variable rather than a compulsory >=70 entry rule. Price, trend and excessive-run-up checks remain. A matched momentum comparison and the old downside veto are reported separately. No old weights, probabilities, registrations or files change.

This is a selector experiment, not a newly trained predictor. A company can satisfy past-profit checks and still make future losses. Reported PAT does not establish recurring earnings; financial-sector cash/debt interpretation and exceptional-item reconciliation remain unresolved. No output is approved for investment.

## Fixed run results

The protocol and code hash were recorded before running this experiment. The historical period has already been examined, so these are development results, not an unseen test. There was no search over thresholds after the run.

| Variant | Six months | Twelve months |
|---|---:|---:|
| Profit-first, rank inherited doubling score, up to 5 | 2/47 (4.3%) | 0/20 (0%) |
| Same pool, rank relative strength, up to 5 | 2/47 (4.3%) | 0/20 (0%) |
| Profit-first plus legacy loss score <=45% | 0/37 (0%) | 0/19 (0%) |

There are 22 six-month calendar decision dates, with 11 selecting for the first two variants, and four twelve-month dates. Selected labels in this run are all known. Repeated issuers and overlapping horizons are not independent trials. These results do not demonstrate accuracy improvement. On the twelve-month profit-first selections, 10/20 experienced the model-defined 30% pre-target loss event, illustrating that profitable businesses can still have losing shares.

Current accounting proof across 639 NSE rows: 56 pass the reported-profit evidence, 30 have an observed failure and 553 remain unknown. Only three pass the further growth/debt/cash/market gate. None passes the old downside veto. The BSE current sample remains blocked by missing dated financial proof. No private stock names are included in this public aggregate note.

## Loss-score interpretation

`p_loss30_12` estimates an adverse price move of at least 30% before a 2x closing target or the twelve-month deadline, based on adjusted-price research proxies. It does not predict accounting losses or represent a percentage expected return. A stock can first fall and later double. Do not compute a supposed joint probability as `p_double * (1-p_loss)` without a defensible dependence model.

V12.3 `enriched_top10` has zero loss-score ranking weight. `enriched_selective_max5` applies a hard <=45% veto, not a weighted penalty. The 70% logistic / 30% gradient-boosting blend is the learner mixture within each prediction head; it is not a risk-versus-return allocation. Feature weights are learned rather than manually assigned financial/technical percentages.

The historical NSE risk head averages 37.9% predicted versus 48.9% observed over 5,000 rows; BSE averages 48.7% versus 68.6% over 102. Current high risk scores are not calibrated forecasts of those exact loss frequencies. Replacing a weakly calibrated veto with an arbitrary penalty would not prove improvement.

## Next model specification — not implemented or validated by this experiment

1. Keep the primary goals of 2x by six/twelve calendar months. Add a separately fitted joint path target, 2x before a specified adverse loss, if avoiding a large interim loss is the intended trade-off. Do not silently change a doubling target into low volatility.
2. Add point-in-time revenue and recurring diluted EPS acceleration over successive quarters, trailing profit consistency, operating-cash conversion, accruals, margin change and incremental return on capital. Ratios and exclusions must respect industry accounting; banks/NBFCs need their own variables.
3. Replace announcement counts with verified capacity commissioned, utilisation/ramp timing, executable order revenue over the next twelve months, credible margins and funding. Announced capex or gross order value is not incremental profit. Treat unknown dates, margins and probability of execution as unknown.
4. Add dated valuation and dilution: sector-relative multiples, shares outstanding and plausible earnings/multiple scenarios. A 2x scenario needs a credible earnings-per-share path or re-rating; a low rupee share price is not cheap valuation.
5. Retain relative strength, trend, volume/turnover, liquidity and run-up as candidate predictive variables. Test them in a few registered group ablations, rather than searching numerous RSI cutoffs on the same outcomes.
6. Evaluate precision at a fixed selection budget, coverage/abstention, PR-AUC, baseline lift, calibration, realised endpoint returns and adverse loss. Use chronological matured-label splits and date/issuer-aware uncertainty. Fit calibration on distinct earlier predictions; do not inflate scores to reach a requested probability.
7. Preserve these development results and the existing prospective registry. Use a new registered version for changed weights/targets. More trials on the same data cannot become independent confirmation. The four-year limit constrains the number of independent twelve-month observations.

Quality/profitability research supports testing these inputs, not a guarantee of 2x Indian stock returns: https://www.aqr.com/Insights/Research/Working-Paper/Quality-Minus-Junk . Calibration guidance: https://scikit-learn.org/stable/modules/calibration.html . Chronological evaluation: https://scikit-learn.org/stable/modules/cross_validation.html . Original dated inputs: https://www.nseindia.com/companies-listing/corporate-integrated-filing .

## Reproduction

Use the V12.3 completed evidence archive for the unchanged input scores and financial facts. Run in a fresh output directory:

```sh
PYTHONPATH=src python src/v12_4_profit_first.py \
  --scores /path/results/evaluation/chronological_development_scores.parquet \
  --facts /path/results/financial/dated_primary_financial_facts.parquet \
  --bse /path/results/BSE_current/current_BSE_RESEARCH_scores.parquet \
  --output /path/new-v12-4-results
```

Nine new regression checks cover future filings, quarter gaps, wrong identities, conflicting versions, reporting-basis mixing, negative profits/cash flow and separation of risk/RSI filters.
