# V11.4 standalone COVID shock / mature-history research — 2026-10-08

## Scope and decision

Research only. No previous model predictions or shortlists are used. Original market-derived outcome labels are used exclusively as historical supervised targets and never as features. The standalone V11.4 technical/NSE disclosure research matrix includes 18,569 stock-date rows across 18 frozen dates. Annual verified numeric feature history only covers two year-end folds and remains inactive. Production is **NOT** approved; historical model development influenced design of the four-fold warmup gate, so a newly untouched chronological confirmation is needed.

## Why COVID is an explicitly separate cohort

Two point-in-time six-month event-window regimes were audited: 2020-02-20 to 2020-06-30 (initial market shock/rebound), and 2020-02-20 to 2021-12-31 (broad pandemic). Dates were evaluated by overlap with the actual six-month outcome maturation window, not only stock-selection date. Selection on 2019-06-28 (six-month outcome ending before COVID) is NOT a COVID observation.

Original 14-fold model: 14/140 stocks doubled (10.00%); retraining mean Top10 Jaccard 0.88620 and worst-fold p05 0.11111; failed original 0.60 p05 gate. Filtering test outcomes without retraining: initial market shock overlap excluded => 11/120 (9.17%), mean Jaccard 0.88091, worst p05 0.11111. Broad pandemic overlap excluded => 8/90 (8.89%), mean Jaccard 0.86451, worst p05 0.11111 and only 9 folds.

## Legitimate root cause

For 2019-06-28, available historical mature base training consisted of only 2 folds, 1,974 stock-date rows and 75 positive labels. In two of twelve 95%-retained training resamples, 8 of 10 stocks changed and Jaccard fell to 0.11111. June 2019 is pre-COVID. June 2023 displayed especially narrow Top10/11 probability difference (about 0.000024), making the bottom of the Top10 rank sensitive to small parameter changes.

## Exploratory research variants, four mature training folds + earlier mature calibration fold

Both variants retained the same fixed feature list, model hyperparameters, future-label maturity cutoff, 12 independent perturbation seeds, six-month doubling definition, price/liquidity screens, and unchanged minimum 12-fold / mean Jaccard >=0.80 / worst-fold p05 >=0.60 thresholds.

| Research-only warmup candidate | Clean forward test folds | 2x successes / selections | Precision | Mean Top10 Jaccard | Worst-fold p05 | Research technical gates |
|---|---:|---:|---:|---:|---:|---|
| Four mature prior training folds, all history | 12 | 13 / 120 | 10.83% | 0.89094 | 0.60897 | Pass |
| Four mature prior training folds, exclude initial 2020 COVID-shock-overlapping six-month training labels | 12 | 15 / 120 | 12.50% | 0.92319 | 0.66667 | Pass |

The COVID-outcome training blackout does **not** remove COVID-exposed evaluation folds or reconstruct counterfactual pre-pandemic market technical features. It excludes only overlapping training label windows, so reported 12-fold backtest is still a full-history test. Fourfold warmup reduces the set of evaluated folds by not issuing early selection lists with insufficient prior mature labels. These thresholds were chosen after observing earlier instability. Passing exploratory gates is NOT confirmatory evidence and does not authorize production/promotion.

## Reproducible GitHub Actions evidence

- Original all-fold standalone test (expected hard stability gate failure): https://github.com/ayushverma1988/v9-multibagger-calibration/actions/runs/37767869111
- Correct COVID outcome overlap and original-history non-overwrite audit: https://github.com/ayushverma1988/v9-multibagger-calibration/actions/runs/37783414547
- Two COVID training-outcome blackout experiments, no mature-history guard (both fail worst p05): https://github.com/ayushverma1988/v9-multibagger-calibration/actions/runs/37783626271
- June 2019 and June 2023 twelve-seed ranking/feature cause: https://github.com/ayushverma1988/v9-multibagger-calibration/actions/runs/37784033275
- **Four-fold warmup independent research matrix** with and without initial COVID-shock training exclusion: https://github.com/ayushverma1988/v9-multibagger-calibration/actions/runs/37784322754

## Required next gates

1. Freeze the candidate and provenance/hyperparameters, then test on a newly unseen chronological period; do not optimize old 2019/2023 dates repeatedly.
2. Reverify leakage, calibration monotonicity, probability Brier/PR-AUC, lift, tail drawdown, small/midcap liquidity, and actual stock corporate action identity.
3. Continue historical original-source revenue/PAT/quarterly acceleration financial backfill (older pre-2024 original NSE annual XBRL contexts not yet trustworthy). Inactive until >=12 folds.
4. Improve original exchange-catalyst event materiality (order size against revenue, commissioning evidence, new-product approvals), keeping source timestamps and avoiding synthetic facts.
5. Preserve original all-history metrics, pandemic stress regime and model rejection state; maintain private research archive once validation quality is confirmed.
