# V12.5 expanded evidence experiment

This adds source-checked financial and catalyst variables to the four-year
research model. It does not promote a trading model, replace the original
forecast registry, or establish that any share will double.

## Changes

- Reparse original XBRL with explicit currency, shares and per-share units,
  reporting basis, economic period, original publication clock and ISIN.
- Add sales/PAT acceleration, margin change, annual cash conversion change,
  accruals, interest coverage, ROCE and incremental ROCE proxies, reported
  share-count change, exceptional items, other income and receivables ratios.
- EPS growth requires both periods in the same filing version. Reported EPS
  is never called recurring EPS. Unavailable fields remain unknown and are
  dropped from a training fold if entirely absent.
- Share-count change is not automatically dilution; bonus issues can change
  share count without economic dilution. Face-value changes block this proxy.
- ROCE uses assets less current liabilities. PBT plus finance cost is an EBIT
  proxy; neither is silently treated as a precise operating-profit measure.
- Add prior-only 63-session breakout and liquidity/volume interaction to
  existing relative-strength, turnover, trend and prior-run-up inputs.
- Review catalyst documents against immutable original bytes. Commissioning
  and commercial production are separate. Funding plans are not proof of
  adequate funding. Contract amount is not revenue or expected profit.
- Catalyst absence in a document sample remains unknown, never zero.
- Collect current company PE and sector-peer valuation as a companion. Current
  membership/PE cannot be backfilled into historical training. Required EPS
  growth for a 2x price at a given PE is a scenario, not an earnings forecast.
- Fit a separate joint event: 2x before a 30% loss within 12 months. Do not
  multiply independently estimated upside and downside probabilities.
  Incoherent joint probabilities are flagged and blocked from joint ranking.

## Fixed experiment

The learner remains 70% logistic regression and 30% histogram gradient boosting.
This is a learner blend, not a 70/30 return/risk score. Hyperparameters are fixed.
Training observations must mature before calibration begins; calibration
observations must mature before the scoring date. Imputation and clipping use
training rows only. All four-year results were previously available to development
and are not an untouched test set.

Compare old upside, expanded upside and momentum on the same profit-qualified
pool, selecting at most five per date. Add a separately reported 12-month joint
ranking. Eligibility requires four consecutive positive quarterly PAT values,
positive sales/PAT growth and margin, CFO/PAT >= 0.6, debt/equity between 0 and 1,
price 20–2000, above the 50-session average and prior 63-session gain <= 75%.
Missing financials fail eligibility. A 45% downside-veto count is reported
separately; it does not inflate the upside probability.

BSE scores are out-of-venue transfer diagnostics using NSE-trained heads for
each date. They are not independently validated BSE probabilities. Missing
BSE financials cannot pass the profit gate.

## Reproduction

Install both `requirements_v12_frozen.txt` and `requirements_v11_4_systematic.txt`.
Set `PYTHONPATH=src`. Run these modules with their `--help` arguments in order:

1. `v12_5_expanded_facts.py`: original filing metadata and immutable raw cache.
2. `v12_5_features.py`: dated market panel, old enrichment, base and expanded facts,
   and adjusted daily histories. Run separately for NSE and BSE.
3. `v12_5_collect_companions.py`: whole-market current valuation and a public,
   source-defined catalyst-document sample; no requests based on model picks.
4. `v12_5_catalyst_join.py`: reviewed ledger and original documents; validate
   source identities, timestamps and content hashes before joining.
5. `v12_5_expanded_model.py`: final NSE matrix, old baseline scores and optional
   BSE matrix. Write to a new output directory; the protocol uses exclusive
   creation and records input hashes before fitting.

Run `PYTHONPATH=src python -m unittest discover -s tests`.
Large source archives, security-level scores and reviewed evidence are private
deliverables, not committed here. No paid data or new account is required for
the implemented collection path; endpoint availability and completeness are
not guaranteed.

## Release limitations

No guaranteed-live BSE feed, complete current/historical exchange universe,
independent corporate-action reconciliation, verified recurring EPS, historical
sector-relative PE, complete executable-catalyst profit/funding chain, or mature
unseen outcomes has been established. Repeatedly changing rules on the same
four-year archive cannot create independent validation. Keep the existing
forward registry intact and report future outcomes without backfilling picks.
