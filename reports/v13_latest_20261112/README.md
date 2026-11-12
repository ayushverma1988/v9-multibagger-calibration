# V13 results — run 12 November 2026

The fixed V13 model completed a new research run using **11 November 2026 closing prices**, the latest completed daily session at the run time. It scored **1,622 shares**, with **173 eligible for RSI >70** and **447 eligible for RSI >50**. Each screen selected 25 names.

**Assessment:** these are ranked research candidates. The strongest six-month doubling estimate in either active list is only **13.9%** (NITTAGELA); 17 of the 25 RSI >70 names have a model loss-event estimate above 50%. A 95% ranking score is not a 95% chance of doubling. No six-month live accuracy claim is supported by this run.

## Data and run coverage

| Item | Outcome |
|---|---|
| Data session | 11 November 2026, completed close; no 12 November intraday prices |
| Model calculation completed | 12 November 2026, 09:08:52 IST |
| Tracked universe | 3,218 issuers |
| Histories extended with actual NSE session data | 2,257 |
| Shares passing modelling input checks | 1,622 (50.4% of tracked issuers) |
| Minimum breadth required | 337; passed |
| Scored rows with complete financial inputs | 1,109 |
| Scored rows passing model financial-quality checks | 529 |
| Normal collector's independently reconciled current input | 3 shares; failed breadth guard |
| Latest-session independent vendor validation for recovery | Not claimed |
| New formal frozen forecast registration | None; latest formal cohort remains source 9 November |
| Production ranking / model / archive changes | None |
| Trial weights adopted | No |
| Frozen/protected files checked by this run | 60, all unchanged |
| Extension tests | 6 passed |
| Delivery validation | Passed: artifact hash, prices, ISIN/symbol identities, ranks, thresholds and probability coherence |

The normal history feed had left too few verified current prices for a broad run. This separate research run used actual official NSE 11 November OHLC and volume on retained histories that ended on 9 November and matched that day's official prices and volume. NSE's calendar verifies 10 November as a trading holiday. Histories were extended only over that single next-session interval, with the original adjustment scale retained, stable symbol/ISIN, valid primary prices and volume, and no listed corporate action for the security in the interval. Earlier history values were preserved.

This is a partial-universe, single-primary-source recovery, not a claim that the normal collector is repaired or that the whole NSE universe was independently reconciled. No SME history was extended without a separate verified action feed. Already-current histories could enter through normal checks. Financial and catalyst joins use the restored information available to the model and its dated cutoff; today's run does not make every fundamental field freshly reported or complete.

History exclusions: 22 missing primary identities, 4 histories not ending on the exact anchor session, 7 anchor price mismatches, 19 histories requiring explicit corporate-action adjustments. Eight histories were already current.

## Active weights and interpretation

The active research ranking used **70% six-month doubling-probability percentile + 25% twelve-month joint-success percentile + 5% optional proximity to the 200-day moving average**. It ranks within the eligible pool and is not sorted solely by raw six-month probability.

- **2× / 6m** and **2× / 12m** are the model estimates of a doubling event within those horizons. They are not expected returns.
- **Loss-event risk** is the model's probability of a 30% adverse price move before a 2× closing-price event or the twelve-month deadline. It is not the probability of any loss at month twelve.
- The twelve-month joint-success component also considers avoiding that adverse event; its exact values remain in the JSON results.
- These probabilities remain unvalidated research estimates. Financial checks do not establish a source-supported 2× valuation or a completed catalyst chain.
- “Incomplete” is an explicitly incomplete financial-discovery candidate. “Current available” has some usable current financial evidence; “Seven checks pass” passes the available seven-check screen, not the user's full multi-year fundamental checklist. “Turnaround” is a separate research exception.
- BE-series names are retained as requested. RSI >50 includes stocks above 70; the two lists are overlapping screens.

## Active list: RSI strictly above 70

| Rank | NSE symbol | Series | 11 Nov close ₹ | RSI(14) | 2× / 6m % | 2× / 12m % | 30% loss-event risk % | Financial tier |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 1 | ICICIAMC | EQ | 3113.00 | 74.5 | 7.5 | 13.9 | 28.9 | Incomplete |
| 2 | NITTAGELA | EQ | 1805.90 | 80.7 | 13.9 | 21.9 | 37.6 | Incomplete |
| 3 | MARSONS | EQ | 152.21 | 76.0 | 12.9 | 20.2 | 64.0 | Current available |
| 4 | SUDEEPPHRM | EQ | 1207.70 | 77.9 | 11.0 | 19.0 | 55.6 | Current available |
| 5 | GROWW | EQ | 221.22 | 72.6 | 10.7 | 17.6 | 36.4 | Incomplete |
| 6 | ONELIFECAP | BE | 64.98 | 88.6 | 10.5 | 19.3 | 67.9 | Incomplete |
| 7 | SPECTRUM | EQ | 4443.20 | 82.6 | 9.3 | 17.0 | 64.4 | Current available |
| 8 | LENSKART | EQ | 723.10 | 77.1 | 8.1 | 15.2 | 34.3 | Seven checks pass |
| 9 | MENONBE | EQ | 327.60 | 70.5 | 8.1 | 16.0 | 71.5 | Seven checks pass |
| 10 | MATRIMONY | EQ | 609.90 | 75.9 | 5.6 | 13.7 | 60.4 | Seven checks pass |
| 11 | UYFINCORP | BE | 30.00 | 79.7 | 5.4 | 13.8 | 67.7 | Incomplete |
| 12 | AMANTA | EQ | 196.04 | 70.7 | 6.1 | 14.0 | 69.5 | Seven checks pass |
| 13 | BIL | EQ | 1167.40 | 79.3 | 5.5 | 13.4 | 66.2 | Current available |
| 14 | ROLEXRINGS | EQ | 208.66 | 80.2 | 5.2 | 13.1 | 66.3 | Current available |
| 15 | MANCREDIT | EQ | 264.78 | 84.7 | 5.1 | 12.8 | 62.3 | Incomplete |
| 16 | JAGRAN | EQ | 66.83 | 88.8 | 5.4 | 12.1 | 27.1 | Current available |
| 17 | RUBICON | EQ | 1723.30 | 73.1 | 5.0 | 12.6 | 60.9 | Seven checks pass |
| 18 | GUFICBIO | EQ | 467.35 | 81.1 | 4.9 | 12.6 | 59.8 | Seven checks pass |
| 19 | ALLDIGI | EQ | 832.95 | 83.3 | 4.6 | 11.3 | 29.2 | Seven checks pass |
| 20 | JLHL | EQ | 286.50 | 80.5 | 4.9 | 10.7 | 48.3 | Incomplete |
| 21 | WEL | EQ | 78.29 | 81.0 | 5.3 | 11.0 | 69.6 | Seven checks pass |
| 22 | MOLDTKPAC | EQ | 352.05 | 71.8 | 4.9 | 11.3 | 58.7 | Seven checks pass |
| 23 | SURAKSHA | EQ | 356.45 | 89.3 | 4.8 | 11.9 | 59.6 | Seven checks pass |
| 24 | CSLFINANCE | EQ | 231.08 | 83.1 | 4.6 | 10.1 | 38.2 | Incomplete |
| 25 | TFCILTD | EQ | 166.31 | 74.0 | 4.4 | 12.3 | 51.5 | Incomplete |

Nine of these names have incomplete financial inputs. Across this selection, mean model estimates are 7.0% for doubling within six months, 14.3% within twelve months, and 54.2% for the defined adverse event.

## Active list: RSI strictly above 50

| Rank | NSE symbol | Series | 11 Nov close ₹ | RSI(14) | 2× / 6m % | 2× / 12m % | 30% loss-event risk % | Financial tier |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 1 | TMCV | EQ | 446.05 | 62.6 | 8.3 | 15.3 | 30.3 | Turnaround |
| 2 | ICICIAMC | EQ | 3113.00 | 74.5 | 7.5 | 13.9 | 28.9 | Incomplete |
| 3 | NITTAGELA | EQ | 1805.90 | 80.7 | 13.9 | 21.9 | 37.6 | Incomplete |
| 4 | MARSONS | EQ | 152.21 | 76.0 | 12.9 | 20.2 | 64.0 | Current available |
| 5 | GAUDIUMIVF | EQ | 110.29 | 69.2 | 11.9 | 19.5 | 56.2 | Current available |
| 6 | CORONA | EQ | 2367.70 | 69.3 | 11.4 | 18.9 | 40.7 | Seven checks pass |
| 7 | SUDEEPPHRM | EQ | 1207.70 | 77.9 | 11.0 | 19.0 | 55.6 | Current available |
| 8 | TIMEX | EQ | 751.80 | 66.9 | 11.0 | 19.3 | 57.9 | Incomplete |
| 9 | GROWW | EQ | 221.22 | 72.6 | 10.7 | 17.6 | 36.4 | Incomplete |
| 10 | ONELIFECAP | BE | 64.98 | 88.6 | 10.5 | 19.3 | 67.9 | Incomplete |
| 11 | CEINSYS | EQ | 725.60 | 67.4 | 10.3 | 17.9 | 58.2 | Current available |
| 12 | FRACTAL | EQ | 789.15 | 50.6 | 9.5 | 16.6 | 50.8 | Current available |
| 13 | PNBGILTS | EQ | 77.77 | 68.7 | 6.2 | 11.7 | 54.2 | Incomplete |
| 14 | SPECTRUM | EQ | 4443.20 | 82.6 | 9.3 | 17.0 | 64.4 | Current available |
| 15 | LENSKART | EQ | 723.10 | 77.1 | 8.1 | 15.2 | 34.3 | Seven checks pass |
| 16 | NRL | BE | 171.75 | 60.8 | 8.0 | 16.7 | 51.1 | Seven checks pass |
| 17 | MENONBE | EQ | 327.60 | 70.5 | 8.1 | 16.0 | 71.5 | Seven checks pass |
| 18 | HAWKINCOOK | EQ | 8145.50 | 65.5 | 7.6 | 15.2 | 21.0 | Incomplete |
| 19 | SCPL | BE | 644.90 | 67.3 | 7.5 | 15.6 | 56.7 | Seven checks pass |
| 20 | KOVAI | EQ | 6129.00 | 69.3 | 7.2 | 14.1 | 24.9 | Incomplete |
| 21 | REGAAL | EQ | 93.05 | 62.5 | 7.9 | 14.0 | 65.7 | Current available |
| 22 | ESAFSFB | EQ | 43.43 | 55.6 | 6.5 | 14.7 | 59.1 | Incomplete |
| 23 | DBOL | EQ | 138.94 | 52.9 | 7.0 | 13.6 | 66.8 | Turnaround |
| 24 | MAHASTEEL | BE | 1523.10 | 65.7 | 6.5 | 14.7 | 74.3 | Seven checks pass |
| 25 | MANAKSTEEL | BE | 119.08 | 59.1 | 6.5 | 14.6 | 69.9 | Seven checks pass |

This list contains nine incomplete financial-discovery names and two turnaround names. Its average model estimates are 9.0% for six-month doubling, 16.5% for twelve-month doubling and 51.9% for the defined adverse event.

## Separate weight trial

The trial used **60% six-month probability percentile + 20% twelve-month joint-success percentile + 10% lower estimated loss-risk percentile + 5% price discipline + 5% optional 200-day-average proximity**. It only changes the ranking of the same scored inputs. It does not retrain the model or change any stock's underlying probabilities.

| Screen | Active/trial overlap | Trial additions | Trial removals | Active mean loss-event estimate | Trial mean loss-event estimate |
|---|---:|---|---|---:|---:|
| RSI >70 | 23/25 | ROTO, MBAPL | SURAKSHA, TFCILTD | 54.2% | 53.6% |
| RSI >50 | 17/25 | CAPITALSFB, TAJGVK, TVTODAY, BLAL, JAGRAN, GULPOLY, ALLDIGI, NINSYS | MARSONS, ONELIFECAP, SPECTRUM, MENONBE, REGAAL, DBOL, MAHASTEEL, MANAKSTEEL | 51.9% | 43.3% |

The lower average is a model selection effect, not evidence of lower realised losses. The RSI >50 trial also increases incomplete-financial names from nine to twelve. No claim of improved returns or calibration can be made from this comparison.

### Trial list: RSI >70

| Rank | NSE symbol | Series | 11 Nov close ₹ | RSI(14) | 2× / 6m % | 2× / 12m % | 30% loss-event risk % | Financial tier |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 1 | ICICIAMC | EQ | 3113.00 | 74.5 | 7.5 | 13.9 | 28.9 | Incomplete |
| 2 | GROWW | EQ | 221.22 | 72.6 | 10.7 | 17.6 | 36.4 | Incomplete |
| 3 | NITTAGELA | EQ | 1805.90 | 80.7 | 13.9 | 21.9 | 37.6 | Incomplete |
| 4 | LENSKART | EQ | 723.10 | 77.1 | 8.1 | 15.2 | 34.3 | Seven checks pass |
| 5 | JAGRAN | EQ | 66.83 | 88.8 | 5.4 | 12.1 | 27.1 | Current available |
| 6 | SUDEEPPHRM | EQ | 1207.70 | 77.9 | 11.0 | 19.0 | 55.6 | Current available |
| 7 | ALLDIGI | EQ | 832.95 | 83.3 | 4.6 | 11.3 | 29.2 | Seven checks pass |
| 8 | MARSONS | EQ | 152.21 | 76.0 | 12.9 | 20.2 | 64.0 | Current available |
| 9 | JLHL | EQ | 286.50 | 80.5 | 4.9 | 10.7 | 48.3 | Incomplete |
| 10 | MATRIMONY | EQ | 609.90 | 75.9 | 5.6 | 13.7 | 60.4 | Seven checks pass |
| 11 | SPECTRUM | EQ | 4443.20 | 82.6 | 9.3 | 17.0 | 64.4 | Current available |
| 12 | ONELIFECAP | BE | 64.98 | 88.6 | 10.5 | 19.3 | 67.9 | Incomplete |
| 13 | CSLFINANCE | EQ | 231.08 | 83.1 | 4.6 | 10.1 | 38.2 | Incomplete |
| 14 | MENONBE | EQ | 327.60 | 70.5 | 8.1 | 16.0 | 71.5 | Seven checks pass |
| 15 | MANCREDIT | EQ | 264.78 | 84.7 | 5.1 | 12.8 | 62.3 | Incomplete |
| 16 | RUBICON | EQ | 1723.30 | 73.1 | 5.0 | 12.6 | 60.9 | Seven checks pass |
| 17 | GUFICBIO | EQ | 467.35 | 81.1 | 4.9 | 12.6 | 59.8 | Seven checks pass |
| 18 | AMANTA | EQ | 196.04 | 70.7 | 6.1 | 14.0 | 69.5 | Seven checks pass |
| 19 | BIL | EQ | 1167.40 | 79.3 | 5.5 | 13.4 | 66.2 | Current available |
| 20 | UYFINCORP | BE | 30.00 | 79.7 | 5.4 | 13.8 | 67.7 | Incomplete |
| 21 | ROLEXRINGS | EQ | 208.66 | 80.2 | 5.2 | 13.1 | 66.3 | Current available |
| 22 | ROTO | EQ | 67.24 | 78.1 | 4.5 | 10.2 | 46.6 | Seven checks pass |
| 23 | MOLDTKPAC | EQ | 352.05 | 71.8 | 4.9 | 11.3 | 58.7 | Seven checks pass |
| 24 | WEL | EQ | 78.29 | 81.0 | 5.3 | 11.0 | 69.6 | Seven checks pass |
| 25 | MBAPL | EQ | 192.14 | 79.1 | 4.4 | 12.3 | 47.6 | Incomplete |

### Trial list: RSI >50

| Rank | NSE symbol | Series | 11 Nov close ₹ | RSI(14) | 2× / 6m % | 2× / 12m % | 30% loss-event risk % | Financial tier |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 1 | TMCV | EQ | 446.05 | 62.6 | 8.3 | 15.3 | 30.3 | Turnaround |
| 2 | ICICIAMC | EQ | 3113.00 | 74.5 | 7.5 | 13.9 | 28.9 | Incomplete |
| 3 | KOVAI | EQ | 6129.00 | 69.3 | 7.2 | 14.1 | 24.9 | Incomplete |
| 4 | GROWW | EQ | 221.22 | 72.6 | 10.7 | 17.6 | 36.4 | Incomplete |
| 5 | CORONA | EQ | 2367.70 | 69.3 | 11.4 | 18.9 | 40.7 | Seven checks pass |
| 6 | PNBGILTS | EQ | 77.77 | 68.7 | 6.2 | 11.7 | 54.2 | Incomplete |
| 7 | CAPITALSFB | EQ | 276.65 | 60.2 | 5.2 | 11.7 | 42.1 | Incomplete |
| 8 | LENSKART | EQ | 723.10 | 77.1 | 8.1 | 15.2 | 34.3 | Seven checks pass |
| 9 | NITTAGELA | EQ | 1805.90 | 80.7 | 13.9 | 21.9 | 37.6 | Incomplete |
| 10 | TAJGVK | EQ | 342.35 | 67.7 | 5.3 | 11.1 | 37.9 | Incomplete |
| 11 | TVTODAY | EQ | 105.97 | 55.7 | 5.5 | 11.0 | 45.8 | Turnaround |
| 12 | HAWKINCOOK | EQ | 8145.50 | 65.5 | 7.6 | 15.2 | 21.0 | Incomplete |
| 13 | BLAL | EQ | 165.37 | 64.5 | 5.3 | 10.7 | 34.5 | Incomplete |
| 14 | SUDEEPPHRM | EQ | 1207.70 | 77.9 | 11.0 | 19.0 | 55.6 | Current available |
| 15 | JAGRAN | EQ | 66.83 | 88.8 | 5.4 | 12.1 | 27.1 | Current available |
| 16 | SCPL | BE | 644.90 | 67.3 | 7.5 | 15.6 | 56.7 | Seven checks pass |
| 17 | NRL | BE | 171.75 | 60.8 | 8.0 | 16.7 | 51.1 | Seven checks pass |
| 18 | GAUDIUMIVF | EQ | 110.29 | 69.2 | 11.9 | 19.5 | 56.2 | Current available |
| 19 | GULPOLY | EQ | 169.70 | 59.9 | 5.1 | 11.6 | 57.9 | Incomplete |
| 20 | FRACTAL | EQ | 789.15 | 50.6 | 9.5 | 16.6 | 50.8 | Current available |
| 21 | ALLDIGI | EQ | 832.95 | 83.3 | 4.6 | 11.3 | 29.2 | Seven checks pass |
| 22 | TIMEX | EQ | 751.80 | 66.9 | 11.0 | 19.3 | 57.9 | Incomplete |
| 23 | ESAFSFB | EQ | 43.43 | 55.6 | 6.5 | 14.7 | 59.1 | Incomplete |
| 24 | CEINSYS | EQ | 725.60 | 67.4 | 10.3 | 17.9 | 58.2 | Current available |
| 25 | NINSYS | EQ | 897.35 | 54.5 | 6.2 | 14.5 | 54.6 | Seven checks pass |

## Reproducibility and evidence

- [Completed model run](https://github.com/ayushverma1988/v9-multibagger-calibration/actions/runs/43148531308)
- [Passed delivery verification](https://github.com/ayushverma1988/v9-multibagger-calibration/actions/runs/43150057869)
- [Original run data package](https://github.com/ayushverma1988/v9-multibagger-calibration/actions/runs/43148531308/artifacts/13912504824), 1,958,668 bytes; GitHub retention expires 12 December 2026.
- [NSE 11 November bhavcopy](https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_20261111_F_0000.csv.zip)
- [NSE 9 November bhavcopy](https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_20261109_F_0000.csv.zip)
- [NSE corporate-action catalogue](https://www.nseindia.com/api/corporates-corporateActions?index=equities&from_date=10-11-2026&to_date=11-11-2026), captured 12 November; 21 actions were parsed and affected securities excluded from extension.
- [NSE 2026 trading calendar](https://nsearchives.nseindia.com/content/circulars/CMTR71775.pdf).

| Evidence | SHA-256 / identifier |
|---|---|
| Source code commit | 279f939672cc138e35479dd5ebf8197373c4b9b8 |
| Frozen model | c9208f89c46d043ee1d74e44dee1c85b9721b7a094e40a0fabf2c5506c246571 |
| Restored state manifest | e6bf6da4b8e0c16dab3b4082941909b2bcbf3a83840c2e3bd8f5e4621e8cc0b6 |
| Run artifact ZIP | 33196362aff11a4b385d23d13aad5bb9fb86a22a3c4d9d4d0624becfe3998079 |
| Corporate-action capture | 862d6d097a385aa0704aadbe63b4fc2faffe084be6e70fa317c64616c572ac73 |

The adjacent `latest_RSI70.json`, `latest_RSI50.json`, `run_receipt.json` and `delivery_validation.json` retain exact values and run evidence. Earlier frozen forecasts, registered decisions and the production model were not rewritten.
