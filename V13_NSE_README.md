# V13 NSE research collector and learner

V12.5's 45% risk veto left no fully verified picks. V13 raises the research ceiling to 95%, ranks by 95% upside percentile plus 5% inverse-risk percentile, and lowers the daily liquidity requirement from INR 1 crore to INR 25 lakh. These are user risk preferences, not accuracy improvements. Known financial failures remain excluded; unknown financials are a separate discovery tier. BSE is optional.

The local run uses only 2022-10-09 through 2026-10-09 data: 69,410 monthly/security rows and 670 currently scoreable, official-price-matched NSE shares. Four pass the verified financial gate, while a separate 20-name discovery list needs verification. The complete public request universe has 3,182 names; current coverage is not complete.

Fourteen added market/interaction variables cover 126-session momentum, 252-session momentum excluding the latest 21 sessions, relative strength, volatility contraction, range compression, ATR, closing location, accumulation, turnover growth, illiquidity, breakout/volume, regime and earnings confirmation. Existing dated earnings, cash conversion, accruals, ROCE, debt, interest coverage, share changes and catalyst fields remain. Missing recurring EPS, executable profit/funding and historical sector valuation are not invented.

Four conditional heads produce nested 6/12-month doubling probabilities and coherent loss/joint probabilities. Blend weights are chosen on the first two earlier calibration dates and probability calibration on the final date; forecast outcomes are never used. Exported standardized coefficients are model coefficients, not percentage investment allocations. V10's old target/horizon weights are not transferable to this calendar target.

The development archive was already examined. Six-month verified selections achieved 6/109 doubles (5.50%), and twelve-month selections 4/43 (9.30%), with only four mature twelve-month test dates. Discovery results include unknown outcomes. These results do not establish a reliable 2x predictor or justify accuracy approval. The target remains an adjusted closing total-return proxy net of 0.5% per side, not a guaranteed price-only double or an endpoint return.

## Scheduled operation

`.github/workflows/v13_research_worker.yml` is a minimal default-branch scheduler that checks out this research branch. Its weekday cron is 16:30 UTC / 22:00 IST. It uses the already configured `HF_ARCHIVE_TOKEN` to restore and atomically archive versioned checkpoints in the existing private dataset. The worker refuses a public archive destination and refuses to overwrite a state whose restore fails.

Each run collects the official whole-market bhavcopy, public NSE masters, original dated financial catalogs and bounded original filings, recent announcements/corporate actions, and up to 400 public-universe provider histories. Financial/history stages have ten-minute budgets and rotate failed requests so unavailable names cannot stall coverage. It checkpoints after phases, uploads each immutable object by content hash, and publishes an atomic manifest. Old raw bytes remain archived without being uploaded again in every daily snapshot. It discloses incomplete coverage. The initial files are non-secret public-source research seeds, hash checked before use; pickle loading is restricted to these trusted model checkpoints.

Every scoreable forecast is frozen, including unselected names. Entry is the first verified session open after actual registration; no entry or outcome is backdated to the older source-asof date. It monitors raw doublings, but action-affected raw prices are not confirmed target outcomes. Missing/unfinished follow-up remains UNKNOWN. Full 6/12-month maturity is required.

Monthly refitting archives new learned research heads while the formal paper challenger stays fixed until its outcomes mature. Promotion infrastructure requires at least six mature decision months (one predeclared cohort per month), 500 paired mature observations, 20 positives, 60 selections per model, a date-clustered Brier improvement with a Student-t confidence bound and a two-percentage-point precision gain, plus independent source/action verification. Overlapping horizons still limit inference strength. Independent adjustment reconciliation is currently incomplete; automatic champion promotion stays blocked. This is automated collection and challenger learning, not verified autonomous accuracy improvement. Existing V10/V12 freezes and monitoring tasks are preserved.

Resume development from the git branch and the latest private checkpoint. ChatGPT token resets cannot be detected and do not trigger an immediate coding session; the scheduled worker operates independently, and a separate maintenance task can resume at its next scheduled opportunity.

## Run and research references

```bash
PYTHONPATH=src python -m v13_features --history HISTORY --previous-matrix PREVIOUS.parquet --output features --asof YYYY-MM-DD
PYTHONPATH=src python -m v13_model --matrix features/matrix.parquet --output evaluation --asof YYYY-MM-DD
PYTHONPATH=src python -m v13_runner --state v13_state --seed v13_seed
PYTHONPATH=src python -m unittest discover -s tests
```

The feature families are motivated by William O'Neil's published earnings/relative-strength/price-volume definitions and AQR's quality research; neither proves Indian equities will double within a year. No proprietary investment-firm model is reproduced.

- https://www.williamoneil.com/about-us/legal/oneil-proprietary-rating-and-rankings
- https://www.aqr.com/Insights/Research/Working-Paper/Quality-Minus-Junk
- https://www.nseindia.com/companies-listing/corporate-filings-actions
