# V11.4 standalone — historical source activation and FY2023 backfill audit (2026-10-09)

This is a **research** source/readiness audit. No changes to standalone predictive features, coefficients, earlier 6-month Top-10 predictions, labels or the first-seen archive. No comparisons with other predictive model versions.

## Verified 18-fold source gate

[GitHub Actions source activation audit](https://github.com/ayushverma1988/v9-multibagger-calibration/actions/runs/37873597302) passed its 6 integrity regression tests, scanning 18,569 company-date rows on the original 18 fold dates.

* Confirmed annual 1-year revenue **and** PAT coverage >=70% of a fold universe: **2/18 folds** (2024-12-31 and 2025-12-31), against minimum 12 folds before financial activation.
* Every original user-requested **four-screen family** is undercovered in this combined historical source matrix. The current RSI live screener is separate and is not disproved by RSI being absent here.
* Annual 1y revenue/PAT cannot be substituted for 3y, 5y or 7y user filters; current NSE company P/E and sector proxies cannot be retroactively used as if timestamped at historical 15:30 close.
* A per-fold missing-original-financial-input queue has been exported as a workflow artifact. Missing metric is UNKNOWN; do not convert it to FAIL/PASS or zero.

## FY2023 original NSE archive feasibility

[Historical FY2022/FY2023 NSE pilot](https://github.com/ayushverma1988/v9-multibagger-calibration/actions/runs/37873849356) completed, with 4 new source-identity tests and 6 original financial-context tests passing:

* Original 2023-12-29 universe: 1,276 stocks.
* Valid index pairs of pre-cutoff FY2022 and FY2023 XBRL URLs from consistent reporting mode: 1,116 companies.
* Deterministically sampled 24 company pairs, fetched 48 original XML documents, 0 HTTP/source download errors.
* FY2023 original XBRL revenue+PAT extracts: 24/24 core documents verified.
* FY2022 revenue+PAT extracts: 0/24; so paired one-year growth computations: **0**, not trainable.

[First original-context diagnostic](https://github.com/ayushverma1988/v9-multibagger-calibration/actions/runs/37874008087) and [comparative/context followup](https://github.com/ayushverma1988/v9-multibagger-calibration/actions/runs/37874180054) confirm FY2022 XML values include source INR tags, but its `OneD` and `FourD` fact references are not backed by recognized date/duration `context` metadata in four inspected files. The same sample's FY2023 XBRL contains valid 2023 contexts, but no independently verified prior-year 2022 context in its revenue/PAT tags. **Do not make up fiscal-period dates or treat FY2023 as FY2022.**

## Next authorized research direction

1. Recover a complete authoritative original FY2022 XBRL document with a matching fiscal context, or locate BSE's original company financial-result XBRL for the FY2022 filing with its original submission timestamp. Use the company ISIN or authenticated exchange scrip mapping; do not match a similarly named firm alone.
2. Verify the FY2022 exact fiscal period, INR units, statement mode, publication before the frozen 2023-12-29 15:30 IST cutoff, and original file SHA256. Repeat pair extraction on the same original company sample.
3. Expand to the full 1,116 indexed FY2023 eligible company pairs only after a minimum reproducible pilot demonstrates actual *two-year* core numerics. Keep 3y/5y/7y, ROCE/ROE and the other screener fields UNKNOWN until independently sourced for at least 12 historical folds.
4. Then perform untouched strictly chronological forward leakage/calibration/stability tests of a separately predeclared financial challenger. Do not optimize historical test folds or compare V11.4 against older model versions.

Research promotion state: **NOT APPROVED**. Original standalone predictive model remains unmodified.
