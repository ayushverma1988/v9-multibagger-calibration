# V11.4 strict free NSE three-fiscal-year financial source recovery — October 9, 2026

## User decision and scope
Investigate **only 3 years of annual financial statements** rather than spending time on 5/7-year growth reconstruction. Use original NSE Yearly annual XBRL, historical cut-off, a verified legacy FourD parser, plus FY2025 integrated Q4 FourD **annual YTD** (not quarterly OneD). This is independent V11.4 work and MUST NOT use an older model as a benchmark.

## Completed: December 29, 2023 historical snapshot
- Original frozen 2023 stock universe: **1,276** stocks.
- Three source URLs with consistent mode and publication before 2023 15:30 IST: **965** companies.
- Strict annual FY2021 FY2022 FY2023 revenue and PAT all verified on original NSE XML: **960 companies**. Rejected: **5**. Coverage **75.24%** of original 2023 universe.
- Actual FY2021→FY2023 2-year revenue CAGR calculable on 960; profit CAGR with valid positive prior PAT on **709**.
- Twelve full strict numerical shards passed: [full 2023 source reconstruction run](https://github.com/ayushverma1988/v9-multibagger-calibration/actions/runs/37905297457).
- Reconciliation passed, all 1,276 original stock identities retained, source SHA256 hashes and rejection records verified: [run 37906034342](https://github.com/ayushverma1988/v9-multibagger-calibration/actions/runs/37906034342).
- Data saved in owner-private HF dataset **ayushverma1988/v10-multibagger-archive**, path:
  `v11_4/historical_source_original_NSE/FY2021_FY2022_FY2023_asof_2023_12_29/verified_run_37906034342`

## Completed: December 31, 2025 historical snapshot
- Original frozen 2025 market universe: **1,314** stocks.
- Older NSE annual endpoint **does not index FY2025**. It must be bridged to the *Integrated Filing – Financials* source. See source provenance pilot and [integrated 2025 link audit run](https://github.com/ayushverma1988/v9-multibagger-calibration/actions/runs/37906558270).
- Three original FY2023 FY2024 FY2025 eligible NSE source link sets: **1,029**.
- Strict annual numerics verified in all three years: **1,013** companies. Rejected: **16**. Coverage **77.09%** of original 2025 universe.
- Actual FY2023→FY2025 2-year revenue CAGR calculable for 1,013 companies, profit CAGR on **857** with valid positive profit baselines.
- FY2025 accepted only original Integrated March-Q4 **FourD annual YTD** context, never quarter-only OneD. Twelve shard jobs successful: [run 37907016134](https://github.com/ayushverma1988/v9-multibagger-calibration/actions/runs/37907016134).
- Full exact original-stock and numeric/filing reconciliation, plus permanent private archive: [run 37907820467](https://github.com/ayushverma1988/v9-multibagger-calibration/actions/runs/37907820467).
- Private HF path:
  `v11_4/historical_source_original_NSE/FY2023_FY2024_FY2025_asof_2025_12_31/verified_run_37907820467`

## Frozen 2026-10-08 independent V11.4 stock demo
- Original model Top-10 predictions were previously sealed in private HF `v11_4/forward_observations/2026-10-08/`. All frozen ranks and scores remain unchanged.
- Verified current-relevant FY2023-FY2025 financial source matched **4 out of 10** original Top10 stocks; **6 lack all three verified annual records in this 2025 historical universe**. Their missing financial ratios MUST remain **UNKNOWN**, not false or pass and not estimated.
- Private overlay was successfully generated without rewriting original predictions: [run 37907987027](https://github.com/ayushverma1988/v9-multibagger-calibration/actions/runs/37907987027).
- Owner-private source overlay path:
  `v11_4/forward_3FY_fiscal_overlay/2026-10-08_NSE_FY2023_FY2025/numeric_reconciliation_run_37907820467`

## Quantitative / forward-testing guard
Three consecutive *fiscal-year financial reports* represent **two growth intervals**. The correct compounded 2023→2025 growth exponent is **1/2**. A real **3-year sales CAGR** requires four fiscal-year endpoints (FY2022→FY2025). Likewise do not substitute 2-year averages for the original user-specified 3/5/7-year ROCE/ROE or original four rule groups.

Both reconstructed years are historical source demos and each independently passes a one-fold 70% company completeness bar, but they DO NOT meet the frozen minimum >=12 historic folds with the actual originally requested long-term inputs. The **financial-enhanced V11.4 model is not yet trained**, its probabilities not calibrated on financial data, and there is no verified prospective six-month-double prediction improvement. No past selection/ranking, labels or integrity windows were changed.

## Next necessary steps
1. Repair the 6/10 current-demonstration picks missing complete prior three fiscal-year records using their **actual 2026 symbol identities** and original NSE Integrated/annual sources, including companies newly listed since 2025. Never backfill by company-name guesses.
2. Populate additional original historical chronological folds with the same audited *rolling three-fiscal-year* sources, especially FY2022→FY2024 for December 2024.
3. Redefine a **separate** 3-FY research selector using actual 2-year CAGR, successive YoY sales/PAT and validated market/catalyst inputs. Do not silently overwrite original 3-, 5- or 7-year threshold conditions.
4. Do chronological matured six-month labels, distinct train/calibration/test folds, and explicit false-positive, stability, liquidity and no-lookahead quality checks before claiming calibrated stock-doubling odds.
5. The existing frozen forward NSE Top-10 remains available as a **research demo only**, not an approved trading model.
